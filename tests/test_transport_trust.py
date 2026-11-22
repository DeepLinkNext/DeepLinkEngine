"""Real CA parsing and public entry handoff checks; no model or accelerator."""
from pathlib import Path
import hashlib
import json
import os
import ssl
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from official import enable_module_imports

enable_module_imports()
from engine import transport_trust as trust


class Trust(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        certificates = ssl.create_default_context().get_ca_certs(binary_form=True)
        if not certificates:
            self.skipTest('The CPU test host has no trusted public CA certificate')
        self.raw = ssl.DER_cert_to_PEM_cert(certificates[0]).encode()
        self.bundle = self.root / 'trusted.pem'
        self.bundle.write_bytes(self.raw)

    def test_explicit_ca_is_validated_copied_and_inherited_without_mutating_source(self):
        env = dict(SSL_CERT_FILE=str(self.bundle.resolve()))
        result = trust.configure(self.root / 'cache', env=env)
        self.assertEqual(result['sha256'], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(self.bundle.read_bytes(), self.raw)
        self.assertTrue(result['tls_verification_enabled'])
        for name in ('SSL_CERT_FILE', 'PIP_CERT', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE'):
            self.assertEqual(Path(env[name]).read_bytes(), self.raw)
        self.assertGreater(ssl.create_default_context().cert_store_stats()['x509_ca'], 0)
        # Re-entering with the inherited cache path preserves the original provenance.
        trust.configure(self.root / 'cache', env=env)
        receipt = Path(result['bundle']).with_suffix('.json')
        self.assertEqual(json.loads(receipt.read_bytes())['source'], str(self.bundle.resolve()))

    def test_official_bundle_must_match_installed_files(self):
        sdk = self.root / 'sdk'
        name = 'rootfs/usr/local/python3.12.13/lib/python3.12/site-packages/certifi/cacert.pem'
        path = sdk / name
        path.parent.mkdir(parents=True)
        path.write_bytes(self.raw)
        inventory = {name: dict(bytes=len(self.raw), sha256=hashlib.sha256(self.raw).hexdigest())}
        (sdk / 'installed-files.json').write_text(json.dumps(inventory))
        env = {}
        result = trust.configure(self.root / 'cache', sdk, env=env)
        self.assertEqual(result['source_kind'], 'installed-official-sdk-public-ca')
        inventory[name]['sha256'] = '0' * 64
        (sdk / 'installed-files.json').write_text(json.dumps(inventory))
        with self.assertRaisesRegex(ValueError, 'differs from installed-files'):
            trust.configure(self.root / 'other-cache', sdk, env={})

    def test_private_key_and_invalid_explicit_ca_are_rejected_without_fallback(self):
        for raw in (b'-----BEGIN PRIVATE KEY-----\n', b'not a CA bundle'):
            self.bundle.write_bytes(raw)
            with self.assertRaises((ValueError, ssl.SSLError)):
                trust.configure(self.root / 'cache', env=dict(SSL_CERT_FILE=str(self.bundle.resolve())))
            self.assertEqual(self.bundle.read_bytes(), raw)
        self.assertFalse((self.root / 'cache').exists())

    def test_changed_cached_bundle_is_preserved_and_rejected(self):
        result = trust.configure(self.root / 'cache', env=dict(SSL_CERT_FILE=str(self.bundle.resolve())))
        target = Path(result['bundle'])
        target.chmod(0o644)
        target.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Cached public CA bundle changed'):
            trust.configure(self.root / 'cache', env=dict(SSL_CERT_FILE=str(self.bundle.resolve())))
        self.assertEqual(target.read_bytes(), b'changed')

    def test_host_default_fallback_requires_an_existing_valid_bundle(self):
        with patch.object(ssl, 'get_default_verify_paths', return_value=type('Defaults', (), dict(cafile=str(self.bundle.resolve())))()):
            result = trust.configure(self.root / 'cache', env={})
        self.assertEqual(result['source_kind'], 'host-default-public-ca')
        with patch.object(ssl, 'get_default_verify_paths', return_value=type('Defaults', (), dict(cafile=None))()), patch.object(trust, 'SYSTEM_CA_BUNDLES', ()):
            with self.assertRaisesRegex(ValueError, 'No verified public CA'):
                trust.configure(self.root / 'missing', env={})

    def test_system_bundle_when_openssl_default_is_missing(self):
        defaults = ssl.DefaultVerifyPaths(None, None, 'SSL_CERT_FILE', '/absent/ca.pem',
                                         'SSL_CERT_DIR', '/absent/certs')
        with patch.object(ssl, 'get_default_verify_paths', return_value=defaults), patch.object(
            trust, 'SYSTEM_CA_BUNDLES', (self.root / 'absent.pem', self.bundle)
        ):
            result = trust.configure(self.root / 'cache', env={})
        self.assertEqual(result['source_kind'], 'host-system-public-ca')
        self.assertEqual(Path(result['bundle']).read_bytes(), self.raw)

    def test_public_protocol_entry_selects_trust_before_private_handoff(self):
        import run_all
        model = self.root / 'model'
        model.mkdir()
        config = self.root / 'evaluation.json'
        config.write_text(json.dumps(dict(model=str(model), cache=str(self.root / 'cache'), out=str(self.root / 'out'), physical_devices=[0, 1], environment=dict(official_sdk=str(self.root / 'sdk')))))
        events = []
        class Handoff(Exception):
            pass
        def configure(*args):
            events.append(('trust', args))
        def enter(*args):
            events.append(('enter', args))
            raise Handoff()
        with patch('official_sdk_bytecode.verify_or_repair', return_value=None), patch.object(sys, 'argv', ['run_all.py', '--config', str(config), '--prepare-official-protocol-only']), patch('official_execution.needed', return_value=True), patch('engine.transport_trust.configure', side_effect=configure), patch('official_execution.maybe_enter_config', side_effect=enter), self.assertRaises(Handoff):
            run_all.main()
        self.assertEqual([e[0] for e in events], ['trust', 'enter'])
        self.assertFalse((self.root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
