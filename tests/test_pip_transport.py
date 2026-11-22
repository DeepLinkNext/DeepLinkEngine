"""Bounded HTTP probes and real binary pip report/source-admission checks."""
from pathlib import Path
import contextlib
import hashlib
import http.server
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmark/one-click-kit/scripts')]
from official import enable_module_imports

enable_module_imports()
import pip_transport as transport
import official_build_assets as authority
import environment_setup as setup


class Transport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def test_real_http_range_probe_checks_size_prefix_and_never_admits_whole(self):
        body = b'public-wheel-prefix' * 100
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(206)
                self.send_header('Content-Range', 'bytes 0-255/' + str(len(body)))
                self.send_header('Content-Length', '256')
                self.end_headers()
                self.wfile.write(body[:256])
            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            pin = dict(url='http://127.0.0.1:' + str(server.server_port), artifact_bytes=len(body),
                       prefix_bytes=256, prefix_sha256=hashlib.sha256(body[:256]).hexdigest())
            result = transport.probe(pin)
            self.assertTrue(result['verified'])
            self.assertFalse(result['whole_artifact_qualified'])
            self.assertFalse(transport.probe(dict(pin, prefix_sha256='0' * 64))['verified'])
            self.assertFalse(transport.probe(dict(pin, artifact_bytes=len(body) + 1))['verified'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_explicit_user_index_is_not_probed_or_replaced(self):
        env = dict(PIP_INDEX_URL='https://user.invalid/simple')
        with patch.object(transport, 'probe', side_effect=AssertionError('User choice wins')):
            args, child, pending = transport.plan(['python', '-m', 'pip', 'install', '--only-binary=:all:', 'sample'], env, self.root)
        self.assertIsNone(pending)
        self.assertEqual(child['PIP_INDEX_URL'], env['PIP_INDEX_URL'])
        self.assertNotIn('--report', args)

    def test_user_pip_config_source_is_honored_without_printing_values(self):
        config = type('Result', (), dict(returncode=0, stdout='["global.index-url"]'))()
        with patch.object(transport.subprocess, 'run', return_value=config), patch.object(transport, 'probe', side_effect=AssertionError('No probe')):
            result, path = transport.selection(sys.executable, {}, self.root)
        self.assertEqual(result['mode'], 'user-configured')
        self.assertEqual(result['explicit_source_keys'], ['global.index-url'])

    def test_verified_mirror_is_child_only_and_adds_an_owned_report(self):
        config = type('Result', (), dict(returncode=0, stdout='[]'))()
        env = {}
        with patch.object(transport.subprocess, 'run', return_value=config), patch.object(transport, 'probe', return_value=dict(verified=True)):
            args, child, pending = transport.plan(['python', '-m', 'pip', 'install', '--only-binary=:all:', 'sample==1'], env, self.root)
        self.assertEqual(child['PIP_INDEX_URL'], transport.INDEX)
        self.assertNotIn('PIP_INDEX_URL', env)
        self.assertIn('--report', args)
        self.assertEqual(Path(pending['report']).parent, self.root)

    def test_slow_or_invalid_probe_keeps_original_index(self):
        config = type('Result', (), dict(returncode=0, stdout='[]'))()
        with patch.object(transport.subprocess, 'run', return_value=config), patch.object(transport, 'probe', return_value=dict(verified=False, error='bounded failure')):
            _, child, pending = transport.plan(['python', '-m', 'pip', 'install', '--only-binary=:all:', 'sample'], {}, self.root)
        self.assertNotIn('PIP_INDEX_URL', child)
        self.assertIsNone(pending)

    def report(self, digest):
        item = dict(metadata=dict(name='sample', version='1.0'), download_info=dict(url='https://mirror.invalid/sample-1.0-py3-none-any.whl', archive_info=dict(hashes=dict(sha256=digest))))
        path = self.root / 'report.json'
        path.write_text(json.dumps(dict(install=[item])))
        return dict(report=str(path), transport='fixture'), item

    def test_original_pypi_hash_is_required_and_mismatch_is_rejected(self):
        pending, _ = self.report('a' * 64)
        def metadata(digest):
            return json.dumps(dict(urls=[dict(filename='sample-1.0-py3-none-any.whl', size=12, url='https://files.pythonhosted.org/sample.whl', digests=dict(sha256=digest))])).encode()
        with patch.object(authority, '_read', return_value=metadata('a' * 64)):
            result = transport.audit(pending, {})
        self.assertEqual(result['artifacts'], 1)
        with patch.object(authority, '_read', return_value=metadata('b' * 64)), self.assertRaisesRegex(ValueError, 'absent from the official'):
            transport.audit(pending, {})
        pending, item = self.report('a' * 64)
        item['download_info']['archive_info'] = {}
        Path(pending['report']).write_text(json.dumps(dict(install=[item])))
        with self.assertRaisesRegex(ValueError, 'actual whole SHA'):
            transport.audit(pending, {})

    def test_fixed_vendor_keeps_its_original_local_hash_without_public_resolution(self):
        pending, _ = self.report('a' * 64)
        row = dict(filename='sample-1.0-py3-none-any.whl', sha256='a' * 64, bytes=12, url='https://original.invalid/sample.whl')
        with patch.object(authority, '_read', side_effect=AssertionError('Do not re-resolve vendor')):
            self.assertEqual(transport.audit(pending, dict(environment_sources=dict(sample=row)))['artifacts'], 1)

    @unittest.skipIf(sys.platform == 'win32', 'Owned child runner uses Linux process groups')
    def test_actual_public_environment_command_audits_real_pip_body_before_qualification(self):
        wheel = self.root / 'deeplink_transport_fixture-1.0-py3-none-any.whl'
        prefix = 'deeplink_transport_fixture-1.0.dist-info/'
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr(prefix + 'METADATA', 'Metadata-Version: 2.1\nName: deeplink-transport-fixture\nVersion: 1.0\n')
            archive.writestr(prefix + 'WHEEL', 'Wheel-Version: 1.0\nGenerator: cpu-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
            archive.writestr(prefix + 'RECORD', '')
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        env = dict(os.environ)
        for key in transport.SOURCE_OPTIONS:
            env.pop(key, None)
        env['PIP_REQUIRE_VIRTUALENV'] = 'false'
        chosen = dict(mode='verified-public-mirror', automatic_index=transport.INDEX)
        metadata = json.dumps(dict(urls=[dict(filename=wheel.name, size=wheel.stat().st_size, url='https://files.pythonhosted.org/' + wheel.name, digests=dict(sha256=digest))])).encode()
        report = dict(steps=[])
        args = [sys.executable, '-I', '-B', '-m', 'pip', 'install', '--only-binary=:all:', '--no-deps', '--no-index', '--target', str(self.root / 'installed'), str(wheel)]
        with patch.object(transport, 'selection', return_value=(chosen, self.root / 'selection.json')), patch.object(authority, '_read', return_value=metadata), contextlib.redirect_stdout(io.StringIO()):
            setup.command(args, env, self.root, 'actual-binary-pip', report, self.root / 'environment.json')
        self.assertEqual(report['steps'][0]['status'], 'completed')
        self.assertEqual(report['steps'][0]['publisher_admission']['artifacts'], 1)
        raw = json.loads(Path(report['steps'][0]['pip_source_audit']['report']).read_bytes())
        self.assertEqual(raw['install'][0]['download_info']['archive_info']['hashes']['sha256'], digest)
        self.assertNotIn('torch', sys.modules)
        invalid = json.dumps(dict(urls=[dict(filename=wheel.name, size=wheel.stat().st_size,
                                            url='https://files.pythonhosted.org/' + wheel.name,
                                            digests=dict(sha256='0' * 64))])).encode()
        failed = dict(steps=[])
        args[args.index('--target') + 1] = str(self.root / 'unqualified')
        with patch.object(transport, 'selection', return_value=(chosen, self.root / 'selection.json')), patch.object(authority, '_read', return_value=invalid), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'absent from the official'):
            setup.command(args, env, self.root, 'bad-publisher-pip', failed, self.root / 'failed-environment.json')
        self.assertEqual(failed['steps'][0]['status'], 'failed')
        self.assertNotIn('publisher_admission', failed['steps'][0])


if __name__ == '__main__':
    unittest.main()
