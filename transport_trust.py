"""Carry a verified public CA bundle into the private execution namespace."""
import hashlib
import json
import os
from pathlib import Path
import ssl
import time
import uuid


def _public_bundle(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    if not path.is_file() or not 0 < before.st_size <= 8 * 1024**2:
        raise ValueError('Expected a bounded public CA bundle: ' + str(path))
    raw = path.read_bytes()
    if b'PRIVATE KEY' in raw:
        raise ValueError('A transport CA bundle must not contain a private key')
    context = ssl.create_default_context(cafile=str(path))
    if not context.cert_store_stats()['x509_ca']:
        raise ValueError('The public CA bundle contains no trusted CA certificates')
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Public CA bundle changed while being read')
    return path, raw


def configure(cache, sdk=None, *, env=None):
    """Set child-process TLS trust; never change a system or installed SDK file."""
    env = os.environ if env is None else env
    explicit = env.get('SSL_CERT_FILE') or env.get('REQUESTS_CA_BUNDLE') or env.get('PIP_CERT')
    source = None
    kind = 'explicit-public-ca'
    if explicit:
        source, raw = _public_bundle(explicit)
    elif sdk is not None:
        sdk = Path(sdk).resolve()
        inventory = sdk / 'installed-files.json'
        if inventory.is_file():
            files = json.loads(inventory.read_bytes())
            for relative in ('certifi/cacert.pem', 'pip/_vendor/certifi/cacert.pem'):
                for candidate in sorted((sdk / 'rootfs/usr/local').glob('python3.12*/lib/python3.12/site-packages/' + relative)):
                    name = candidate.relative_to(sdk).as_posix()
                    expected = files.get(name, {})
                    source, raw = _public_bundle(candidate)
                    if (not source.is_relative_to(sdk) or expected.get('bytes') != len(raw)
                            or expected.get('sha256') != hashlib.sha256(raw).hexdigest()):
                        raise ValueError('The SDK public CA bundle differs from installed-files: ' + name)
                    kind = 'installed-official-sdk-public-ca'
                    break
                if source is not None:
                    break
    if source is None:
        source = ssl.get_default_verify_paths().cafile
        if not source:
            raise ValueError('No verified public CA bundle is available before private execution')
        source, raw = _public_bundle(source)
        kind = 'host-default-public-ca'
    digest = hashlib.sha256(raw).hexdigest()
    folder = Path(cache).resolve() / 'transport-trust'
    if folder.is_symlink():
        raise ValueError('Transport trust cache must not be a symlink')
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (digest + '.pem')
    receipt = folder / (digest + '.json')
    if target.is_symlink() or receipt.is_symlink():
        raise ValueError('Transport trust cache files must not be symlinks')
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError('Cached public CA bundle changed; preserve ' + str(target))
    else:
        temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
        temporary.write_bytes(raw)
        temporary.chmod(0o444)
        temporary.replace(target)
    record = dict(schema='deeplinkengine-transport-public-ca/v1', source=str(source), source_kind=kind,
                  bundle=str(target), bytes=len(raw), sha256=digest, selected_at=time.time(),
                  tls_verification_enabled=True, system_trust_modified=False, sdk_modified=False)
    if receipt.exists():
        existing = json.loads(receipt.read_bytes())
        if existing.get('sha256') != digest or existing.get('bytes') != len(raw):
            raise ValueError('Public CA provenance differs; preserve ' + str(receipt))
    else:
        temporary = receipt.with_name(receipt.name + '.' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        temporary.replace(receipt)
    for key in ('SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE', 'PIP_CERT', 'CURL_CA_BUNDLE'):
        env[key] = str(target)
    return record
