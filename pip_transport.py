"""Select a bounded public pip transport and audit installed wheel provenance."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
INDEX = 'https://pypi.tuna.tsinghua.edu.cn/simple/'
PROBE = dict(
    url='https://pypi.tuna.tsinghua.edu.cn/packages/79/f8/97f10e6755e2a7d027ca783f63044d5b1bc1ae7acb12afe6a9b4286eac17/numpy-1.26.4-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl',
    publisher_url='https://pypi.org/pypi/numpy/1.26.4/json',
    artifact_bytes=13925109,
    artifact_sha256='9fad7dcb1aac3c7f0584a5a8133e3a43eeb2fe127f47e3632d43d677c66c102b',
    prefix_bytes=262144,
    prefix_sha256='9227ebbc59cfbe8e6df201084f0f0774a5f46745dc979dabc03914099e779706')
SOURCE_OPTIONS = ('PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL', 'PIP_NO_INDEX', 'PIP_FIND_LINKS', 'PIP_CONFIG_FILE')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def probe(pin=PROBE):
    """This public prefix measures transport only; no whole wheel is admitted."""
    began = time.monotonic()
    result = dict(pin, whole_artifact_qualified=False)
    try:
        count = pin['prefix_bytes']
        request = urllib.request.Request(pin['url'], headers={'Range': 'bytes=0-' + str(count - 1)})
        with urllib.request.urlopen(request, timeout=3) as response:
            if (response.status != 206 or response.headers.get('Content-Range') != 'bytes 0-' + str(count - 1) + '/' + str(pin['artifact_bytes'])
                    or response.headers.get('Content-Length') != str(count)):
                raise ValueError('Public transport probe requires its exact bounded range')
            blocks = []
            size = 0
            while size < count:
                if time.monotonic() - began > 5:
                    raise TimeoutError('Public transport probe exceeded five seconds')
                block = getattr(response, 'read1', response.read)(min(64 * 1024, count - size))
                if not block:
                    raise ValueError('Public transport probe was truncated')
                blocks.append(block)
                size += len(block)
        raw = b''.join(blocks)
        if hashlib.sha256(raw).hexdigest() != pin['prefix_sha256']:
            raise ValueError('Public transport prefix differs from the original wheel')
        result.update(verified=True, bytes=size)
    except (OSError, ValueError) as error:
        result.update(verified=False, error=type(error).__name__ + ': ' + str(error))
    result['elapsed_seconds'] = time.monotonic() - began
    return result


def selection(python, env, folder):
    folder = Path(folder)
    explicit = [key for key in SOURCE_OPTIONS if env.get(key)]
    if not explicit:
        code = "import json;from pip._internal.configuration import Configuration;c=Configuration(isolated=False);c.load();print(json.dumps(list(dict(c.items()))))"
        checked = subprocess.run([str(python), '-I', '-B', '-c', code], env=env, capture_output=True, text=True, timeout=20)
        if checked.returncode:
            explicit = ['unreadable-pip-configuration']
        else:
            keys = json.loads(checked.stdout)
            explicit = [key for key in keys if key.endswith(('.index-url', '.extra-index-url', '.no-index', '.find-links'))]
    binding = dict(helper_sha256=digest(__file__), authority_helper_sha256=digest(ROOT / 'official_build_assets.py'))
    inherited = env.get('DEEPLINKENGINE_PIP_TRANSPORT')
    if inherited and not explicit:
        value = json.loads(Path(inherited).read_bytes())
        if all(value.get(key) == expected for key, expected in binding.items()):
            return value, Path(inherited)
    path = folder / 'pip-transport.json'
    value = dict(schema='deeplinkengine-private-pip-transport/v1', **binding, selected_at=time.time(),
                 explicit_source_keys=explicit, tls_verification_enabled=True, model_started=False)
    if explicit:
        value.update(mode='user-configured', automatic_index=None)
    else:
        result = probe()
        value.update(probe=result, mode='verified-public-mirror' if result['verified'] else 'original-index',
                     automatic_index=INDEX if result['verified'] else None)
    save(path, value)
    env['DEEPLINKENGINE_PIP_TRANSPORT'] = str(path)
    return value, path


def primary_sources(lock):
    rows = [*lock.get('environment_sources', {}).values(), *lock.get('runtime_compatibility_sources', {}).values(), *lock.get('assets', [])]
    return {row.get('filename', row.get('name')): row for row in rows if row.get('filename', row.get('name'))}


def plan(args, env, folder):
    args = list(map(str, args))
    if not any(args[index:index + 3] == ['-m', 'pip', 'install'] for index in range(len(args))) or '--only-binary=:all:' not in args:
        return args, env, None
    choice, receipt = selection(args[0], env, folder)
    if choice['mode'] != 'verified-public-mirror':
        return args, env, None
    if '--report' in args:
        raise ValueError('Private pip source audit requires its own immutable report')
    report = Path(folder) / ('pip-report-' + uuid.uuid4().hex + '.json')
    child = dict(env, PIP_INDEX_URL=choice['automatic_index'])
    return [*args, '--report', str(report)], child, dict(report=str(report), transport=str(receipt))


def audit(pending, lock):
    """pip checks the downloaded body hash; bind it to the original publisher."""
    from official_build_assets import authority
    path = Path(pending['report'])
    report = json.loads(path.read_bytes())
    primary = primary_sources(lock)
    rows = []
    for item in report['install']:
        actual = item['download_info'].get('archive_info', {}).get('hashes', {}).get('sha256')
        if not re.fullmatch(r'[0-9a-f]{64}', actual or ''):
            raise ValueError('Installed pip wheel report lacks its actual whole SHA-256')
        row = authority(item, primary)
        if row['sha256'] != actual:
            raise ValueError('Installed wheel SHA differs from its original publisher')
        filename = urllib.parse.unquote(urllib.parse.urlsplit(item['download_info']['url']).path.rsplit('/', 1)[-1])
        if filename not in primary and not row.get('publisher_metadata_url', '').startswith('https://pypi.org/pypi/'):
            raise ValueError('Ordinary runtime dependency requires the official PyPI publisher SHA')
        rows.append(dict(row, package=item['metadata']['name'], version=item['metadata']['version'], actual_transport_url=item['download_info']['url']))
    result = dict(schema='deeplinkengine-private-pip-source-audit/v1', pip_report=str(path), pip_report_sha256=digest(path),
                  transport=pending['transport'], artifacts=rows, all_publisher_hashes_verified=True,
                  fixed_vendor_versions_changed=False, runtime_import_started=False, model_started=False)
    target = path.with_suffix('.source-audit.json')
    save(target, result)
    return dict(path=str(target), sha256=digest(target), artifacts=len(rows))


def run_stage(args, *, cache, phase, env=None, **kwargs):
    """Use the existing child owner and preserve a separate source audit."""
    from release_selection import runtime_lock
    from child_runner import run_stage as owned
    folder = Path(cache) / 'private-pip-audits' / (phase + '-' + uuid.uuid4().hex)
    folder.mkdir(parents=True)
    environment = dict(os.environ if env is None else env)
    command, child, pending = plan(args, environment, folder)
    record = dict(status='running', started=time.time(), model_started=False, source_audit=pending)
    try:
        result = owned(command, env=child, **kwargs)
        record['returncode'] = result.returncode
        if result.returncode == 0 and pending:
            record['publisher_admission'] = audit(pending, runtime_lock(ROOT))
        record['status'] = 'completed' if result.returncode == 0 else 'failed'
        return result
    except BaseException as error:
        record.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        record['ended'] = time.time()
        save(folder / 'result.json', record)
