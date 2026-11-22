"""Resolve one official stable engine and retain its identity across all stages."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SELECTION = 'DEEPLINKENGINE_RELEASE_SELECTION'
SELECTION_SHA = SELECTION + '_SHA256'
PROJECT = 'vllm-hust-engine'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(url):
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read(16 * 1024**2))


def stable_sdist(index, version=None):
    from packaging.version import Version
    candidates = []
    for text, rows in index['releases'].items():
        parsed = Version(text)
        if (parsed.is_prerelease or parsed.is_devrelease or parsed.local
                or not re.fullmatch(r'\d{4}\.\d{1,2}\.\d{1,2}\.\d+(?:\.post\d+)?', text)
                or (version is not None and text != version)):
            continue
        sdists = [r for r in rows if r.get('packagetype') == 'sdist' and not r.get('yanked')]
        if len(sdists) == 1:
            candidates.append((parsed, text, sdists[0]))
    if not candidates:
        raise ValueError('No eligible official stable engine sdist')
    _, text, row = max(candidates, key=lambda x: x[0])
    filename = row['filename']
    if (filename != 'vllm_hust_engine-' + text + '.tar.gz'
            or not row['url'].startswith('https://files.pythonhosted.org/')
            or not re.fullmatch('[0-9a-f]{64}', row['digests']['sha256'])
            or type(row['size']) is not int or not 0 < row['size'] < 1024**3):
        raise ValueError('Invalid official PyPI source identity')
    return text, dict(name=filename, filename=filename, url=row['url'],
                      bytes=row['size'], sha256=row['digests']['sha256'])


def metadata(archive, version):
    prefix = 'vllm_hust_engine-' + version
    wanted = prefix + '/src/vllm_hust_engine/payload/frozen.json'
    result = None
    seen = set()
    with tarfile.open(archive) as source:
        for member in source:
            path = PurePosixPath(member.name)
            if (not (member.isfile() or member.isdir()) or path.is_absolute() or '..' in path.parts
                    or not path.parts or path.parts[0] != prefix or member.name in seen):
                raise ValueError('Unsafe or duplicate sdist member')
            seen.add(member.name)
            if member.isdir():
                continue
            if member.name == wanted:
                if member.size > 1024**2:
                    raise ValueError('Oversized frozen metadata')
                result = json.load(source.extractfile(member))
    if result is None or result.get('version') != version:
        raise ValueError('sdist/frozen engine version differs')
    parts = version.split('.post')[0].split('.')
    release = f'full-answer-b1-{int(parts[0]):04d}{int(parts[1]):02d}{int(parts[2]):02d}-r{int(parts[3])}'
    if result.get('release') != release:
        raise ValueError('Frozen release does not match the selected stable version')
    return result


def runtime_artifacts(frozen):
    public = frozen.get('public_runtime', {})
    release = frozen['release']
    tag = 'engine-wheels-' + release.rsplit('-', 1)[1] + '-v1'
    if (public.get('schema') != 'vllm-hust-public-runtime-artifacts/v1'
            or public.get('release') != release
            or public.get('mirror_repository') != 'vLLM-HUST/vllm-hust-website'
            or public.get('mirror_release') != tag):
        raise ValueError('Selected engine lacks the supported public runtime artifact contract')
    result, names = [], set()
    for row in public.get('artifacts', []):
        name = row['name']
        kind = name.split('-', 1)[0]
        if (kind not in ('vllm', 'vllm_ascend', 'vllm_hust_runtime') or kind in names
                or Path(name).name != name or not name.endswith('-cp312-cp312-linux_aarch64.whl')
                or not re.fullmatch('[0-9a-f]{64}', row['sha256'])
                or type(row['bytes']) is not int or not 0 < row['bytes'] < 1024**3
                or row['url'] != 'https://github.com/vLLM-HUST/vllm-hust-website/releases/download/' + tag + '/' + __import__('urllib.parse', fromlist=['quote']).quote(name)):
            raise ValueError('Invalid frozen public runtime source')
        names.add(kind)
        result.append(dict(row, filename=name))
    if len(result) != 3:
        raise ValueError('Expected all three frozen runtime wheels')
    if frozen.get('native_abi') != dict(python='cp312', platform='linux_aarch64', cann='9.1.0', hardware='TP2 Ascend910B2'):
        raise ValueError('Selected stable engine requires a new vendor environment adapter')
    # API asset URLs are usable from hosts where github.com release redirects
    # time out. Resolve them anonymously, then bind name, size and digest to
    # the producer's frozen package identity before using either route.
    base = 'https://api.github.com/repos/vLLM-HUST/vllm-hust-website/'
    try:
        release_index = read_json(base + 'releases/tags/' + tag)
    except OSError:
        # Public API rate limits must not disable the package's already pinned
        # browser route. Hash/size validation remains mandatory on its bytes.
        return result
    if release_index.get('tag_name') != tag:
        raise ValueError('Public runtime mirror release differs')
    assets = release_index.get('assets', [])
    for record in result:
        matches = [a for a in assets if a.get('name') == record['name']]
        if len(matches) != 1:
            raise ValueError('Public runtime mirror lacks a unique frozen asset')
        asset = matches[0]
        if (type(asset.get('id')) is not int or asset['id'] <= 0
                or asset.get('size') != record['bytes']
                or asset.get('digest') != 'sha256:' + record['sha256']
                or asset.get('browser_download_url') != record['url']):
            raise ValueError('Public runtime mirror asset identity differs')
        record['api_url'] = base + 'releases/assets/' + str(asset['id'])
    return result


def active():
    name = os.environ.get(SELECTION)
    if not name:
        return None
    path = Path(name)
    raw = path.read_bytes()
    if path.is_symlink() or digest(raw) != os.environ.get(SELECTION_SHA):
        raise ValueError('Engine selection changed across the interpreter/stage handoff')
    record = json.loads(raw)
    if record.get('schema') != 'deeplinkengine-stable-selection/v1':
        raise ValueError('Unknown stable selection schema')
    return record


def bind(path):
    path = Path(path).resolve()
    if path.is_symlink():
        raise ValueError('Engine selection cannot be a symlink')
    os.environ[SELECTION] = str(path)
    os.environ[SELECTION_SHA] = digest(path.read_bytes())
    return active()


def select(cache, *, version=None):
    prior = active()
    if prior is not None:
        if version is not None and prior['version'] != version:
            raise ValueError('Requested version differs from the already selected engine')
        return prior
    cache = Path(cache).resolve()
    text, sdist = stable_sdist(read_json('https://pypi.org/pypi/' + PROJECT + '/json'), version)
    from engine.environment_setup import cached_wheel
    archive = cached_wheel(sdist, [], cache / 'wheels')
    frozen = metadata(archive, text)
    old = json.loads((ROOT / 'engine/r8-lock.json').read_bytes())
    legacy = text == old['public_distribution'].split('==')[1]
    if legacy:
        matches = [r for r in old['assets'] if r['name'] == sdist['name']]
        if len(matches) != 1 or matches[0]['sha256'] != sdist['sha256']:
            raise ValueError('Legacy source differs from its frozen lock')
        artifacts = None
        client = None
    else:
        artifacts = runtime_artifacts(frozen)
        for name in ('torch','torch-npu','transformers'):
            if frozen['public_runtime']['versions'][name] != old['required_python_packages'][name]:
                raise ValueError('Selected stable engine requires new vendor prerequisites: '+name)
        from benchmark.evaluation_client import prepare
        client = prepare(cache, archive, frozen)
    record = dict(schema='deeplinkengine-stable-selection/v1', version=text, release=frozen['release'],
                  engine_profile=frozen['release'].rsplit('-', 1)[1], source='https://pypi.org/pypi/' + PROJECT + '/json',
                  sdist=sdist, runtime_artifacts=artifacts, frozen=frozen, client=client, legacy=legacy)
    folder = cache / 'release-selections'
    folder.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(record, sort_keys=True, indent=2) + '\n').encode()
    path = folder / (text + '-' + digest(raw) + '.json')
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Existing release selection was modified')
    else:
        with path.open('xb') as stream:
            stream.write(raw)
    return bind(path)


def save(out):
    record = active()
    if record is None:
        return
    path = Path(out) / 'engine-selection.json'
    raw = Path(os.environ[SELECTION]).read_bytes()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('Output already belongs to a different engine release')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('xb') as stream:
            stream.write(raw)


def runtime_lock(root=None):
    old = json.loads(((Path(root) if root else ROOT) / 'engine/r8-lock.json').read_bytes())
    record = active()
    if record and not record['legacy']:
        old.update(upstream_release=record['release'], public_distribution=PROJECT + '==' + record['version'])
    return old


def kit_path():
    client = selected_client()
    if client:
        return Path(client['path'])
    return ROOT / 'benchmark/one-click-kit'


def profile():
    return active()['engine_profile'] if active() else 'r8'


METADATA_CLIENT = 'DEEPLINKENGINE_METADATA_ASSESSMENT_CLIENT'


def selected_client():
    record = active()
    if record is None:
        return None
    raw = os.environ.get(METADATA_CLIENT)
    if raw:
        value = json.loads(raw)
        if value.get('selection_sha256') != os.environ[SELECTION_SHA] or value['client']['release'] != record['release']:
            raise ValueError('CPU metadata assessment client belongs to a different engine selection')
        from benchmark.evaluation_client import verify
        verify(value['client'])
        return value['client']
    return record.get('client')


def ensure_metadata_client(cache):
    """Bind an explicit derived client without rewriting a historical selection."""
    record = active()
    if record is None:
        return None
    from benchmark.evaluation_client import metadata_client
    client = metadata_client(cache, release=record['release'], record=selected_client())
    os.environ[METADATA_CLIENT] = json.dumps(dict(selection_sha256=os.environ[SELECTION_SHA], client=client), sort_keys=True)
    return client
