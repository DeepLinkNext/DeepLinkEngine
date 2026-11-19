"""Restore byte-identical fixed datasets from this repository's release or offline archives."""
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'data/bundles.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def manifest():
    value = json.loads(MANIFEST.read_bytes())
    if (value['schema'] != 'deeplinkengine-fixed-dataset-bundles/v1'
            or value['standard_id'] != 'five-datasets-20261113-v1'
            or set(value['components']) != {'mmlu', 'hle', 'frontier', 'swe', 'terminal'}):
        raise ValueError('Unknown fixed dataset bundle')
    for name, expected in value['frozen_protocols'].items():
        if not name.startswith('datasets/') or '..' in PurePosixPath(name).parts or digest(ROOT / name) != expected:
            raise ValueError('Fixed dataset protocol changed: ' + name)
    for record in value['components'].values():
        if not record['url'].startswith('https://github.com/DeepLinkNext/DeepLinkEngine/releases/download/' + value['release'] + '/'):
            raise ValueError('Dataset body must come from this repository release')
        if record.get('api_url') and not record['api_url'].startswith('https://api.github.com/repos/DeepLinkNext/DeepLinkEngine/releases/assets/'):
            raise ValueError('Dataset API source is outside this repository')
        if Path(record['name']).name != record['name'] or not record['files']:
            raise ValueError('Invalid dataset archive name or inventory')
        for name, item in record['files'].items():
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or str(p) != name or item['bytes'] < 0:
                raise ValueError('Unsafe dataset inventory path')
    return value


def valid_file(path, item):
    path = Path(path)
    return (path.is_file() and not path.is_symlink() and path.stat().st_size == item['bytes']
            and digest(path) == item['sha256'])


def verify_files(root, record):
    root = Path(root).resolve()
    for name, item in record['files'].items():
        p = root / name
        if not p.resolve().is_relative_to(root) or not valid_file(p, item):
            raise ValueError('Dataset bytes missing or changed: ' + name)
    return root


def download(record, folder):
    target = folder / record['name']
    if valid_file(target, record):
        return target
    if target.is_symlink():
        raise ValueError('Dataset archive cache must not be a symlink')
    last = None
    urls = list(dict.fromkeys([record.get('api_url', record['url']), record['url']]))
    for _ in range(2):
        for url in urls:
            temporary = None
            try:
                print('DATA_DOWNLOAD', record['name'], flush=True)
                request = urllib.request.Request(url, headers={
                    'Accept': 'application/octet-stream', 'User-Agent': 'DeepLinkEngine-fixed-data'})
                with tempfile.NamedTemporaryFile(dir=folder, prefix='download-', delete=False) as output:
                    temporary = Path(output.name)
                    with urllib.request.urlopen(request, timeout=45) as response:
                        while block := response.read(1024 * 1024):
                            output.write(block)
                            if output.tell() > record['bytes']:
                                raise ValueError('Dataset download exceeds fixed size')
                if not valid_file(temporary, record):
                    raise ValueError('Dataset archive SHA-256 or size differs')
                temporary.replace(target)
                return target
            except (OSError, ValueError) as error:
                last = error
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    raise OSError('Cannot obtain repository dataset archive ' + record['name'] + ': ' + str(last))


def component(cache, name, archives=None):
    record = manifest()['components'][name]
    base = Path(cache).resolve() / 'deeplinkengine-data'
    base.mkdir(parents=True, exist_ok=True)
    target = base / (name + '-' + record['sha256'][:16])
    with (base / (name + '.lock')).open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_symlink():
            raise ValueError('Dataset destination must not be a symlink')
        if target.exists():
            return verify_files(target, record)
        folder = base / 'archives'
        folder.mkdir(exist_ok=True)
        archives = archives or os.environ.get('DEEPLINKENGINE_DATA_ARCHIVES')
        if archives:
            archive = Path(archives).expanduser().resolve() / record['name']
            if not valid_file(archive, record):
                raise ValueError('Offline dataset archive missing or changed: ' + str(archive))
        else:
            archive = download(record, folder)
        with tempfile.TemporaryDirectory(prefix=name + '-unpack-', dir=base) as temporary:
            stage = Path(temporary) / 'data'
            stage.mkdir()
            seen = set()
            # Stream extraction; admit only exact regular members from the frozen inventory.
            with tarfile.open(archive, mode='r|gz') as source:
                for member in source:
                    name_in_archive = member.name
                    if (name_in_archive not in record['files'] or name_in_archive in seen
                            or not member.isfile() or member.size != record['files'][name_in_archive]['bytes']):
                        raise ValueError('Unexpected or unsafe dataset archive member: ' + name_in_archive)
                    seen.add(name_in_archive)
                    p = stage / name_in_archive
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as stream, p.open('xb') as output:
                        shutil.copyfileobj(stream, output)
            if seen != set(record['files']):
                raise ValueError('Dataset archive inventory incomplete')
            verify_files(stage, record)
            stage.rename(target)
    return target


def install_files(source, destination, record, source_prefix=""):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name, item in record['files'].items():
        target = destination / name
        if not target.parent.resolve().is_relative_to(destination):
            raise ValueError('Dataset destination escaped cache')
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(target):
            if not valid_file(target, item):
                raise ValueError('Existing dataset differs; original retained: ' + name)
            continue
        # Copies keep the sealed component cache independent of user-editable asset views.
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='data-', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with (source / name.removeprefix(source_prefix)).open('rb') as raw:
                    shutil.copyfileobj(raw, stream)
                stream.flush()
                if not valid_file(temporary, item):
                    raise ValueError('Dataset materialization changed bytes')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return verify_files(destination, record)


def prepare(cache, datasets, *, archives=None, assets=None, context=False, deployment=None):
    selected = set(datasets) | {'mmlu'}
    if context:
        selected |= {'swe', 'terminal'}
    if selected - set(manifest()['components']):
        raise ValueError('Unknown dataset selection')
    notice = manifest().get('notice')
    if notice and not valid_file(ROOT / 'data' / notice['name'], notice):
        raise ValueError('Dataset source notice changed')
    cache = Path(cache).expanduser().resolve()
    base = cache / 'deeplinkengine-data'
    base.mkdir(parents=True, exist_ok=True)
    content_id = hashlib.sha256(json.dumps({n:r['sha256'] for n,r in manifest()['components'].items()}, sort_keys=True).encode()).hexdigest()[:16]
    asset_root = base / ('assets-' + content_id)
    supplied = Path(assets).expanduser().resolve() if assets else None
    with (base / 'materialize.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name in sorted(selected):
            record = manifest()['components'][name]
            source, source_prefix = None, ''
            if name == 'mmlu' and deployment:
                source = Path(deployment).resolve()
                source_prefix = 'mmlu/'
                for relative, item in record['files'].items():
                    if not valid_file(source / relative.removeprefix(source_prefix), item):
                        raise ValueError('Installed MMLU input differs from repository data: ' + relative)
            else:
                candidates = [p for p in (supplied, asset_root, cache if name in ('swe','terminal') else None) if p is not None]
                for candidate in candidates:
                    complete = True
                    for relative, item in record['files'].items():
                        p = candidate / relative
                        if os.path.lexists(p) and not valid_file(p,item):
                            raise ValueError('Existing dataset differs; original retained: ' + relative)
                        complete = complete and p.is_file()
                    if complete:
                        source = candidate
                        break
                if source is None:
                    source = component(cache, name, archives)
            if source.resolve() != asset_root.resolve():
                install_files(source, asset_root, record, source_prefix)
            else:
                verify_files(asset_root,record)
            if context and name in ('swe', 'terminal'):
                install_files(source, cache, record)
        if notice:
            install_files(ROOT / 'data', asset_root, dict(files={notice['name']: notice}))
    print('DATA_READY', ','.join(sorted(selected)), str(asset_root), flush=True)
    return asset_root
