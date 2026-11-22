"""Restore an explicitly pinned upstream Ascend image into a private SDK."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import time
import uuid

from benchmark.baseline_setup import fetch as sequential_fetch

PREFIXES = ('usr/local/', 'etc/Ascend/', 'vllm-workspace/')
SYSTEM_LIBS = ('libbz2.so', 'libnuma.so', 'libgomp.so', 'libffi.so', 'liblzma.so',
               'libsqlite3.so', 'libssl.so', 'libcrypto.so', 'libreadline.so',
               'libtinfo.so', 'libncursesw.so', 'libuuid.so', 'libz.so',
               'libstdc++.so', 'libgcc_s.so', 'libexpat.so', 'libcrypt.so', 'libatomic.so')


def download_workers():
    try:
        workers=int(os.environ.get('DEEPLINKENGINE_DOWNLOAD_RANGE_WORKERS','1'))
    except ValueError as error:
        raise ValueError('Range workers must be an integer between 1 and 4') from error
    if not 1<=workers<=4:
        raise ValueError('Range workers must be between 1 and 4')
    if workers>1 and not shutil.which('curl'):
        raise ValueError('Parallel official ranges require native curl')
    return workers


def fetch(cache,registry,kind,identity,size):
    workers=download_workers()
    if workers>1 and size>=2*1024**2:
        from official_range_download import fetch as ranges
        return ranges(cache,registry,kind,identity,size,workers=workers)
    return sequential_fetch(cache,registry,kind,identity,size)


def digest(path):
    with Path(path).open('rb') as stream:
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            hasher.update(block)
        result = hasher.hexdigest()
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        return result


def read_lock(path):
    lock = json.loads(Path(path).read_bytes())
    if (lock.get('schema') != 'deeplinkengine-official-baseline-image/v1'
            or lock.get('registry') != 'https://quay.io/v2/ascend/vllm-ascend/'
            or lock.get('repository') != 'vllm-project/vllm-ascend'
            or lock.get('platform') != dict(os='linux', architecture='arm64')
            or not re.fullmatch(r'\d+\.\d+\.\d+', lock.get('upstream_version', ''))
            or not re.fullmatch(r'sha256:[0-9a-f]{64}', lock.get('manifest_digest', ''))
            or lock.get('ascend_version') != lock['upstream_version']):
        raise ValueError('Invalid explicitly pinned official ARM64 image')
    return lock


def destination(root, name):
    pure = PurePosixPath(name)
    if pure.is_absolute() or '..' in pure.parts or '\\' in name:
        raise ValueError('Unsafe image member')
    path = root.joinpath(*pure.parts)
    if not path.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError('Image parent escapes private root')
    return path


def remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def selected(name):
    pure = PurePosixPath(name)
    if pure.is_absolute() or '..' in pure.parts:
        raise ValueError('Unsafe image path')
    if any((str(pure) + '/').startswith(prefix) for prefix in PREFIXES):
        return True
    return str(pure.parent) in ('lib/aarch64-linux-gnu', 'usr/lib/aarch64-linux-gnu') and pure.name.startswith(SYSTEM_LIBS)


def extract(blob, root):
    """Apply OCI whiteouts and links; never write outside the new rootfs."""
    with tarfile.open(blob) as archive:
        members = [m for m in archive if selected(m.name)]
        for member in members:
            path = destination(root, member.name)
            if path.name == '.wh..wh..opq':
                if path.parent.is_dir():
                    for prior in path.parent.iterdir():
                        remove(prior)
            elif path.name.startswith('.wh.'):
                remove(path.with_name(path.name[4:]))
        links = []
        for member in members:
            path = destination(root, member.name)
            if path.name.startswith('.wh.'):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                if path.is_symlink() or path.is_file():
                    remove(path)
                path.mkdir(exist_ok=True)
            elif member.isfile():
                if path.is_symlink() or path.is_dir():
                    remove(path)
                with archive.extractfile(member) as source, path.open('wb') as output:
                    shutil.copyfileobj(source, output, 4 * 1024**2)
                    if hasattr(os, 'posix_fadvise'):
                        os.posix_fadvise(output.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                path.chmod(member.mode & 0o777)
            elif member.issym():
                remove(path)
                target = root / member.linkname.lstrip('/') if member.linkname.startswith('/') else path.parent / member.linkname
                if not target.resolve().is_relative_to(root.resolve()):
                    raise ValueError('Image symlink escapes private root')
                path.symlink_to(os.path.relpath(target, path.parent))
            elif member.islnk():
                links.append((path, destination(root, member.linkname)))
            else:
                raise ValueError('Unsupported selected image member')
        for path, target in links:
            if not target.resolve().is_relative_to(root.resolve()) or not target.is_file():
                raise ValueError('Image hardlink target unavailable')
            remove(path)
            os.link(target, path)


def file_map(root):
    result = {}
    for path in sorted((root / 'rootfs').rglob('*')):
        name = str(path.relative_to(root))
        if path.is_symlink():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('Restored link escaped private SDK')
            result[name] = dict(symlink=os.readlink(path))
        elif path.is_file():
            result[name] = dict(bytes=path.stat().st_size, sha256=digest(path))
    return result


def configure_fla_opp(root, site, env):
    """Register the locked 0.30 wheel's OPP before CANN/Torch initialization."""
    lock = root / 'image-lock.json'
    if not lock.is_file() or json.loads(lock.read_bytes()).get('upstream_version') != '0.30.0':
        return
    package = (site / 'fla_npu').resolve()
    vendor = (package / 'opp/vendors/fla_npu_transformer').resolve()
    library = (vendor / 'op_api/lib/libcust_opapi.so').resolve()
    if not package.is_relative_to(root.resolve()) or not vendor.is_relative_to(package) or not library.is_relative_to(package):
        raise ValueError('FLA custom operators must remain inside the admitted package')
    if not (package / '__init__.py').is_file() or not library.is_file() or not (vendor / 'op_impl').is_dir():
        raise ValueError('The official 0.30 runtime requires its complete embedded FLA OPP')
    # Match the package's pre-import recommendation. Its op_api is opened by
    # absolute path; never shadow CANN's libopapi.so in LD_LIBRARY_PATH.
    entries = [str(vendor), str(vendor / 'op_api/lib')]
    entries += [value for value in env.get('ASCEND_CUSTOM_OPP_PATH', '').split(':') if value]
    env['ASCEND_CUSTOM_OPP_PATH'] = ':'.join(dict.fromkeys(entries))


def environment(root, cache, *, devices=None):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    config = json.loads((root / 'image-config.json').read_bytes())
    rootfs = root / 'rootfs'
    blocked = ('ASCEND', 'ATB', 'VLLM', 'HUST', 'BETTERSCALE', 'MTP', 'KVMAT', 'HCCL', 'TRITON', 'TORCH', 'FLA_NPU')
    env = {k: v for k, v in os.environ.items() if not k.startswith(blocked)
           and k not in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'LD_PRELOAD')
           and not k.endswith(('_TOKEN', '_API_KEY', '_PASSWORD', '_SECRET'))
           and k not in ('SSH_AUTH_SOCK', 'GIT_ASKPASS')}
    def relocate(value):
        return ':'.join(str(rootfs) + item if item.startswith(('/usr/local/', '/vllm-workspace/'))
                        and not item.startswith('/usr/local/Ascend/driver/') else item
                        for item in value.split(':') if item)
    for item in config['config']['Env']:
        key, value = item.split('=', 1)
        env[key] = relocate(value) if key in ('PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'CMAKE_PREFIX_PATH') or value.startswith('/usr/local/') else value
    # The execution context binds the live driver utility at its host path.
    driver_utility = shutil.which('npu-smi')
    if driver_utility:
        paths = env.get('PATH', os.defpath).split(os.pathsep)
        paths.append(str(Path(driver_utility).absolute().parent))
        env['PATH'] = os.pathsep.join(dict.fromkeys(paths))
    homes = sorted((rootfs / 'usr/local').glob('python3.12*'))
    homes = [p for p in homes if (p / 'bin/python3').is_file()]
    if len(homes) != 1:
        raise ValueError('Expected one image CPython 3.12 runtime')
    home = homes[0]
    sites = [home / 'lib/python3.12/site-packages']
    sources = [p for p in [rootfs / 'vllm-workspace/vllm', rootfs / 'vllm-workspace/vllm-ascend'] if p.is_dir()]
    libs = [home / 'lib', *[p / name / 'lib' for p in sites for name in ('torch', 'torch_npu')],
            rootfs / 'lib/aarch64-linux-gnu', rootfs / 'usr/lib/aarch64-linux-gnu']
    env.update(PYTHONHOME=str(home), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               VLLMHUST_NO_SITE='1',
               PYTHONPATH=':'.join(map(str, sources + sites)) + ':' + env.get('PYTHONPATH', ''),
               LD_LIBRARY_PATH=env.get('LD_LIBRARY_PATH', '') + ':' + ':'.join(map(str, libs)),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    for key, name in [('XDG_CACHE_HOME', 'xdg'), ('XDG_CONFIG_HOME', 'config'),
                      ('VLLM_CONFIG_ROOT', 'config/vllm'), ('ASCEND_CACHE_PATH', 'ascend'), ('TRITON_CACHE_DIR', 'triton'),
                      ('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('NUMBA_CACHE_DIR', 'numba'), ('VLLM_CACHE_ROOT', 'vllm'),
                      ('HF_HOME', 'huggingface'), ('TORCH_HOME', 'torch'), ('TORCH_EXTENSIONS_DIR', 'torch-extensions'),
                      ('MPLCONFIGDIR', 'matplotlib'), ('TMPDIR', 'tmp')]:
        folder = cache / name
        folder.mkdir(parents=True, exist_ok=True)
        env[key] = str(folder)
    if devices is not None:
        env['ASCEND_RT_VISIBLE_DEVICES'] = devices
    configure_fla_opp(root, sites[0], env)
    return home / 'bin/python3', env


def probe(root, cache, lock):
    python, env = environment(root, cache)
    code = """import importlib.metadata as m,importlib.util as u,json,sys,os
names=('vllm','vllm-ascend','torch','torch-npu','transformers','triton-ascend')
plugins=[dict(group=g,name=p.name,value=p.value) for g in ('vllm.general_plugins','vllm.platform_plugins') for p in m.entry_points(group=g)]
print(json.dumps(dict(python=sys.executable,prefix=sys.prefix,no_site=sys.flags.no_site,versions={n:m.version(n) for n in names},origins={n:u.find_spec(n).origin for n in ('torch','torch_npu','vllm','vllm_ascend')},plugins=plugins,execution_context=json.loads(os.environ.get('DEEPLINK_EXECUTION_CONTEXT','null')))))"""
    from official_execution import sdk_run
    result = sdk_run([str(python), '-S', '-B', '-c', code], sdk=root, env=env, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise ValueError('Official image metadata probe failed: ' + result.stderr[-2000:])
    observed = json.loads(result.stdout)
    if (not matches_version('vllm', observed['versions']['vllm'], lock['upstream_version'])
            or not matches_version('vllm-ascend', observed['versions']['vllm-ascend'], lock['ascend_version'])
            or not Path(observed['prefix']).is_relative_to(Path(root).resolve())):
        raise ValueError('Actual official package identities differ from image selection')
    for origin in observed['origins'].values():
        if not Path(origin).resolve().is_relative_to(Path(root).resolve()):
            raise ValueError('Official package resolves outside the private image')
    return observed


def matches_version(distribution, actual, requested):
    # The digest-pinned upstream Ascend image ships vLLM's CUDA-less backend
    # build as +empty. Preserve that exact identity; accept no arbitrary local
    # fork/dev marker and never rewrite the installed package metadata.
    return actual == requested or (distribution == 'vllm' and actual == requested + '+empty')


def installation_record(out, lock, raw, files, observed):
    result = dict(schema='deeplinkengine-official-sdk/v1', baseline_id=lock['id'],
                  image_manifest=lock['manifest_digest'], installed_files_sha256=hashlib.sha256(raw).hexdigest(),
                  files=len(files), metadata=observed, host_driver_retained=True, native_packages_imported=False,
                  model_started=False, source_code_patched=False,
                  distribution_versions_retained_exactly=True)
    path = out / 'installation.json'
    if path.exists():
        raise ValueError('Do not overwrite an existing installation receipt')
    path.write_text(json.dumps(result, indent=2) + '\n')
    return result


def finish_restoration(lock_path, out):
    """Recover a fully extracted image after a metadata-only setup failure."""
    out = Path(out).resolve()
    lock = read_lock(lock_path)
    if (out / 'image-lock.json').read_bytes() != Path(lock_path).read_bytes():
        raise ValueError('Restored image belongs to another immutable selection')
    if digest(out / 'image-manifest.json') != lock['manifest_digest'].split(':')[1]:
        raise ValueError('Restored official image manifest differs')
    if digest(out / 'image-config.json') != lock['config']['digest'].split(':')[1]:
        raise ValueError('Restored official image configuration differs')
    raw = (out / 'installed-files.json').read_bytes()
    files = json.loads(raw)
    if file_map(out) != files:
        raise ValueError('Restored image files differ from the original extraction map')
    observed = probe(out, out / 'recovery-probe-cache', lock)
    result = installation_record(out, lock, raw, files, observed)
    (out / 'restoration-recovery.json').write_text(json.dumps(dict(
        reason='Complete original image extraction retained; metadata qualification repaired',
        all_original_extracted_files_reverified=True, source_code_patched=False,
        model_started=False, installed_files_sha256=result['installed_files_sha256']), indent=2) + '\n')
    return result


def quarantine_partial_sdk(lock_path, out):
    """Preserve an identifiable interrupted restoration; never move unknown data."""
    out = Path(out).absolute()
    if not os.path.lexists(out):
        return None
    if out.is_symlink() or not out.is_dir():
        raise ValueError('Official SDK must be a private ordinary directory')
    installed = out / 'installed-files.json'
    if os.path.lexists(installed):
        if installed.is_symlink() or not installed.is_file():
            raise ValueError('Invalid official installed-file manifest')
        return None  # Complete extraction: retain metadata-only recovery.
    requested = Path(lock_path).read_bytes()
    lock = out / 'image-lock.json'
    if lock.is_symlink() or not lock.is_file() or lock.read_bytes() != requested:
        raise ValueError('Interrupted SDK lock does not match; retain unknown directory')
    names = {p.name for p in out.iterdir()}
    marker = out / 'preparation.json'
    if names == {'image-lock.json'}:
        raise ValueError('Legacy SDK preparation owner is unknown; preserve its directory')
    if names != {'image-lock.json'}:
        if marker.is_symlink() or not marker.is_file():
            raise ValueError('Unknown incomplete SDK has no owned preparation marker')
        proof = json.loads(marker.read_bytes())
        if (proof.get('schema') != 'deeplinkengine-official-image-preparation/v1'
                or proof.get('destination') != str(out)
                or proof.get('lock_sha256') != hashlib.sha256(requested).hexdigest()
                or names - {'image-lock.json', 'preparation.json', 'rootfs', 'image-manifest.json', 'image-config.json'}):
            raise ValueError('Unrecognized incomplete official SDK; originals retained')
        for name in names - {'image-lock.json', 'preparation.json'}:
            if (out / name).is_symlink():
                raise ValueError('Incomplete SDK top-level link refused')
    from official_sdk_state import refuse_active_owner
    owner_state = refuse_active_owner(proof)
    parent = out.parent.resolve()
    if out.resolve().parent != parent:
        raise ValueError('Incomplete SDK escapes its admitted parent')
    suffix = uuid.uuid4().hex
    saved = parent / (out.name + '.interrupted-' + suffix)
    assert not os.path.lexists(saved)
    info = out.stat()
    out.rename(saved)
    result = dict(schema='deeplinkengine-official-interrupted-restoration/v1', at=time.time(),
                  previous_sdk=str(saved), new_sdk=str(out), inode=info.st_ino,
                  lock_sha256=hashlib.sha256(requested).hexdigest(), files_deleted=False,
                  reason='Identifiable interrupted extraction has no installed-file manifest', owner_state=owner_state)
    receipt = parent / (out.name + '.recovery-' + suffix + '.json')
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    return dict(receipt=str(receipt), **result)


def prepare(lock_path, cache, out):
    lock = read_lock(lock_path)
    workers=download_workers()
    out, cache = Path(out).resolve(), Path(cache).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(out).free < 40 * 1024**3:
        raise ValueError('Official image restoration requires 40 GiB free disk')
    (out / 'image-lock.json').write_bytes(Path(lock_path).read_bytes())
    from official_sdk_state import owner_info
    (out / 'preparation.json').write_text(json.dumps(dict(
        schema='deeplinkengine-official-image-preparation/v1', destination=str(out),
        lock_sha256=digest(lock_path), started=time.time(), owner=owner_info()), indent=2) + '\n')
    registry = lock['registry']
    manifest_path = fetch(cache, registry, 'manifests', lock['manifest_digest'], lock['manifest_bytes'])
    manifest = json.loads(manifest_path.read_bytes())
    if manifest['config'] != lock['config'] or manifest['layers'] != lock['layers']:
        raise ValueError('Image manifest differs from locked layer/config identities')
    config_path = fetch(cache, registry, 'blobs', lock['config']['digest'], lock['config']['size'])
    config = json.loads(config_path.read_bytes())
    if (config.get('architecture'), config.get('os')) != ('arm64', 'linux'):
        raise ValueError('Image configuration platform mismatch')
    with ThreadPoolExecutor(max_workers=1 if workers>1 else 3) as pool:
        blobs = list(pool.map(lambda row: fetch(cache, registry, 'blobs', row['digest'], row['size']), lock['layers']))
    rootfs = out / 'rootfs'
    rootfs.mkdir()
    for index, blob in enumerate(blobs):
        extract(blob, rootfs)
        print('OFFICIAL_IMAGE_LAYER', lock['id'], index + 1, len(blobs), flush=True)
    shutil.copyfile(manifest_path, out / 'image-manifest.json')
    shutil.copyfile(config_path, out / 'image-config.json')
    files = file_map(out)
    raw = (json.dumps(files, sort_keys=True, indent=2) + '\n').encode()
    (out / 'installed-files.json').write_bytes(raw)
    observed = probe(out, out / 'probe-cache', lock)
    return installation_record(out, lock, raw, files, observed)
