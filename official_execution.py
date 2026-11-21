"""Private glibc execution context for official SDKs on older Linux hosts.

The immutable SDK remains outside this root. Only a private mount namespace
is created: host PIDs, boot identity, NPU devices and cooperative lock inodes
are retained. This is an ABI compatibility context, not a security sandbox.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

BASE_DIGEST = '119d19e001bafa21919289095e1dbfac64f1e16d2469dd14c2d2a520039d26d9'
MS_RDONLY, MS_REMOUNT, MS_BIND, MS_REC, MS_PRIVATE = 1, 32, 4096, 16384, 1 << 18
CLONE_NEWNS = 0x00020000
GLIBC_LIBRARIES = {'libc.so.6', 'libm.so.6', 'libpthread.so.0', 'libdl.so.2',
                   'librt.so.1', 'libutil.so.1', 'libresolv.so.2', 'libanl.so.1',
                   'libnss_files.so.2', 'libnss_dns.so.2', 'ld-linux-aarch64.so.1'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def libc_version():
    if sys.platform != 'linux':
        return None
    library = ctypes.CDLL(None)
    function = library.gnu_get_libc_version
    function.restype = ctypes.c_char_p
    return function().decode()


def needed():
    version = libc_version()
    return version is not None and tuple(int(v) for v in version.split('.')[:2]) < (2, 35)


def context_identity():
    raw = os.environ.get('DEEPLINK_EXECUTION_CONTEXT')
    if not raw:
        return None
    context = json.loads(raw)
    if (context.get('schema') != 'deeplinkengine-private-glibc-context/v1'
            or context.get('base_layer_sha256') != BASE_DIGEST
            or context['root_dev_inode'] != [os.stat('/').st_dev, os.stat('/').st_ino]
            or context['mount_namespace'] != os.readlink('/proc/self/ns/mnt')
            or context['host']['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or needed()):
        raise ValueError('Inherited private execution context does not match this process')
    return context


def destination(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError('Unsafe execution-root archive path')
    result = root.joinpath(*path.parts)
    if not result.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError('Execution-root member escaped its private root')
    return result


def prepare_root(blob, output):
    """Restore the exact official Ubuntu base layer, independently of SDKs."""
    blob, output = Path(blob).resolve(), Path(output).resolve()
    if sha(blob) != BASE_DIGEST:
        raise ValueError('Official Ubuntu execution base digest differs')
    if output.exists():
        raise ValueError('Execution-root output already exists')
    from official_sdk_state import owner_info
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + '.preparing-', dir=output.parent))
    marker = dict(schema='deeplinkengine-private-glibc-preparation/v1',
                  layer_sha256=BASE_DIGEST, destination=str(output), owner=owner_info())
    (stage / 'preparation.json').write_text(json.dumps(marker, indent=2) + '\n')
    # Publish the owner marker with the directory, before any extraction starts.
    stage.rename(output)
    root = output / 'rootfs'
    root.mkdir()
    links = []
    inventory = {}
    with tarfile.open(blob) as archive:
        for member in archive:
            path = destination(root, member.name)
            if path.name.startswith('.wh.'):
                raise ValueError('An initial base layer must not contain whiteouts')
            path.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                path.mkdir(exist_ok=True)
                path.chmod(member.mode & 0o777)
            elif member.isfile():
                with archive.extractfile(member) as source, path.open('xb') as target:
                    shutil.copyfileobj(source, target)
                path.chmod(member.mode & 0o777)
                inventory[member.name] = {'sha256': sha(path)}
            elif member.issym():
                target = (root / member.linkname.lstrip('/')) if member.linkname.startswith('/') else path.parent / member.linkname
                if not target.resolve().is_relative_to(root):
                    raise ValueError('Execution-root symlink escaped')
                path.symlink_to(os.path.relpath(target, path.parent))
                inventory[member.name] = {'symlink': os.readlink(path)}
            elif member.islnk():
                links.append((path, destination(root, member.linkname), member.name))
            elif not member.name.startswith(('dev/', 'proc/', 'sys/')):
                raise ValueError('Unexpected special member in official execution base')
    for path, target, name in links:
        if not target.is_file():
            raise ValueError('Missing execution-base hardlink target')
        os.link(target, path)
        inventory[name] = {'sha256': sha(path)}
    for name in ('usr/lib/aarch64-linux-gnu/libc.so.6', 'usr/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1', 'bin/sh'):
        if not (root / name).is_file():
            raise ValueError('Official execution base lacks required glibc/shell')
    (output / 'static-files.json').write_text(json.dumps(inventory, sort_keys=True) + '\n')
    receipt = dict(schema='deeplinkengine-private-glibc-base/v1', layer_sha256=BASE_DIGEST,
                   root=str(root), files=len(inventory), static_inventory_sha256=sha(output / 'static-files.json'),
                   sdk_modified=False, host_glibc_modified=False, model_started=False)
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return root


def verify_root(output):
    output = Path(output).resolve()
    receipt = json.loads((output / 'receipt.json').read_bytes())
    if (receipt.get('schema') != 'deeplinkengine-private-glibc-base/v1'
            or receipt.get('layer_sha256') != BASE_DIGEST
            or receipt.get('static_inventory_sha256') != sha(output / 'static-files.json')):
        raise ValueError('Private execution root identity changed')
    root = output / 'rootfs'
    inventory = json.loads((output / 'static-files.json').read_bytes())
    for name, entry in inventory.items():
        path = destination(root, name)
        if 'sha256' in entry:
            if not path.is_file() or path.is_symlink() or sha(path) != entry['sha256']:
                raise ValueError('Private execution root file changed: ' + name)
        elif not path.is_symlink() or os.readlink(path) != entry['symlink']:
            raise ValueError('Private execution root link changed: ' + name)
    return root


def ensure_for_sdk(sdk):
    """Use the image cache's pinned base layer, outside every admitted SDK."""
    sdk = Path(sdk).resolve()
    cache = sdk.parent.parent
    output = cache / 'execution-contexts' / ('ubuntu-' + BASE_DIGEST[:16])
    if output.is_relative_to(sdk):
        raise ValueError('Execution roots must remain outside immutable SDKs')
    output.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output.parent / (output.name + '.lock')).open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if output.is_symlink():
            raise ValueError('Execution-root output must not be a link')
        if output.exists():
            if (output / 'receipt.json').exists():
                # Completed roots may already be in use. Never repair drift in place.
                return verify_root(output)
            from official_sdk_state import refuse_active_owner
            marker = json.loads((output / 'preparation.json').read_bytes())
            if (marker.get('schema') != 'deeplinkengine-private-glibc-preparation/v1'
                    or marker.get('layer_sha256') != BASE_DIGEST
                    or marker.get('destination') != str(output)):
                raise ValueError('Unknown partial execution root; preserve original directory')
            stopped = refuse_active_owner(marker)
            quarantine = output.with_name(output.name + '.interrupted-' + str(time.time_ns()))
            output.rename(quarantine)
            (quarantine / 'interrupted-recovery.json').write_text(json.dumps(
                dict(original=str(output), preserved=str(quarantine), **stopped), indent=2) + '\n')
        from baseline_setup import fetch
        lock = json.loads((Path(__file__).parent / 'engine/official-baselines/0.23.0.json').read_bytes())
        layers = [item for item in lock['layers'] if item['digest'] == 'sha256:' + BASE_DIGEST]
        if len(layers) != 1:
            raise ValueError('Execution base is absent from the pinned official image')
        blob = fetch(cache / 'official-oci', lock['registry'], 'blobs', layers[0]['digest'], layers[0]['size'])
        return prepare_root(blob, output)


def _call(function, *arguments):
    result = function(*arguments)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _mount(source, target, flags):
    library = ctypes.CDLL(None, use_errno=True)
    library.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_void_p]
    _call(library.mount, os.fsencode(source) if source is not None else None, os.fsencode(target), None, flags, None)


def _bind(root, source, target=None, readonly=False):
    source = Path(source).resolve()
    target = Path(target or source)
    if not source.exists() or not target.is_absolute() or target == Path('/'):
        raise ValueError('Invalid execution-context bind')
    destination_path = destination(root, str(target).lstrip('/')).resolve()
    if not destination_path.is_relative_to(root):
        raise ValueError('Bind destination escaped private execution root')
    if source.is_dir():
        destination_path.mkdir(parents=True, exist_ok=True)
    else:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if not destination_path.exists():
            destination_path.touch()
    _mount(source, destination_path, MS_BIND | (MS_REC if source.is_dir() else 0))
    if readonly:
        _mount(None, destination_path, MS_BIND | MS_REMOUNT | MS_RDONLY)


def _neutral_environment():
    environment = {k: v for k, v in os.environ.items()
                   if k not in ('PYTHONHOME', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD')}
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    return environment


def host_tools(toolchain=True):
    """Expose required host tools read-only, retaining the private glibc."""
    files = set()
    directories = set()
    binaries = set()
    names = ('npu-smi', 'gcc', 'g++', 'cc', 'c++', 'as', 'ld', 'ar', 'ranlib',
                 'make', 'ninja', 'cmake', 'pkg-config', 'curl', 'wget', 'git',
                 'bash', 'tar', 'gzip', 'unzip', 'readelf', 'nm', 'strip', 'flock') if toolchain else ('npu-smi',)
    for name in names:
        binary = shutil.which(name)
        if not binary:
            continue
        files.add(binary)
        binaries.add(binary)
    for compiler in (('gcc', 'g++') if toolchain else ()):
        binary = shutil.which(compiler)
        if binary:
            for name in ('cc1', 'cc1plus', 'collect2', 'lto1', 'lto-wrapper'):
                value = subprocess.run([binary, '-print-prog-name=' + name], env=_neutral_environment(),
                                       capture_output=True, text=True, timeout=10).stdout.strip()
                if Path(value).is_absolute() and Path(value).is_file():
                    binaries.add(value)
    driver = Path('/usr/local/Ascend/driver')
    if driver.is_dir():
        for name in ('libdcmi.so', 'libascend_hal.so', 'libdrvdsmi_host.so', 'libtsdclient.so'):
            binaries.update(str(p) for p in driver.rglob(name) if p.is_file())
    for binary in sorted(binaries):
        result = subprocess.run(['/usr/bin/ldd', binary], env=_neutral_environment(),
                                capture_output=True, text=True, timeout=15)
        for value in re.findall(r'(?:=>\s*|^\s*)(/[^\s]+)', result.stdout, re.MULTILINE):
            path = Path(value)
            if path.name not in GLIBC_LIBRARIES and path.is_file():
                files.add(str(path))
    # GCC's subordinate compilers, headers and link-time objects are needed by
    # Triton and source builds. None of the host's glibc ELF files is exposed.
    for name in (('/usr/lib/gcc', '/usr/libexec/gcc', '/usr/include') if toolchain else ()):
        if Path(name).is_dir():
            directories.add(name)
    for folder in (('/usr/lib64', '/lib64') if toolchain else ()):
        if Path(folder).is_dir():
            for pattern in ('crt*.o', 'Scrt*.o', '*.a', 'libc.so', 'libm.so', 'libpthread.so',
                            'libgcc_s.so*', 'libstdc++.so*', 'libgomp.so*', 'libatomic.so*'):
                for path in Path(folder).glob(pattern):
                    if path.is_file():
                        with path.open('rb') as stream:
                            if path.name in ('libc.so', 'libm.so', 'libpthread.so') and stream.read(4) == b'\x7fELF':
                                continue
                        files.add(str(path))
    bindings = [(name, name, True) for name in sorted(directories)]
    bindings += [(name, name, True) for name in sorted(files)
                 if not any(Path(name).is_relative_to(Path(d)) for d in directories)]
    library_dirs = sorted({str(Path(name).parent) for name in files if '.so' in Path(name).name})
    library_dirs += [str(p) for p in (driver / 'lib64', driver / 'lib64/common', driver / 'lib64/driver') if p.is_dir()]
    return bindings, library_dirs


def _private_glibc_aliases(root):
    directory = root / 'lib64'
    directory.mkdir(exist_ok=True)
    for name in GLIBC_LIBRARIES:
        source = root / 'usr/lib/aarch64-linux-gnu' / name
        target = directory / name
        if source.is_file() and not target.exists() and not target.is_symlink():
            target.symlink_to(os.path.relpath(source, target.parent))


def execute_request(request):
    """Called before chroot by the host interpreter, then execs the child."""
    root = Path(request['root']).resolve()
    if os.geteuid() != 0:
        raise PermissionError('Private mount/chroot execution requires root on this host')
    before = dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  host_libc=libc_version(), lock_directories={p: [os.stat(p).st_dev, os.stat(p).st_ino] for p in ('/root', '/tmp', '/run/lock')})
    before['start_ticks'] = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    before['os_release'] = Path('/etc/os-release').read_text()
    root_identity = [root.stat().st_dev, root.stat().st_ino]
    root_receipt = root.parent / 'receipt.json'
    root_receipt_sha256 = sha(root_receipt)
    library = ctypes.CDLL(None, use_errno=True)
    _call(library.unshare, CLONE_NEWNS)
    _mount(None, '/', MS_REC | MS_PRIVATE)
    _private_glibc_aliases(root)
    # Mount parents before kernel virtual filesystems. The SDK itself is only
    # visible through its original host workspace path, never used as /.
    for source, target, readonly in request['binds']:
        _bind(root, source, target, readonly)
    mount_namespace = os.readlink('/proc/self/ns/mnt')
    os.chroot(root)
    os.chdir(request['cwd'])
    readonly_anchor = request.get('readonly_sdk_anchor')
    if readonly_anchor and not os.statvfs(readonly_anchor).f_flag & os.ST_RDONLY:
        raise ValueError('Admitted SDK anchor did not become read-only in the private namespace')
    for path, expected in before['lock_directories'].items():
        if [os.stat(path).st_dev, os.stat(path).st_ino] != expected:
            raise ValueError('Cooperative lock directory inode changed: ' + path)
    if os.getpid() != before['pid'] or Path('/proc/sys/kernel/random/boot_id').read_text().strip() != before['boot_id']:
        raise ValueError('Host PID or boot identity changed')
    context = dict(schema='deeplinkengine-private-glibc-context/v1', base_layer_sha256=BASE_DIGEST,
                   host=before, mount_namespace=mount_namespace, shared_lock_inodes_verified=True,
                   pid_namespace_created=False, sdk_root_used_as_execution_root=False,
                   root_dev_inode=root_identity,
                   readonly_sdk_anchor=readonly_anchor,
                   sdk_anchor_readonly=bool(readonly_anchor),
                   execution_os_release=Path('/etc/os-release').read_text(),
                   source_build_system_root=request.get('source_build_system_root', False),
                   execution_root_receipt=str(root_receipt), execution_root_receipt_sha256=root_receipt_sha256)
    environment = request['env']
    environment['DEEPLINK_EXECUTION_CONTEXT'] = json.dumps(context, separators=(',', ':'))
    os.execvpe(request['command'][0], request['command'], environment)


def request_for(command, root, *, env, cwd=None, extra_paths=()):
    root = Path(root).resolve()
    if root == Path('/') or not (root / 'usr/lib/aarch64-linux-gnu/libc.so.6').is_file():
        raise ValueError('A prepared private execution root is required')
    binds = [(p, p, False) for p in ('/root', '/tmp', '/run/lock')]
    binds += [(p, p, False) for p in ('/workspace', '/data', '/models', '/home') if Path(p).is_dir()]
    binds += [(p, p, p != '/dev') for p in ('/proc', '/sys', '/dev')]
    for path in ('/etc/resolv.conf', '/etc/hosts', '/usr/local/Ascend/driver', '/usr/local/Ascend/firmware', '/etc/ascend_install.info'):
        if Path(path).exists():
            binds.append((str(Path(path).resolve()), path, True))
    workspace = Path(cwd or os.getcwd()).resolve()
    extra_paths = [*extra_paths, *([] if workspace == Path('/') else [workspace])]
    for original in extra_paths:
        path = Path(original).resolve()
        if not path.exists():
            raise ValueError('Required host workspace is missing: ' + str(path))
        if any(path == Path(target) or path.is_relative_to(Path(target)) for _, target, _ in binds):
            continue
        if str(path) in ('/', '/usr', '/lib', '/lib64', '/bin', '/sbin', '/etc', '/run'):
            raise ValueError('Do not replace private execution system directories')
        binds.append((str(path), str(path), False))
    receipt = json.loads((root.parent / 'receipt.json').read_bytes())
    system_tools = receipt.get('source_build_system_root') is True
    tools, library_dirs = host_tools(toolchain=not system_tools)
    binds += tools
    environment = dict(env)
    current = environment.get('LD_LIBRARY_PATH', '').split(':')
    environment['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys([p for p in current if p] + library_dirs))
    return dict(root=str(root), command=[str(v) for v in command], env=environment,
                cwd=str(Path(cwd or os.getcwd()).resolve()), binds=binds,
                source_build_system_root=system_tools)


def run_in_context(command, root, *, env, cwd=None, extra_paths=(), **kwargs):
    request = request_for(command, root, env=env, cwd=cwd, extra_paths=extra_paths)
    # The wrapper starts with the host interpreter. Image PYTHONHOME and shared
    # libraries apply only after chroot/exec, not to that host interpreter.
    outer = _neutral_environment()
    text_mode = kwargs.get('text') or kwargs.get('encoding') or kwargs.get('errors') or kwargs.get('universal_newlines')
    payload = json.dumps(request)
    return subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--exec-json'],
                          env=outer, input=payload if text_mode else payload.encode(), **kwargs)


def popen_in_context(command, root, *, env, cwd=None, extra_paths=(), **kwargs):
    request = request_for(command, root, env=env, cwd=cwd, extra_paths=extra_paths)
    process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--exec-json'],
                               env=_neutral_environment(), stdin=subprocess.PIPE, **kwargs)
    try:
        text_mode = kwargs.get('text') or kwargs.get('encoding') or kwargs.get('errors') or kwargs.get('universal_newlines')
        payload = json.dumps(request)
        with process.stdin as pipe:
            pipe.write(payload if text_mode else payload.encode())
    except BrokenPipeError:
        pass
    except BaseException:
        # A failed bootstrap write must not leave an unowned waiting child.
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stdin = None
    return process


def sdk_run(command, *, sdk, env, **kwargs):
    if not needed():
        return subprocess.run(command, env=env, **kwargs)
    root = ensure_for_sdk(sdk)
    return run_in_context(command, root, env=env, extra_paths=(Path(sdk), Path(__file__).parent), **kwargs)


def sdk_popen(command, *, sdk, env, **kwargs):
    if not needed():
        return subprocess.Popen(command, env=env, **kwargs)
    root = ensure_for_sdk(sdk)
    return popen_in_context(command, root, env=env, extra_paths=(Path(sdk), Path(__file__).parent), **kwargs)


def enter_cli(sdk, script, arguments, *, cache=None, execution_root=None, extra_paths=()):
    """Replace this controller with SDK Python, retaining PID and birth time."""
    if not needed() and execution_root is None:
        return False
    from official_image import environment
    sdk = Path(sdk).resolve()
    root = Path(execution_root).resolve() if execution_root is not None else ensure_for_sdk(sdk)
    python, env = environment(sdk, Path(cache or sdk.parent.parent / 'execution-controller'))
    request = request_for([str(python), '-B', str(Path(script).resolve()), *arguments], root,
                          env=env, extra_paths=(sdk, Path(script).resolve().parent, *extra_paths))
    readonly_sdk_anchor(request, sdk)
    execute_request(request)
    raise AssertionError('Execution-context exec unexpectedly returned')


def readonly_sdk_anchor(request, sdk):
    """Protect a completed raw image only in this new private mount namespace."""
    sdk = Path(sdk).resolve()
    receipt = sdk / 'installation.json'
    if not receipt.is_file() or json.loads(receipt.read_bytes()).get('schema') != 'deeplinkengine-official-sdk/v1':
        return False
    if str(sdk) in ('/', '/root', '/tmp', '/usr', '/lib', '/lib64', '/etc', '/run'):
        raise ValueError('An SDK anchor must have its own private directory')
    request['binds'].append((str(sdk), str(sdk), True))
    request['readonly_sdk_anchor'] = str(sdk)
    return True


def private_mounts():
    """A verified inherited context must not propagate mounts to its parent."""
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split(' - ', 1)[0].split()
        if any(value.startswith(('shared:', 'master:', 'propagate_from:')) for value in fields[6:]):
            raise ValueError('Inherited execution context has non-private mount propagation')


def execute_tokenizer_alias(request):
    """Inherit the loader/root and create only a new private tokenizer alias."""
    inherited = context_identity()
    if inherited is None:
        raise ValueError('Tokenizer alias requires a verified inherited private context')
    private_mounts()
    source = Path(request['model']).resolve()
    stage = Path(request['stage']).resolve()
    target = Path('/models')
    if not source.is_dir() or not stage.is_dir() or target.is_symlink() or stage.is_relative_to(target):
        raise ValueError('Tokenizer alias staging must be outside /models')
    library = ctypes.CDLL(None, use_errno=True)
    _call(library.unshare, CLONE_NEWNS)
    # The inherited namespace was already made recursively private. A chroot
    # directory need not be a mountpoint, so do not remount or rebind / here.
    private_mounts()
    raw = dict(binds=[])
    readonly_sdk_anchor(raw, request['sdk'])
    for original, destination_path, readonly in raw['binds']:
        _bind(Path('/'), original, destination_path, readonly)
    # Preserve the source before shadowing an existing /models mount.
    _bind(Path('/'), source, stage / 'source-model', True)
    _bind(Path('/'), stage / 'models', '/models', False)
    _bind(Path('/'), stage / 'source-model', '/models/Qwen3.5-35B-A3B', True)
    if not os.statvfs('/models/Qwen3.5-35B-A3B').f_flag & os.ST_RDONLY:
        raise ValueError('Tokenizer model alias is not read-only')
    context = dict(inherited, mount_namespace=os.readlink('/proc/self/ns/mnt'),
                   tokenizer_alias=dict(source=str(source), target='/models/Qwen3.5-35B-A3B', read_only=True),
                   inherited_private_root=True)
    context['host'] = dict(inherited['host'], pid=os.getpid(),
                           start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    if raw.get('readonly_sdk_anchor'):
        context.update(readonly_sdk_anchor=raw['readonly_sdk_anchor'],
                       sdk_anchor_readonly=bool(os.statvfs(raw['readonly_sdk_anchor']).f_flag & os.ST_RDONLY))
    environment = dict(request['env'], DEEPLINK_EXECUTION_CONTEXT=json.dumps(context, separators=(',', ':')))
    os.chdir(request['cwd'])
    os.execve(request['command'][0], request['command'], environment)


def tokenizer_alias(command, *, model, sdk, cache, env, cwd):
    """Retain the current private glibc instead of entering its chroot again."""
    if context_identity() is None:
        return None
    parent = Path(cache) / 'official-tokenizer-aliases'
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='alias-', dir=parent))
    (stage / 'source-model').mkdir()
    (stage / 'models/Qwen3.5-35B-A3B').mkdir(parents=True)
    request = dict(command=list(map(str, command)), model=str(Path(model).resolve()),
                   sdk=str(Path(sdk).resolve()), stage=str(stage), env=env, cwd=str(Path(cwd).resolve()))
    return subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--tokenizer-alias-json'],
                          input=json.dumps(request), env=os.environ.copy(), text=True).returncode


def prepare_anchor(config):
    """Prepare the first selected image via the ordinary public CPU entry."""
    cache = Path(config['cache']).resolve()
    versions = [v for v in config['versions'] if v in ('0.23.0', '0.30.0')]
    version = versions[0] if versions else '0.23.0'
    if '0.28.0rc1-source' in config['versions']:
        version = '0.23.0'
    if not any(key in config for key in ('protocol_run', 'protocol_deployment', 'protocol_python')):
        version = '0.23.0'
    lock = json.loads((Path(__file__).parent / 'engine/official-baselines' / (version + '.json')).read_bytes())
    sdk = cache / 'official-sdks' / (version + '-' + lock['manifest_digest'].split(':')[1][:16])
    # Even an existing complete SDK is admitted again by the public entry.
    bootstrap = cache / 'execution-bootstrap' / str(time.time_ns())
    bootstrap.mkdir(parents=True, exist_ok=False)
    model = bootstrap / 'empty-model-cpu-only'
    model.mkdir()
    plan = dict(model=str(model), physical_devices=config.get('physical_devices', [0, 1]),
                versions=[version], cache=str(cache), out=str(bootstrap / 'result'))
    path = bootstrap / 'evaluation.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    with (bootstrap / 'prepare.log').open('xb') as stream:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).parent / 'run_all.py'),
                                    '--official-baselines', '--config', str(path), '--prepare-environment'],
                                   env=_neutral_environment(), stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    if code:
        raise RuntimeError('Private execution anchor preparation failed; preserve ' + str(bootstrap))
    return sdk


def config_workspaces(config, path):
    """Bind declared workspace parents, including paths outside /root or /data."""
    path = Path(path).resolve()
    paths = [path.parent]
    keys = ('model','cache','out','protocol_run','protocol_deployment','protocol_python',
            'context_contract','judge_installation','model_manifest','baseline_deployment','dataset_assets','env_file')
    values = [config[key] for key in keys if key in config]
    options = config.get('environment', {})
    values += [options[key] for key in ('official_sdk','wheelhouse','cann_home') if key in options]
    for value in values:
        parent = (path.parent / Path(value).expanduser()).absolute().parent
        parent.mkdir(parents=True, exist_ok=True)
        if parent == Path('/') or str(parent) in ('/usr','/bin','/sbin','/lib','/lib64','/etc','/run'):
            raise ValueError('Use workspace paths outside private system directories')
        paths.append(parent)
    return tuple(dict.fromkeys(paths))


def argument_config(arguments):
    for index, value in enumerate(arguments):
        if value == '--config' and index + 1 < len(arguments):
            return Path(arguments[index + 1])
        if value.startswith('--config='):
            return Path(value.split('=',1)[1])
    raise ValueError('Public execution context requires its configuration path')


def maybe_enter_official(config, arguments):
    source_candidate = '0.28.0rc1-source' in config['versions']
    current = context_identity()
    build_root_needed = source_candidate and not (current and current.get('source_build_system_root'))
    if needed() or build_root_needed:
        sdk = prepare_anchor(config)
        root = None
        if source_candidate:
            from official_build_context import ensure
            root = ensure(sdk)
        paths = config_workspaces(config, argument_config(arguments))
        enter_cli(sdk, Path(__file__).parent / 'run_all.py', ['--official-baselines', *arguments],
                  execution_root=root, extra_paths=paths)


def maybe_enter_config(path, arguments):
    """Normal protocol/full CLI: enter before creating any output directory."""
    if not needed():
        return
    config = json.loads(Path(path).read_bytes())
    sdk = config.get('environment', {}).get('official_sdk')
    if sdk:
        sdk = Path(sdk)
        if not sdk.is_absolute():
            sdk = Path(path).resolve().parent / sdk
        enter_cli(sdk, Path(__file__).parent / 'run_all.py', arguments,
                  extra_paths=config_workspaces(config, path))


if __name__ == '__main__':
    if sys.argv[1:] == ['--tokenizer-alias-json']:
        execute_tokenizer_alias(json.load(sys.stdin))
    elif sys.argv[1:] == ['--exec-json']:
        execute_request(json.load(sys.stdin))
    else:
        raise SystemExit('Internal private execution wrapper; invoke public run_all.py')
