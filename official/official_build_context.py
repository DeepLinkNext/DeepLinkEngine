"""Restore the pinned image's system tools for private upstream source builds.

System tools enter this root. The pinned CANN package also has its original
canonical path inside this private root because its unchanged setenv script
uses that path. SDK bytes stay in their admitted directory; host PIDs, drivers
and cooperative locks remain shared through the execution context.
"""
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import tempfile
import time

import official_execution as execution

BASE_MANIFEST = 'sha256:4ee78def8f33d59d48f116d1dfa793332c23c99ecab4f0d7dd5cd62d0fb4e6c1'
SYSTEM_PREFIXES = ('bin', 'sbin', 'lib', 'lib64', 'usr/bin', 'usr/sbin', 'usr/lib',
                   'usr/lib64', 'usr/libexec', 'usr/include', 'usr/share', 'etc',
                   'var/lib/dpkg', 'var/lib/apt', 'var/cache/apt')
UBUNTU_PORTS_MIRRORS = ('https://mirrors.ustc.edu.cn/ubuntu-ports',
                        'https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports')
CANN_CANONICAL_PREFIX = '/usr/local/Ascend/cann-9.1.0'


def canonical_cann_view(root, base_sdk, *, create=False):
    """Expose the fixed raw SDK package only in a new private system root."""
    root, base_sdk = Path(root).resolve(), Path(base_sdk).resolve()
    installed_path = base_sdk / 'installation.json'
    installed = json.loads(installed_path.read_bytes())
    manifest_path = base_sdk / 'installed-files.json'
    manifest_sha = execution.sha(manifest_path)
    if (installed.get('schema') != 'deeplinkengine-official-sdk/v1'
            or installed.get('image_manifest') != BASE_MANIFEST
            or installed.get('installed_files_sha256') != manifest_sha):
        raise ValueError('Canonical CANN view requires the admitted fixed official base')
    source = base_sdk / 'rootfs' / CANN_CANONICAL_PREFIX.lstrip('/')
    if not source.is_dir() or not source.resolve().is_relative_to(base_sdk / 'rootfs'):
        raise ValueError('Canonical CANN source escaped the fixed SDK')
    script = (source / 'bin/setenv.bash').resolve()
    if not script.is_relative_to(source.resolve()):
        raise ValueError('Canonical CANN setenv escaped its package')
    name = script.relative_to(base_sdk).as_posix()
    record = json.loads(manifest_path.read_bytes()).get(name)
    if record != dict(bytes=script.stat().st_size, sha256=execution.sha(script)):
        raise ValueError('Canonical CANN setenv differs from the original installed manifest')
    declarations = re.findall(r'^[ \t]*version_dirpath="([^"]+)"[ \t]*$', script.read_text(), re.MULTILINE)
    if declarations != [CANN_CANONICAL_PREFIX]:
        raise ValueError('Original CANN setenv canonical prefix is unrecognized')
    target = execution.destination(root, CANN_CANONICAL_PREFIX.lstrip('/'))
    if create:
        if target.exists() or target.is_symlink():
            raise ValueError('Canonical CANN target already exists in the new private root')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)
    if not target.is_symlink() or os.readlink(target) != str(source):
        raise ValueError('Canonical CANN view differs from its admitted SDK binding')
    return dict(canonical_prefix=CANN_CANONICAL_PREFIX, source_sdk=str(base_sdk),
        source_package=str(source), installation_sha256=execution.sha(installed_path),
        installed_files_sha256=manifest_sha, original_setenv=dict(path=name, **record),
        scope='private execution root only; original SDK bytes and vendor script retained',
        host_global_path_modified=False)


def private_apt_sources(root, logs):
    """Keep suites/components/keyrings; replace only the upstream ports URI."""
    root = Path(root).resolve()
    paths = [root/'etc/apt/sources.list']
    paths += sorted((root/'etc/apt/sources.list.d').glob('*.list'))
    paths += sorted((root/'etc/apt/sources.list.d').glob('*.sources'))
    sources = {}
    for path in paths:
        if not path.exists():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('APT source escaped the private build root')
        sources[str(path.relative_to(root))] = path.read_text()
    if not any(re.search(r'https?://ports\.ubuntu\.com/ubuntu-ports/?', text)
               for text in sources.values()):
        raise ValueError('Pinned build image has no recognized Ubuntu ports source')
    (logs/'original-sources.json').write_text(json.dumps(sources, indent=2)+'\n')
    return sources


def set_private_mirror(root, sources, mirror):
    if mirror not in UBUNTU_PORTS_MIRRORS:
        raise ValueError('Unrecognized Ubuntu ports mirror')
    hashes = {}
    for name, original in sources.items():
        path = execution.destination(root, name)
        if path.is_symlink():
            raise ValueError('APT source became a link')
        text = re.sub(r'https?://ports\.ubuntu\.com/ubuntu-ports/?', mirror+'/', original)
        path.write_text(text)
        hashes[name] = dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                           effective_sha256=execution.sha(path))
    return hashes


def selected(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('Unsafe system-root archive path')
    return any(str(path) == prefix or str(path).startswith(prefix + '/') for prefix in SYSTEM_PREFIXES)


def remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def apply_system_layer(blob, root):
    import tarfile
    links = []
    with tarfile.open(blob) as archive:
        members = [member for member in archive if selected(member.name)]
        for member in members:
            path = execution.destination(root, member.name)
            if path.name == '.wh..wh..opq':
                if path.parent.is_dir():
                    for child in path.parent.iterdir():
                        remove(child)
            elif path.name.startswith('.wh.'):
                remove(path.with_name(path.name[4:]))
        for member in members:
            path = execution.destination(root, member.name)
            if path.name.startswith('.wh.'):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                if path.is_symlink() or path.is_file():
                    remove(path)
                path.mkdir(exist_ok=True)
            elif member.isfile():
                remove(path)
                with archive.extractfile(member) as source, path.open('xb') as target:
                    shutil.copyfileobj(source, target, 4 * 1024**2)
                path.chmod(member.mode & 0o777)
            elif member.issym():
                target = root / member.linkname.lstrip('/') if member.linkname.startswith('/') else path.parent / member.linkname
                if not target.resolve().is_relative_to(root.resolve()):
                    raise ValueError('System-root symlink escaped private root')
                remove(path)
                path.symlink_to(os.path.relpath(target, path.parent))
            elif member.islnk():
                links.append((path, execution.destination(root, member.linkname)))
            else:
                raise ValueError('Unsupported system-root member')
        for path, target in links:
            if not target.resolve().is_relative_to(root.resolve()) or not target.is_file():
                raise ValueError('System-root hardlink target is unavailable')
            remove(path)
            os.link(target, path)


def inventory(root):
    values = {}
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in list(directories) + files:
            path = Path(parent) / name
            if path.is_symlink():
                values[str(path.relative_to(root))] = {'symlink': os.readlink(path)}
                if name in directories:
                    directories.remove(name)
            elif path.is_file():
                values[str(path.relative_to(root))] = {'sha256': execution.sha(path)}
            elif not path.is_dir():
                raise ValueError('Unexpected private system-root special file')
    return values


def provision_protobuf(root, staging):
    """Apply the upstream recipe's Ubuntu build packages inside this new root."""
    if (root/'usr/bin/protoc').exists() and (root/'usr/include/google/protobuf/struct.proto').is_file():
        return dict(already_in_official_image=True)
    logs=staging/'provision';logs.mkdir()
    archives=logs/'apt-archives';archives.mkdir();(archives/'partial').mkdir()
    sources=private_apt_sources(root,logs)
    environment=execution._neutral_environment()
    environment.update(PATH='/usr/sbin:/usr/bin:/sbin:/bin',DEBIAN_FRONTEND='noninteractive',
                       HOME=str(logs),TMPDIR=str(logs))
    common=['apt-get','-o','APT::Keep-Downloaded-Packages=true',
            '-o','Dir::Cache::archives='+str(archives),'-o','Acquire::Retries=1',
            '-o','Acquire::http::Timeout=15','-o','Acquire::https::Timeout=15',
            '-o','APT::Update::Error-Mode=any']
    commands=[common+['update'],common+['install','--yes','--no-install-recommends',
                                      'protobuf-compiler','libprotobuf-dev']]
    records=[]
    deadline=time.monotonic()+600
    completed=False
    for attempt,mirror in enumerate(UBUNTU_PORTS_MIRRORS):
        source_hashes=set_private_mirror(root,sources,mirror)
        for index,command in enumerate(commands):
            budget=min(180 if index==0 else 120,deadline-time.monotonic())
            if budget<=0:
                break
            record=dict(command=command,mirror=mirror,source_files=source_hashes,
                        timeout_seconds=budget,status='running',model_started=False)
            records.append(record)
            (logs/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
            print('Private Ubuntu build packages:',mirror,'update' if index==0 else 'install',
                  'timeout='+str(round(budget,1))+'s','log='+str(logs),flush=True)
            process=None
            with (logs/('apt-'+str(attempt)+'-'+str(index)+'.log')).open('xb') as stream:
                try:
                    process=execution.popen_in_context(command,root,env=environment,
                        extra_paths=(staging,Path(__file__).resolve().parents[1]),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    record['returncode']=process.wait(timeout=budget)
                    record['status']='completed' if record['returncode']==0 else 'failed'
                except subprocess.TimeoutExpired:
                    record['status']='timeout'
                except BaseException as error:
                    record.update(status='error',error=repr(error))
                    raise
                finally:
                    if process is not None and process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                    record['process_reaped']=process is None or process.poll() is not None
                    (logs/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
            if record['status']!='completed':
                break
            if index==1:
                completed=True
        if completed:
            break
    if not completed:
        raise RuntimeError('Private Ubuntu build-package preparation failed within its bounded mirror attempts; preserve '+str(logs))
    packages=execution.run_in_context(['dpkg-query','-W','-f=${binary:Package}\t${Version}\n',
        'protobuf-compiler','libprotobuf-dev','libprotobuf23','libprotoc23'],root,
        env=environment,extra_paths=(staging,),capture_output=True,text=True,timeout=30)
    if packages.returncode:
        raise ValueError('Private build-package version readback failed')
    return dict(already_in_official_image=False,package_versions=packages.stdout,
        source='Ubuntu ports mirror transport for original image suites/components/keyring; apt signature/hash verification retained',
        mirror=mirror,original_sources_sha256=execution.sha(logs/'original-sources.json'),
        archives={path.name:dict(bytes=path.stat().st_size,sha256=execution.sha(path)) for path in archives.glob('*.deb')},
        commands=records,host_packages_modified=False)


def ensure(base_sdk):
    """Publish only a complete root; interrupted staging roots remain intact."""
    from benchmark.baseline_setup import fetch, download_lock
    base_sdk = Path(base_sdk).resolve()
    cache = base_sdk.parent.parent
    lock = json.loads((base_sdk / 'image-lock.json').read_bytes())
    installed = json.loads((base_sdk / 'installation.json').read_bytes())
    if (lock['manifest_digest'] != BASE_MANIFEST or installed['image_manifest'] != BASE_MANIFEST
            or execution.sha(base_sdk / 'installed-files.json') != installed['installed_files_sha256']):
        raise ValueError('Build tools require the admitted fixed official base')
    execution_sha = execution.sha(execution.__file__)
    key = hashlib.sha256((BASE_MANIFEST + execution.sha(__file__) + execution_sha).encode()).hexdigest()[:20]
    output = cache / 'execution-contexts' / ('official-build-system-' + key)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / (output.name + '.lock')).open('a+b') as stream:
        with download_lock(stream):
            if output.is_symlink():
                raise ValueError('Private build system root must not be a link')
            if output.exists():
                root = execution.verify_root(output)
                receipt = json.loads((output / 'receipt.json').read_bytes())
                if (receipt.get('source_build_system_root') is not True
                        or receipt.get('image_manifest') != BASE_MANIFEST
                        or receipt.get('composer_sha256') != execution.sha(__file__)
                        or receipt.get('execution_controller_sha256') != execution_sha):
                    raise ValueError('Private build system root identity differs')
                if receipt.get('canonical_cann_view') != canonical_cann_view(root, base_sdk):
                    raise ValueError('Private build system root CANN binding differs')
                return root
            blobs=[]
            for layer in lock['layers']:
                blob=fetch(cache/'official-oci',lock['registry'],'blobs',layer['digest'],layer['size'])
                blobs.append(blob)
            if lock['layers'][0]['digest'] != 'sha256:' + execution.BASE_DIGEST:
                raise ValueError('Unexpected pinned build-system Ubuntu layer')
            staging_parent = Path(tempfile.mkdtemp(prefix=output.name + '.staging-', dir=output.parent))
            staging = staging_parent / 'complete-root'
            root = execution.prepare_root(blobs[0], staging)
            for blob in blobs[1:]:
                apply_system_layer(blob, root)
            # This staging root is not published or used by any model process.
            # Its marker selects image tools instead of host compiler overlays.
            pending=json.loads((staging/'receipt.json').read_bytes())
            pending['source_build_system_root']=True
            (staging/'receipt.json').write_text(json.dumps(pending)+'\n')
            provision=provision_protobuf(root,staging)
            cann_view=canonical_cann_view(root,base_sdk,create=True)
            values=inventory(root)
            (staging/'static-files.json').write_text(json.dumps(values,sort_keys=True)+'\n')
            receipt=dict(schema='deeplinkengine-private-glibc-base/v1',layer_sha256=execution.BASE_DIGEST,
                static_inventory_sha256=execution.sha(staging/'static-files.json'),source_build_system_root=True,
                image_manifest=BASE_MANIFEST,composer_sha256=execution.sha(__file__),execution_controller_sha256=execution_sha,
                layers=[layer['digest'] for layer in lock['layers']],sdk_root_used_as_execution_root=False,
                additional_build_packages=provision,canonical_cann_view=cann_view)
            (staging/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            execution.verify_root(staging)
            staging.rename(output)
            staging_parent.rmdir()
            return execution.verify_root(output)


def base_for(sdk):
    return Path(sdk).resolve().parent.parent/'official-sdks/0.23.0-4ee78def8f33d59d'


def popen(command, *, sdk, env, **kwargs):
    context = execution.context_identity()
    if context is not None and context.get('source_build_system_root'):
        return subprocess.Popen(command, env=env, **kwargs)
    root = ensure(base_for(sdk))
    return execution.popen_in_context(command, root, env=env,
        extra_paths=(Path(sdk),Path(__file__).resolve().parents[1]), **kwargs)


def run(command, *, sdk, env, **kwargs):
    context = execution.context_identity()
    if context is not None and context.get('source_build_system_root'):
        return subprocess.run(command, env=env, **kwargs)
    root = ensure(base_for(sdk))
    return execution.run_in_context(command, root, env=env,
        extra_paths=(Path(sdk),Path(__file__).resolve().parents[1]), **kwargs)
