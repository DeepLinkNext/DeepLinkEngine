"""Restore the pinned image's system tools for private upstream source builds.

Only system prefixes enter this root. Python/CANN/model SDKs retain their
original absolute paths, and host PIDs, drivers and cooperative locks remain
shared through the ordinary execution-context implementation.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import tempfile

import official_execution as execution

BASE_MANIFEST = 'sha256:4ee78def8f33d59d48f116d1dfa793332c23c99ecab4f0d7dd5cd62d0fb4e6c1'
SYSTEM_PREFIXES = ('bin', 'sbin', 'lib', 'lib64', 'usr/bin', 'usr/sbin', 'usr/lib',
                   'usr/lib64', 'usr/libexec', 'usr/include', 'usr/share', 'etc',
                   'var/lib/dpkg', 'var/lib/apt', 'var/cache/apt')


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
    environment=execution._neutral_environment()
    environment.update(PATH='/usr/sbin:/usr/bin:/sbin:/bin',DEBIAN_FRONTEND='noninteractive',
                       HOME=str(logs),TMPDIR=str(logs))
    common=['apt-get','-o','APT::Keep-Downloaded-Packages=true',
            '-o','Dir::Cache::archives='+str(archives),'-o','Acquire::Retries=2',
            '-o','Acquire::http::Timeout=45','-o','Acquire::https::Timeout=45']
    commands=[common+['update'],common+['install','--yes','--no-install-recommends',
                                      'protobuf-compiler','libprotobuf-dev']]
    records=[]
    for index,command in enumerate(commands):
        process=None
        with (logs/('apt-'+str(index)+'.log')).open('xb') as stream:
            try:
                process=execution.popen_in_context(command,root,env=environment,
                    extra_paths=(staging,Path(__file__).parent),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                code=process.wait(timeout=900)
                records.append(dict(command=command,returncode=code,model_started=False))
                (logs/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
                if code:
                    raise RuntimeError('Private Ubuntu build-package preparation failed; preserve '+str(logs))
            finally:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
    packages=execution.run_in_context(['dpkg-query','-W','-f=${binary:Package}\t${Version}\n',
        'protobuf-compiler','libprotobuf-dev','libprotobuf23','libprotoc23'],root,
        env=environment,extra_paths=(staging,),capture_output=True,text=True,timeout=30)
    if packages.returncode:
        raise ValueError('Private build-package version readback failed')
    return dict(already_in_official_image=False,package_versions=packages.stdout,
        source='Official Ubuntu apt repositories configured in the pinned upstream image; apt signature/hash verification retained',
        archives={path.name:dict(bytes=path.stat().st_size,sha256=execution.sha(path)) for path in archives.glob('*.deb')},
        commands=records,host_packages_modified=False)


def ensure(base_sdk):
    """Publish only a complete root; interrupted staging roots remain intact."""
    from baseline_setup import fetch, download_lock
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
            values=inventory(root)
            (staging/'static-files.json').write_text(json.dumps(values,sort_keys=True)+'\n')
            receipt=dict(schema='deeplinkengine-private-glibc-base/v1',layer_sha256=execution.BASE_DIGEST,
                static_inventory_sha256=execution.sha(staging/'static-files.json'),source_build_system_root=True,
                image_manifest=BASE_MANIFEST,composer_sha256=execution.sha(__file__),execution_controller_sha256=execution_sha,
                layers=[layer['digest'] for layer in lock['layers']],sdk_root_used_as_execution_root=False,
                additional_build_packages=provision)
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
        extra_paths=(Path(sdk),Path(__file__).parent), **kwargs)


def run(command, *, sdk, env, **kwargs):
    context = execution.context_identity()
    if context is not None and context.get('source_build_system_root'):
        return subprocess.run(command, env=env, **kwargs)
    root = ensure(base_for(sdk))
    return execution.run_in_context(command, root, env=env,
        extra_paths=(Path(sdk),Path(__file__).parent), **kwargs)
