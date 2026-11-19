"""Keep a reconstructible SDK on local disk while retaining public output on NFS."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent


def filesystem(path):
    path = Path(path).resolve()
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ',1)
        mount = Path(left.split()[4])
        if path.is_relative_to(mount):
            mounts.append((len(str(mount)),right.split()[0]))
    return max(mounts)[1] if mounts else None


def storage_target(logical, frozen):
    logical = Path(os.path.abspath(Path(logical).expanduser()))
    logical = logical.parent.resolve() / logical.name
    if filesystem(logical.parent) not in ('nfs','nfs4','cifs'):
        return logical
    explicit = os.environ.get('DEEPLINKENGINE_RUNTIME_CACHE')
    bases = [Path(explicit).expanduser()] if explicit else [Path('/tmp'),Path('/var/tmp')]
    for base in bases:
        if not base.exists():
            if explicit:base.mkdir(parents=True)
            else:continue
        kind = filesystem(base)
        if kind in ('nfs','nfs4','cifs','tmpfs') or (kind or '').startswith('fuse'):
            continue
        if shutil.disk_usage(base).free < 2*2**30:
            continue
        home = base / ('deeplinkengine-runtime-'+str(os.getuid()))
        if home.is_symlink():
            raise ValueError('Runtime storage root must not be a symlink')
        home.mkdir(mode=0o700,exist_ok=True)
        key = hashlib.sha256(json.dumps(dict(logical=str(logical),python=sys.executable,
            version=frozen['version'],archives=frozen['archives']),sort_keys=True).encode()).hexdigest()[:20]
        parent = home / key
        parent.mkdir(exist_ok=True)
        if parent.is_symlink():
            raise ValueError('Runtime storage key must not be a symlink')
        return parent / 'installation'
    return logical


def marker(logical):
    return logical.with_name(logical.name+'.storage.json')


def audit(target, frozen):
    from vllm_hust_engine.cli import audit_install, verify_tools
    receipt = json.loads((target/'installation.json').read_text())
    if (receipt['release'] != frozen['release'] or receipt['package_version'] != frozen['version']
            or receipt['python'] != sys.executable
            or Path(receipt['deployment']).resolve() != (target/'sdk').resolve()):
        raise ValueError('Runtime installation identity differs')
    verify_tools(target/'tools')
    result = audit_install(target/'sdk',target/'tools')
    if result['verified_runtime_files'] != frozen['runtime_files']:
        raise ValueError('Runtime file inventory differs')
    return receipt


def ensure(logical, *, reuse=False):
    from vllm_hust_engine.cli import verify
    frozen = verify()
    logical = Path(os.path.abspath(Path(logical).expanduser()))
    logical = logical.parent.resolve() / logical.name
    logical.parent.mkdir(parents=True,exist_ok=True)
    target = storage_target(logical,frozen)
    if os.path.lexists(logical) and not reuse:
        raise ValueError('Choose a new installation directory: '+str(logical))
    lease = logical.with_name(logical.name+'.storage.lock')
    with lease.open('a+b') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX)
        print('RUNTIME_STORAGE', 'local' if target!=logical else 'requested', target, flush=True)
        if target.is_symlink():
            raise ValueError('Physical runtime installation must not be a symlink')
        if target.exists() and not (target/'installation.json').is_file():
            if target==logical:
                raise ValueError('Incomplete direct installation requires a new output directory')
            target.rename(target.with_name('failed-installation-'+uuid.uuid4().hex))
        if not target.exists():
            command=[sys.executable,'-c','from vllm_hust_engine.cli import main; raise SystemExit(main())',
                     'install','--out',str(target)]
            result=subprocess.run(command)
            if result.returncode:
                raise RuntimeError('Runtime installation failed; original staging retained')
        receipt=audit(target,frozen)
        archived=None
        if target!=logical and (not logical.is_symlink() or logical.resolve()!=target.resolve()):
            if os.path.lexists(logical):
                archive=logical.parent/'attempts'/('runtime-storage-'+uuid.uuid4().hex)
                archive.mkdir(parents=True)
                archived=archive/logical.name
                logical.rename(archived)
            logical.symlink_to(target,target_is_directory=True)
        record=dict(schema='deeplinkengine-runtime-storage/v1',logical=str(logical),physical=str(target),
                    python=sys.executable,release=frozen['release'],package_version=frozen['version'],
                    archives=frozen['archives'],runtime_files=frozen['runtime_files'],
                    archived_previous=str(archived) if archived else None,
                    results_moved=False,verified=True)
        marker(logical).write_text(json.dumps(record,indent=2)+'\n')
        print('RUNTIME_STORAGE_VERIFIED', frozen['runtime_files'], logical, flush=True)
        return receipt


def restore_missing(logical):
    """Rebuild only a managed missing ephemeral runtime, using its recorded venv."""
    logical=Path(os.path.abspath(Path(logical).expanduser()))
    logical=logical.parent.resolve()/logical.name
    info=marker(logical)
    if not (logical.is_symlink() and not logical.exists() and info.is_file()):
        return
    record=json.loads(info.read_text())
    if record.get('logical')!=str(logical) or record.get('verified') is not True:
        raise ValueError('Runtime storage marker differs')
    python=Path(record['python'])
    if not (python.parent.parent/'pyvenv.cfg').is_file():
        raise ValueError('Restore the recorded private environment before rebuilding runtime storage')
    from environment_setup import cann_environment
    env,_=cann_environment({})
    result=subprocess.run([str(python),str(ROOT/'runtime_storage.py'),'--out',str(logical),'--reuse'],env=env)
    if result.returncode:
        raise RuntimeError('Managed runtime reconstruction failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--reuse',action='store_true')
    args=parser.parse_args()
    ensure(args.out,reuse=args.reuse)
