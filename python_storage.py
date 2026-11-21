"""Put private Python files on local disk; keep artifacts and setup evidence durable."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

from runtime_storage import filesystem


class LocalPythonStorageUnavailable(ValueError):
    """No local directory meets the capacity required for relocation."""


def canonical(path):
    path=Path(os.path.abspath(Path(path).expanduser()))
    return path.parent.resolve()/path.name


def marker(path):return path.with_name(path.name+'.storage.json')


def location(managed):
    key=hashlib.sha256(str(managed).encode()).hexdigest()[:24]
    info=marker(managed)
    if info.is_file():
        record=json.loads(info.read_text());target=Path(record['physical'])
        if (record.get('schema')!='deeplinkengine-python-storage/v1' or record.get('logical')!=str(managed)
            or target.name!='venv' or target.parent.name!=key
            or target.parent.parent.name!='deeplinkengine-python-'+str(os.getuid())):
            raise ValueError('Private Python storage identity differs')
        return target
    explicit=os.environ.get('DEEPLINKENGINE_RUNTIME_CACHE')
    bases=[Path(explicit).expanduser()] if explicit else [Path('/tmp'),Path('/var/tmp')]
    for base in bases:
        base.mkdir(parents=True,exist_ok=True)
        kind=filesystem(base)
        if kind in ('nfs','nfs4','cifs','tmpfs') or (kind or '').startswith('fuse'):continue
        if shutil.disk_usage(base).free<10*2**30:continue
        home=base/('deeplinkengine-python-'+str(os.getuid()))
        if home.is_symlink():raise ValueError('Private Python cache root must not be a symlink')
        home.mkdir(mode=0o700,exist_ok=True)
        return home/key/'venv'
    raise LocalPythonStorageUnavailable('NFS Python setup requires 10GiB free on local disk; set DEEPLINKENGINE_RUNTIME_CACHE')


def active_users(managed):
    users=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:command=(proc/'cmdline').read_bytes().split(b'\0')[0].decode()
        except (OSError,UnicodeError):continue
        if command and Path(command).parent==managed/'bin':users.append(int(proc.name))
    return users


def copy_verified(source,target):
    files=[];records=[]
    target.mkdir()
    for folder,dirs,names in os.walk(source,followlinks=False):
        relative=Path(folder).relative_to(source);destination=target/relative
        destination.mkdir(parents=True,exist_ok=True)
        for name in list(dirs):
            original=Path(folder)/name
            if original.is_symlink():
                (destination/name).symlink_to(os.readlink(original),target_is_directory=True)
                records.append(dict(path=str(relative/name),symlink=os.readlink(original)));dirs.remove(name)
        for name in names:
            original=Path(folder)/name
            if original.is_symlink():
                (destination/name).symlink_to(os.readlink(original))
                records.append(dict(path=str(relative/name),symlink=os.readlink(original)))
            else:files.append((original,destination/name,relative/name))
    def transfer(item):
        original,destination,relative=item;digest=hashlib.sha256()
        with original.open('rb') as src,destination.open('xb') as dst:
            before=os.fstat(src.fileno())
            while block:=src.read(2**20):digest.update(block);dst.write(block)
            after=os.fstat(src.fileno())
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError('Private Python source changed during relocation: '+str(relative))
        shutil.copystat(original,destination)
        with destination.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=digest.hexdigest():raise ValueError('Local Python file hash differs: '+str(relative))
        return dict(path=str(relative),bytes=before.st_size,sha256=actual)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index,record in enumerate(executor.map(transfer,files),1):
            records.append(record)
            if index%2000==0:print('PYTHON_STORAGE_COPY',index,len(files),flush=True)
    return sorted(records,key=lambda r:r['path'])


def ensure(managed, *, allow_current=False):
    managed=canonical(managed)
    # A restart can remount the same persistent workspace as a local filesystem.
    # Existing managed aliases retain their recorded owner and original hashes.
    if filesystem(managed.parent) not in ('nfs','nfs4','cifs') and not marker(managed).is_file():
        if managed.is_symlink():raise ValueError('Unmanaged private Python symlink')
        return managed
    if not os.path.lexists(managed):
        if not marker(managed).is_file():return managed
        target=location(managed)
        record=json.loads(marker(managed).read_text())
        original=Path(record.get('archived_previous') or '')
        if original.parent!=managed.parent or not original.is_dir():
            raise ValueError('Persistent private Python recovery identity differs')
        managed.symlink_to(target,target_is_directory=True)
    managed.parent.mkdir(parents=True,exist_ok=True)
    with managed.with_name(managed.name+'.storage.lock').open('a+b') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        try:
            target=location(managed)
        except LocalPythonStorageUnavailable:
            if (os.environ.get('DEEPLINKENGINE_RUNTIME_CACHE') or marker(managed).exists()
                    or managed.is_symlink()):
                raise
            print('PYTHON_STORAGE nfs',managed,'local disk capacity unavailable; keeping persistent environment',flush=True)
            return managed
        if target.is_symlink() or target.parent.is_symlink():raise ValueError('Physical Python storage must not be a symlink')
        if managed.is_symlink():
            if not marker(managed).is_file() or managed.resolve()!=target.resolve():
                raise ValueError('Unmanaged private Python symlink')
            if not target.exists():
                record=json.loads(marker(managed).read_text())
                original=Path(record.get('archived_previous') or '')
                if original.parent!=managed.parent or not original.name.startswith('venv-nfs-original-') or not original.is_dir() or original.is_symlink():
                    raise ValueError('Persistent original private Python is missing')
                target.parent.mkdir(parents=True,exist_ok=True)
                staging=target.with_name('restore-'+uuid.uuid4().hex)
                restored=copy_verified(original,staging)
                manifest=json.dumps(restored,sort_keys=True,separators=(',',':')).encode()
                if hashlib.sha256(manifest).hexdigest()!=record['copied_files_sha256']:
                    raise ValueError('Persistent original private Python hash differs')
                staging.rename(target)
                print('PYTHON_STORAGE_RESTORED',len(restored),flush=True)
            return managed
        users=active_users(managed)
        if allow_current:users=[pid for pid in users if pid!=os.getpid()]
        if managed.exists() and users:
            raise ValueError('Private Python is in use; relocation requires an idle environment')
        if target.exists():
            target.rename(target.with_name('interrupted-'+uuid.uuid4().hex))
        target.parent.mkdir(parents=True,exist_ok=True)
        staging=target.with_name('staging-'+uuid.uuid4().hex)
        print('PYTHON_STORAGE local',target,flush=True)
        if managed.exists():records=copy_verified(managed,staging)
        else:staging.mkdir();records=[]
        staging.rename(target)
        archived=managed.with_name('venv-nfs-original-'+uuid.uuid4().hex) if managed.exists() else None
        manifest=json.dumps(records,sort_keys=True,separators=(',',':')).encode()
        (target.parent/'copied-files.json').write_bytes(manifest)
        record=dict(schema='deeplinkengine-python-storage/v1',logical=str(managed),physical=str(target),
                    archived_previous=str(archived) if archived else None,files_copied=len(records),
                    bytes_copied=sum(r.get('bytes',0) for r in records),copied_files_sha256=hashlib.sha256(manifest).hexdigest(),
                    relocation_verified=True,results_moved=False)
        temporary=marker(managed).with_name(marker(managed).name+'.'+uuid.uuid4().hex+'.tmp')
        temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(marker(managed))
        if archived is not None:managed.rename(archived)
        try:managed.symlink_to(target,target_is_directory=True)
        except BaseException:
            if archived is not None:archived.rename(managed)
            raise
        print('PYTHON_STORAGE_READY',record['files_copied'],record['bytes_copied'],flush=True)
        return managed


def admitted(python,cache):
    managed=canonical(Path(python).parent.parent)
    root=(Path(cache)/'environments').resolve()
    if managed.name!='venv' or not managed.parent.is_relative_to(root):return False
    if not managed.is_symlink() and managed.exists():return managed.is_dir()
    if not marker(managed).is_file():return False
    target=location(managed)
    if managed.is_symlink():return managed.resolve()==target.resolve()
    record=json.loads(marker(managed).read_text());original=Path(record.get('archived_previous') or '')
    return original.parent==managed.parent and original.is_dir() and not original.is_symlink()
