"""Bounded optional OCI ranges; publish only the complete locked SHA256."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import uuid


def fetch(cache, registry, kind, identity, size, *, workers=4, chunk_size=1024**2):
    from baseline_setup import digest, download_lock, fetch as sequential
    if (not re.fullmatch(r'sha256:[0-9a-f]{64}',identity)
            or type(size) is not int or size<=0 or type(workers) is not int or not 1<=workers<=4
            or type(chunk_size) is not int or not 1<=chunk_size<=4*1024**2):
        raise ValueError('Invalid bounded official range geometry')
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    target=cache/identity.replace(':','-');pin=identity.split(':')[1]
    # A sequential prefix is deliberately preserved in its original format.
    # Do not preallocate a second large object or reinterpret that prefix.
    if target.with_suffix('.download.partial').exists():
        print('BASELINE_RANGE_SEQUENTIAL_PREFIX_RETAINED',identity,flush=True)
        return sequential(cache,registry,kind,identity,size,attempts=0)
    lock=target.with_suffix('.download.lock')
    if lock.is_symlink():raise ValueError('Range lock must not be a link')
    with lock.open('a+b') as stream, download_lock(stream):
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file() or target.stat().st_size!=size or digest(target)!=pin:
                raise ValueError('Existing official object differs')
            return target
        partial=target.with_suffix('.range.partial');state_path=target.with_suffix('.range-state.json')
        parts=target.with_suffix('.range-parts')
        if any(p.is_symlink() for p in (partial,state_path,parts)):
            raise ValueError('Range cache paths must not be links')
        signature=dict(schema='deeplinkengine-official-range-state/v1',digest=identity,bytes=size,chunk_size=chunk_size,
                       source=registry+kind+'/'+identity)
        if state_path.exists():
            state=json.loads(state_path.read_bytes())
            if any(state.get(k)!=v for k,v in signature.items()) or not isinstance(state.get('chunks'),dict):
                raise ValueError('Range checkpoint identity differs')
        else:
            if partial.exists():raise ValueError('Unowned range partial retained')
            if shutil.disk_usage(cache).free<size+2*1024**3:
                raise OSError('Range download requires one complete object plus 2 GiB free')
            state=dict(signature,chunks={})
            temporary=state_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(state,sort_keys=True)+'\n',encoding='utf-8');temporary.replace(state_path)
        if not partial.exists():
            if state['chunks']:raise ValueError('Committed range data is missing')
            with partial.open('xb') as f:f.truncate(size)
        elif not partial.is_file() or partial.stat().st_size!=size:
            if state['chunks'] or not partial.is_file() or partial.stat().st_size>size:
                raise ValueError('Range partial geometry differs')
            with partial.open('r+b') as f:f.truncate(size)
        count=(size+chunk_size-1)//chunk_size
        def bounds(index):
            start=index*chunk_size
            return start,min(size,start+chunk_size)-1
        with partial.open('rb') as f:
            for key,row in state['chunks'].items():
                if not key.isdecimal() or not 0<=int(key)<count:raise ValueError('Invalid range checkpoint index')
                start,end=bounds(int(key));f.seek(start);raw=f.read(end-start+1)
                if row!=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()):
                    raise ValueError('Committed range bytes changed; originals retained')
        parts.mkdir(exist_ok=True)
        mutex=threading.Lock();stop=threading.Event();failures=[0]
        def save():
            temporary=state_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(state,sort_keys=True)+'\n',encoding='utf-8');temporary.replace(state_path)
        save()
        def one(index):
            start,end=bounds(index);length=end-start+1;last=None
            for attempt in range(3):
                if stop.is_set():raise OSError('Range download stopped after a bounded failure')
                prefix=parts/(str(index)+'-'+uuid.uuid4().hex)
                body=prefix.with_suffix('.body');headers=prefix.with_suffix('.headers')
                try:
                    result=subprocess.run(['curl','--fail','--location','--silent','--show-error',
                        '--connect-timeout','15','--max-time','60','--range',str(start)+'-'+str(end),
                        '--max-filesize',str(length),'--dump-header',str(headers),'--output',str(body),
                        '--write-out','%{http_code}',signature['source']],capture_output=True,text=True,timeout=70)
                    text=headers.read_text(encoding='utf-8',errors='replace') if headers.exists() else ''
                    values=re.findall(r'(?im)^content-range:\s*([^\r\n]+)',text)
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',values[-1].strip()) if values else None
                    if (result.returncode or result.stdout.strip()!='206' or not match
                            or tuple(map(int,match.groups()))!=(start,end,size)
                            or not body.is_file() or body.stat().st_size!=length):
                        raise OSError('Range HTTP status, boundary, total or bytes differ')
                    raw=body.read_bytes();row=dict(bytes=length,sha256=hashlib.sha256(raw).hexdigest())
                    with mutex:
                        with partial.open('r+b') as f:f.seek(start);f.write(raw);f.flush()
                        state['chunks'][str(index)]=row;save()
                    body.unlink();headers.unlink()
                    return
                except (OSError,subprocess.SubprocessError) as error:
                    last=error
                    with mutex:
                        failures[0]+=1
                        if failures[0]>=12:stop.set()
            raise OSError('Range failed after bounded attempts: '+str(index)) from last
        missing=iter(i for i in range(count) if str(i) not in state['chunks'])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending=set()
            for _ in range(workers):
                index=next(missing,None)
                if index is not None:pending.add(pool.submit(one,index))
            try:
                while pending:
                    done,pending=wait(pending,return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                        index=next(missing,None)
                        if index is not None:pending.add(pool.submit(one,index))
            except BaseException:
                stop.set()
                for future in pending:future.cancel()
                raise
        if len(state['chunks'])!=count or partial.stat().st_size!=size or digest(partial)!=pin:
            raise ValueError('Complete ranged official OCI SHA256 differs; partial retained')
        partial.replace(target)
        print('BASELINE_RANGE_VERIFIED',identity,size,'workers',workers,flush=True)
        return target
