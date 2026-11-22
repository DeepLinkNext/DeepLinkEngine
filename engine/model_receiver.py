"""Immutable model transport with retained ranges, receipts and one output owner."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,os,re,shutil,threading,time,urllib.request,uuid
from pathlib import Path
from urllib.parse import quote,urlsplit,urlunsplit
RANGE_BYTES=16*1024**2
NETWORK_TIMEOUT=45
MAX_FAILURES=3

def transport_base_url(value):
    if value is None:return None
    if not isinstance(value,str):raise ValueError('Model transport base URL must be HTTP(S)')
    p=urlsplit(value)
    if p.scheme not in ('http','https') or not p.hostname or p.username is not None or p.password is not None or p.query or p.fragment:
        raise ValueError('Model transport URL requires HTTP(S), without credentials, query or fragment')
    try:p.port
    except ValueError:raise ValueError('Invalid model transport port') from None
    return urlunsplit((p.scheme,p.netloc,p.path.rstrip('/'),'',''))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with tmp.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def regular(path):return path.is_file() and not path.is_symlink()

def restore(lock,output,*,transport_base_url=None):
    from engine.model_restore import manifest
    from benchmark.baseline_setup import download_lock
    value=manifest(lock);pin=digest(lock);base=globals()['transport_base_url'](transport_base_url);output=Path(output).absolute()
    if output.is_symlink():raise ValueError('Model destination must not be a symlink')
    output.mkdir(parents=True,exist_ok=True);lease_path=output.resolve().parent/(output.name+'.model-restoration.lock')
    if lease_path.is_symlink():raise ValueError('Model lease must not be a symlink')
    with lease_path.open('a+b') as lease,download_lock(lease):return _restore(value,pin,output,base)

def _restore(value,pin,output,base):
    attempt=uuid.uuid4().hex;history=output/'model-restoration-attempts';canonical=output/'model-restoration.json'
    if history.is_symlink() or canonical.is_symlink():raise ValueError('Unsafe model receipt path')
    history.mkdir(exist_ok=True)
    if canonical.exists():shutil.copyfile(canonical,history/('previous-'+attempt+'.json'))
    receipt=dict(schema='deeplinkengine-model-restoration/v1',started=time.time(),manifest_sha256=pin,repository=value['repository'],transport_base_url=base,files={},status='running',attempt=attempt,expected_file_count=len(value['files']),model_service_started=False,formal_client_started=False)
    guard=threading.RLock();stop=threading.Event()
    def report(name=None,facts=None):
        with guard:
            if name is not None:receipt['files'][name]=dict(facts)
            save(history/(attempt+'.json'),receipt);save(canonical,receipt)
    report();pool=None;futures=[]
    try:
        stage_root=output.resolve().parent/('.'+output.name+'-model-downloads')
        if stage_root.is_symlink():raise ValueError('Model staging root must not be a symlink')
        stage_root.mkdir(exist_ok=True);stages=stage_root/pin
        if stages.is_symlink():raise ValueError('Model staging must not be a symlink')
        stages.mkdir(parents=True,exist_ok=True)
        def one(item):
            name,row=item;origin='https://modelscope.cn/models/'+value['repository']+'/resolve/'+row['revision']+'/'+quote(name);url=base+'/'+quote(name) if base is not None else origin
            summary=dict(bytes=row['bytes'],sha256=row['sha256'],revision=row['revision'],url=origin,transport_url=url,status='running');report(name,summary)
            def progress(facts):summary.update(facts);report(name,summary)
            try:
                if stop.is_set():raise InterruptedError('Other model file failed; retain transport state')
                result=_one(name,row,origin,url,pin,output,stages,stop,progress)
                report(name,dict(summary,**result,status='verified'));print('MODEL_FILE_VERIFIED',name,row['sha256'],flush=True)
            except BaseException as error:
                if summary.get('state_path'):
                    state_path=Path(summary['state_path']);state=json.loads(state_path.read_bytes());state.update(lifecycle_status='failed',ended=time.time(),error=type(error).__name__+': '+str(error));save(state_path,state)
                report(name,dict(summary,status='failed',error=type(error).__name__+': '+str(error)));raise
        pool=ThreadPoolExecutor(max_workers=3);futures=[pool.submit(one,item) for item in value['files'].items()]
        for f in as_completed(futures):f.result()
    except BaseException as error:
        stop.set()
        for f in futures:f.cancel()
        if pool:pool.shutdown(wait=True)
        receipt.update(status='failed',ended=time.time(),error=type(error).__name__+': '+str(error));report();raise
    else:
        pool.shutdown(wait=True);receipt.update(status='completed',ended=time.time());report();return receipt

def _one(name,row,origin,url,pin,output,stages,stop,report):
    target=output/name
    if target.is_symlink() or (target.exists() and not target.is_file()):raise ValueError('Unsafe existing model file: '+name)
    if target.exists():
        if target.stat().st_size!=row['bytes'] or digest(target)!=row['sha256']:raise ValueError('Existing model file differs; preserve it and choose a new directory: '+name)
        return dict(admission='verified-existing')
    legacy=[output/(name+s) for s in ('.download-part','.download.partial')]
    for p in legacy:
        if p.is_symlink() or (p.exists() and not p.is_file()):raise ValueError('Unsafe legacy model partial')
        if p.exists() and p.stat().st_size==row['bytes']:
            before=p.stat();h=digest(p);after=p.stat()
            if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('Legacy model partial changed')
            if h!=row['sha256']:raise ValueError('Full-size model partial has wrong whole SHA-256; preserved')
            os.link(p,target);return dict(admission='whole-SHA-verified-full-size-partial',legacy_partial=str(p))
        if p.exists() and p.stat().st_size>row['bytes']:raise ValueError('Oversized legacy model partial; preserved')
    folder=stages/name
    if folder.is_symlink():raise ValueError('Model file staging must not be a symlink')
    folder.mkdir(exist_ok=True);path=folder/'state.json';expected=dict(manifest_sha256=pin,name=name,bytes=row['bytes'],sha256=row['sha256'],revision=row['revision'],origin_url=origin)
    if path.exists():
        if path.is_symlink():raise ValueError('Model state must not be a symlink')
        state=json.loads(path.read_bytes())
        if state.get('schema')!='deeplinkengine-model-file-transport/v1' or state.get('identity')!=expected:raise ValueError('Model partial has a different pinned identity')
    else:
        if list(folder.iterdir()):raise ValueError('Unowned model staging data; preserve it')
        state=dict(schema='deeplinkengine-model-file-transport/v1',identity=expected,segments=[],attempts=[],status='preparing');save(path,state)
    if state.get('lifecycle_status')=='failed':
        state.setdefault('previous_failures',[]).append(dict(ended=state.get('ended'),error=state.get('error')))
        state.pop('error',None);state.pop('ended',None)
    state['lifecycle_status']='running'
    def checkpoint():save(path,state);report(dict(state_path=str(path),retained_bytes=sum(s['bytes'] for s in state['segments'])))
    used={s['file'] for s in state['segments']}
    for request in state['attempts']:
        if request.get('status')=='receiving' and request.get('file') not in used:
            p=folder/request['file']
            if p.exists():
                if not regular(p) or Path(request['file']).name!=request['file']:raise ValueError('Unsafe unsealed model response')
                kept=p.with_name(p.name+'.unsealed-'+uuid.uuid4().hex);p.rename(kept);request.update(status='preserved-unsealed-after-interruption',preserved_file=kept.name,bytes=kept.stat().st_size)
    if not state['segments']:
        candidates=[p for p in legacy if p.exists() and p.stat().st_size]
        if candidates:
            p=max(candidates,key=lambda p:p.stat().st_size);before=p.stat();snapshot=folder/('legacy-prefix-'+uuid.uuid4().hex);shutil.copyfile(p,snapshot);after=p.stat();h=digest(snapshot)
            if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or snapshot.stat().st_size!=before.st_size or digest(p)!=h:raise ValueError('Legacy model prefix changed during capture')
            state['segments'].append(dict(start=0,bytes=before.st_size,sha256=h,file=snapshot.name,kind='unqualified-legacy-prefix',source_file=str(p)))
    offset=0
    for segment in state['segments']:
        p=folder/segment['file']
        if Path(segment['file']).name!=segment['file'] or segment['start']!=offset or not regular(p) or p.stat().st_size!=segment['bytes'] or digest(p)!=segment['sha256']:raise ValueError('Retained model prefix changed; preserve transport state')
        offset+=segment['bytes']
    if offset>row['bytes']:raise ValueError('Model prefix exceeds pinned size')
    checkpoint();failures=0;opener=urllib.request.build_opener(urllib.request.ProxyHandler({})) if urlsplit(url).scheme=='http' else urllib.request.build_opener()
    while offset<row['bytes']:
        if stop.is_set():raise InterruptedError('Model restoration interrupted; prefix retained')
        whole_small=row['bytes']<=RANGE_BYTES;start=0 if whole_small else offset
        end=min(row['bytes']-1,start+RANGE_BYTES-1);part=folder/('range-'+str(start)+'-'+uuid.uuid4().hex+'.partial');count=0;error=None;valid=False;began=time.monotonic()
        request=dict(start=start,end=end,total=row['bytes'],origin_url=origin,transport_url=url,file=part.name,status='requesting',started=time.time(),retained_prefix_before_request=offset);state['attempts'].append(request);checkpoint()
        try:
            req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
            with opener.open(req,timeout=NETWORK_TIMEOUT) as response:
                request.update(response_status=response.status,content_range=response.headers.get('Content-Range'),content_length=response.headers.get('Content-Length'))
                expected_range=f'bytes {start}-{end}/{row["bytes"]}'
                bounded_whole_200=(whole_small and start==0 and end==row['bytes']-1 and response.status==200 and response.headers.get('Content-Range') in (None,expected_range))
                if ((not bounded_whole_200 and (response.status!=206 or response.headers.get('Content-Range')!=expected_range)) or response.headers.get('Content-Length')!=str(end-start+1)):raise ValueError('Model response requires exact HTTP206/Content-Range/total size or bounded whole-small HTTP200')
                valid=True;request.update(status='receiving',headers_validated=True,response_mode='bounded-whole-small-200' if bounded_whole_200 else 'exact-range-206');checkpoint()
                with part.open('xb') as stream:
                    try:
                        while count<end-start+1:
                            if stop.is_set():raise InterruptedError('Model restoration interrupted; retain range prefix')
                            b=getattr(response,'read1',response.read)(min(1024**2,end-start+1-count))
                            if not b:raise OSError('Model response truncated; retain range prefix')
                            stream.write(b);count+=len(b);elapsed=time.monotonic()-began
                            if elapsed>=45 and count/elapsed<1024:raise TimeoutError('Model source below 1 KiB/s; retain range prefix')
                    finally:stream.flush();os.fsync(stream.fileno())
        except BaseException as failure:error=failure
        if valid and part.exists() and part.stat().st_size:
            n=part.stat().st_size;h=digest(part);request['received_sha256']=h
            if n>end-start+1:raise ValueError('Model range exceeded requested extent')
            if whole_small:
                # A small provider object may use HTTP200. It is never appended
                # to a prefix: retain old pieces, then admit only a whole hash.
                if n==row['bytes'] and h==row['sha256']:
                    if state['segments']:state.setdefault('superseded_small_prefixes',[]).append(state['segments'])
                    state['segments']=[dict(start=0,bytes=n,sha256=h,file=part.name,kind='range-response',transport_url=url,request_index=len(state['attempts'])-1,interrupted=error is not None)];offset=n
                elif error is None:error=ValueError('Small whole response SHA-256 mismatch; preserved without prefix append')
            else:
                state['segments'].append(dict(start=offset,bytes=n,sha256=h,file=part.name,kind='range-response',transport_url=url,request_index=len(state['attempts'])-1,interrupted=error is not None));offset+=n
        request.update(status='completed' if error is None else 'failed-prefix-retained',ended=time.time(),received_bytes=count,error=None if error is None else type(error).__name__+': '+str(error));state['status']='downloading';checkpoint()
        if error is not None:
            if isinstance(error,(KeyboardInterrupt,SystemExit,InterruptedError)):raise error
            failures=failures+1 if whole_small else (0 if count>=1024**2 else failures+1)
            if failures>=MAX_FAILURES:raise OSError('Model transport failed; prefix and request facts preserved: '+str(error)) from error
        else:failures=0
    assembly=folder/('assembled-'+uuid.uuid4().hex+'.partial');state.update(status='assembling',assembly=assembly.name);checkpoint()
    with assembly.open('xb') as stream:
        for segment in state['segments']:
            p=folder/segment['file']
            if not regular(p) or p.stat().st_size!=segment['bytes'] or digest(p)!=segment['sha256']:raise ValueError('Model range changed before assembly')
            with p.open('rb') as source:shutil.copyfileobj(source,stream,4*1024**2)
        stream.flush();os.fsync(stream.fileno())
    if assembly.stat().st_size!=row['bytes'] or digest(assembly)!=row['sha256']:
        state['status']='failed-whole-SHA';checkpoint();raise ValueError('Model download whole SHA-256 mismatch; all bytes preserved')
    if target.exists():raise ValueError('Model destination appeared while restoring')
    assembly.replace(target);state.update(status='completed',lifecycle_status='completed',completed=time.time(),canonical=str(target));checkpoint()
    retired=[]
    for segment in state['segments']:
        if segment.get('kind')=='range-response' and not segment.get('interrupted'):
            p=folder/segment['file']
            if regular(p) and p.stat().st_size==segment['bytes'] and digest(p)==segment['sha256']:p.unlink();retired.append(segment['file'])
    state['successful_response_files_retired_after_whole_admission']=retired;checkpoint()
    return dict(admission='whole-SHA-verified-restoration',state_path=str(path))
