"""Deliver verified binary wheels to pip's native cache; never claim a build."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from urllib.parse import urldefrag, urlsplit, unquote, urljoin, parse_qs
import re

PIP_VERSION = '26.2'


def configure_cargo_transport(environ=None):
    """Use Cargo's native HTTP/1 transport; retain an explicit user setting."""
    environ = os.environ if environ is None else environ
    previous = environ.get('CARGO_HTTP_MULTIPLEXING')
    environ.setdefault('CARGO_HTTP_MULTIPLEXING', 'false')
    return dict(variable='CARGO_HTTP_MULTIPLEXING', value=environ['CARGO_HTTP_MULTIPLEXING'],
                explicit_user_setting=previous is not None, cargo_native_configuration=True,
                registry_or_source_replacement=False, dependency_lock_modified=False,
                original_command_timeout_changed=False, tls_verification_retained=True)


def sha(path):
    with Path(path).open('rb') as stream:
        h=hashlib.sha256()
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def prime(request):
    """Run with the pinned SDK's pip; cache origins name original wheel URLs."""
    import email
    import zipfile
    import pip
    import pip._internal.cache as cache_module
    import pip._internal.models.link as link_module
    import pip._internal.models.direct_url as direct_module
    from pip._internal.cache import WheelCache
    from pip._internal.models.link import Link
    from pip._internal.models.direct_url import DirectUrl
    from pip._internal.utils.compatibility_tags import get_supported
    from pip._internal.utils.temp_dir import global_tempdir_manager
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name,parse_wheel_filename
    from pip._vendor.packaging.version import Version
    if request.get('pip_version')!=PIP_VERSION or pip.__version__!=PIP_VERSION:
        raise ValueError('Verified wheel delivery requires the pinned actual pip '+PIP_VERSION)
    if not Path(pip.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError('The cache implementation must come from the selected SDK interpreter')
    cache_root=Path(request['cache_root']).resolve();canonical=cache_root/'wheels'
    raw_targets=[Path(p) for p in request['pip_caches']]
    if any(p.is_symlink() for p in raw_targets):raise ValueError('Pip delivery cache must not be a link')
    targets=[p.resolve() for p in raw_targets]
    for target in targets:
        if not target.is_relative_to(cache_root/'source-build-downloads'):
            raise ValueError('Pip delivery cache must stay in its owned cache subtree')
        if target.is_symlink():raise ValueError('Pip delivery cache must not be a link')
    admitted=[];skipped=[]
    for row in request['artifacts']:
        filename=row['filename']
        if not filename.endswith('.whl'):
            skipped.append(dict(filename=filename,reason='Not a binary wheel; no build origin fabricated'));continue
        if Path(filename).name!=filename:raise ValueError('Invalid wheel filename')
        path=canonical/filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Canonical binary wheel differs from its verified whole SHA: '+filename)
        url=urldefrag(row.get('upstream_url') or row['url'])[0]
        parsed=urlsplit(url)
        if (parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in ('127.0.0.1','::1','localhost'))) or parsed.username or parsed.password:
            raise ValueError('Cache origin must be the canonical HTTPS wheel URL')
        if not parsed.path.endswith('.whl'):raise ValueError('A source/build origin cannot be substituted for a binary wheel')
        with zipfile.ZipFile(path) as archive:
            members=[n for n in archive.namelist() if n.count('/')==1 and n.endswith('.dist-info/METADATA')]
            if len(members)!=1:raise ValueError('Ambiguous binary wheel metadata')
            metadata=email.message_from_bytes(archive.read(members[0]))
        name=canonicalize_name(metadata['Name']);version=Version(metadata['Version'])
        filename_name,filename_version,_,tags=parse_wheel_filename(filename)
        if (name!=canonicalize_name(row['package']) or name!=filename_name or version!=filename_version
                or ('version' in row and version!=Version(row['version']))
                or not tags.intersection(get_supported())):
            raise ValueError('Binary wheel package/version/tag differs from its declared source')
        expected=row.get('requirement') or (row['package']+'=='+row['version'])
        requirement=Requirement(expected)
        if canonicalize_name(requirement.name)!=name or version not in requirement.specifier:
            raise ValueError('Binary wheel version does not satisfy the original requirement')
        provenance=[]
        for suffix in ('.download.json','.range-receipt.json'):
            item=path.with_name(filename+suffix)
            if item.is_file() and not item.is_symlink():provenance.append(dict(path=str(item),sha256=sha(item)))
        nohash=row.get('no_hash_link_attestation')
        if nohash:
            receipt=Path(nohash['range_receipt'])
            if receipt.is_symlink() or not receipt.is_file() or sha(receipt)!=nohash['range_receipt_sha256']:
                raise ValueError('Unhashed vendor Link lacks its actual verified whole-transfer receipt')
            transfer=json.loads(receipt.read_bytes());publisher=nohash['published_metadata'];published=Path(publisher['path'])
            if (transfer.get('kind')!='verified-public-four-worker-ranges' or transfer.get('source_url')!=url
                    or transfer.get('sha256')!=row['sha256'] or nohash['published_sha256']!=row['sha256']
                    or sha(published)!=publisher['sha256']):
                raise ValueError('Unhashed vendor Link differs from its actual publisher-SHA-qualified body')
            release=json.loads(published.read_bytes())
            if not any(a.get('name')==filename and a.get('size')==row['bytes'] and a.get('digest')=='sha256:'+row['sha256'] for a in release.get('assets',[])):
                raise ValueError('Unhashed vendor Link lacks the exact official release artifact digest')
            index=nohash['index'];index_file=Path(index['index_path'])
            if sha(index_file)!=index['index_sha256'] or index.get('link_has_digest') is not False:
                raise ValueError('Unhashed vendor Link index bytes or no-fragment identity changed')
            from official_build_assets import _Links
            parser=_Links();parser.feed(index_file.read_text(encoding='utf-8'))
            if url not in [urljoin(index['index'],href) for href in parser.links]:
                raise ValueError('Unhashed vendor Link is absent from the original recipe index')
        keys=[(url,None,not bool(nohash))]
        for alias in row.get('cache_aliases',[]):
            key=alias['url'];proof=alias['attestation'];part=urlsplit(key)
            loopback=part.scheme=='http' and part.hostname in ('127.0.0.1','::1','localhost')
            if ((part.scheme!='https' and not loopback) or not part.hostname or part.username or part.password or part.fragment
                    or unquote(part.path.rsplit('/',1)[-1])!=filename
                    or proof.get('advertised_whole_sha256')!=row['sha256'] or proof.get('filename')!=filename
                    or key not in proof.get('matching_urls',[]) or not re.fullmatch(r'[0-9a-f]{64}',proof.get('index_sha256',''))):
                raise ValueError('Binary cache alias lacks its matching index filename and whole SHA')
            index=Path(proof['index_path'])
            if index.is_symlink() or not index.is_file() or sha(index)!=proof['index_sha256']:
                raise ValueError('Binary cache alias index differs from its recorded whole SHA')
            from official_build_assets import _Links
            parser=_Links();parser.feed(index.read_text(encoding='utf-8'));advertised=[]
            for href in parser.links:
                link=urlsplit(urljoin(proof['index'],href))
                if (unquote(link.path.rsplit('/',1)[-1])==filename
                        and parse_qs(link.fragment).get('sha256')==[row['sha256']]):
                    advertised.append(link._replace(fragment='').geturl())
            if key not in advertised:raise ValueError('Binary cache alias was not advertised with this whole SHA')
            if (key,True) not in [(k,h) for k,_,h in keys]:keys.append((key,proof,True))
        for key,attestation,hashed in keys:
            admitted.append(dict(row=row,path=path,url=url,key_url=key,alias_attestation=attestation,
                                 link=Link(key+('#sha256='+row['sha256'] if hashed else '')),cache_key_has_hash=hashed,
                                 name=name,version=str(version),provenance=provenance))
    result=dict(schema='deeplinkengine-verified-wheel-delivery/v1',pip_version=pip.__version__,python=sys.executable,
        pip_api_sources={str(Path(module.__file__).resolve()):sha(module.__file__) for module in (cache_module,link_module,direct_module)},
        helper_sha256=sha(__file__),model_started=False,sdk_payload_modified=False,upstream_requirements_modified=False,
        http_response_or_publisher_headers_fabricated=False,local_build_claimed=False,skipped=skipped,entries=[],started=time.time())
    from baseline_setup import download_lock
    with global_tempdir_manager():
        for target in targets:
            target.mkdir(parents=True,exist_ok=True)
            manager=WheelCache(str(target))
            for item in admitted:
                row=item['row'];destination=Path(manager.get_path_for_link(item['link']))
                if not destination.resolve().is_relative_to(target):raise ValueError('Native cache destination escaped its owned root')
                destination.parent.mkdir(parents=True,exist_ok=True)
                with destination.with_name(destination.name+'.delivery.lock').open('a+b') as lease,download_lock(lease):
                    if destination.is_symlink():raise ValueError('Native wheel cache entry must not be a link')
                    existed=destination.exists()
                    if existed:
                        wheel=destination/row['filename'];origin=destination/'origin.json'
                        if (not destination.is_dir() or wheel.is_symlink() or origin.is_symlink()
                                or not wheel.is_file() or not origin.is_file()
                                or {p.name for p in destination.glob('*.whl')}!={row['filename']}
                                or wheel.stat().st_size!=row['bytes'] or sha(wheel)!=row['sha256']):
                            raise ValueError('Existing native wheel cache differs; preserve '+str(destination))
                        data=json.loads(origin.read_bytes())
                        if data.get('url')!=item['url'] or data.get('archive_info',{}).get('hashes',{}).get('sha256')!=row['sha256']:
                            raise ValueError('Existing native wheel origin differs; preserve '+str(origin))
                    else:
                        staging=Path(tempfile.mkdtemp(prefix=destination.name+'.delivery-',dir=destination.parent))
                        shutil.copyfile(item['path'],staging/row['filename'])
                        if sha(staging/row['filename'])!=row['sha256']:raise ValueError('Wheel delivery copy changed')
                        manager.record_download_origin(str(staging),DirectUrl.from_json(json.dumps(dict(
                            url=item['url'],archive_info=dict(hashes={'sha256':row['sha256']})))))
                        staging.rename(destination)
                    entry=manager.get_cache_entry(item['link'],item['name'],get_supported())
                    if entry is None or Path(entry.link.file_path)!=destination/row['filename']:
                        raise ValueError('Actual pip did not admit the delivered wheel')
                    result['entries'].append(dict(filename=row['filename'],version=item['version'],canonical_source_url=item['url'],
                        native_cache_key_url=item['key_url'],transport_alias_attestation=item['alias_attestation'],
                        cache_key_has_hash=item['cache_key_has_hash'],no_hash_link_attestation=row.get('no_hash_link_attestation'),
                        source_kind='published-binary-wheel',source_path=str(item['path']),sha256=row['sha256'],bytes=row['bytes'],
                        acquisition_kind='verified-local-cache',acquisition_receipts=item['provenance'],
                        observed_http_fetch=False,local_build_claimed=False,cache_path=str(destination),reused=existed,
                        origin_json_sha256=sha(destination/'origin.json')))
    result.update(status='completed',ended=time.time());return result


def invoke(python,env,sdk,cache,rows,caches,out):
    from official_execution import sdk_run
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    request=dict(pip_version=PIP_VERSION,cache_root=str(Path(cache).resolve()),artifacts=rows,pip_caches=list(map(str,caches)))
    path=out/'request.json';save(path,request)
    with (out/'prime.log').open('xb') as log:
        child=sdk_run([str(python),'-B','-S',str(Path(__file__).resolve()),'--prime',str(path),str(out/'result.json')],
                      sdk=sdk,env=env,stdout=log,stderr=-2,timeout=180)
    if child.returncode or not (out/'result.json').is_file():
        raise ValueError('Verified wheel delivery failed; preserve '+str(out))
    result=json.loads((out/'result.json').read_bytes())
    return dict(path=str(out/'result.json'),sha256=sha(out/'result.json'),entries=len(result['entries']),pip_version=result['pip_version'])


def invoke_collection(python,env,sdk,cache,out,*,timeout=3600):
    """The actual admitted SDK owns markers, tags, packaging and tomllib."""
    from official_execution import sdk_popen
    out=Path(out);control=out.with_name(out.name+'-control');control.mkdir(parents=True,exist_ok=False)
    request=control/'request.json';result=control/'result.json'
    save(request,dict(cache=str(cache),out=str(out)))
    command=[str(python),'-B','-S',str(Path(__file__).resolve()),'--collect',str(request),str(result)]
    record=dict(status='running',command=command,selected_sdk=str(sdk),python_sha256=sha(python),
        timeout_seconds=timeout,original_graph_metadata_timeout_changed=False,model_started=False,started=time.time())
    process=None
    try:
        with (control/'collect.log').open('xb') as log:
            process=sdk_popen(command,sdk=sdk,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record['pid']=process.pid
        stat=Path('/proc',str(process.pid),'stat')
        record['start_ticks']=stat.read_text().rsplit(')',1)[1].split()[19] if stat.exists() else None
        save(control/'receipt.json',record)
        code=process.wait(timeout=timeout);record['returncode']=code
        if code or not result.is_file():raise ValueError('SDK-only build input collection failed; preserve '+str(control))
        value=json.loads(result.read_bytes());actual=value['runtime']
        if (value.get('status')!='completed' or Path(actual['python']).resolve()!=Path(python).resolve()
                or (env.get('PYTHONHOME') and Path(actual['prefix']).resolve()!=Path(env['PYTHONHOME']).resolve())):
            raise ValueError('Build-input metadata runtime differs from the selected SDK')
        record.update(status='completed',runtime=actual,result_sha256=sha(result))
        return value['artifacts']
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=type(error).__name__+': '+str(error));raise
    finally:
        if process is not None and process.poll() is None:
            try:
                if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
                else:process.terminate()
            except ProcessLookupError:pass
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
                    else:process.kill()
                except ProcessLookupError:pass
                process.wait(timeout=10)
        record.update(ended=time.time(),quiescent=process is None or process.poll() is not None)
        save(control/'receipt.json',record)


def primary_aliases(rows,out,*,scope='fixed-primary-binary-inputs'):
    """Retain original wheel origins and attest only same-byte transport keys."""
    from official_build_assets import transports,_read
    out=Path(out);out.mkdir(parents=True,exist_ok=False);result=[];observations=[]
    for row in rows:
        aliases=[]
        if row['filename'].endswith('.whl'):
            _,proofs=transports(row)
            for proof in proofs:
                if not proof.get('matching_urls'):continue
                try:
                    raw=_read(proof['index']);digest=hashlib.sha256(raw).hexdigest()
                    if digest!=proof['index_sha256']:raise ValueError('Mirror index changed between transport selection and attestation')
                    index=out/(digest+'.html')
                    if not index.exists():index.write_bytes(raw)
                    attestation=dict(proof,index_path=str(index),filename=row['filename'],advertised_whole_sha256=row['sha256'])
                    aliases.extend(dict(url=url,attestation=attestation) for url in proof['matching_urls'])
                    observations.append(dict(status='attested',**attestation))
                except (OSError,ValueError) as error:
                    observations.append(dict(filename=row['filename'],status='unavailable',index=proof['index'],error=type(error).__name__+': '+str(error)))
        result.append(dict(row,cache_aliases=aliases))
    save(out/'results.json',dict(scope=scope,observations=observations,
        actual_mirror_body_download_claimed=False,canonical_origins_retained=True))
    return result


def retain_delivery_qualification(row,fixed):
    if fixed is None:return row
    result=dict(row,cache_aliases=fixed.get('cache_aliases',[]))
    if fixed.get('no_hash_link_attestation'):result['no_hash_link_attestation']=fixed['no_hash_link_attestation']
    return result


def closure_aliases(rows,primary,out):
    """Attest new resolver transports while retaining earlier qualifications."""
    fixed={(row['filename'],row['sha256']) for row in primary}
    pending=[row for row in rows if (row['filename'],row['sha256']) not in fixed]
    enriched=primary_aliases(pending,out,scope='new-resolved-closure-binary-inputs')
    by_key={(row['filename'],row['sha256']):row for row in enriched}
    return [by_key.get((row['filename'],row['sha256']),row) for row in rows]


def prepare(cache,sdk,out):
    """Owned caller invokes this under the existing SDK mutex, before new prepare."""
    import official_source_candidate as candidate
    from official_image import environment
    from official_build_assets import fetch,prefetch_all
    from official_sdk_state import sdk_preparation
    from official_image import file_map
    cache=Path(cache).resolve();sdk=Path(sdk).resolve();out=Path(out)
    key=candidate.key()
    if sdk!=cache/'source-candidate-sdks'/(candidate.ID+'-'+key[:16]) or sdk.exists():
        raise ValueError('Delivery preparation requires this recipe\'s new, absent SDK')
    out.mkdir(parents=True,exist_ok=False)
    base=cache/'official-sdks/0.23.0-4ee78def8f33d59d'
    primary=list(candidate.build_dependency_artifacts().values())
    record=dict(schema='deeplinkengine-source-delivery-preparation/v1',status='running',started=time.time(),
        recipe_key=key,bound_preparation_sources=candidate.identity()['preparation_sources'],
        candidate_controller_sha256=candidate.identity()['controller_sha256'],
        helper_sha256=sha(__file__),sdk_payload_modified=False,source_graph_modified=False,
        original_constraints_and_pep517_isolation_retained=True,model_started=False,steps=[])
    caches=[cache/'source-build-downloads/verified-metadata-pip',cache/'source-build-downloads'/key/'pip']
    try:
        record['cargo_transport'] = configure_cargo_transport()
        save(out/'results.json',record)
        # Apply the original candidate's raw-base admission before any metadata
        # subprocess. This does not repair, copy, or re-sign an SDK payload.
        with sdk_preparation(base):
            installed=json.loads((base/'installation.json').read_bytes())
            lock=Path(candidate.__file__).parent/'engine/official-baselines/0.23.0.json'
            if (installed['image_manifest']!=candidate.BASE_MANIFEST
                    or (base/'image-lock.json').read_bytes()!=lock.read_bytes()
                    or sha(base/'installed-files.json')!=installed['installed_files_sha256']
                    or file_map(base)!=json.loads((base/'installed-files.json').read_bytes())):
                raise ValueError('Official source candidate base integrity failed before delivery')
        record['base_whole_map_verified']=True;save(out/'results.json',record)
        python,env=environment(base,out/'runtime-cache');env.update(ASCEND_RT_VISIBLE_DEVICES='',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
        for row in primary:
            if row['filename'].endswith('.whl'):fetch(row,cache/'wheels')
        primary=invoke_collection(python,env,base,cache,out/'static-build-input-prefetch')
        record['static_build_input_prefetch']=dict(path=str(out/'static-build-input-prefetch/results.json'),sha256=sha(out/'static-build-input-prefetch/results.json'))
        primary=primary_aliases(primary,out/'primary-alias-indices')
        record['primary_alias_attestation']=dict(path=str(out/'primary-alias-indices/results.json'),sha256=sha(out/'primary-alias-indices/results.json'))
        record['steps'].append(invoke(python,env,base,cache,primary,caches,out/'primary-delivery'));save(out/'results.json',record)
        closure=out/'original-closure'
        if prefetch_all(dict(cache=str(cache),out=str(closure),versions=[candidate.VERSION]),[])!=0:
            raise ValueError('Original source-build closure preflight failed')
        closed=json.loads((closure/'results.json').read_bytes())
        if closed['status']!='completed':raise ValueError('Original closure preflight is incomplete')
        wheels={}
        for stage in closed['stages']:
            lock=Path(stage['lock'])
            if sha(lock)!=stage['lock_sha256']:raise ValueError('Original closure lock changed')
            data=json.loads(lock.read_bytes())
            for row in data['approved_artifacts']:
                fixed=next((p for p in primary if p['filename']==row['filename'] and p['sha256']==row['sha256']),None)
                row=retain_delivery_qualification(row,fixed)
                wheels[(row['filename'],row['sha256'])]=row
        record['steps'].append(dict(closure=str(closure/'results.json'),sha256=sha(closure/'results.json')))
        resolved=closure_aliases(list(wheels.values()),primary,out/'closure-alias-indices')
        record['closure_alias_attestation']=dict(path=str(out/'closure-alias-indices/results.json'),
            sha256=sha(out/'closure-alias-indices/results.json'))
        record['steps'].append(invoke(python,env,base,cache,resolved,caches,out/'closure-delivery'))
        record['status']='completed'
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=type(error).__name__+': '+str(error));raise
    finally:
        record['ended']=time.time();save(out/'results.json',record)
    return dict(path=str(out/'results.json'),sha256=sha(out/'results.json'))


if __name__=='__main__':
    if len(sys.argv)!=4 or sys.argv[1] not in ('--prime','--collect'):raise SystemExit('Internal verified-wheel delivery; use public run_all.py')
    request=json.loads(Path(sys.argv[2]).read_bytes())
    try:
        if sys.argv[1]=='--collect':
            from official_source_prefetch import collect
            artifacts=collect(request['cache'],request['out']);result=json.loads((Path(request['out'])/'results.json').read_bytes())
        else:result=prime(request)
    except BaseException as error:
        save(sys.argv[3],dict(status='failed',error=type(error).__name__+': '+str(error),helper_sha256=sha(__file__)))
        raise
    save(sys.argv[3],result)
