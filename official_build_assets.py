"""Resolve original isolated build requirements and cache verified source bytes."""
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from email.parser import BytesParser

PYPI_MIRRORS=('https://pypi.tuna.tsinghua.edu.cn/simple/',
              'https://mirror.sjtu.edu.cn/pypi/web/simple/')
CPU_MIRROR='https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/'
CPU_ORIGIN='https://download.pytorch.org/whl/cpu/'
ASCEND_ORIGIN='https://mirrors.huaweicloud.com/ascend/repos/pypi/'


def sha(path):
    from official_image import digest
    return digest(Path(path))


def _read(url,limit=8*1024**2):
    began=time.monotonic();chunks=[];size=0
    with urllib.request.urlopen(url,timeout=30) as response:
        while True:
            if time.monotonic()-began>120:raise TimeoutError('Build metadata exceeded its 120 second budget')
            data=getattr(response,'read1',response.read)(64*1024)
            if not data:break
            chunks.append(data);size+=len(data)
            if size>limit:raise ValueError('Build index exceeded its metadata bound')
    return b''.join(chunks)


class _Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if tag=='a' and values.get('href'):self.links.append(values['href'])


def matching_links(index,row):
    """Select transports only when both filename and advertised SHA are equal."""
    raw=_read(index);parser=_Links();parser.feed(raw.decode('utf-8'));links=[]
    for href in parser.links:
        url=urllib.parse.urljoin(index,href);parsed=urllib.parse.urlsplit(url)
        if (parsed.scheme=='https' and parsed.username is None and parsed.password is None
                and urllib.parse.unquote(parsed.path.rsplit('/',1)[-1])==row['filename']
                and urllib.parse.parse_qs(parsed.fragment).get('sha256')==[row['sha256']]):
            links.append(urllib.parse.urlunsplit(parsed._replace(fragment='')))
    return links,dict(index=index,index_sha256=hashlib.sha256(raw).hexdigest(),matching_urls=links)


def transports(row):
    package=re.sub(r'[-_.]+','-',row['package']).lower();urls=[];proof=[]
    indexes=[CPU_MIRROR+package+'/'] if '+cpu' in row['filename'] else [base+package+'/' for base in PYPI_MIRRORS]
    for index in indexes:
        try:
            found,record=matching_links(index,row);urls+=found;proof.append(record)
        except (OSError,ValueError) as error:proof.append(dict(index=index,error=type(error).__name__+': '+str(error)))
    urls+= [row.get('api_url',row['url']),row['url']]
    return list(dict.fromkeys(urls)),proof


def fetch(row,cache):
    """Use the existing Range-resume downloader and its standard wheel mutex."""
    from baseline_setup import download_lock
    from environment_setup import _download_dependency,wheel_digest
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);target=cache/row['filename']
    if Path(row['filename']).name!=row['filename']:raise ValueError('Unsafe build artifact filename')
    def valid(path):return path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and wheel_digest(path)==row['sha256']
    with (cache/(row['filename']+'.lock')).open('a+b') as lease,download_lock(lease):
        if valid(target):return target,dict(cached=True,filename=row['filename'],sha256=row['sha256'])
        if target.is_symlink():raise ValueError('Build cache artifact is a symlink')
        if target.exists():raise ValueError('Build cache artifact has wrong whole SHA; preserve '+str(target))
        if shutil.disk_usage(cache).free<row['bytes']+2*1024**3:
            raise OSError('Build artifact prefetch requires one whole artifact plus 2 GiB free')
        urls,proof=transports(row)
        if row['bytes']>=16*1024**2 and shutil.which('curl'):
            return fetch_ranges(row,cache,target,urls,proof)
        path=_download_dependency(row,target,urls,valid)
        if not valid(path):raise ValueError('Build artifact failed fresh whole SHA admission')
        return path,dict(cached=False,filename=row['filename'],sha256=row['sha256'],mirror_indexes=proof)


def fetch_ranges(row,cache,target,urls,index_proof):
    """Reuse the public four-worker verified downloader for large build assets."""
    from official_range_download import fetch as ranged
    import tempfile
    attempts=[]
    for url in urls:
        folder=cache.parent/'source-build-downloads/ranges'/hashlib.sha256(url.encode()).hexdigest()[:20]
        try:
            path=ranged(folder,'','','sha256:'+row['sha256'],row['bytes'],workers=4,source_url=url)
            if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
                raise ValueError('Ranged build artifact differs from its publisher whole SHA')
            with tempfile.NamedTemporaryFile(dir=cache,prefix=row['filename']+'.ranged-',delete=False) as stream:
                temporary=Path(stream.name)
                with path.open('rb') as reader:shutil.copyfileobj(reader,stream,1024**2)
            if temporary.stat().st_size!=row['bytes'] or sha(temporary)!=row['sha256']:
                raise ValueError('Ranged cache copy changed; preserve '+str(temporary))
            temporary.replace(target)
            proof=dict(cached=False,kind='verified-public-four-worker-ranges',filename=row['filename'],
                sha256=row['sha256'],source_url=url,source_path=str(path),workers=4,mirror_indexes=index_proof,
                attempts=attempts,sequential_prefix_preserved=str(target.with_name(row['filename']+'.download.partial')))
            from official_baselines import save
            save(target.with_name(row['filename']+'.source-ranges.json'),proof)
            return target,proof
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            attempts.append(dict(url=url,error=type(error).__name__+': '+str(error)))
    raise OSError('Pinned build artifact failed all bounded Range transports: '+row['filename']+'; '+str(attempts))


def admit_pip_cache(row,cache,*,pip_cache,report_url):
    """Copy only this report URL's verified whole body from our metadata cache."""
    if urllib.parse.urlsplit(report_url).scheme not in ('https','http'):
        return None
    from baseline_setup import download_lock
    from pip._internal.network.cache import SafeFileCache
    from pip._vendor.cachecontrol.controller import CacheController
    import tempfile
    root=Path(pip_cache).resolve();http=SafeFileCache(str(root/'http-v2'))
    key=CacheController.cache_url(report_url)
    source=Path(http._get_cache_path(key)+'.body')
    if (source.is_symlink() or not source.is_file()
            or not source.resolve().is_relative_to(root)
            or source.stat().st_size!=row['bytes'] or sha(source)!=row['sha256']):
        return None
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);target=cache/row['filename']
    if Path(row['filename']).name!=row['filename']:raise ValueError('Unsafe metadata cache artifact filename')
    with (cache/(row['filename']+'.lock')).open('a+b') as lease,download_lock(lease):
        if target.is_symlink():raise ValueError('Build cache artifact is a symlink')
        if target.exists():
            if target.stat().st_size!=row['bytes'] or sha(target)!=row['sha256']:
                raise ValueError('Build cache artifact has wrong whole SHA; preserve '+str(target))
            return target,dict(cached=True,filename=row['filename'],sha256=row['sha256'])
        if shutil.disk_usage(cache).free<row['bytes']+2*1024**3:
            raise OSError('Metadata cache admission requires one whole artifact plus 2 GiB free')
        with tempfile.NamedTemporaryFile(dir=cache,prefix=row['filename']+'.metadata-cache-',delete=False) as output:
            temporary=Path(output.name)
            with source.open('rb') as stream:shutil.copyfileobj(stream,output,1024**2)
        if temporary.stat().st_size!=row['bytes'] or sha(temporary)!=row['sha256']:
            raise ValueError('Metadata cache body changed; preserve '+str(temporary))
        temporary.replace(target)
        return target,dict(cached=True,kind='verified-metadata-http-cache',source_path=str(source),
            report_url=report_url,filename=row['filename'],sha256=row['sha256'],
            pip_version=__import__('pip').__version__,sdk_installation_performed=False)


def authority(install,primary):
    """Check a resolver choice against the original publisher's artifact list."""
    info=install['download_info'];url=info['url'];filename=urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit('/',1)[-1]);meta=install['metadata'];name=re.sub(r'[-_.]+','-',meta['name']).lower();version=meta['version'];digest=info.get('archive_info',{}).get('hashes',{}).get('sha256')
    if digest is not None and not re.fullmatch(r'[0-9a-f]{64}',digest):raise ValueError('Resolver artifact has an invalid whole SHA-256')
    if filename in primary:
        row=dict(primary[filename]);
        if digest is not None and row['sha256']!=digest:raise ValueError('Resolver changed a fixed stage artifact')
        return dict(row,resolver_advertised_sha256=digest is not None)
    if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]*',name):raise ValueError('Unsafe resolved package name')
    source='https://pypi.org/pypi/'+urllib.parse.quote(name,safe='')+'/'+urllib.parse.quote(version,safe='')+'/json'
    try:
        raw=_read(source);data=json.loads(raw);matches=[r for r in data['urls'] if r['filename']==filename and (digest is None or r['digests']['sha256']==digest)]
        if len(matches)!=1:raise ValueError('Resolver artifact is absent from the official PyPI release')
        selected=matches[0];publisher_digest=selected['digests']['sha256']
        if not re.fullmatch(r'[0-9a-f]{64}',publisher_digest):raise ValueError('Publisher artifact has an invalid whole SHA-256')
        return dict(filename=filename,bytes=selected['size'],sha256=publisher_digest,url=selected['url'],upstream_url=selected['url'],package=name,version=version,publisher_metadata_url=source,publisher_metadata_sha256=hashlib.sha256(raw).hexdigest(),resolver_advertised_sha256=digest is not None,publisher_digest_advertised=True)
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
    if digest is None:
        raise ValueError('Resolver and official PyPI publisher supply no unique pinned artifact SHA-256: '+filename)
    # CPU variants and Triton-Ascend are also published in the exact upstream
    # indexes used by the Docker recipe. They must advertise the same digest.
    row=dict(filename=filename,sha256=digest,package=name)
    indexes=[CPU_ORIGIN+name+'/',ASCEND_ORIGIN+name+'/']
    for index in indexes:
        try:
            links,proof=matching_links(index,row)
            if links:
                request=urllib.request.Request(links[0],method='HEAD')
                with urllib.request.urlopen(request,timeout=30) as response:size=int(response.headers['Content-Length'])
                return dict(row,bytes=size,url=links[0],upstream_url=links[0],version=version,publisher_index=proof)
            # Some official Ascend indexes do not advertise archive digests.
            # Accept only the exact HTTPS publisher URL already downloaded and
            # hashed by pip, and label the absence of a publisher digest.
            raw=_read(index);parser=_Links();parser.feed(raw.decode('utf-8'))
            for href in parser.links:
                link=urllib.parse.urljoin(index,href);parsed=urllib.parse.urlsplit(link)
                if (not parsed.fragment and parsed.scheme=='https' and link==url
                        and urllib.parse.unquote(parsed.path.rsplit('/',1)[-1])==filename):
                    with urllib.request.urlopen(urllib.request.Request(link,method='HEAD'),timeout=30) as response:size=int(response.headers['Content-Length'])
                    return dict(row,bytes=size,url=link,upstream_url=link,version=version,
                        publisher_index=dict(index=index,index_sha256=hashlib.sha256(raw).hexdigest(),publisher_digest_advertised=False,digest_source='pip-computed-whole-source-download'))
        except (OSError,ValueError):continue
    raise ValueError('No publisher index attests the resolved build artifact: '+filename)


def verify_wheel_metadata(path,metadata):
    """PEP 658 resolver metadata must agree with the admitted wheel itself."""
    if not path.name.endswith('.whl'):
        return dict(kind='pip-generated-source-metadata',source_archive_sha256=sha(path))
    try:from packaging.requirements import Requirement
    except ImportError:from pip._vendor.packaging.requirements import Requirement
    with zipfile.ZipFile(path) as wheel:
        names=[n for n in wheel.namelist() if n.count('/')==1 and n.endswith('.dist-info/METADATA')]
        if len(names)!=1:raise ValueError('Ambiguous wheel metadata')
        raw=wheel.read(names[0]);actual=BytesParser().parsebytes(raw)
    normalize=lambda value:re.sub(r'[-_.]+','-',value).lower()
    if normalize(actual['Name'])!=normalize(metadata['name']) or actual['Version']!=metadata['version']:
        raise ValueError('Resolver identity differs from actual wheel metadata')
    expected=sorted(str(Requirement(v)) for v in metadata.get('requires_dist',[]))
    observed=sorted(str(Requirement(v)) for v in actual.get_all('Requires-Dist',[]))
    if observed!=expected:raise ValueError('Resolver dependency metadata differs from admitted wheel')
    return dict(kind='verified-wheel-metadata',member=names[0],metadata_sha256=hashlib.sha256(raw).hexdigest())


def verify_closure(requirements,rows,environment):
    try:from packaging.requirements import Requirement
    except ImportError:from pip._vendor.packaging.requirements import Requirement
    normalize=lambda name:re.sub(r'[-_.]+','-',name).lower()
    selected={normalize(row['metadata']['name']):row['metadata'] for row in rows};extras={};queue=[]
    def admit(text):
        requirement=Requirement(text);name=normalize(requirement.name)
        if name not in selected or not requirement.specifier.contains(selected[name]['version'],prereleases=True):
            raise ValueError('Incomplete isolated build dependency closure: '+text)
        before=extras.get(name)
        desired=set(requirement.extras)|(before or set())
        if before is None or desired!=before:extras[name]=desired;queue.append(name)
    for text in requirements:
        requirement=Requirement(text)
        if requirement.marker is None or requirement.marker.evaluate(dict(environment,extra='')):admit(text)
    while queue:
        name=queue.pop()
        for text in selected[name].get('requires_dist',[]):
            requirement=Requirement(text)
            if requirement.marker is None or any(requirement.marker.evaluate(dict(environment,extra=value)) for value in extras[name]|{''}):admit(text)
    return dict(projects=len(selected),reachable_projects=len(extras),original_requirements_verified=True)


def _run(argv,*,sdk,env,cwd,folder):
    from official_execution import sdk_popen
    from official_baselines import save
    process=None;record=dict(command=list(map(str,argv)),started=time.time(),sdk_installation_performed=False)
    try:
        with (folder/'resolver.log').open('xb') as stream:
            process=sdk_popen(argv,sdk=sdk,env=env,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record['pid']=process.pid;record['start_ticks']=Path('/proc',str(process.pid),'stat').read_text().rsplit(')',1)[1].split()[19];save(folder/'resolver.json',record)
        record['returncode']=process.wait(timeout=900)
        if record['returncode']:raise RuntimeError('Original isolated build requirements did not resolve; preserve '+str(folder/'resolver.log'))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        record.update(ended=time.time(),quiescent=process is None or process.poll() is not None);save(folder/'resolver.json',record)


def prepare_stage(python,core,cache,build,env,*,sdk,stage):
    import tomllib
    from official_baselines import save
    from official_source_candidate import primary_build_constraint,build_dependency_artifacts
    from transport_trust import configure
    cache=Path(cache);base=cache/'official-sdks/0.23.0-4ee78def8f33d59d'
    tls=configure(cache,base);child=dict(env);configure(cache,base,env=child)
    # The primary wheels are fetched through the same advertised-hash mirrors.
    primary={r['filename']:r for r in build_dependency_artifacts().values() if r['stage']==stage}
    for row in primary.values():fetch(row,cache/'wheels')
    primary_constraint,primary_proof=primary_build_constraint(python,core,cache,build,child,sdk=sdk,stage=stage)
    original=tomllib.loads((core/'pyproject.toml').read_text())['build-system']['requires']
    folder=build/(stage+'-dependency-plan');folder.mkdir();report=folder/'pip-report.json'
    child.update(PIP_CONFIG_FILE=os.devnull,PIP_CACHE_DIR=str(cache/'source-build-downloads/verified-metadata-pip'),PIP_BUILD_CONSTRAINT=str(primary_constraint))
    child.setdefault('PIP_DEFAULT_TIMEOUT','120');child.setdefault('PIP_RETRIES','3')
    argv=[str(python),'-S','-m','pip','install','--dry-run','--ignore-installed','--report',str(report),'--constraint',str(primary_constraint),'--index-url',PYPI_MIRRORS[0],'--extra-index-url',CPU_MIRROR]
    if stage=='03-ascend':argv+=['--extra-index-url',ASCEND_ORIGIN]
    argv+=original
    _run(argv,sdk=sdk,env=child,cwd=core,folder=folder)
    resolved=json.loads(report.read_bytes());rows=[];lines=[]
    for install in resolved['install']:
        row=authority(install,primary)
        admitted=admit_pip_cache(row,cache/'wheels',pip_cache=Path(child['PIP_CACHE_DIR']),report_url=install['download_info']['url'])
        path,transport=admitted if admitted is not None else fetch(row,cache/'wheels')
        binding=verify_wheel_metadata(path,install['metadata']);rows.append(dict(row,path=str(path),transport=transport,metadata=install['metadata'],metadata_binding=binding))
        lines.append(row['package']+' @ '+path.as_uri()+'#sha256='+row['sha256'])
    closure=verify_closure(original,rows,resolved.get('environment',{}))
    constraint=build/(stage+'-verified-closure-constraints.txt');constraint.write_text('\n'.join(lines)+'\n')
    lock=dict(schema='deeplinkengine-isolated-build-dependency-closure/v1',stage=stage,original_requirements=original,pyproject_sha256=sha(core/'pyproject.toml'),resolver_report_sha256=sha(report),tls=tls,primary=primary_proof,closure=closure,artifacts=rows,constraint_sha256=sha(constraint),sdk_runtime_constraints_added=False,temporary_metadata_build_environments_possible=True,model_started=False)
    save(folder/'lock.json',lock)
    return constraint,dict(stage=stage,lock=str(folder/'lock.json'),lock_sha256=sha(folder/'lock.json'),constraint=str(constraint),constraint_sha256=sha(constraint),artifacts=len(rows),runtime_constraint_added=False)


def prefetch_all(config,arguments):
    """Use an existing verified SDK/context; never repair or unpack that SDK."""
    import official_execution as execution
    import official_build_context as context
    from official_image import environment
    from official_baselines import save
    from official_source_candidate import SOURCES,source_archive
    from transport_trust import configure
    import platform
    if sys.platform!='linux' or platform.machine()!='aarch64':
        raise ValueError('Resolve complete build closures in an already prepared Linux aarch64 SDK; individual artifacts are cross-platform')
    cache=Path(config['cache']);base=cache/'official-sdks/0.23.0-4ee78def8f33d59d'
    if not (base/'installation.json').is_file():raise ValueError('Build closure prefetch requires an already prepared official 0.23 SDK')
    key=hashlib.sha256((context.BASE_MANIFEST+sha(context.__file__)+sha(execution.__file__)).encode()).hexdigest()[:20];root_folder=cache/'execution-contexts'/('official-build-system-'+key)
    if not root_folder.is_dir():raise ValueError('Build closure prefetch requires the completed private build system root')
    root=execution.verify_root(root_folder);python,env=environment(base,cache/'source-build-downloads/prefetch-runtime');env['ASCEND_RT_VISIBLE_DEVICES']='';configure(cache,base,env=env)
    identity=execution.context_identity()
    if not identity or not identity.get('source_build_system_root'):
        request=execution.request_for([str(python),'-B',str(Path(__file__).parent/'run_all.py'),'--official-baselines',*arguments],root,env=env,extra_paths=(cache,Path(config['out']).parent,base,Path(__file__).parent))
        execution.execute_request(request);raise AssertionError('Build prefetch exec returned')
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=False);record=dict(schema='deeplinkengine-source-build-closure-prefetch/v1',status='running',started=time.time(),model_execution_authorized=False,sdk_installation_performed=False,temporary_metadata_build_environments_possible=True,stages=[])
    try:
        save(out/'results.json',record)
        for project,stage in [('vllm','01-core'),('vllm-ascend','03-ascend')]:
            item=SOURCES[project];archive=source_archive(item,cache/'source-candidate-archives');metadata=out/(project+'-metadata');metadata.mkdir()
            with tarfile.open(archive) as tar:
                candidates=[m for m in tar if m.isfile() and m.name.count('/')==1 and m.name.endswith('/pyproject.toml')]
                if len(candidates)!=1:raise ValueError('Original source has ambiguous build metadata')
                (metadata/'pyproject.toml').write_bytes(tar.extractfile(candidates[0]).read())
            constraint,proof=prepare_stage(python,metadata,cache,out,env,sdk=base,stage=stage);record['stages'].append(proof);save(out/'results.json',record)
        record['status']='completed';return 0
    except BaseException as error:record.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:record['ended']=time.time();save(out/'results.json',record)
