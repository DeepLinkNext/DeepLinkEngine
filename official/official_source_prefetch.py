"""Prefetch original build inputs; leave resolution and PEP 517 isolation intact."""
import email
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
from urllib.parse import quote,urljoin,urlsplit,unquote,parse_qs
import zipfile


def sha(path):
    from official_build_assets import sha as digest
    return digest(Path(path))


def save(path,value):
    from official_baselines import save as write
    write(path,value)


def runtime():
    import pip
    from pip._vendor.packaging.markers import default_environment
    from pip._internal.utils.compatibility_tags import get_supported
    if not Path(pip.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):raise ValueError('Build-input metadata must use the selected SDK pip')
    return dict(markers=default_environment(),tags=[str(tag) for tag in get_supported()],python=sys.executable,
        prefix=sys.prefix,pip_version=pip.__version__,pip_origin=str(Path(pip.__file__).resolve()))


def requirements(projects,environment):
    from pip._vendor.packaging.requirements import Requirement
    result=[]
    for project in projects:
        for text in project['requires']:
            item=Requirement(text)
            if item.marker and not item.marker.evaluate(dict(environment,extra='')):continue
            result.append(dict(project=project['project'],requirement=text,name=item.name,
                extras=sorted(item.extras),exact=(next(iter(item.specifier)).version
                if len(list(item.specifier))==1 and next(iter(item.specifier)).operator=='=='
                and '*' not in next(iter(item.specifier)).version else None)))
    return result


def original_projects(cache,out):
    import official_source_candidate as candidate
    import official_build_assets as assets
    from official_source_build_graph import source_metadata
    import tomllib
    cache=Path(cache);out=Path(out);out.mkdir(parents=True,exist_ok=False);projects=[]
    for name,item in candidate.SOURCES.items():
        archive=candidate.source_archive(item,cache/'source-candidate-archives')
        folder,proof=source_metadata(dict(item,path=str(archive),package=name),out/name)
        raw=(folder/'pyproject.toml').read_bytes();projects.append(dict(project=name,
            requires=tomllib.loads(raw.decode())['build-system']['requires'],source=proof))
    for item in candidate.NESTED_BUILD_ARTIFACTS.values():
        if item['filename'].endswith('.whl'):continue
        path,_=assets.fetch(item,cache/'wheels')
        folder,proof=source_metadata(dict(item,path=str(path)),out/item['package'])
        raw=(folder/'pyproject.toml').read_bytes();projects.append(dict(project=item['package'],
            requires=tomllib.loads(raw.decode())['build-system']['requires'],source=proof))
    save(out/'projects.json',projects);return projects


def compatible(row,environment):
    from pip._vendor.packaging.utils import parse_wheel_filename
    from pip._vendor.packaging.specifiers import SpecifierSet
    try:_,_,_,tags=parse_wheel_filename(row['filename'])
    except ValueError:return False
    return (bool({str(t) for t in tags}&set(environment['tags']))
        and (not row.get('requires_python') or environment['markers']['python_full_version'] in SpecifierSet(row['requires_python'])))


def metadata(url,folder):
    from official_build_assets import _read
    raw=_read(url);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(raw).hexdigest();path=folder/(digest+'.json')
    if not path.exists():path.write_bytes(raw)
    return json.loads(raw),dict(url=url,path=str(path),sha256=digest)


def published_wheels(name,version,environment,out):
    """Select bytes only for preloading, never constrain the original resolver."""
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.utils import parse_wheel_filename
    from pip._vendor.packaging.specifiers import SpecifierSet
    package=canonicalize_name(name);url='https://pypi.org/pypi/'+quote(package,safe='')+'/'+quote(version,safe='')+'/json'
    def selected_filename(row):
        if not compatible(row,environment):return False
        found,actual,_,_=parse_wheel_filename(row['filename'])
        return canonicalize_name(found)==package and actual in SpecifierSet('=='+version)
    try:release,proof=metadata(url,Path(out)/'publishers')
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
        if package!='triton-ascend':return [],dict(status='delegated',reason='No PyPI release JSON; original resolver remains responsible',url=url)
        release,proof=metadata('https://api.github.com/repos/triton-lang/triton-ascend/releases/tags/v'+quote(version,safe=''),Path(out)/'publishers')
        rows=[]
        for asset in release.get('assets',[]):
            digest=asset.get('digest','');row=dict(filename=asset['name'])
            if not re.fullmatch(r'sha256:[0-9a-f]{64}',digest) or not selected_filename(row):continue
            rows.append(dict(row,package=package,version=version,requirement=package+'=='+version,
                bytes=asset['size'],sha256=digest.split(':',1)[1],url=asset['browser_download_url'],
                upstream_url=asset['browser_download_url'],publisher=proof,source_authority='official-github-release-asset',asset_id=asset['id']))
        return rows,dict(status='published',publisher=proof)
    rows=[]
    for asset in release.get('urls',[]):
        if asset.get('yanked') or asset.get('packagetype')!='bdist_wheel' or not selected_filename(asset):continue
        digest=asset['digests']['sha256']
        if not re.fullmatch(r'[0-9a-f]{64}',digest):raise ValueError('Publisher wheel has an invalid SHA-256')
        rows.append(dict(filename=asset['filename'],package=package,version=version,requirement=package+'=='+version,
            bytes=asset['size'],sha256=digest,url=asset['url'],upstream_url=asset['url'],
            requires_python=asset.get('requires_python'),publisher=proof,source_authority='official-pypi-release-json'))
    return rows,dict(status='published' if rows else 'delegated',publisher=proof,
        reason=None if rows else 'No compatible published wheel; source requirements remain with the original Graph')


def recipe_vendor(row,out):
    """An unhashed index link becomes eligible only after real pinned transport."""
    import official_build_assets as assets
    url=assets.ASCEND_ORIGIN+row['package']+'/'
    raw=assets._read(url);parser=assets._Links();parser.feed(raw.decode('utf-8'));matches=[]
    for href in parser.links:
        link=urlsplit(urljoin(url,href))
        if link.scheme=='https' and not link.username and not link.password and unquote(link.path.rsplit('/',1)[-1])==row['filename']:
            matches.append(link.geturl())
    if len(set(matches))!=1:raise ValueError('Original recipe vendor has no unique published wheel filename')
    link=next(iter(set(matches)));part=urlsplit(link)
    advertised=parse_qs(part.fragment).get('sha256')
    if advertised is not None and advertised!=[row['sha256']]:raise ValueError('Recipe vendor link advertises a different publisher SHA')
    if part.fragment and advertised is None:raise ValueError('Unsupported recipe vendor link fragment')
    folder=Path(out)/'vendor-indices';folder.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256(raw).hexdigest();path=folder/(digest+'.html');path.write_bytes(raw)
    return part._replace(fragment='').geturl(),dict(index=url,index_path=str(path),index_sha256=digest,
        link=link,link_has_digest=advertised is not None,publisher_release=row['publisher'])


def fetch_vendor(row,cache,out):
    import official_build_assets as assets
    from baseline_setup import download_lock
    source,index=recipe_vendor(row,out);qualified=Path(cache)/'source-build-downloads/vendor-qualified';qualified.mkdir(parents=True,exist_ok=True)
    target=qualified/row['filename'];receipt=target.with_name(target.name+'.source-ranges.json')
    with target.with_name(target.name+'.lock').open('a+b') as lease,download_lock(lease):
        if target.exists():
            if target.is_symlink() or target.stat().st_size!=row['bytes'] or sha(target)!=row['sha256'] or not receipt.is_file():
                raise ValueError('Preserve unknown vendor qualification bytes; no unhashed cache key is admitted')
            proof=json.loads(receipt.read_bytes())
        else:
            wheels=Path(cache)/'wheels';wheels.mkdir(parents=True,exist_ok=True)
            _,proof=assets.fetch_ranges(row,wheels,target,[source],[index])
        if proof.get('kind')!='verified-public-four-worker-ranges' or proof.get('source_url')!=source or proof.get('sha256')!=row['sha256']:
            raise ValueError('Vendor cache key needs this URL\'s actual publisher-SHA-qualified whole transfer')
    canonical=Path(cache)/'wheels'/row['filename'];canonical.parent.mkdir(parents=True,exist_ok=True)
    with canonical.with_name(canonical.name+'.lock').open('a+b') as lease,download_lock(lease):
        if canonical.exists():
            if canonical.is_symlink() or canonical.stat().st_size!=row['bytes'] or sha(canonical)!=row['sha256']:raise ValueError('Canonical vendor artifact changed; preserve it')
        else:
            with tempfile.NamedTemporaryFile(dir=canonical.parent,prefix=canonical.name+'.vendor-qualified-',delete=False) as stream:temporary=Path(stream.name)
            shutil.copyfile(target,temporary)
            if sha(temporary)!=row['sha256']:raise ValueError('Vendor whole copy changed')
            temporary.replace(canonical)
    result=dict(row,url=source,upstream_url=source,original_publisher_url=row['upstream_url'],
        source_authority='official-release-SHA-and-verified-recipe-vendor-body')
    if not index['link_has_digest']:
        result['no_hash_link_attestation']=dict(index=index,range_receipt=str(receipt),range_receipt_sha256=sha(receipt),
            published_sha256=row['sha256'],published_metadata=row['publisher'],vendor_url=source)
    return result


def wheel_metadata(path):
    with zipfile.ZipFile(path) as archive:
        members=[n for n in archive.namelist() if n.count('/')==1 and n.endswith('.dist-info/METADATA')]
        if len(members)!=1:raise ValueError('Ambiguous cache wheel metadata')
        return email.message_from_bytes(archive.read(members[0]))


def cached_build_inputs(cache,requested,environment,out,primary):
    """Audit relevant cache candidates against publisher bytes, without choosing versions."""
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version
    pending=[Requirement(r['requirement']) for r in requested];candidates=[];accepted=[];observations=[]
    for file in (Path(cache)/'wheels').glob('*.whl'):
        if file.is_symlink() or not compatible(dict(filename=file.name),environment):continue
        try:
            data=wheel_metadata(file);candidates.append((file,data,canonicalize_name(data['Name']),Version(data['Version'])))
        except (ValueError,TypeError,zipfile.BadZipFile,KeyError) as error:
            observations.append(dict(filename=file.name,status='not-primed',reason='Unknown cache metadata: '+str(error)))
    visited=set()
    while True:
        selected=[(f,m,n,v) for f,m,n,v in candidates if f.name not in visited and any(canonicalize_name(r.name)==n and v in r.specifier for r in pending)]
        if not selected:break
        for file,data,name,version in selected:
            visited.add(file.name);fixed=next((r for r in primary if r['filename']==file.name),None)
            if fixed is not None:
                if file.stat().st_size!=fixed['bytes'] or sha(file)!=fixed['sha256']:raise ValueError('Fixed primary cache differs from its original pin')
                row=fixed
            else:
                try:
                    rows,proof=published_wheels(name,str(version),environment,out)
                    row=next((r for r in rows if r['filename']==file.name and r['bytes']==file.stat().st_size and r['sha256']==sha(file)),None)
                    if row is None:
                        observations.append(dict(filename=file.name,status='not-primed',reason='No matching original publisher whole authority'));continue
                except (OSError,ValueError) as error:
                    observations.append(dict(filename=file.name,status='not-primed',reason=type(error).__name__+': '+str(error)));continue
            accepted.append(row)
            for text in data.get_all('Requires-Dist',[]):
                item=Requirement(text)
                if not item.marker or item.marker.evaluate(dict(environment['markers'],extra='')):pending.append(item)
    save(Path(out)/'cached-input-audit.json',dict(observations=observations,accepted=accepted,
        resolver_versions_forced=False,canonical_wheel_source_code_imported=False))
    return accepted


def collect(cache,out):
    import official_source_candidate as candidate
    import official_build_assets as assets
    out=Path(out);out.mkdir(parents=True,exist_ok=False);environment=runtime();projects=original_projects(cache,out/'original-projects')
    requested=requirements(projects,environment['markers']);primary=list(candidate.build_dependency_artifacts().values());rows=[];observations=[]
    primary_names={re.sub(r'[-_.]+','-',r['package']).lower() for r in primary}
    for item in requested:
        if item['exact'] is None or re.sub(r'[-_.]+','-',item['name']).lower() in primary_names:continue
        selected,proof=published_wheels(item['name'],item['exact'],environment,out)
        observations.append(dict(request=item,source=proof))
        for row in selected:
            if row['package']=='triton-ascend':row=fetch_vendor(row,cache,out)
            else:assets.fetch(row,Path(cache)/'wheels')
            rows.append(row)
    rows+=cached_build_inputs(cache,requested,environment,out,primary+rows)
    unique={(r['filename'],r['sha256']):r for r in [*primary,*rows]}
    result=list(unique.values());save(out/'results.json',dict(schema='deeplinkengine-original-build-input-prefetch/v1',
        status='completed',runtime=environment,projects=projects,requests=requested,observations=observations,artifacts=result,
        requirements_or_isolation_modified=False,original_resolver_choices_forced=False,model_started=False,
        helper_sha256=sha(__file__),ended=time.time()))
    return result
