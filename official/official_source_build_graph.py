"""Keep each upstream PEP 517 build node's requirements independent."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile


def requirements(path):
    import tomllib
    raw=Path(path).read_bytes()
    return tomllib.loads(raw.decode())['build-system']['requires']


def primary_inputs(original,cache,build,label,*,required_stage=None):
    import official_build_assets as assets
    from official_source_candidate import build_dependency_artifacts
    registry=list(build_dependency_artifacts().values())
    normalized={re.sub(r'\s+','',text) for text in original}
    if required_stage:
        for row in registry:
            if row['stage']==required_stage and row['requirement'] not in normalized:
                raise ValueError('Official isolated build requirement changed: '+row['requirement'])
    selected=[row for row in registry if row.get('requirement') in normalized]
    rows=[]
    for row in selected:
        path,transport=assets.fetch(row,Path(cache)/'wheels')
        if path.stat().st_size!=row['bytes'] or assets.sha(path)!=row['sha256']:
            raise ValueError('Isolated build input failed fresh whole SHA verification')
        rows.append(dict(row,path=str(path),transport=transport))
    constraint=Path(build)/(label+'-metadata-root-constraints.txt')
    constraint.write_text(''.join(row['package']+' @ '+Path(row['path']).as_uri()+'#sha256='+row['sha256']+'\n' for row in rows))
    return constraint,rows


def approved_links(rows,folder):
    import official_build_assets as assets
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);seen={};manifest=[]
    for row in rows:
        name=row['filename']
        if Path(name).name!=name:raise ValueError('Unsafe approved build filename')
        if name in seen:
            if seen[name]!=row['sha256']:raise ValueError('Conflicting approved build filename: '+name)
            continue
        source=Path(row['path'])
        if source.is_symlink() or not source.is_file() or source.stat().st_size!=row['bytes'] or assets.sha(source)!=row['sha256']:
            raise ValueError('Approved build input changed: '+name)
        target=folder/name
        try:os.link(source,target);kind='same-filesystem-hardlink'
        except OSError:shutil.copyfile(source,target);kind='independent-copy'
        if target.stat().st_size!=row['bytes'] or assets.sha(target)!=row['sha256']:
            raise ValueError('Approved build copy changed: '+name)
        seen[name]=row['sha256'];manifest.append(dict(filename=name,bytes=row['bytes'],sha256=row['sha256'],source=str(source),kind=kind))
    return manifest


def source_metadata(row,folder):
    import official_build_assets as assets
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);archive=Path(row['path'])
    if assets.sha(archive)!=row['sha256']:raise ValueError('Nested source archive changed')
    with tarfile.open(archive) as stream:
        members=[item for item in stream if item.isfile() and item.name.count('/')==1 and item.name.endswith('/pyproject.toml')]
        if len(members)!=1:raise ValueError('Nested source has no unique original PEP 517 pyproject: '+row['filename'])
        member=members[0];raw=stream.extractfile(member).read()
    if row.get('pyproject_sha256') and hashlib.sha256(raw).hexdigest()!=row['pyproject_sha256']:
        raise ValueError('Pinned nested source build requirements changed')
    (folder/'pyproject.toml').write_bytes(raw)
    return folder,dict(archive_sha256=row['sha256'],archive_member=member.name,pyproject_sha256=hashlib.sha256(raw).hexdigest(),upstream_source_modified=False)


def clean_environment(environment):
    child=dict(environment)
    for name in ('PIP_BUILD_CONSTRAINT','PIP_CONSTRAINT'):child.pop(name,None)
    child['PYTHONDONTWRITEBYTECODE']='1'
    excluded=[part for part in child.get('LD_LIBRARY_PATH','').split(':') if part.endswith(('/site-packages/torch/lib','/site-packages/torch.libs','/site-packages/torch_npu/lib'))]
    child['LD_LIBRARY_PATH']=':'.join(part for part in child.get('LD_LIBRARY_PATH','').split(':') if part and part not in excluded)
    return child,excluded


def prepare(python,core,cache,build,environment,*,sdk,stage,ancestry=()):
    import official_build_assets as assets
    from official_baselines import save
    from engine.transport_trust import configure
    cache=Path(cache);build=Path(build);core=Path(core)
    if len(ancestry)>8:raise ValueError('Nested PEP 517 source graph exceeded its bounded depth')
    label=re.sub(r'[^A-Za-z0-9_.-]+','--',stage)
    original=requirements(core/'pyproject.toml')
    child,excluded=clean_environment(environment)
    tls=configure(cache,cache/'official-sdks/0.23.0-4ee78def8f33d59d')
    configure(cache,cache/'official-sdks/0.23.0-4ee78def8f33d59d',env=child)
    constraint,primary=primary_inputs(original,cache,build,label,required_stage=stage if stage in ('01-core','03-ascend') else None)
    nested=[];nested_artifacts=[];prepared_sources=set()
    def nested_source(row):
        if row['sha256'] in prepared_sources:return
        if row['sha256'] in ancestry:raise ValueError('Cyclic upstream PEP 517 source requirements')
        project=row['package'];name=label+'--'+project+'--'+row['sha256'][:8]
        metadata,binding=source_metadata(row,build/(name+'-source-metadata'))
        links,proof=prepare(python,metadata,cache,build,child,sdk=sdk,stage=name,ancestry=(*ancestry,row['sha256']))
        lock=json.loads(Path(proof['lock']).read_bytes())
        nested_artifacts.extend(lock['approved_artifacts'])
        prepared_sources.add(row['sha256']);nested.append(dict(project=project,source=binding,build=proof))
    for row in primary:
        if not row['filename'].endswith('.whl'):nested_source(row)
    seed=build/(label+'-approved-seed')
    seed_manifest=approved_links([*primary,*nested_artifacts],seed)
    folder=build/(label+'-dependency-plan');folder.mkdir();report=folder/'pip-report.json'
    child.update(PIP_CONFIG_FILE=os.devnull,PIP_CACHE_DIR=str(cache/'source-build-downloads/verified-metadata-pip'),PIP_FIND_LINKS=str(seed))
    child.setdefault('PIP_DEFAULT_TIMEOUT','120');child.setdefault('PIP_RETRIES','3')
    argv=[str(python),'-B','-S','-m','pip','install','--dry-run','--ignore-installed','--report',str(report),
          '--constraint',str(constraint),'--find-links',str(seed),'--index-url',assets.PYPI_MIRRORS[0],
          '--extra-index-url',assets.CPU_MIRROR]
    if stage=='03-ascend' or stage.startswith('03-ascend--'):argv+=['--extra-index-url',assets.ASCEND_ORIGIN]
    argv+=original
    assets._run(argv,sdk=sdk,env=child,cwd=core,folder=folder)
    resolved=json.loads(report.read_bytes());rows=[]
    primary_by_name={row['filename']:row for row in primary}
    for item in resolved['install']:
        row=assets.authority(item,primary_by_name)
        admitted=assets.admit_pip_cache(row,cache/'wheels',pip_cache=Path(child['PIP_CACHE_DIR']),report_url=item['download_info']['url'])
        path,transport=admitted if admitted is not None else assets.fetch(row,cache/'wheels')
        row=dict(row,path=str(path),transport=transport,metadata=item['metadata'],metadata_binding=assets.verify_wheel_metadata(path,item['metadata']))
        rows.append(row)
        if not row['filename'].endswith('.whl'):nested_source(row)
    closure=assets.verify_closure(original,rows,resolved.get('environment',{}))
    approved=[*rows,*nested_artifacts]
    links=build/(label+'-approved-build-inputs')
    link_manifest=approved_links(approved,links)
    lock=dict(schema='deeplinkengine-isolated-build-dependency-closure/v2',stage=stage,original_requirements=original,
        pyproject_sha256=assets.sha(core/'pyproject.toml'),resolver_report_sha256=assets.sha(report),tls=tls,
        closure=closure,artifacts=rows,nested_builds=nested,approved_artifacts=approved,approved_wheelhouse=str(links),
        approved_link_manifest=link_manifest,metadata_seed_manifest=seed_manifest,metadata_root_constraint=str(constraint),
        metadata_root_constraint_sha256=assets.sha(constraint),global_build_constraints_added=False,runtime_constraints_added=False,
        upstream_pep517_isolation_retained=True,excluded_parent_torch_libraries=excluded,
        temporary_metadata_build_environments_possible=True,model_started=False)
    save(folder/'lock.json',lock)
    return links,dict(stage=stage,lock=str(folder/'lock.json'),lock_sha256=assets.sha(folder/'lock.json'),
        approved_wheelhouse=str(links),artifacts=len(rows),nested_builds=nested,runtime_constraint_added=False,
        global_build_constraint_added=False,upstream_pep517_isolation_retained=True)
