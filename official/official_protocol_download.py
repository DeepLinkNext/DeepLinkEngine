"""Prefetch one locked common-protocol asset; never install or start a model."""
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def artifacts():
    index=json.loads((ROOT/'data/b0-oci.json').read_bytes())
    lock=json.loads((ROOT/'engine/r8-lock.json').read_bytes())
    rows=[]
    for blob in index['blobs']:
        for part in blob['parts']:
            rows.append(dict(part,filename=part['name'],cache_subdirectory='b0-parts',role='original018-protocol-dependency'))
    for source in lock['environment_sources'].values():
        rows.append(dict(source,cache_subdirectory='wheels',role='common-protocol-environment'))
    for source in lock.get('runtime_compatibility_sources',{}).values():
        rows.append(dict(source,cache_subdirectory='wheels',role='common-protocol-runtime-compatibility'))
    for source in lock['assets']:
        if source.get('deepengine_url'):
            rows.append(dict(source,filename=source['name'],url=source['deepengine_url'],cache_subdirectory='wheels',role='common-protocol-engine-package'))
    result={}
    for row in rows:
        if (Path(row['filename']).name!=row['filename'] or len(row['sha256'])!=64
                or any(c not in '0123456789abcdef' for c in row['sha256'])
                or type(row['bytes']) is not int or row['bytes']<=0
                or not row['url'].startswith('https://')):
            raise ValueError('Invalid pinned common-protocol artifact')
        identity='sha256:'+row['sha256']
        if identity in result and result[identity]!=row:
            raise ValueError('Conflicting pinned common-protocol artifact')
        result[identity]=row
    return result


def run(config,*,artifact):
    from official_baselines import save
    from engine.environment_setup import cached_wheel,wheel_digest
    selected=artifacts()
    if artifact not in selected:
        raise ValueError('Artifact digest is not a pinned common-protocol dependency')
    row=selected[artifact];cache=Path(config['cache'])/row['cache_subdirectory']
    # Individual assets can be transported without keeping the full dependency
    # set on the prefetch computer. The target CLI performs its normal admission.
    cache.mkdir(parents=True,exist_ok=True)
    import shutil
    if shutil.disk_usage(cache).free<row['bytes']+2*1024**3:
        raise OSError('Protocol prefetch requires one asset plus 2 GiB free')
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=False)
    record=dict(schema='deeplinkengine-protocol-dependency-download/v1',started=time.time(),status='running',
                model_execution_authorized=False,extraction_performed=False,installation_performed=False,
                transport='python',artifact=artifact,files=[])
    try:
        save(out/'results.json',record)
        path=cached_wheel(row,[],cache)
        record['files'].append(dict(filename=row['filename'],path=str(path),bytes=path.stat().st_size,
                                   sha256=wheel_digest(path),role=row['role'],cache_subdirectory=row['cache_subdirectory']))
        record['status']='completed';return 0
    except BaseException as error:
        record.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        record['ended']=time.time();save(out/'results.json',record)
