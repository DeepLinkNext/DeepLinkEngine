"""Prefetch pinned source-build wheels without installing or extracting them."""
import json
from pathlib import Path
import platform
import shutil
import time


def run(config, *, artifact=None, transport='auto',arguments=None):
    from official_baselines import save
    from official_source_candidate import VERSION,build_dependency_artifacts,source_archive
    from environment_setup import cached_wheel,wheel_digest
    from baseline_setup import download_lock
    if config['versions']!=[VERSION]:
        raise ValueError('Source-build dependency prefetch requires the explicit source candidate')
    if artifact is None:
        if transport!='auto':raise ValueError('Complete build closure prefetch uses the verified Range downloader')
        from official_build_assets import prefetch_all
        return prefetch_all(config,arguments)
    rows=build_dependency_artifacts()
    if artifact not in rows:
        raise ValueError('Artifact digest is not a pinned source-build dependency')
    if transport not in ('auto','curl'):
        raise ValueError('Unknown source-build download transport')
    use_curl=transport=='curl' or platform.system()=='Windows'
    if use_curl and not shutil.which('curl'):
        raise ValueError('Native curl is unavailable for source-build prefetch')
    row=rows[artifact];cache=Path(config['cache'])/'wheels';cache.mkdir(parents=True,exist_ok=True)
    target=cache/row['filename']
    if target.is_symlink():raise ValueError('Source-build wheel cache must not be a link')
    cached_before=target.is_file() and target.stat().st_size==row['bytes'] and wheel_digest(target)==row['sha256']
    if shutil.disk_usage(cache).free<(0 if cached_before else row['bytes'])+2*1024**3:
        raise OSError('Source-build prefetch requires one whole asset plus 2 GiB free')
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=False)
    record=dict(schema='deeplinkengine-source-build-dependency-download/v1',status='running',started=time.time(),
        model_execution_authorized=False,extraction_performed=False,installation_performed=False,
        source_build_candidate=True,official_release_image=False,artifact=artifact,
        transport='native-curl' if use_curl else 'python',files=[])
    try:
        save(out/'results.json',record)
        if use_curl:
            with (cache/(row['filename']+'.lock')).open('a+b') as lease,download_lock(lease):
                partial=cache/(row['filename']+'.part')
                if partial.is_symlink():raise ValueError('Source-build partial must not be a link')
                if partial.exists() and partial.stat().st_size>=row['bytes']:
                    if partial.stat().st_size!=row['bytes'] or wheel_digest(partial)!=row['sha256']:
                        raise ValueError('Full-size source-build partial has wrong whole SHA; preserve it')
                    if not target.exists():partial.replace(target)
                path=source_archive(row,cache)
        else:path=cached_wheel(row,[],cache)
        if path.is_symlink() or path.stat().st_size!=row['bytes'] or wheel_digest(path)!=row['sha256']:
            raise ValueError('Source-build wheel failed whole SHA admission')
        provenance=cache/(row['filename']+'.download.json')
        provenance_record=None
        if provenance.is_file() and not provenance.is_symlink():
            provenance_record=dict(path=str(provenance),sha256=wheel_digest(provenance))
        record['files'].append(dict(filename=row['filename'],path=str(path),bytes=path.stat().st_size,
            sha256=wheel_digest(path),isolated_stage=row['stage'],package=row['package'],
            source_url=row['upstream_url'],configured_download_url=row['url'],transport_provenance=provenance_record,cached_before=cached_before))
        record['status']='completed';return 0
    except BaseException as error:
        record.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        record['ended']=time.time();save(out/'results.json',record)
