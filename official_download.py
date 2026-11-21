"""Cross-platform download-only route for digest-pinned official OCI files."""
import json
from pathlib import Path
import time
import platform
import shutil


def artifacts(versions):
    from official_baselines import ROOT
    from official_image import read_lock
    result = {}
    for version in versions:
        path = ROOT / 'engine/official-baselines' / (version + '.json')
        if not path.is_file():
            raise ValueError('No admitted official image for download: ' + version)
        lock = read_lock(path)
        entries = [('manifests', lock['manifest_digest'], lock['manifest_bytes']),
                   ('blobs', lock['config']['digest'], lock['config']['size'])]
        entries += [('blobs', row['digest'], row['size']) for row in lock['layers']]
        for kind, identity, size in entries:
            entry = dict(registry=lock['registry'], kind=kind, digest=identity, bytes=size)
            if identity in result and result[identity] != entry:
                raise ValueError('Conflicting official artifact identity')
            result[identity] = entry
    return result


def run(config, *, artifact=None):
    from official_baselines import save
    from baseline_setup import fetch, digest
    selected = artifacts(config['versions'])
    if artifact is not None:
        if artifact not in selected:
            raise ValueError('Artifact digest is not in the selected official lock')
        selected = {artifact: selected[artifact]}
    out = Path(config['out'])
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema='deeplinkengine-official-download/v1', started=time.time(), status='running',
                  model_execution_authorized=False, extraction_performed=False, files=[])
    try:
        save(out / 'results.json', record)
        for identity, row in selected.items():
            # Native Windows curl uses the system TLS transport. Python TLS
            # can trickle through the same configured proxy without timing out.
            attempts = 0 if platform.system() == 'Windows' and shutil.which('curl') else 3
            path = fetch(Path(config['cache']) / 'official-oci', row['registry'], row['kind'], identity, row['bytes'], attempts=attempts)
            record['files'].append(dict(row, path=str(path), sha256=digest(path)))
            save(out / 'results.json', record)
        record['status'] = 'completed'
        return 0
    except BaseException as error:
        record.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        record['ended'] = time.time()
        save(out / 'results.json', record)
