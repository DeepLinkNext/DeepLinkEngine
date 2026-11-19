"""Release clean cached installation archives before allocating model host state."""
import json
import os
from pathlib import Path
import stat
import time

ROOT = Path(__file__).resolve().parent


def memory():
    base = Path('/sys/fs/cgroup')
    return {name: int((base / name).read_text()) for name in ('memory.current', 'memory.max')}


def release(cache, baseline=None):
    """Only advise known repository artifact files; never write or globally flush."""
    from baseline_bundle import inventory
    from tools.verify_dependency_index import build_index
    cache = Path(cache).expanduser().resolve()
    b0 = inventory()
    candidates = {}
    blob_folders = [cache / 'b0-oci', Path.home() / '.cache/vllm-hust-engine/b0-image-018']
    if os.environ.get('DEEPLINKENGINE_B0_OCI_CACHE'):
        blob_folders.append(Path(os.environ['DEEPLINKENGINE_B0_OCI_CACHE']).expanduser().resolve())
    if baseline:
        blob_folders.append(Path(baseline).resolve().parent / 'oci')
    part_folders = [cache / 'b0-parts']
    if os.environ.get('DEEPLINKENGINE_B0_ARCHIVES'):
        part_folders.append(Path(os.environ['DEEPLINKENGINE_B0_ARCHIVES']).expanduser().resolve())
    for row in b0['blobs']:
        for folder in blob_folders:
            candidates[folder / row['digest'].replace(':', '-')] = row['bytes']
        for part in row['parts']:
            for folder in part_folders:
                candidates[folder / part['name']] = part['bytes']
    runtime = json.loads((ROOT / 'engine/r8-lock.json').read_text())
    evaluation = json.loads((ROOT / 'benchmark/one-click-lock.json').read_text())
    for row in build_index(runtime, evaluation)['artifacts']:
        candidates[cache / 'wheels' / row['name']] = row['bytes']
    before = memory()
    files, failures = [], []
    for path, size in sorted(candidates.items()):
        if size < 16 * 1024**2 or not path.exists():
            continue
        try:
            # No traversal or symlink following, and no additional hash read that
            # would refill the very archive cache being released. Integrity was
            # checked by the preceding artifact preparation stage.
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                prior = os.fstat(fd)
                if not stat.S_ISREG(prior.st_mode) or prior.st_size != size:
                    raise ValueError('Known artifact identity differs: ' + str(path))
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                after = os.fstat(fd)
                if (prior.st_dev, prior.st_ino, prior.st_size, prior.st_mtime_ns) != (
                        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                    raise ValueError('Artifact changed during cache hint: ' + str(path))
                files.append(dict(path=str(path), bytes=size, inode=prior.st_ino, mtime_ns=prior.st_mtime_ns))
            finally:
                os.close(fd)
        except OSError as error:
            failures.append(dict(path=str(path), error=str(error)))
    result = dict(schema='deeplinkengine-artifact-cache-hints/v1', time=time.time(),
                  operation='POSIX_FADV_DONTNEED on known installation archive descriptors',
                  file_content_written=False, global_drop_caches=False,
                  memory_limits_changed=False, cache_reclamation_guaranteed=False,
                  before=before, after=memory(), files=files, hint_failures=failures)
    path = cache / 'artifact-cache-admission.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print('ARTIFACT_CACHE_HINT', len(files), before['memory.current'], result['after']['memory.current'], flush=True)
    return result
