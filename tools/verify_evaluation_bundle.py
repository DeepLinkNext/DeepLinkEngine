"""Verify declared evaluation workflow patches against the original tools."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PATCH_FILES = {
    'scripts/baseline_entry.py',
    'scripts/checkpoint_cache.py',
    'scripts/child_runner.py',
    'scripts/engine_entry.py',
    'scripts/run_b1_r4.py',
    'scripts/run_native_reference.py',
    'scripts/run_paired.py',
    'scripts/third_party_report.py',
    'tests/test_baseline_entry.py',
    'tests/test_checkpoint_cache.py',
    'tests/test_child_runner.py',
    'tests/test_deployment.py',
    'tests/test_native_checkpoint_reserve.py',
    'tests/test_pre_service_device_admission.py',
    'tests/test_startup_cache_admission.py',
}


def inventory(path):
    result = {}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            pure = Path(member.name)
            if (not member.isfile() or not pure.parts or pure.parts[0] != 'vllm-hust-engine'
                    or '..' in pure.parts or pure.is_absolute()):
                raise ValueError('Unexpected evaluation archive member: ' + member.name)
            name = Path(*pure.parts[1:]).as_posix()
            if name in result:
                raise ValueError('Duplicate evaluation archive member: ' + name)
            result[name] = archive.extractfile(member).read()
    manifest = json.loads(result['DEPLOYMENT-TOOLS.json'])
    actual = {n: hashlib.sha256(b).hexdigest() for n, b in result.items() if n != 'DEPLOYMENT-TOOLS.json'}
    if manifest['files'] != actual:
        raise ValueError('Evaluation archive differs from its internal source manifest')
    return result, manifest


def verify(cache):
    from engine.environment_setup import cached_wheel
    from one_click import verify_kit
    lock = verify_kit()['tools']
    candidate = cached_wheel(dict(lock, filename=lock['archive']), [], Path(cache) / 'evaluation')
    base = lock['base_archive']
    original = cached_wheel(dict(base, filename=base['archive']), [], Path(cache) / 'evaluation-base')
    new, manifest = inventory(candidate)
    old, old_manifest = inventory(original)
    if hashlib.sha256(new['DEPLOYMENT-TOOLS.json']).hexdigest() != lock['manifest_sha256']:
        raise ValueError('Published evaluation manifest differs from the lock')
    if hashlib.sha256(old['DEPLOYMENT-TOOLS.json']).hexdigest() != base['manifest_sha256']:
        raise ValueError('Original evaluation manifest differs from its frozen identity')
    if len(manifest['files']) != lock['files'] or set(new) != set(old):
        raise ValueError('Evaluation workflow patch changed the frozen evaluation inventory')
    differences = {n for n in new if n != 'DEPLOYMENT-TOOLS.json' and new[n] != old[n]}
    patch = manifest['local_patchset']
    if differences != PATCH_FILES or set(patch['files']) != PATCH_FILES:
        raise ValueError('Evaluation workflow patch changed files outside its declared preparation scope')
    if (patch['base_manifest_sha256'] != base['manifest_sha256']
            or manifest['commit'] != old_manifest['commit']):
        raise ValueError('Evaluation workflow patch lost its original source binding')
    for name in PATCH_FILES:
        if patch['files'][name] != dict(before_sha256=old_manifest['files'][name], after_sha256=manifest['files'][name]):
            raise ValueError('Evaluation workflow patch before/after hashes differ: ' + name)
    local = ROOT / 'benchmark/one-click-kit'
    for name, body in new.items():
        if (local / name).read_bytes() != body:
            raise ValueError('Repository evaluation source differs from its downloadable archive: ' + name)
    print('EVALUATION_BUNDLE_VERIFIED', len(manifest['files']), 'files;', len(PATCH_FILES), 'declared startup/memory/lease patch files differ from the original', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    verify(parser.parse_args().cache)
