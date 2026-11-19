"""Admit an explicitly selected, sealed original200 arm without relabelling it."""

import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(root):
    return {name:{key:row[key] for key in ('sha256', 'bytes')}
        for name, row in read(Path(root) / 'model-weights.json').items()}


def expected(deployment, baseline, model, physical, logical, qualified_weights):
    deployment = Path(deployment).resolve()
    model = Path(model).resolve()
    if {p.name for p in model.glob('*.safetensors')} != set(qualified_weights):
        raise ValueError('Current checkpoint membership differs from the qualified C8 run')
    for name, row in qualified_weights.items():
        stat = (model / name).stat()
        if (stat.st_size, stat.st_ino, stat.st_mtime_ns) != (row['bytes'], row['inode'], row['mtime_ns']):
            raise ValueError('Current checkpoint changed since C8; obtain a fresh qualification before reuse')
    return dict(model=str(Path(model).resolve()), physical_cards=list(physical),
        logical_cards=[int(x) for x in logical.split(',')],
        baseline_deployment=str(Path(baseline).resolve()),
        optimized_release=read(deployment / 'deployment.json')['release'],
        optimized_deployment_sha256=sha(deployment / 'deployment.json'),
        optimized_installed_files_sha256=sha(deployment / 'provenance/installed-files.json'),
        task_plan_sha256=sha(deployment / 'inputs/mmlu-quality018-r2/quality-request-plan.json'),
        scorer_sha256=sha(deployment / 'scorers/mmlu_pro_pyramidkv.py'),
        model_weights={name:{key:row[key] for key in ('bytes', 'sha256')}
            for name, row in qualified_weights.items()})


def validate_conditions(root, role, conditions, inspected):
    """Check observed execution identity; an optimized companion map is not B0's runtime."""
    from baseline_admission import PINNED_SOURCE_MAP_SHA256
    root = Path(root)
    receipt = read(root / 'receipt.json')
    if (role not in ('B0', 'B1') or receipt.get('role', 'B1') != role
            or receipt.get('kind') != 'joint' or not receipt.get('completed')
            or receipt.get('status') != 'completed' or not receipt.get('quiescent')):
        raise ValueError('Reuse requires a completed released original200 arm of the requested role')
    if (inspected['role'] != role or inspected['recorded_requests'] != 200
            or inspected['successful_requests'] != 200 or inspected['failed_requests']
            or inspected['quality']['unreturned']):
        raise ValueError('Reuse requires all original200 responses, including the original quality outcomes')
    mapping = read(root / 'device-mapping.json')
    if (mapping['physical_cards'] != conditions['physical_cards']
            or mapping['logical_cards'] != conditions['logical_cards']):
        raise ValueError('Reused arm belongs to another physical/logical device pair')
    requests = read(root / 'service/request-plan.json')
    if (requests['request_count'] != 200 or requests['request_rate'] != 'inf'
            or requests['warmups'] != 0 or len(requests['requests']) != 200
            or sha(requests['source']) != conditions['task_plan_sha256']
            or requests['scorer_sha256'] != conditions['scorer_sha256']
            or inspected['quality']['scorer_sha256'] != conditions['scorer_sha256']):
        raise ValueError('Reused task source, ordered population, scorer or original arrival policy differs')
    # inspect_run already validates original raw request/SSE, generation budget,
    # actual worker identity, shared warmup and both post-measurement boundaries.
    if not inspected.get('serving_warmup', {}).get('completed'):
        raise ValueError('Reused arm lacks the completed current shared warmup')
    if weights(root) != conditions['model_weights']:
        raise ValueError('Reused arm weights differ from the current qualified model')
    if role == 'B0':
        baseline = read(root / 'baseline-source-admission.json')
        digest = hashlib.sha256(json.dumps(baseline['files'], sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        if (digest != PINNED_SOURCE_MAP_SHA256
                or baseline['source_map_sha256'] != PINNED_SOURCE_MAP_SHA256
                or str(Path(baseline['deployment']).resolve()) != conditions['baseline_deployment']):
            raise ValueError('Reused B0 actual pinned image/source identity differs')
    elif (sha(root / 'deployment.json') != conditions['optimized_deployment_sha256']
            or sha(root / 'installed-files.json') != conditions['optimized_installed_files_sha256']
            or read(root / 'deployment.json')['release'] != conditions['optimized_release']):
        raise ValueError('Reused B1 is not the exact current optimized deployment')


def admit(root, role, conditions):
    root = Path(root).resolve()
    receipt = read(root / 'receipt.json')
    # Reject a known failed attempt before any expensive raw reconstruction.
    if not receipt.get('completed') or not receipt.get('quiescent'):
        raise ValueError('Failed or running arms are never reusable')
    from report_execution import inspect_run
    data = inspect_run(root)
    validate_conditions(root, role, conditions, data)
    binding = dict(run=str(root), role=role, conditions=conditions,
        verified_at=time.time(), original_started=receipt['started'],
        original_release=read(root / 'deployment.json')['release'],
        seal_sha256=sha(root / 'checksums.json'),
        receipt_sha256=sha(root / 'receipt.json'),
        controller_sources_sha256=sha(root / 'controller-source.json'),
        weights=weights(root), original_quality=data['quality'],
        scope='Explicit earlier sealed arm; original dates/runtime remain unchanged; no new inference.')
    return binding


def recheck(binding):
    root = Path(binding['run'])
    for name, key in [('checksums.json', 'seal_sha256'), ('receipt.json', 'receipt_sha256'),
                      ('controller-source.json', 'controller_sources_sha256')]:
        if sha(root / name) != binding[key]:
            raise ValueError('Reused arm changed after prospective admission: ' + name)
    current = admit(root, binding['role'], binding['conditions'])
    for key in ('original_started', 'original_release', 'weights', 'original_quality'):
        if current[key] != binding[key]:
            raise ValueError('Reused arm provenance differs: ' + key)
    return root


def resolve(pair_root, role, plan):
    binding = plan.get('reused_arms', {}).get(role)
    return recheck(binding) if binding else Path(pair_root) / role
