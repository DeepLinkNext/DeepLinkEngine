"""Admission of the fixed200 workload and the original R10 native configuration."""

import hashlib
import json
from pathlib import Path

RELEASE = 'full-answer-b1-20261121-r10'
VERSION = '2026.11.21.10'
SDIST_SHA = '904ea0146969d59b3e6ca150972c43dab263936c4d1bead7bf897e00697d8413'
FORMAL_SHA = 'c36b966384a220b9060c452e632ac636da59a92dce5a6ded9fe931794c187851'
WARMUP_SHA = 'b4527c75c156653b856a097375c4db99634b8373b974e4986de0323b33b7934b'
WARMUP_WIDTHS = [1, 2, 4, 8, 16, 24, 16, 1, 2, 4, 8, 16, 24, 32, 36]
MEASURED_RUNTIME = {
    'release': 'full-answer-b1-20261121-r10-contiguous-restore',
    'installed_files_canonical_sha256': '993b9a6fc29ae24dbfe707d570060de42ef861884a0d7e12ffa68e1e7252668c',
    'runtime_overlay_sha256': '96ecd0418ada8f8f7ea6a91af875676f2249c34a5559efcaf707496cf5ecb749',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def compact(value):
    return (json.dumps(value, separators=(',', ':')) + '\n').encode()


def workload(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    prompts, warmup = data['request_token_ids'], data['warmup_cohorts']
    if (sha(compact(prompts)) != FORMAL_SHA or len(prompts) != 200
            or sum(map(len, prompts)) != 282265 or sha(compact(warmup)) != WARMUP_SHA
            or list(map(len, warmup)) != WARMUP_WIDTHS):
        raise ValueError('Native input IDs or synthetic warmup differ from the two patched B0 runs')
    return data, dict(input_file_sha256=sha(raw), request_token_ids_sha256=FORMAL_SHA,
                      warmup_cohorts_sha256=WARMUP_SHA, requests=200, input_tokens=282265,
                      warmup_widths=WARMUP_WIDTHS)


def runtime(profile, deployment, installed_files):
    from runtime_geometry import validate_profile

    pins = json.dumps(installed_files, sort_keys=True, separators=(',', ':')).encode()
    if (profile.get('release') != RELEASE or profile.get('measured_runtime') != MEASURED_RUNTIME
            or len(installed_files) != 9256 or sha(pins) != MEASURED_RUNTIME['installed_files_canonical_sha256']
            or sha((deployment / 'provenance/runtime-overlay.json').read_bytes()) != MEASURED_RUNTIME['runtime_overlay_sha256']):
        raise ValueError('Installed SDK differs from the stable R10 contiguous-restore runtime')
    geometry = validate_profile(profile)
    if (geometry['id'], geometry['execution_seats'], geometry['resident_seats']) != (
            'r10-contiguous-restore-E36-R54', 36, 54):
        raise ValueError('Native R10 runtime geometry differs')
    return dict(release=RELEASE, measured_runtime=MEASURED_RUNTIME, runtime_geometry=geometry)


def model_weights(model, model_lock):
    data = json.loads(Path(model_lock).read_bytes())
    if data.get('repository') != 'Qwen/Qwen3.5-35B-A3B':
        raise ValueError('Native R10 requires the pinned Qwen3.5-35B-A3B checkpoint')
    weights = {}
    for name, entry in data['files'].items():
        if not name.endswith('.safetensors'):
            continue
        stat = (model / name).stat()
        weights[name] = dict(bytes=entry['bytes'], sha256=entry['sha256'], mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)
    if len(weights) != 14 or sum(row['bytes'] for row in weights.values()) != 71903878016:
        raise ValueError('Native R10 checkpoint shard contract differs')
    return weights


def selected_processes(sample, cards):
    """Parse npu-smi process rows; the process column is separate from memory."""
    for line in sample.splitlines():
        columns = [column.strip() for column in line.strip('|').split('|')]
        indices = columns[0].split()
        if len(columns) >= 4 and len(indices) == 2 and indices[0].isdigit() and indices[1] == '0':
            if int(indices[0]) in cards and columns[1].isdigit():
                yield int(columns[1])
