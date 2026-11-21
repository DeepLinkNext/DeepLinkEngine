"""Reuse one complete fixed-B0 native arm while measuring the selected B1 anew."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def inspect(source, expected_seal, plan, *, location=True):
    source = Path(source).resolve()
    if sha(source / 'checksums.json') != expected_seal:
        raise ValueError('Fixed B0 native source seal changed')
    seal = read(source / 'checksums.json')
    for name, pin in seal.items():
        relative = PurePosixPath(name)
        path = source / name
        if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(source) or sha(path) != pin):
            raise ValueError('Fixed B0 native raw evidence changed: ' + name)
    def sealed(name):
        if seal.get(name) != sha(source / name):
            raise ValueError('Required fixed B0 native record is unsealed: ' + name)
        return read(source / name)
    receipt = sealed('receipt.json')
    if (receipt.get('role') != 'B0' or receipt.get('completed') is not True
            or receipt.get('quiescent') is not True or receipt.get('preflight_only') is not False
            or receipt.get('failure') is not None or receipt.get('returncode') != 0
            or receipt.get('formal_repetitions') != plan['capture_profile']['formal_repetitions']):
        raise ValueError('A complete quiescent fixed B0 native arm is required')
    native = sealed('service/native-plan.json')
    from reference_capture_profile import validate_native_plan
    if (validate_native_plan(native) != plan['capture_profile'] or native['role'] != 'B0'
            or native['engine_release'] != 'original018-cann8.5.1'
            or Path(native['model']).resolve() != Path(plan['binding']['model']).resolve()):
        raise ValueError('Fixed B0 native model, profile or output budget differs')
    mapping = sealed('device-mapping.json')
    if mapping['logical_cards'] != list(map(int, plan['logical_devices'].split(','))):
        raise ValueError('Fixed B0 native physical/logical device pair differs')
    device_change = None
    if mapping['physical_cards'] != plan['physical_cards']:
        from device_reuse import compare
        actual = plan.get('device_reuse_mapping')
        device_change = compare(mapping, actual or {'physical_cards': plan['physical_cards']},
                                allowed=actual is not None)
        if actual['physical_cards'] != plan['physical_cards']:
            raise ValueError('Current native device mapping differs from the plan')
    # These are original source bytes, including its real companion release.
    # They are retained rather than relabelled as the new B1's implementation.
    for name, pin in native['files'].items():
        if sha(name) != pin:
            raise ValueError('Original fixed B0 native implementation changed: ' + name)
    weights = sealed('model-weights.json')
    if len(weights) != 14:
        raise ValueError('Fixed B0 native checkpoint inventory is incomplete')
    for name, row in weights.items():
        identity = (Path(plan['binding']['model']) / name).stat()
        if (identity.st_size, identity.st_ino, identity.st_mtime_ns) != (
                row['bytes'], row['inode'], row['mtime_ns']):
            raise ValueError('Fixed B0 native checkpoint identity differs: ' + name)
    # Rebuild ordinary public output geometry, qualify the original018 argv,
    # source domain and clean client, and retain each actual native clock.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third-party-test-kit'))
    from full_reference_reuse import qualify_native
    applicability = qualify_native(source, 'B0', plan['capture_profile'])
    from qualify_baseline_capacity import qualify as qualify_baseline
    original_baseline = Path(native['baseline_deployment']).resolve()
    timings = sealed('service/client/timings.json')
    expected_reps = list(range(1, plan['capture_profile']['formal_repetitions'] + 1))
    if (timings.get('role') != 'B0'
            or [r['repetition'] for r in timings['repetitions']] != expected_reps
            or receipt['result_sha256'] != sha(source / receipt['result'])):
        raise ValueError('Fixed B0 native repetitions or timing receipt differ')
    rows = []
    previous_end = None
    for row in timings['repetitions']:
        number = row['repetition']
        qualify_baseline(original_baseline, source, repetition=number)
        if (type(row['start_ns']) is not int or type(row['end_ns']) is not int
                or row['end_ns'] <= row['start_ns']
                or abs(row['T_seconds'] - (row['end_ns'] - row['start_ns']) / 1e9) > 1e-9
                or row.get('ordinary_public_outputs_retained') is not True
                or (previous_end is not None and row['start_ns'] <= previous_end)
                or len(row['boundary']) != 2 or {r['rank'] for r in row['boundary']} != {0, 1}
                or any(r.get('sealed') is not True or r.get('failure') is not None for r in row['boundary'])):
            raise ValueError('Fixed B0 native clock, population or fences differ')
        previous_end = row['end_ns']
        rows.append(dict(repetition=number, T_seconds=row['T_seconds'],
                         start_ns=row['start_ns'], end_ns=row['end_ns']))
    baseline = sealed('service/baseline-source-admission.json')
    from baseline_admission import PINNED_SOURCE_MAP_SHA256
    if baseline['source_map_sha256'] != PINNED_SOURCE_MAP_SHA256:
        raise ValueError('Fixed B0 native image pin differs')
    if location:
        from baseline_relocation import verify
        verify(baseline, plan['baseline_deployment'])
    result = dict(schema='deeplinkengine-fixed-b0-native-reuse/v1', source=str(source),
        checksums_sha256=expected_seal, receipt_sha256=sha(source / 'receipt.json'),
        original_baseline_deployment=str(original_baseline),
        original_paired_engine_release=native['paired_engine_release'],
        original_companion_deployment=native['deployment'],
        original_started=receipt['started'], original_ended=receipt['ended'],
        capture_profile=plan['capture_profile'], applicability=applicability,
        repetitions=rows, model_inference_rerun=False)
    if device_change:
        result['device_change'] = device_change
    return result


def resolve(folder):
    folder = Path(folder)
    marker = folder / 'baseline-native-reuse.json'
    if not marker.exists():
        return None
    plan = read(folder / 'automatic-plan.json')
    identity = plan.get('baseline_reuse')
    if not identity:
        raise ValueError('Fixed B0 native reuse lacks a prospective selection')
    proof = inspect(identity['source'], identity['checksums_sha256'], plan, location=False)
    if read(marker) != proof:
        raise ValueError('Fixed B0 native reuse proof changed')
    return proof
