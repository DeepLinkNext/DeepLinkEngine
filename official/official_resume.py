"""Retain six sealed candidate solver phases after a pre-ready judge failure."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat

from official import source_path
from official.official_rebuild import Evidence, digest, member


PHASES = ('continuous', 'mmlu', 'frontier', 'hle', 'swe', 'terminal',
          'frontier-judge', 'hle-judge', 'context', 'native-timing', 'necessary-work')
RETAIN = PHASES[:6]
CONTROLLERS = {'official/official_suite.py', 'official/official_judge.py',
               'official/official_report.py'}


def _same_path(left, right):
    return Path(left).resolve() == Path(right).resolve()


def _closed(receipt):
    if receipt.get('quiescent') is not True:
        raise ValueError('Original phase is not quiescent')
    for action in receipt.get('cleanup_actions', []):
        pid, ticks = action.get('pid'), action.get('start_ticks')
        if (not isinstance(pid, int) or pid <= 0 or isinstance(ticks, bool)
                or not isinstance(ticks, (int, str)) or not str(ticks).isdigit() or int(ticks) <= 0):
            raise ValueError('Invalid original cleanup process identity')
        path = Path('/proc') / str(pid) / 'stat'
        try:
            current = path.read_text().rsplit(')', 1)[1].split()[19]
        except FileNotFoundError:
            continue
        if current == str(ticks):
            raise ValueError('An original owned process still exists')


def _phase(evidence, name):
    prefix = name + '/'
    seal = evidence.read(prefix + 'checksums.json')
    if not isinstance(seal, dict) or 'receipt.json' not in seal:
        raise ValueError('Missing original phase receipt/seal: ' + name)
    outer = {key[len(prefix):]: pin for key, pin in evidence.seal.items()
             if key.startswith(prefix) and key != prefix + 'checksums.json'}
    if seal != outer:
        raise ValueError('Inner and outer phase sealed populations differ: ' + name)
    return seal, evidence.read(prefix + 'receipt.json')


def _source_bindings(evidence, original, sources, old_root, current_root):
    if not isinstance(original, dict) or not original:
        raise ValueError('Original published sources are required')
    changes = {}
    for label, pin in original.items():
        if not isinstance(pin, str) or not re.fullmatch('[a-f0-9]{64}', pin):
            raise ValueError('Invalid original source digest')
        # Check the actual original execution bytes, including external protocol tools.
        path = evidence.auxiliary(label, pin).resolve()
        if path.is_relative_to(old_root):
            name = path.relative_to(old_root).as_posix()
            target = source_path(current_root, name).resolve()
        else:
            name, target = None, path
        if not target.is_file() or target.is_symlink():
            raise ValueError('Current published source is missing: ' + str(target))
        now = digest(target)
        if sources.get(str(target)) != now:
            raise ValueError('Current source admission differs: ' + str(target))
        if now != pin:
            if name not in CONTROLLERS:
                raise ValueError('Non-controller source changed: ' + str(name or path))
            changes[name] = dict(original_sha256=pin, current_sha256=now,
                                 original_path=label, current_path=str(target))
    return changes


def _pre_ready(evidence, receipt):
    _closed(receipt)
    if (receipt.get('schema') != 'deeplinkengine-official-fixed-judge/v1'
            or receipt.get('dataset') != 'frontier' or receipt.get('status') != 'failed'
            or receipt.get('completed') is not False
            or any(receipt.get(key, False) is not False for key in ('model_ready', 'client_started'))):
        raise ValueError('Only a failed pre-ready frontier judge may be resumed')
    # Include unsealed names in this refusal: omitted client results must not hide a repeat.
    root = evidence.root / 'frontier-judge'
    forbidden = {'ready-models.json', 'client.log', 'summarize.log'}
    outputs = {'evaluation', 'quality-summary', 'requests', 'responses'}
    for parent, directories, files in os.walk(root, followlinks=False):
        if any((Path(parent) / name).is_symlink() for name in directories + files):
            raise ValueError('Symlinks in failed judge evidence are not admitted')
        for name in files:
            path = Path(parent) / name
            if (name in forbidden or set(path.relative_to(root).parts) & outputs
                    or name.startswith(('request-', 'response-'))):
                raise ValueError('Judge readiness or client request evidence already exists')


def admit(source, destination, *, expected_seal, config, image_lock, runtime, sources,
          weights, mapping, inputs, current_root, phases):
    """Verify a closed whole campaign, then copy only its six unchanged sealed phases."""
    source, destination, current_root = map(lambda value: Path(value).absolute(),
                                             (source, destination, current_root))
    if (source.is_symlink() or not source.is_dir() or destination.is_symlink()
            or destination.resolve().is_relative_to(source.resolve())
            or source.resolve().is_relative_to(destination.resolve())
            or not re.fullmatch('[a-f0-9]{64}', expected_seal or '')
            or tuple(phases) != PHASES):
        raise ValueError('Select a closed campaign, a separate output and the fixed phase order')
    if any(os.path.lexists(destination / name) for name in RETAIN):
        raise ValueError('Retained phase destinations must be absent')
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError('Destination ancestors must not be symlinks')
    seal_raw = member(source, 'checksums.json').read_bytes()
    if hashlib.sha256(seal_raw).hexdigest() != expected_seal:
        raise ValueError('External whole campaign seal differs')
    pins = json.loads(seal_raw)
    if not isinstance(pins, dict) or not pins:
        raise ValueError('A nonempty whole campaign seal is required')
    for name in pins:
        if not stat.S_ISREG(member(source, name).stat().st_mode):
            raise ValueError('Only ordinary sealed files may be retained')
    evidence = Evidence(source)  # Verify every sealed file, not only retained phases.
    campaign = evidence.read('campaign.json')
    rows = campaign.get('stages', [])
    if (campaign.get('schema') != 'deeplinkengine-official-full-suite/v1'
            or campaign.get('status') != 'failed'
            or type(campaign.get('ended')) not in (int, float)
            or not math.isfinite(campaign['ended']) or campaign['ended'] <= 0
            or campaign.get('baseline_image') != image_lock
            or image_lock.get('manifest_kind') != 'source-build-derivation'
            or image_lock.get('version') != '0.28.0rc1-source'
            or runtime.get('runtime_kind') != 'source-build-candidate'
            or campaign.get('runtime_identity') != runtime
            or campaign.get('physical_mapping') != mapping
            or campaign.get('continuous_input_sha256') != digest(inputs)
            or [row.get('name') for row in rows] != list(PHASES)
            or [row.get('status') for row in rows] != ['completed'] * 6 + ['failed'] + ['not_attempted'] * 4):
        raise ValueError('Original terminal candidate campaign identity or stage boundary differs')
    if evidence.read('model-weights.json') != weights:
        raise ValueError('Original model weights/readback differ')
    expected_input = Path(config['protocol_run']) / 'continuous-B0/inputs/continuous-inputs-r1.json'
    if not _same_path(inputs, expected_input):
        raise ValueError('Continuous input is not from this protocol run')
    frozen_pin = digest(Path(config['protocol_run']) / 'frozen-execution-plan.json')
    from official import enable_module_imports
    enable_module_imports()
    from official_suite import public_arguments
    _, failed = _phase(evidence, 'frontier-judge')
    _pre_ready(evidence, failed)
    admitted, old_root = {}, None
    for name in RETAIN:
        seal, receipt = _phase(evidence, name)
        _closed(receipt)
        if (receipt.get('schema') != 'deeplinkengine-official-arm/v1'
                or receipt.get('baseline_id') != image_lock['id'] or receipt.get('phase') != name
                or receipt.get('role') != 'B0' or receipt.get('status') != 'completed'
                or any(receipt.get(key) is not True for key in ('completed', 'model_ready', 'client_started'))
                or receipt.get('input_plan_sha256') != frozen_pin
                or rows[PHASES.index(name)].get('receipt') != receipt):
            raise ValueError('Original completed solver receipt differs: ' + name)
        plan = evidence.read(name + '/service/plan.json')
        kind = 'continuous' if name == 'continuous' else 'joint' if name == 'mmlu' else 'dataset'
        root = Path(plan['client_tools']).resolve().parents[1]
        if old_root is not None and root != old_root:
            raise ValueError('Original execution source roots differ')
        old_root = root
        if (plan.get('baseline_id') != image_lock['id'] or plan.get('role') != 'B0'
                or plan.get('runtime_identity') != runtime
                or not _same_path(plan['model'], config['model'])
                or not _same_path(plan['protocol_deployment'], config['protocol_deployment'])
                or not _same_path(plan['continuous_inputs'], inputs)
                or plan.get('input_sha256') != digest(inputs)
                or plan.get('kind') != kind
                or plan.get('serve_arguments') != public_arguments(kind)
                or evidence.read(name + '/model-weights.json') != weights
                or evidence.raw(name + '/resource-admission.json') != evidence.raw('resource-admission.json')
                or evidence.read(name + '/deployment.json') != dict(release=image_lock['id'], **runtime)
                or plan.get('client_sources') != campaign['sources']):
            raise ValueError('Original solver runtime, model or protocol plan differs: ' + name)
        if kind == 'dataset':
            suffix = 'trace' if name in ('swe', 'terminal') else 'solver'
            spec = Path(config['protocol_run']) / 'plans' / (name + '-' + suffix + '-B0.json')
            if (not _same_path(plan['dataset_spec'], spec)
                    or evidence.seal.get(name + '/dataset-spec.json') != digest(spec)):
                raise ValueError('Original fixed dataset workload differs: ' + name)
        if not isinstance(plan.get('worker_sources'), dict) or not plan['worker_sources']:
            raise ValueError('Original worker/tool bindings are required: ' + name)
        for label, pin in plan['worker_sources'].items():
            if campaign['sources'].get(label) != pin:
                raise ValueError('Original worker/tool binding differs: ' + name)
        admitted[name] = (seal, receipt)
    changes = _source_bindings(evidence, campaign['sources'], sources, old_root, current_root.resolve())
    # No writes happen until the whole failed campaign and every retained phase are admitted.
    result = {}
    for name, (seal, receipt) in admitted.items():
        target = destination / name
        target.mkdir(parents=True, exist_ok=False)
        for item, pin in seal.items():
            path = member(target, item)
            path.parent.mkdir(parents=True, exist_ok=True)
            original_path = member(source, name + '/' + item)
            with original_path.open('rb') as original, path.open('xb') as stream:
                shutil.copyfileobj(original, stream, length=1024 * 1024)
            if digest(path) != pin:
                raise ValueError('Copied original phase changed: ' + name + '/' + item)
            shutil.copystat(original_path, path)
        with (target / 'checksums.json').open('xb') as stream:
            stream.write(evidence.raw(name + '/checksums.json'))
        shutil.copystat(member(source, name + '/checksums.json'), target / 'checksums.json')
        result[name] = dict(schema='deeplinkengine-reused-completed-solvers/v1',
            source=str((source / name).resolve()), original_campaign=str(source.resolve()),
            original_checksums_sha256=evidence.seal[name + '/checksums.json'],
            original_campaign_checksums_sha256=expected_seal,
            original_receipt_sha256=seal['receipt.json'],
            original_dates_and_outputs_preserved=True, inference_repeated=False,
            reviewed_controller_changes=changes)
    return result
