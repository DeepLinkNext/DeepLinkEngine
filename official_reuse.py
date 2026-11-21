"""Explicitly retain a sealed C8 result after a later pre-model failure."""
import hashlib
import json
from pathlib import Path
import shutil


REVIEWED_CONTROLLERS = {
    'official_suite.py': {
        '0e005084411b8497d84c003bebbdbe699ffe6c713d2f9f39dd767abcbc2daa38':
        '63b8e6157a5bcbc803973b8099128cb1df28fd7d4cb8d6d9f5b4a335ad26fdcc',
    },
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified(root, name, seal):
    path = root / name
    if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve()) or seal.get(name) != sha(path)):
        raise ValueError('Completed-phase evidence changed or is unsealed: ' + name)
    return path


def read(root, name, seal):
    return json.loads(verified(root, name, seal).read_bytes())


def admit(source, destination, *, config, image_lock, sources, weights, mapping, inputs, current_root, arguments):
    source, destination, current_root = map(Path, (source, destination, current_root))
    if source.is_symlink() or source.name != 'continuous' or destination.exists():
        raise ValueError('Select an original continuous phase and a new destination')
    previous = source.parent
    outer = json.loads((previous / 'checksums.json').read_bytes())
    campaign = read(previous, 'campaign.json', outer)
    if (campaign['schema'] != 'deeplinkengine-official-full-suite/v1'
            or campaign['status'] != 'failed' or campaign['baseline_image'] != image_lock
            or campaign['physical_mapping'] != mapping
            or campaign['continuous_input_sha256'] != sha(inputs)):
        raise ValueError('Completed continuous runtime, inputs or physical mapping differ')
    phases = campaign['stages']
    if (len(phases) != 10 or phases[0]['name'] != 'continuous' or phases[0]['status'] != 'completed'
            or phases[1]['name'] != 'mmlu' or phases[1]['status'] != 'failed'
            or any(v['status'] != 'not_attempted' for v in phases[2:])):
        raise ValueError('Reuse is limited to a completed C8 followed by a pre-model MMLU failure')
    failed = read(previous, 'mmlu/receipt.json', outer)
    if (failed.get('model_service_started') is not False or failed.get('client_started') is not False
            or failed.get('quiescent') is not True):
        raise ValueError('A later model may have started; this narrow recovery refuses a repeat')
    before_weights = read(previous, 'model-weights.json', outer)
    signature = lambda value: {name: {k: row[k] for k in ('bytes', 'sha256')} for name, row in value.items()}
    if signature(before_weights) != signature(weights):
        raise ValueError('Completed continuous checkpoint weights differ')
    seal = read(previous, 'continuous/checksums.json', outer)
    for name in seal:
        verified(source, name, seal)
    receipt = read(source, 'receipt.json', seal)
    if any(receipt.get(k) is not True for k in ('completed', 'quiescent', 'model_ready', 'client_started')):
        raise ValueError('Only completed quiescent continuous inference can be retained')
    if receipt['baseline_id'] != image_lock['id'] or receipt['phase'] != 'continuous':
        raise ValueError('Completed phase identity differs')
    for action in receipt['cleanup_actions']:
        stat = Path('/proc') / str(action['pid']) / 'stat'
        if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[19] == str(action['start_ticks']):
            raise ValueError('An original owned process is still alive')
    plan = read(source, 'service/plan.json', seal)
    if (Path(plan['model']).resolve() != Path(config['model']).resolve()
            or Path(plan['protocol_deployment']).resolve() != Path(config['protocol_deployment']).resolve()
            or plan['serve_arguments'] != arguments):
        raise ValueError('Completed serving model, protocol or arguments differ')
    old_root = Path(plan['client_tools']).resolve().parents[1]
    source_changes = {}
    for name, pin in campaign['sources'].items():
        path = Path(name)
        if sha(path) != pin:
            raise ValueError('Original completed source changed: ' + name)
        if path.is_relative_to(old_root):
            relative = str(path.relative_to(old_root))
            target = current_root / relative
            target_pin = sha(target)
            if target_pin != pin:
                if REVIEWED_CONTROLLERS.get(relative, {}).get(pin) != target_pin:
                    raise ValueError('Unreviewed completed-phase source difference: ' + relative)
                source_changes[relative] = dict(original_sha256=pin, current_sha256=target_pin)
            if sources.get(str(target.resolve())) != target_pin:
                raise ValueError('Current source admission differs: ' + relative)
        elif sources.get(name) != pin:
            raise ValueError('Completed external protocol source differs: ' + name)
    result = read(source, 'service/continuous.json', seal)
    expected = ['4096-warmup', '4096-formal', '16384-warmup', '16384-formal']
    if result['status'] != 'completed' or [v['name'] for v in result['phases']] != expected:
        raise ValueError('Missing complete original C8 phases')
    for index, row in enumerate(result['phases']):
        if row['seconds'] != (60 if index % 2 == 0 else 300) or row['failed'] != 0 or row['requests'] <= 0:
            raise ValueError('Completed C8 submission window or failures differ')
        records = []
        for name in seal:
            if name.startswith('service/service-') and name.endswith('.json'):
                value = read(source, name, seal)
                if value.get('phase') == row['name']:
                    records.append(value)
        if len(records) != row['requests'] or any(v.get('success') is not True for v in records):
            raise ValueError('Completed C8 raw response population differs')
    destination.mkdir()
    for name in seal:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
        verified(destination, name, seal)
    shutil.copy2(source / 'checksums.json', destination / 'checksums.json')
    return dict(schema='deeplinkengine-reused-completed-continuous/v1', source=str(source.resolve()),
                original_checksums_sha256=sha(source / 'checksums.json'),
                original_campaign_checksums_sha256=sha(previous / 'checksums.json'),
                original_receipt_sha256=seal['receipt.json'], original_dates_and_outputs_preserved=True,
                inference_repeated=False, reviewed_controller_changes=source_changes)
