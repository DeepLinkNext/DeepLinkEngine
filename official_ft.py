"""Public automatic F/T completion for official version campaigns."""
import json
import hashlib
import os
from pathlib import Path
import re
import time

from official_baselines import save
from official_ft_audit import Evidence, sha
ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / 'benchmark/one-click-kit'


def finish(root, name, record):
    """Publish the completed receipt only after its final seal was written."""
    from official_report import seal
    root = Path(root)
    if record.get('completed') is not True:
        save(root / name, record)
        seal(root)
        return
    try:
        files = seal(root)  # On disk the receipt is still running/failed.
        raw = (json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        files[name] = hashlib.sha256(raw).hexdigest()
        save(root / 'checksums.json', files)
        marker = root / (name + '.tmp')
        marker.write_bytes(raw)
        marker.replace(root / name)  # Exact sealed bytes; completion is last.
    except BaseException as error:
        failed = dict(record, status='failed', completed=False,
                      finalization_error=type(error).__name__ + ': ' + str(error))
        save(root / name, failed)
        raise


def verify_model(model, original_weights, out):
    """Reuse unchanged verified files, otherwise hash each required shard."""
    from full_reference_reuse import domain
    expected = domain()['weights']
    if ({n: dict(bytes=v['bytes'], sha256=v['sha256']) for n, v in original_weights.items()} != expected):
        raise ValueError('Native weights differ from the original complete-reference domain')
    checked = {}
    for name, value in expected.items():
        path = Path(model) / name
        before = path.stat()
        old = original_weights[name]
        retained = (before.st_size == old['bytes'] and before.st_mtime_ns == old.get('mtime_ns')
                    and before.st_ino == old.get('inode'))
        if not retained:
            print('OFFICIAL_FT_VERIFY_WEIGHT', name, flush=True)
            if before.st_size != value['bytes'] or sha(path) != value['sha256']:
                raise ValueError('Current companion checkpoint differs: ' + name)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
            raise ValueError('Companion checkpoint changed during verification')
        checked[name] = dict(**value, unchanged_original_verified_stat=retained,
                             mtime_ns=after.st_mtime_ns, inode=after.st_ino)
    save(Path(out) / 'model-verification.json', checked)
    return checked


def complete_native(config, image, native, output, tools, sdk, sources, leases, continuous,
                    *, mapping, weights, native_seal=None, companion_source=None, companion_seal=None):
    """Keep clean native samples and append one separately labelled companion."""
    import sys
    sys.path[:0] = [str(TOOLS / 'scripts'), str(TOOLS / 'third-party-test-kit')]
    from full_reference_reuse import arm_statistics, domain, prepare, qualify_mapping
    from official_report import seal
    from official_ft_audit import audit
    from official_native import run as capture
    output = Path(output)
    output.mkdir(exist_ok=False)
    record = dict(schema='deeplinkengine-official-ft-completion/v1', status='running', completed=False,
        started=time.time(), source_native=str(native), source_native_seal_sha256=sha(Path(native) / 'checksums.json'),
        native_timing_repeated=False, dataset_solvers_repeated=False, independent_third_party=False)
    save(output / 'receipt.json', record)
    try:
        if native_seal is not None and record['source_native_seal_sha256'] != native_seal:
            raise ValueError('Externally bound native evidence changed')
        qualify_mapping(mapping, domain()['accelerator'])
        verify_model(config['model'], weights, output)
        save(output / 'model-weights.json', weights)
        source = Evidence(native, record['source_native_seal_sha256'])
        if source.read('receipt.json')['quiescent'] is not True:
            raise ValueError('Source native workers are not quiescent')
        # Validate the original input file before the controller copies it.
        source.bytes('token-workload.json')
        if companion_source is None:
            capture(config, image, output / 'companion', tools, sdk, sources, leases, continuous,
                    native_source=Path(native))
            companion_source = output / 'companion'
        else:
            retained = Evidence(companion_source, companion_seal)
            if (retained.read('receipt.json')['completed'] is not True
                    or retained.read('receipt.json')['quiescent'] is not True):
                raise ValueError('Only a sealed, closed independent companion may be retained')
        record.update(companion_source=str(companion_source), companion_repeated=False,
                      companion_source_seal_sha256=sha(Path(companion_source) / 'checksums.json'))
        version = json.loads((image / 'image-lock.json').read_bytes())['upstream_version']
        rows = audit(native, companion_source, image, Path(config['model']) / 'config.json',
                     output / 'audit', version=version, source_seal=record['source_native_seal_sha256'])
        # Resolve the existing PUBLIC immutable mirror. The older nested
        # project_evidence default is private; never fall through to it.
        import importlib.util
        public_spec = importlib.util.spec_from_file_location('deeplink_public_reference_restore', ROOT / 'one_click.py')
        one_click = importlib.util.module_from_spec(public_spec)
        public_spec.loader.exec_module(one_click)
        lock = json.loads((ROOT / 'benchmark/one-click-lock.json').read_bytes())
        public_reference = one_click.reference_evidence(lock)
        previous = os.environ.get('VLLM_HUST_PROJECT_EVIDENCE')
        os.environ['VLLM_HUST_PROJECT_EVIDENCE'] = str(public_reference)
        try:
            reference = output / 'reference'
            reference.mkdir()
            statistics = prepare(reference)
        finally:
            if previous is None:
                os.environ.pop('VLLM_HUST_PROJECT_EVIDENCE', None)
            else:
                os.environ['VLLM_HUST_PROJECT_EVIDENCE'] = previous
        if statistics['original_method_frozen_unix'] >= source.read('receipt.json')['started']:
            raise ValueError('Native measurement predates the fixed independent reference method')
        result = arm_statistics(rows, statistics['arms']['B0'])
        result.update(schema='deeplinkengine-qualified-official-m01/v1', role='B0',
            new_F_T_qualification=True, necessary_work_certified=True,
            source_runtime_applicability_certified=True, original_native_timing_retained=True,
            native_timing_repeated=False, independent_third_party=False,
            public_reference_asset=lock['reference_evidence'],
            reference_statistics_sha256=sha(reference / 'qualified-full-reference-statistics.json'),
            source_native_seal_sha256=record['source_native_seal_sha256'],
            arithmetic_companion_seal_sha256=record['companion_source_seal_sha256'],
            hardware_mapping=mapping, hardware_peak_utilization_percent=None,
            original_efficiency_substituted=False, original_results_modified=False)
        save(output / 'qualified-m01.json', result)
        record.update(status='completed', completed=True, new_F_T_qualification=True,
                      qualification_sha256=sha(output / 'qualified-m01.json'))
    except BaseException as error:
        record.update(status='failed', error=type(error).__name__ + ': ' + str(error),
                      new_F_T_qualification=False)
        raise
    finally:
        record['ended'] = time.time()
        finish(output, 'receipt.json', record)
    return record


def continue_closed(config, original, expected_seal, *, companion_source=None, companion_seal=None):
    """Explicit public continuation: only companion and CPU work, new output."""
    import sys
    sys.path.insert(0, str(TOOLS / 'scripts'))
    from official_image import prepare, read_lock
    from official_suite import source_bindings
    from resource_guard import acquire, release, verify_mapping
    if not re.fullmatch('[0-9a-f]{64}', expected_seal or ''):
        raise ValueError('Provide the independently fixed original version seal SHA256')
    if companion_source is not None and not re.fullmatch('[0-9a-f]{64}', companion_seal or ''):
        raise ValueError('Provide the independently fixed retained companion seal SHA256')
    original = Path(original).resolve()
    source = Evidence(original, expected_seal)
    campaign = source.read('campaign.json')
    if (campaign['status'] not in ('measurements_collected', 'completed', 'failed')
            or not isinstance(campaign.get('ended'), (int, float))
            or campaign['ended'] <= 0
            or any(row['status'] in ('running', 'starting', 'planned') for row in campaign['stages'])):
        raise ValueError('Only a terminal original campaign may receive a new F/T continuation')
    source.bytes('native-timing/checksums.json')
    native_receipt = source.read('native-timing/receipt.json')
    if native_receipt['completed'] is not True or native_receipt['quiescent'] is not True:
        raise ValueError('Closed native measurements are required; do not repeat them here')
    version = campaign['baseline_image']['upstream_version']
    if config['versions'] != [version]:
        raise ValueError('Select exactly the original source version')
    lock_path = ROOT / 'engine/official-baselines' / (version + '.json')
    identity = read_lock(lock_path)
    if identity != campaign['baseline_image']:
        raise ValueError('Companion runtime source identity differs from the native original')
    image = Path(config['cache']) / 'official-sdks' / (version + '-' + identity['manifest_digest'].split(':')[1][:16])
    if not (image / 'installation.json').is_file():
        prepare(lock_path, Path(config['cache']) / 'official-oci', image)
    elif json.loads((image / 'image-lock.json').read_bytes()) != identity:
        raise ValueError('Cached companion SDK identity differs')
    out = Path(config['out'])
    out.mkdir(parents=True, exist_ok=False)
    weights = source.read('model-weights.json')
    sdk = Path(config['protocol_deployment'])
    sources = source_bindings(TOOLS, sdk)
    sources.update({str(p): sha(p) for p in ROOT.glob('official*.py')})
    sources.update({str(p): sha(p) for p in (ROOT / 'deeplinkengine_official_ft-1.0.dist-info').iterdir()})
    os.environ['VLLM_HUST_PHYSICAL_DEVICES'] = ','.join(map(str, config['physical_devices']))
    current_mapping = verify_mapping('0,1')
    record = dict(schema='deeplinkengine-closed-official-ft-continuation/v1', completed=False,
        status='running', started=time.time(), original=str(original),
        original_version_seal_sha256=expected_seal, original_campaign_status=campaign['status'],
        original_results_modified=False, source_execution=campaigndata(campaign),
        cpu_and_companion_source_sha256={p.name: sha(p) for p in ROOT.glob('official*.py')},
        companion_physical_mapping=current_mapping, native_timing_repeated=False,
        dataset_solvers_repeated=False, independent_third_party=False)
    save(out / 'continuation.json', record)
    leases = []
    try:
        if companion_source is None:
            leases = acquire(out, 1800)
        result = complete_native(config, image, original / 'native-timing', out / 'necessary-work',
            TOOLS, sdk, sources, leases, original / 'continuous', mapping=campaign['physical_mapping'],
            weights=weights, native_seal=source.seal['native-timing/checksums.json'],
            companion_source=companion_source, companion_seal=companion_seal)
        record.update(status='completed', completed=True, result=result)
        return 0
    except BaseException as error:
        record.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        if leases:
            release(leases)
            os.environ.pop('VLLMHUST_ENGINE_LOCKS', None)
        record['ended'] = time.time()
        finish(out, 'continuation.json', record)


def campaigndata(campaign):
    return dict(status=campaign['status'], started=campaign['started'], ended=campaign['ended'],
                sources=campaign['sources'], qualification=campaign['qualification'])
