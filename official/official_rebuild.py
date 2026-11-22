"""CPU replay and portable custody for sealed official 0.23 C8 originals."""
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import tarfile
import tempfile
import time

from official import source_path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'benchmark/one-click-kit/scripts'
PHASES = ('4096-warmup', '4096-formal', '16384-warmup', '16384-formal')
VERSIONS = {'vllm': '0.23.0+empty', 'vllm-ascend': '0.23.0',
            'torch': '2.10.0+cpu', 'torch-npu': '2.10.0.post4'}
SUPPORT = {'original_audit.py', 'report_execution.py', 'serving_warmup.py', 'compile_boundary.py',
           'runtime_identity.py', 'continuous_client.py', 'official_worker.py',
           'official_kernel_preparation.py', 'official_serve.py', 'official_http.py',
           'official_suite.py'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def relative(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('Unsafe evidence member')
    value = PurePosixPath(name)
    if value.is_absolute() or not value.parts or '..' in value.parts or str(value) != name:
        raise ValueError('Unsafe evidence member: ' + name)
    return value


def member(root, name):
    parts = relative(name).parts
    current = Path(root)
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink evidence is not admitted: ' + name)
    if not current.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Evidence escapes its source')
    return current


class Evidence:
    def __init__(self, root, support=None, logical_root=None, *, c8_subset=False):
        self.root = Path(root)
        self.support = support
        self.logical_root = logical_root or str(self.root.absolute())
        seal_raw = member(self.root, 'checksums.json').read_bytes()
        self.seal_sha256 = hashlib.sha256(seal_raw).hexdigest()
        self.seal = json.loads(seal_raw)
        if not isinstance(self.seal, dict) or not self.seal:
            raise ValueError('A nonempty original version seal is required')
        for name, pin in self.seal.items():
            relative(name)
            if not isinstance(pin, str) or not re.fullmatch('[a-f0-9]{64}', pin):
                raise ValueError('Invalid original seal member: ' + name)
        self.available = c8_members(self.seal) if c8_subset else set(self.seal)
        if c8_subset:
            present = {p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()}
            if present != self.available | {'checksums.json'}:
                raise ValueError('Portable C8 payload must exactly match its original sealed closure')
        for name in self.available:
            pin = self.seal[name]
            if not re.fullmatch('[a-f0-9]{64}', pin) or digest(member(self.root, name)) != pin:
                raise ValueError('Original sealed evidence changed: ' + name)

    def raw(self, name):
        path = member(self.root, name)
        raw = path.read_bytes()
        if name not in self.available or hashlib.sha256(raw).hexdigest() != self.seal[name]:
            raise ValueError('Unsealed evidence: ' + name)
        return raw

    def read(self, name):
        return json.loads(self.raw(name))

    def auxiliary(self, logical, pin):
        if self.support is None:
            path = Path(logical)
            if path.is_relative_to(ROOT) and not (path.exists() or path.is_symlink()):
                path = source_path(ROOT, path.relative_to(ROOT))
        else:
            if logical not in self.support:
                raise ValueError('Portable source binding missing: ' + logical)
            path = self.support[logical]
        if not Path(path).is_file() or digest(path) != pin:
            raise ValueError('Original supporting source changed: ' + logical)
        return Path(path)


def c8_members(seal):
    required = {'campaign.json', 'model-weights.json', 'resource-admission.json'}
    if not required.issubset(seal) or 'continuous/checksums.json' not in seal:
        raise ValueError('Original C8 seal and parent identity metadata are required')
    return required | {name for name in seal if name.startswith('continuous/')}


def trusted(name, evidence, bindings):
    path = TOOLS / name
    matches = [(key, pin) for key, pin in bindings.items() if PurePosixPath(key).name == name]
    if len(matches) != 1 or matches[0][1] != digest(path):
        raise ValueError('Replay consumer differs from original source binding: ' + name)
    evidence.auxiliary(*matches[0])
    spec = importlib.util.spec_from_file_location('_official_replay_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def admit_plan(plan, lock):
    arguments = plan['serve_arguments']
    def option(name):
        if (arguments.count(name) != 1 or any(arg.startswith(name+'=') for arg in arguments)
                or arguments.index(name)+1 >= len(arguments)):
            return None
        return arguments[arguments.index(name) + 1]
    if (plan['baseline_id'] != lock['id'] or plan['role'] != 'B0' or plan['kind'] != 'continuous'
            or plan['runtime_versions'] != VERSIONS or plan['original_baseline_018_relabelled'] is not False
            or plan['necessary_work_certified'] is not False
            or PurePosixPath(plan['model']).name != 'Qwen3.5-35B-A3B'
            or option('--tensor-parallel-size') != '2' or option('--dtype') != 'bfloat16'
            or option('--max-model-len') != '65536'
            or option('--max-num-batched-tokens') != '4096' or option('--max-num-seqs') != '24'
            or option('--kv-cache-memory-bytes') != str(20*1024**3)
            or json.loads(option('--speculative-config') or '{}') != dict(method='mtp', num_speculative_tokens=2)
            or arguments.count('--no-enable-prefix-caching') != 1
            or any(arg.split('=')[0] == '--enable-prefix-caching' or arg.startswith('--no-enable-prefix-caching=') for arg in arguments)
            or json.loads(option('--compilation-config') or '{}').get('cudagraph_mode') != 'FULL'):
        raise ValueError('Original official model/runtime plan differs')


def recorded_hardware(admission, seal_sha):
    text = admission['npu_before']
    tool_version = re.search(r'npu-smi\s+(\S+)', text)
    if (not tool_version or set(re.findall(r'\|\s*\d+\s+(910B\w+)\s*\|', text)) != {'910B2'}
            or len(admission['physical_cards']) != 2 or len(set(admission['physical_cards'])) != 2
            or admission.get('admitted') is not True):
        raise ValueError('Original recorded hardware identity is unavailable')
    return dict(chip_model='Ascend 910B2', driver_version=None, driver_version_status='not-recorded',
                      npu_smi_version=tool_version[1], npu_smi_version_evidence='original npu-smi header; tool version only',
                      physical_devices=admission['physical_cards'], host_identity=None,
                      host_identity_status='Not recorded in original sealed arm', cohort_id=seal_sha)


def replay(evidence):
    campaign = evidence.read('campaign.json')
    lock = json.loads((ROOT / 'engine/official-baselines/0.23.0.json').read_bytes())
    if (campaign.get('schema') != 'deeplinkengine-official-full-suite/v1'
            or campaign.get('baseline_image') != lock
            or campaign.get('status') not in ('failed', 'completed')
            or campaign['stages'][0]['name'] != 'continuous'
            or campaign['stages'][0]['status'] != 'completed'):
        raise ValueError('Select a terminal official 0.23 version with complete original C8')
    prefix = 'continuous/'
    arm_seal = evidence.read(prefix + 'checksums.json')
    original_members = {name[len(prefix):] for name in evidence.seal
                        if name.startswith(prefix) and name != prefix + 'checksums.json'}
    if set(arm_seal) != original_members:
        raise ValueError('Inner and outer C8 sealed populations differ')
    for name, pin in arm_seal.items():
        relative(name)
        if evidence.seal.get(prefix + name) != pin:
            raise ValueError('Inner C8 seal differs from its original version seal')
    arm = evidence.read(prefix + 'receipt.json')
    if (arm.get('schema') != 'deeplinkengine-official-arm/v1'
            or arm.get('baseline_id') != lock['id'] or arm.get('role') != 'B0'
            or arm.get('phase') != 'continuous'
            or any(arm.get(k) is not True for k in ('completed', 'quiescent', 'model_ready', 'client_started'))):
        raise ValueError('Only completed, quiescent official C8 is admitted')
    deployment = evidence.read(prefix + 'deployment.json')
    if deployment != dict(release=lock['id'], official_image_manifest=lock['manifest_digest'],
                          official_runtime=True, source_code_patched=False):
        raise ValueError('Original official image attribution differs')
    plan = evidence.read(prefix + 'service/plan.json')
    admit_plan(plan, lock)
    bindings = campaign['sources']
    if any(bindings.get(name) != pin for name, pin in plan['client_sources'].items()):
        raise ValueError('Original client source binding differs')
    for name, pin in plan['worker_sources'].items():
        if bindings.get(name) != pin:
            raise ValueError('Original worker source binding differs')
    support = {name: evidence.auxiliary(name, pin) for name, pin in bindings.items()
               if PurePosixPath(name).name in SUPPORT}
    if {PurePosixPath(name).name for name in support} != SUPPORT:
        raise ValueError('Complete original C8 controller/client bindings are required')
    audit = trusted('original_audit.py', evidence, bindings)
    warmup_module = trusted('serving_warmup.py', evidence, bindings)
    kernel_pins = [pin for name, pin in bindings.items() if PurePosixPath(name).name == 'official_kernel_preparation.py']
    if kernel_pins != [digest(ROOT / 'official/official_kernel_preparation.py')]:
        raise ValueError('Kernel identity consumer differs from original binding')
    from official_kernel_preparation import POLICY, SOURCES
    ranks = evidence.read(prefix + 'service/official-kernel-preparation.json')
    if sorted(row['rank'] for row in ranks) != [0, 1]:
        raise ValueError('Both original official workers are required')
    for row in ranks:
        name = prefix + 'service/rank-' + str(row['rank']) + '-official-kernel-preparation.json'
        value = evidence.read(name)
        if (evidence.seal[name] != row['sha256'] or value['rank'] != row['rank']
                or value.get('schema') != 'deeplinkengine-official-kernel-preparation/v1'
                or value['policy'] != POLICY or value['official_image_manifest'] != lock['manifest_digest']
                or value['source_sha256'] != SOURCES
                or value['versions'] != {'vllm': VERSIONS['vllm'], 'vllm-ascend': VERSIONS['vllm-ascend'], 'triton-ascend': '3.2.2'}
                or value['completed'] is not True or value['measurement_guard_relaxed'] is not False
                or value['compilation_only'] is not True or value['device_kernels_launched'] is not False
                or value['model_tensors_read'] is not False
                or value['upstream_sources_modified'] is not False):
            raise ValueError('Original worker kernel/source identity differs')
    warmup = evidence.read(prefix + 'service/serving-warmup/receipt.json')
    after = evidence.read(prefix + 'service/compilation-after-measurement.json')
    if (warmup['protocol'] != warmup_module.protocol('continuous')
            or warmup['completed'] is not True or warmup['formal_answers_reused'] is not False
            or len(warmup['requests']) != warmup_module.request_count('continuous')
            or after != warmup['ranks_ready'] or sorted(row['rank'] for row in after) != [0, 1]
            or any(not row['sealed'] or not row['torch_dynamo_guard'] or not row['triton_guard']
                   or row['failure'] is not None for row in after)):
        raise ValueError('Original shared warmup/measurement boundary differs')
    for row in warmup['requests']:
        for suffix, field in (('request.json', 'request_sha256'), ('sse', 'sse_sha256')):
            name = prefix + 'service/serving-warmup/' + f"{row['ordinal']:04d}." + suffix
            if evidence.seal.get(name) != row[field]:
                raise ValueError('Warmup transport differs')
    if sorted(row['ordinal'] for row in warmup['requests']) != list(range(len(warmup['requests']))):
        raise ValueError('Shared warmup request population differs')
    inputs_raw = evidence.raw(prefix + 'service/serving-warmup/input-plan.json')
    if hashlib.sha256(inputs_raw).hexdigest() != warmup['input_plan_sha256'] or warmup['input_plan_sha256'] != plan['input_sha256']:
        raise ValueError('Original input source differs')
    inputs = json.loads(inputs_raw)
    if (set(inputs) != {'4096', '16384'} or any(len(rows) != 8 for rows in inputs.values())
            or inputs != evidence.read(prefix + 'service/request-plan.json')):
        raise ValueError('Original two synthetic C8 populations differ')
    token_checks = [dict(input_tokens=case['input_tokens'], token_ids_sha256=case['token_ids_sha256'])
                    for cases in inputs.values() for case in cases]
    if (warmup['token_checks'] != token_checks
            or evidence.read(prefix + 'service/tokenize-preflight.json') != token_checks):
        raise ValueError('Original tokenizer preflight differs from the sealed input populations')
    phases = evidence.read(prefix + 'service/continuous.json')
    if phases['status'] != 'completed' or tuple(row['name'] for row in phases['phases']) != PHASES:
        raise ValueError('Missing original complete C8 phase clocks')
    records = []
    for name in evidence.seal:
        if name.startswith(prefix + 'service/service-') and name.endswith('.json'):
            row = evidence.read(name)
            if 'success' not in row or 'phase' not in row:
                raise ValueError('Malformed original request record: ' + name)
            if 'success' in row and 'phase' in row:
                first, last, usage, finish = audit.reconstruct(evidence.raw(name[:-5] + '.sse.jsonl.gz'))
                if (row['success'] is not True or row['http_status'] != 200
                        or (first, last, usage, finish) != (row['first_token']['monotonic'], row['last_token']['monotonic'], row['usage'], row['finish_reason'])
                        or finish != 'length' or usage['completion_tokens'] != 256
                        or usage['prompt_tokens'] != row['input_tokens']
                        or usage.get('prompt_tokens_details', {}).get('cached_tokens') not in (None, 0)):
                    raise ValueError('Original C8 request/SSE mismatch: ' + name)
                records.append(row)
    groups = aggregate(records, phases['phases'], inputs)
    admission = evidence.read(prefix + 'resource-admission.json')
    seal_sha = evidence.seal[prefix + 'checksums.json']
    hardware = recorded_hardware(admission, seal_sha)
    observations = dict(schema='vllm-hust-official-baseline-observations/v1', baseline_id='official-0.23.0',
        model='Qwen3.5-35B-A3B', precision='BF16', tensor_parallel_size=2,
        runtime=dict(kind='official-image', image_digest=lock['manifest_digest'], baseline_id=lock['id'],
                     versions=plan['runtime_versions'], requested_graph_mode='FULL',
                     effective_graph_mode='FULL_DECODE_ONLY' if b'setting cudagraph_mode=FULL_DECODE_ONLY' in evidence.raw(prefix + 'service.log') else None),
        hardware=hardware,
        profiles=[dict(id=name, label=('4K' if name.startswith('4096') else '16K') + ' / C8 路 continuous synthetic',
                       workload_kind='continuous-synthetic', raw_phase_id=name,
                       arm=dict(role='B0', source_seal_sha256=seal_sha, **groups[name])) for name in PHASES if name.endswith('-formal')],
        pairs=None, F=None, efficiency_percent=None, necessary_work_certified=False,
        source_execution_status=campaign['status'], original_campaign_seal_sha256=evidence.seal_sha256,
        independent_third_party=False)
    return observations, groups, support, dict(original_campaign_status=campaign['status'],
        original_stage_statuses=campaign['stages'], source_arm_seal_sha256=seal_sha,
        original_source_root=evidence.logical_root, shared_warmup_requests=len(warmup['requests']),
        original_version_all_member_bytes_verified=evidence.available == set(evidence.seal),
        original_version_member_count=len(evidence.seal), included_original_member_count=len(evidence.available),
        original_audit_sha256=digest(TOOLS / 'original_audit.py'),
        original_metric_consumer_sha256=next(pin for name,pin in bindings.items() if PurePosixPath(name).name=='report_execution.py'),
        original_model_weights_manifest_sha256=evidence.seal['model-weights.json'],
        rank_guard_coverage=[{key:row.get(key) for key in ('rank','torch_dynamo_guard','triton_guard','cann_guard')} for row in after],
        cann_compilation_observed=all(row.get('cann_guard') is True for row in after),
        all_tests_completed=False, F=None, efficiency_percent=None)


def aggregate(records, phases, inputs):
    if tuple(row['name'] for row in phases) != PHASES:
        raise ValueError('Missing or reordered original C8 phases')
    identifiers = [row['id'] for row in records]
    if len(set(identifiers)) != len(identifiers) or set(row['phase'] for row in records) != set(PHASES):
        raise ValueError('Duplicate requests or missing C8 populations')
    result = {}
    for index, phase in enumerate(phases):
        name = phase['name']; rows = [row for row in records if row['phase'] == name]
        start, end = phase['start']['monotonic'], phase['end']['monotonic']
        seconds = 60 if index % 2 == 0 else 300
        if (phase['seconds'] != seconds or phase['failed'] != 0 or phase['requests'] != len(rows)
                or not rows or not all(math.isfinite(x) for x in (start, end)) or end <= start
                or phase['submission_deadline_monotonic'] != start + seconds or end < start + seconds):
            raise ValueError('Original phase population/submission clock differs: ' + name)
        if index and start < phases[index-1]['end']['monotonic']:
            raise ValueError('Original C8 phase clocks overlap')
        sequences = {i: [] for i in range(8)}
        for row in rows:
            match = re.fullmatch('service-' + name + r'-lane([0-7])-([0-9]+)', row['id'])
            a, first, last, b = (row[key]['monotonic'] for key in ('start', 'first_token', 'last_token', 'end'))
            if (not match or row['input_tokens'] != inputs[name.split('-')[0]][int(match[1])]['input_tokens']
                    or not all(math.isfinite(x) for x in (a, first, last, b))
                    or not start <= a <= first <= last <= b <= end
                    or a >= phase['submission_deadline_monotonic']):
                raise ValueError('Request lies outside its original phase/inputs')
            sequences[int(match[1])].append(int(match[2]))
        if any(not values or sorted(values) != list(range(len(values))) for values in sequences.values()):
            raise ValueError('Original eight-lane closed-loop population differs')
        tokens = sum(row['usage']['completion_tokens'] for row in rows)
        result[name] = dict(requests=len(rows), failed=0, successful=len(rows), output_tokens=tokens,
            duration_seconds=end-start, output_tokens_per_second=tokens/(end-start),
            mean_ttft_ms=statistics.fmean((row['first_token']['monotonic']-row['start']['monotonic'])*1000 for row in rows),
            mean_tpot_ms=statistics.fmean((row['end']['monotonic']-row['first_token']['monotonic'])*1000/(row['usage']['completion_tokens']-1) for row in rows),
            original_submission_seconds=seconds, original_phase_start=phase['start'], original_phase_end=phase['end'])
    return result


def unpack(archive, destination, expected_version_seal=None):
    with tarfile.open(archive, 'r:gz') as stream:
        for row in stream:
            if not row.isfile():
                raise ValueError('Portable archive permits regular files only')
            target = member(destination, row.name)
            if target.exists():
                raise ValueError('Duplicate portable member')
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(row) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
    manifest = json.loads((destination / 'manifest.json').read_bytes())
    schema = manifest.get('schema')
    if schema != 'deeplinkengine-official-c8-portable/v2':
        raise ValueError('Unknown portable archive')
    if (not isinstance(expected_version_seal, str)
            or not re.fullmatch('[a-f0-9]{64}', expected_version_seal)
            or manifest.get('original_version_seal_sha256') != expected_version_seal):
        raise ValueError('External original version seal SHA256 is required and must match')
    actual = {str(path.relative_to(destination)).replace(os.sep, '/') for path in destination.rglob('*') if path.is_file()}
    if actual != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('Portable file set differs')
    for name, pin in manifest['files'].items():
        if digest(member(destination, name)) != pin:
            raise ValueError('Portable evidence changed: ' + name)
    support = {logical: member(destination, name) for logical, name in manifest['logical_support'].items()}
    evidence = Evidence(destination / 'evidence', support, manifest['original_source_root'], c8_subset=True)
    omitted = {name:pin for name,pin in evidence.seal.items() if name not in evidence.available}
    if (manifest.get('omitted_original_members') != omitted
            or expected_version_seal != evidence.seal_sha256
            or manifest.get('payload_scope') != 'complete-c8-and-parent-identity-only'):
        raise ValueError('Portable original custody scope differs')
    return evidence


def package(evidence, support, destination):
    with tempfile.TemporaryDirectory(prefix='official-c8-export-', dir=destination.parent) as temporary:
        stage = Path(temporary); files = {}; logical = {}
        included = c8_members(evidence.seal)
        for name in sorted(included) + ['checksums.json']:
            path = member(stage, 'evidence/' + name); path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(member(evidence.root, name), path)
            pin = digest(path)
            if pin != (evidence.seal_sha256 if name == 'checksums.json' else evidence.seal[name]):
                raise ValueError('Source changed while archiving')
            files['evidence/' + name] = pin
        for original, source in support.items():
            pin = digest(source); name = 'support/' + pin + '.source'; path = member(stage, name)
            path.parent.mkdir(exist_ok=True)
            if not path.exists():shutil.copyfile(source, path)
            files[name] = pin; logical[original] = name
        write(stage / 'manifest.json', dict(schema='deeplinkengine-official-c8-portable/v2',
            original_source_root=evidence.logical_root, files=files, logical_support=logical,
            original_version_seal_sha256=evidence.seal_sha256,
            payload_scope='complete-c8-and-parent-identity-only',
            omitted_original_members={name:pin for name,pin in evidence.seal.items() if name not in included},
            sdk_runtime_and_model_weights_included=False, supporting_source_files_included=True,
            model_requests_executed=0))
        with tarfile.open(destination, 'x:gz') as archive:
            for name in sorted(set(files) | {'manifest.json'}):archive.add(stage / name, arcname=name, recursive=False)


def run(configuration):
    config_path = Path(configuration).absolute(); config = json.loads(config_path.read_bytes())
    if isinstance(config, dict) and config.get('kind') == 'sealed-report-projection':
        from official_projection import run as project_report
        return project_report(config_path)
    if not isinstance(config, dict) or set(config) - {'source', 'out', 'source_sha256', 'original_version_seal_sha256'} or not all(isinstance(config.get(key), str) and config[key] for key in ('source', 'out')):
        raise ValueError('Report-only config requires source and a new out; archive input also requires source_sha256')
    source = Path(os.path.abspath(config_path.parent / config['source'])); out = Path(os.path.abspath(config_path.parent / config['out']))
    if source.is_symlink() or os.path.lexists(out):raise ValueError('Select immutable input and a new output; symlinks/overwrites are refused')
    if source.is_dir() and out.resolve().is_relative_to(source.resolve()):
        raise ValueError('Output must be outside the immutable original source directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='official-c8-import-', dir=out.parent) as temporary:
        if source.is_file():
            if not re.fullmatch('[a-f0-9]{64}', config.get('source_sha256', '')) or digest(source) != config['source_sha256']:
                raise ValueError('Portable archive SHA256 is required and must match')
            evidence = unpack(source, Path(temporary), config.get('original_version_seal_sha256'))
        else:
            if not re.fullmatch('[a-f0-9]{64}', config.get('source_sha256', '')) or digest(source/'checksums.json') != config['source_sha256']:
                raise ValueError('Original directory checksums.json SHA256 is required and must match')
            evidence = Evidence(source)
            if evidence.seal_sha256 != config['source_sha256']:
                raise ValueError('Original seal changed during admission')
            if config.get('original_version_seal_sha256', evidence.seal_sha256) != evidence.seal_sha256:
                raise ValueError('External original version seal differs')
        observations, groups, support, provenance = replay(evidence)
        out.mkdir(exist_ok=False)
        write(out / 'observations.json', observations)
        write(out / 'report.json', dict(schema='deeplinkengine-official-c8-replay-report/v1',
            status='derived-partial-report', phases=groups, provenance=provenance,
            source_execution_status=provenance['original_campaign_status'], model_requests_executed=0,
            original_results_modified=False, F=None, efficiency_percent=None))
        rows=''.join('<tr><td>'+html.escape(name)+'</td><td>'+str(row['requests'])+'</td><td>'+format(row['output_tokens_per_second'], '.6f')+'</td><td>'+str(row['failed'])+'</td></tr>' for name,row in groups.items())
        (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Official C8 replay</title><h1>Official 0.23 C8 路 CPU replay</h1><p>Original campaign: '+html.escape(provenance['original_campaign_status'])+'. Partial results; F and efficiency remain unqualified. No model requests.</p><table><tr><th>Original phase</th><th>Requests</th><th>Output token/s (including drain)</th><th>Failed</th></tr>'+rows+'</table><p><a href="report.json">Report</a> 路 <a href="observations.json">Observations</a></p>', encoding='utf-8')
        archive = out / 'portable-c8.tar.gz'; package(evidence, support, archive)
        write(out / 'derivation.json', dict(schema='deeplinkengine-official-report-only/v1', status='completed',
            scope='CPU replay and archive only, not completed full-suite execution', generated_at=time.time(),
            source_execution_status=provenance['original_campaign_status'], original_results_modified=False,
            model_requests_executed=0, portable_archive_sha256=digest(archive),
            input_sha256=config['source_sha256'], input_kind='portable-archive' if source.is_file() else 'original-version-directory',
            original_audit_sha256=provenance['original_audit_sha256'],
            derivation_sources={name:digest(source_path(ROOT, name)) for name in ('run_all.py','official_baselines.py','official_rebuild.py')},
            F=None, efficiency_percent=None))
        write(out / 'checksums.json', {path.name:digest(path) for path in out.iterdir() if path.is_file()})
    print('OFFICIAL_REPORT_ONLY_COMPLETED', str(out), 'model_requests=0', flush=True)
    return 0
