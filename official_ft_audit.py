"""CPU qualification of each official native repetition and its own F/T.

The untimed companion proves producer/source applicability. Its token counts
and time never substitute for native geometry or any of the three clocks.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
KIT = ROOT / 'benchmark/one-click-kit'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Evidence:
    def __init__(self, root, expected_seal=None):
        self.root = Path(root).resolve()
        self.seal_path = self.root / 'checksums.json'
        if expected_seal is not None and sha(self.seal_path) != expected_seal:
            raise ValueError('Externally fixed original seal differs')
        self.seal = json.loads(self.seal_path.read_bytes())
        self.selected = {}

    def bytes(self, name):
        path = self.root / name
        if (path.is_symlink() or not path.resolve().is_relative_to(self.root)
                or self.seal.get(name) != sha(path)):
            raise ValueError('Unsealed or changed arithmetic evidence: ' + name)
        self.selected[name] = self.seal[name]
        return path.read_bytes()

    def read(self, name):
        return json.loads(self.bytes(name))

    def lines(self, name):
        return [json.loads(row) for row in self.bytes(name).splitlines() if row]


def runtime(value, version):
    expected = dict(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1,
        prefill_context_parallel_size=1, decode_context_parallel_size=1, enable_expert_parallel=False,
        async_scheduling=True, maximum_sequences=24, maximum_batched_tokens=4096,
        maximum_model_length=65536, kv_cache_memory_bytes=20 * 1024**3, prefix_caching=False,
        speculative_method='mtp', speculative_tokens=2, parallel_drafting=False,
        speculative_tokens_per_batch_size=None, dynamic_spec_active=False,
        tensor_values_observed=False, per_step_observer_installed=False)
    for key, target in expected.items():
        if type(value.get(key)) is not type(target) or value[key] != target:
            raise ValueError('Unsupported resolved official arithmetic setting: ' + key)
    # 0.23 source advances positions unconditionally. 0.30 introduces the
    # explicit constant-position branch, which must be observed as disabled.
    constant = value.get('constant_draft_positions')
    if constant is not None and constant is not False or version == '0.30.0' and constant is not False:
        raise ValueError('Unmapped constant-position MTP source branch')
    return value


def source_domain(sdk, native_plan, companion_plan, native_inventories, companion_inventories, observed_sources, domain):
    sdk = Path(sdk).resolve()
    root = sdk / 'rootfs'
    if sha(ROOT / 'engine/official-ft-source-domains.json') != SOURCE_DOMAINS_SHA256:
        raise ValueError('Reviewed official arithmetic source domains changed')
    for relative, pin in domain['source_sha256'].items():
        if sha(root / relative) != pin:
            raise ValueError('Official arithmetic implementation differs: ' + relative)
    old_root = PurePosixPath(native_plan['sdk'])
    companion_prefix = PurePosixPath(companion_plan['sdk'])
    loaded = {}
    for inventories, prefix in ((native_inventories, old_root),
                                (companion_inventories, companion_prefix)):
        for inventory in inventories:
            for role in ('target', 'draft'):
                for filename, pin in inventory[role]['source_sha256'].items():
                    try:
                        relative = str(PurePosixPath(filename).relative_to(prefix))
                    except ValueError as error:
                        raise ValueError('Loaded official source escaped its SDK') from error
                    if sha(sdk / relative) != pin:
                        raise ValueError('Loaded official implementation changed: ' + relative)
                    if relative.startswith('rootfs/vllm-workspace/') and domain['source_sha256'].get(relative[7:]) != pin:
                        raise ValueError('Unreviewed loaded official implementation: ' + relative)
                    loaded[relative] = pin
    for filename, pin in observed_sources.items():
        relative = str(PurePosixPath(filename).relative_to(companion_prefix))
        if not relative.startswith('rootfs/') or domain['source_sha256'].get(relative[7:]) != pin:
            raise ValueError('Unreviewed actual scheduler/proposer implementation: ' + relative)
    return dict(source_family=domain['id'], source_sha256=domain['source_sha256'],
                loaded_source_sha256=loaded, upstream_model_sources_modified=False)


def native_implementation(native, plan, workers, core, version, domain, sdk):
    for name in ('native_timing_session.py', 'native_public_client.py', 'native_output_geometry.py',
                 'reference_capture_profile.py', 'baseline_native_entry.py'):
        pins = [pin for filename, pin in plan['files'].items()
                if PurePosixPath(filename).parts[-2:] == ('scripts', name)]
        if pins != [sha(KIT / 'scripts' / name)]:
            raise ValueError('Native clock/public-input source differs: ' + name)
    inventory_name = 'third-party-test-kit/proc_capture/m01_model_inventory.py'
    inventory_pins = [pin for filename, pin in plan['files'].items() if filename.endswith('/' + inventory_name)]
    if inventory_pins != [sha(KIT / inventory_name)]:
        raise ValueError('Actual native loaded-inventory collector source differs')
    if len(workers) != 2 or {w['rank'] for w in workers} != {0, 1}:
        raise ValueError('Both original native worker observations required')
    for worker in workers:
        binding = worker['loaded_inventory']
        if (binding['sha256'] != native.seal[f"rank-{worker['rank']}-loaded-inventory.json"]
                or worker['native_per_step_observer'] is not False
                or worker['extra_host_observer'] is not False or worker['host_prefill_accounting'] is not False):
            raise ValueError('Native loaded-inventory/clean observer binding differs')
        resolved = worker.get('resolved_configuration')
        if resolved is not None:
            runtime(resolved, version)
            if resolved['num_blocks'] != core['num_blocks']:
                raise ValueError('Native and companion fixed KV pool geometry differ')
            if resolved['gdn_backend']['selected'] not in domain['supported_prefill_families']:
                raise ValueError('Actual native GDN backend requires its arithmetic adapter')
            if resolved['gdn_backend']['selected'] == 'fla-npu':
                from official_ft_fla import validate
                validate(resolved['gdn_backend']['operator_identity'], sdk)
        elif version == '0.23.0':
            # Original v5 predates the explicit resolved-config record. Its
            # fixed source family has unconditional sequential MTP2, and the
            # source-qualified runtime log independently records resolution.
            log = native.bytes('native.log').decode('utf-8', errors='replace')
            capacities = set(re.findall(r'GPU KV cache size: ([0-9,]+) tokens', log))
            expected = int(core['capacity_at_max_length'] * core['max_model_len'])
            if (capacities != {f'{expected:,}'} or 'Asynchronous scheduling is enabled.' not in log
                    or 'Asynchronous scheduling is disabled.' in log):
                raise ValueError('Original v5 resolved async/capacity does not bind to the same source configuration')
        else:
            raise ValueError('Actual resolved native arithmetic configuration is missing')
        for method in worker['methods'].values():
            source = PurePosixPath(method['source'])
            relative = str(source.relative_to(PurePosixPath(plan['sdk'])))
            if not relative.startswith('rootfs/') or domain['source_sha256'].get(relative[7:]) != method['sha256']:
                raise ValueError('Native actually loaded runner source is unreviewed')


def companion(source, text, version):
    from official_ft_inventory import validate
    from official_ft_inputs import resolve_records
    from audit_public_work_correspondence import audit
    from native_public_client import validate_frames
    from m01_observer_worker import key
    receipt = source.read('receipt.json')
    child = source.read('client/companion-receipt.json')
    if (receipt['completed'] is not True or receipt['quiescent'] is not True
            or receipt['purpose'] != 'arithmetic-companion' or child['completed'] is not True
            or child['timing_eligible'] is not False or child['native_counts_or_times_replaced'] is not False
            or child['observers_detached'] is not True):
        raise ValueError('A closed independently observed official companion is required')
    window = child['window']
    core = source.read('client/raw-work/core-end.json')
    workers = source.read('client/raw-work/workers-end.json')
    if (core['completed'] is not True or core['window'] != window
            or len(workers) != 2 or {w['rank'] for w in workers} != {0, 1}
            or any(w['completed'] is not True or w['window'] != window for w in workers)):
        raise ValueError('Companion scheduler/TP observers did not close')
    inventories = [source.read(f'rank-{rank}-loaded-inventory.json') for rank in (0, 1)]
    audits = [validate(value, text) for value in inventories]
    identities = source.read('client/loaded-inventory-bindings-after.json')
    if (len(identities) != 2 or {r['rank'] for r in identities} != {0, 1}):
        raise ValueError('Actual loaded companion identities are incomplete')
    for value in identities:
        rank = value['rank']
        binding = value['loaded_inventory']
        if binding['sha256'] != source.seal[f'rank-{rank}-loaded-inventory.json']:
            raise ValueError('Companion inventory differs from its actual worker binding')
        runtime(value['resolved_configuration'], version)
    raw = source.read('client/public/raw-public-output-frames.json')
    prompts = source.read('token-workload.json')['request_token_ids']
    responses = validate_frames(raw, list(map(len, prompts)), 256)
    requests = [dict(request=key(identity), input_tokens=len(tokens), output_tokens=len(row['output_ids']))
        for identity, tokens, row in zip(raw['request_ids'], prompts, responses, strict=True)]
    names = ['scheduler-waves.jsonl', 'rank-0-waves.jsonl', 'rank-1-waves.jsonl',
             'rank-0-drafts.jsonl', 'rank-1-drafts.jsonl']
    streams = [source.lines('client/raw-work/' + name) for name in names]
    if (len(streams[0]) != core['waves'] or any(r['window'] != window for s in streams for r in s)
            or any(len(streams[rank + 1]) != workers[rank]['waves'] for rank in (0, 1))):
        raise ValueError('Companion stream closure differs from acknowledged counts')
    for records in streams[3:]:
        for row in records:
            strategy = row['semantic_strategy']
            if (strategy['actual_draft_width'] != 2 or strategy['method'] != 'mtp'
                    or strategy['parallel_drafting'] is not False or strategy['dynamic_spec_active'] is not False
                    or strategy['sampling_all_greedy'] is not True
                    or strategy['constant_draft_positions'] not in (None, False)):
                raise ValueError('Unmapped actual companion MTP strategy')
    resolved, changes = resolve_records(*streams)
    proof = audit(requests, resolved[0], resolved[3], resolved[4])
    proof.update(loaded_inventory_audits=audits, async_placeholder_changes_count=len(changes),
        async_placeholder_changes_sha256=hashlib.sha256(
            (json.dumps(changes, separators=(',', ':')) + '\n').encode()).hexdigest(),
        companion_source_seal_sha256=sha(source.seal_path),
        native_counts_or_times_replaced=False,
        application='Source/producer cross-check only. Each native repetition independently reconstructs its own geometry and necessary work.')
    return proof, inventories, source.read('client/resolved-core.json')


def audit(native_root, companion_root, sdk, model_config, out, *, version, source_seal=None):
    sys.path[:0] = [str(KIT / 'scripts'), str(KIT / 'third-party-test-kit'),
                    str(KIT / 'third-party-test-kit/proc_capture')]
    from official_baselines import save
    from official_ft_inventory import validate
    from official_ft_capacity import envelope
    from native_output_geometry import geometry
    from public_work_ledger import ledger
    from reference_final_stages import validate_native_timing
    from full_reference_reuse import domain as reference_domain
    native, observed = Evidence(native_root, source_seal), Evidence(companion_root)
    out = Path(out)
    out.mkdir(exist_ok=False)
    domains = json.loads((ROOT / 'engine/official-ft-source-domains.json').read_bytes())
    if version not in domains['domains']:
        raise ValueError('A registered actual-source arithmetic domain is required for version ' + version)
    domain = domains['domains'][version]
    if domain['reviewed'] is not True:
        raise ValueError('Official arithmetic source family is not yet reviewed')
    for name, pin in domains['formula_sha256'].items():
        if sha(KIT / 'third-party-test-kit' / name) != pin:
            raise ValueError('Necessary-work algorithm differs: ' + name)
    plan = native.read('native-plan.json')
    if (plan['role'] != 'B0' or plan['baseline_id'] != 'official-vllm-' + version
            or plan['formal_repetitions'] != 3):
        raise ValueError('Official native role/version/profile differs')
    from reference_capture_profile import validate_native_plan
    validate_native_plan(plan)
    actual = native.read('client/actual-identity.json')
    if (actual['versions'] != plan['expected_versions'] or actual['baseline_id'] != plan['baseline_id']
            or actual['official_sources_patched'] is not False or actual['no_optimized_mods'] is not True
            or actual['native_per_step_observer'] is not False):
        raise ValueError('Native actual source identity differs from its admitted plan')
    receipt = native.read('receipt.json')
    if receipt['completed'] is not True or receipt['quiescent'] is not True:
        raise ValueError('Native source is not closed')
    full = reference_domain()
    if sha(model_config) != full['model_config_sha256']:
        raise ValueError('Original complete-reference model differs')
    text = json.loads(Path(model_config).read_bytes())['text_config']
    inventories = [native.read(f'rank-{rank}-loaded-inventory.json') for rank in (0, 1)]
    for inventory in inventories:
        validate(inventory, text)
    proof, companion_inventories, core = companion(observed, text, version)
    for value in observed.read('client/loaded-inventory-bindings-after.json'):
        if value['resolved_configuration']['gdn_backend']['selected'] not in domain['supported_prefill_families']:
            raise ValueError('Actual companion GDN backend requires its arithmetic adapter')
        if value['resolved_configuration']['gdn_backend']['selected'] == 'fla-npu':
            from official_ft_fla import validate as validate_fla
            validate_fla(value['resolved_configuration']['gdn_backend']['operator_identity'], sdk)
    sources = dict(core['source_sha256'])
    for identity in observed.read('client/resolved-worker-identities.json'):
        sources.update(identity['model_sources'])
    companion_plan = observed.read('native-plan.json')
    if companion_plan['baseline_id'] != plan['baseline_id'] or companion_plan['expected_versions'] != plan['expected_versions']:
        raise ValueError('Companion and native runtime identities differ')
    admitted = source_domain(sdk, plan, companion_plan, inventories, companion_inventories, sources, domain)
    workload = native.read('token-workload.json')
    tokens = workload['request_token_ids']
    token_hash = hashlib.sha256((json.dumps(tokens, separators=(',', ':')) + '\n').encode()).hexdigest()
    if token_hash != full['prompt_tokens_sha256'] or len(tokens) != 200 or sum(map(len, tokens)) != 282265:
        raise ValueError('Native prompt population differs from the original reference')
    capacity = envelope(core, list(map(len, tokens)))
    if not capacity['sufficient_page_capacity']:
        raise ValueError('Source-qualified official shared-pool admission is insufficient')
    save(out / 'companion-correspondence.json', proof)
    companion_binding = dict(companion_source_seal_sha256=proof['companion_source_seal_sha256'],
        correspondence_sha256=sha(out / 'companion-correspondence.json'),
        requests=len(proof['requests']), all_request_closures_contained=proof['all_request_closures_contained'],
        native_counts_or_times_replaced=False, application=proof['application'])
    args = native.read('client/effective-engine-args.json')
    if (args['tensor_parallel_size'] != 2 or args['max_num_seqs'] != 24
            or args['max_num_batched_tokens'] != 4096 or args['max_model_len'] != 65536
            or args['kv_cache_memory_bytes'] != 20 * 1024**3 or args['enable_prefix_caching'] is not False
            or args['speculative_config'] != dict(method='mtp', num_speculative_tokens=2)):
        raise ValueError('Native source arithmetic arguments differ')
    all_times = native.read('client/timings.json')['repetitions']
    if len(all_times) != 3:
        raise ValueError('All three original native repetitions are required')
    previous_end, identifiers, rows = None, set(), []
    for index in range(1, 4):
        reset = native.read(f'client/formal-{index}-reset.json')
        if reset != dict(schema='baseline-native-prefix-reset/v1', reset=True,
                         b1_state_cache_present=False, shared_request_cache_salts_unique=True):
            raise ValueError('Native fresh-prefix reset is incomplete')
        workers = native.read(f'client/formal-{index}-pre-audit.json')
        native_implementation(native, plan, workers, core, version, domain, sdk)
        raw = native.read(f'client/formal-{index}/raw-public-output-frames.json')
        original = native.read(f'client/formal-{index}/public-output-geometry.json')
        rebuilt = geometry(raw, tokens, maximum=256)
        if rebuilt != original or any(r['output_tokens'] != 256 or r['finish_reason'] != 'length' for r in rebuilt['requests']):
            raise ValueError('Native own public geometry does not independently reproduce')
        work = ledger(SimpleNamespace(**text), rebuilt, allow_recurrent_prompt=True)
        work.update(necessary_work_certified=True, source_runtime_applicability_certified=True,
            native_checksums_sha256=sha(native.seal_path), source_domain=admitted,
            capacity=capacity, companion=companion_binding, qualification_source='official-versioned-source-adapter/v1')
        timing = native.read(f'client/formal-{index}-timing.json')
        elapsed = validate_native_timing(raw, timing, workers, work, repetition=index)
        if (timing != all_times[index - 1] or previous_end is not None and raw['start_ns'] < previous_end
                or len(set(raw['request_ids'])) != 200 or identifiers.intersection(raw['request_ids'])):
            raise ValueError('Native repetitions overlap or reuse request identities')
        previous_end = raw['end_ns']
        identifiers.update(raw['request_ids'])
        save(out / f'formal-{index}-necessary-work.json', work)
        rows.append(dict(repetition=index, F_interval=work['F_interval'], T_seconds=str(elapsed),
                         geometry_sha256=rebuilt['geometry_sha256'],
                         work_sha256=sha(out / f'formal-{index}-necessary-work.json'),
                         timing_sha256=native.selected[f'client/formal-{index}-timing.json']))
    save(out / 'selected-source-evidence.json', dict(native=native.selected, companion=observed.selected,
         native_source_seal_sha256=sha(native.seal_path), companion_source_seal_sha256=sha(observed.seal_path),
         full_payload_verified=False, original_source_modified=False))
    return rows


# Updated only with the independently reviewed versioned domain file.
SOURCE_DOMAINS_SHA256 = '897b013b9e449910353551f3a041bfc3dedd2a29e2df72611f8a11f0325647ba'
