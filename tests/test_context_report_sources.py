"""Full-report consumers must follow sealed context arms outside the new run."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmark/one-click-kit/scripts')]
from benchmark import evaluation_client, full_report as report


def fixture(tmp_path):
    run = tmp_path / 'new-run'
    folder = run / 'context-tenant'
    arms = {'B0': tmp_path / 'original-context/B0', 'B1': folder / 'B1'}
    rows = []
    for role, arm in arms.items():
        arm.mkdir(parents=True)
        receipt = arm / 'receipt.json'
        receipt.write_text(json.dumps(dict(role=role, completed=True, quiescent=True)))
        (arm / 'source-admission.json').write_text(json.dumps(dict(release='full-answer-b1-20261119-r9' if role=='B1' else 'full-answer-b1-20261117-r8')))
        (arm / 'model-weights.json').write_text(json.dumps({'weight': dict(bytes=17, sha256='a'*64)}))
        seal = arm / 'checksums.json'
        seal.write_text(json.dumps({name:hashlib.sha256((arm/name).read_bytes()).hexdigest() for name in ['receipt.json','source-admission.json','model-weights.json']}))
        rows.append(dict(role=role, run=str(arm), reused=role == 'B0',
                         status='completed', returncode=0,
                         seal_sha256=hashlib.sha256(seal.read_bytes()).hexdigest()))
    state = dict(schema='vllm-hust-acceptance-supplement-execution/v1',
                 status='completed', stages=rows)
    stored = dict(schema='vllm-hust-paired-context-tenant-supplement/v1',
                  arms={r: str(p) for r, p in arms.items()},
                  arm_seals={x['role']: x['seal_sha256'] for x in rows},
                  execution_complete=True, comparison=[])
    (folder / 'execution.json').write_text(json.dumps(state))
    (folder / 'paired-summary.json').write_text(json.dumps(stored))
    return run, folder, arms, state, stored


def test_external_baseline_is_not_assumed_to_have_a_new_local_directory(tmp_path):
    run, folder, arms, state, stored = fixture(tmp_path)
    assert not (folder / 'B0').exists()
    assert report.context_sources(folder, stored) == arms
    assert not (folder / 'B0').exists()


@pytest.mark.parametrize('change', ['duplicate-role', 'nonzero-exit', 'incomplete',
                                   'stage-seal', 'pair-seal', 'pair-path', 'unreleased'])
def test_external_sources_remain_bound_to_both_execution_and_pair(tmp_path, change):
    run, folder, arms, state, stored = fixture(tmp_path)
    if change == 'duplicate-role':
        state['stages'][1]['role'] = 'B0'
    elif change == 'nonzero-exit':
        state['stages'][0]['returncode'] = 1
    elif change == 'incomplete':
        state['stages'][0]['status'] = 'running'
    elif change == 'stage-seal':
        state['stages'][0]['seal_sha256'] = '0'*64
    elif change == 'pair-seal':
        stored['arm_seals']['B0'] = '0'*64
    elif change == 'pair-path':
        stored['arms']['B0'] = '/different/source'
    else:
        (arms['B0'] / 'receipt.json').write_text(json.dumps(dict(role='B0', completed=True, quiescent=False)))
    (folder / 'execution.json').write_text(json.dumps(state))
    with pytest.raises(ValueError):
        report.context_sources(folder, stored)


def test_fresh_arm_cannot_be_relabelled_as_external_evidence(tmp_path):
    run, folder, arms, state, stored = fixture(tmp_path)
    state['stages'][0]['reused'] = False
    (folder / 'execution.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='outside its owned output'):
        report.context_sources(folder, stored)


def test_changed_original_seal_is_rejected(tmp_path):
    run, folder, arms, state, stored = fixture(tmp_path)
    seal = arms['B0'] / 'checksums.json'
    seal.write_text(seal.read_text() + ' ')
    with pytest.raises(ValueError, match='seal differs'):
        report.context_sources(folder, stored)


def collect_fixture(tmp_path, changed=None, actual_cost=None):
    run, folder, arms, state, stored = fixture(tmp_path)
    cost_path = None
    if actual_cost:
        default = run / 'actual-cost.json'
        default.write_text('{}')
        cost_path = tmp_path / 'override.json' if actual_cost == 'override' else None
        if cost_path:
            cost_path.write_text('{}')
    fixed = run / 'fixed-five'; fixed.mkdir()
    for name in ['campaign-checksums.json', 'frozen-execution-plan.json']:
        (fixed / name).write_text('{}')
    kit = tmp_path / 'kit'; (kit / 'mods').mkdir(parents=True)
    release = 'full-answer-b1-20261119-r9'
    (kit / 'mods/active-stack-r9.json').write_text(json.dumps(dict(release=release, enabled_mods=[])))
    mmlu=run/'mmlu-B1';mmlu.mkdir()
    (mmlu/'model-weights.json').write_bytes((arms['B1']/'model-weights.json').read_bytes())
    if changed:
        name='source-admission.json' if changed=='release' else 'model-weights.json'
        (arms['B1']/name).write_text(json.dumps(dict(release='full-answer-b1-20261117-r8')) if changed=='release' else json.dumps({'weight':dict(bytes=17,sha256='b'*64)}))
        seal=arms['B1']/'checksums.json';record=json.loads(seal.read_text());record[name]=hashlib.sha256((arms['B1']/name).read_bytes()).hexdigest();seal.write_text(json.dumps(record))
        digest=hashlib.sha256(seal.read_bytes()).hexdigest();state['stages'][1]['seal_sha256']=digest;stored['arm_seals']['B1']=digest
        (folder/'execution.json').write_text(json.dumps(state));(folder/'paired-summary.json').write_text(json.dumps(stored))
    observed = dict(plan=dict(datasets=list(report.DATASETS), engine_release=release),
                    receipt=dict(selected_execution_complete=True),current_mmlu_pair=dict(arms=dict(B1=dict(run=str(mmlu)))))
    audits = []

    def paired(actual):
        assert actual == arms
        return stored

    def audit(path):
        audits.append(path)
        return dict(clean=True, errors=[])

    # This is a path/consumer contract fixture, not a measured performance run.
    # Real sealed-context replay is exercised separately on the actual originals.
    with patch('engine.runtime_storage.restore_missing'), patch.object(report, 'kit_path', return_value=kit), \
            patch('third_party_report.inspect', return_value=observed), \
            patch('acceptance_suite.paired', side_effect=paired), \
            patch.object(report, 'audit_context', side_effect=audit), \
            patch.object(report, 'targets', return_value={}) as targets, patch.object(report, 'workload_pairs', return_value={}), \
            patch.object(report, 'workload_sources', return_value={}), patch.object(report, 'apply_v2_context_tails'):
        result = report.collect(run, cost_path)
        assert targets.call_args.args[-1] == cost_path
    return result, audits, arms, folder, stored


def test_collect_replays_audits_and_hashes_the_original_external_arm(tmp_path):
    result, audits, arms, folder, stored=collect_fixture(tmp_path)
    assert audits == [arms['B0'], arms['B1']]
    assert result['context_arm_sources']['B0'] == str(arms['B0'])
    assert str(arms['B0'] / 'checksums.json') in result['evidence_sha256']
    assert 'context-tenant/B1/checksums.json' in result['evidence_sha256']
    assert not (folder / 'B0').exists()
    assert (folder / 'paired-summary.json').read_text() == json.dumps(stored)


@pytest.mark.parametrize('changed',['release','weights'])
def test_other_candidate_or_model_cannot_supply_context_results(tmp_path,changed):
    with pytest.raises(ValueError,match='differ[s]? from the fixed-five candidate'):
        collect_fixture(tmp_path,changed)


@pytest.mark.parametrize('mode', ['unselected', 'override'])
def test_sealed_report_only_selects_an_explicit_fee_file(tmp_path, mode):
    collect_fixture(tmp_path, actual_cost=mode)


def test_dataset_source_is_taken_from_the_audited_cohort(tmp_path):
    base = tmp_path / 'acceptance-inputs'
    base.mkdir()
    def case(task_id, tokens):
        return dict(task_id=task_id, input_tokens=tokens, source_index=7, source_sha256='a'*64,
                    source_body_sha256='b'*64, token_ids_sha256='c'*64, request={'secret': 'unpublished body'})
    inputs = {'4096': [case('terminal:dna-insert', 7366)], '16384': [case('swe:repo-issue', 19500)]}
    path = base / 'acceptance-inputs.json'
    path.write_bytes(json.dumps(inputs).encode())
    contract = dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_protocol_sha256='d'*64,
                    output_tokens=256,
                    cohorts=[dict(min_tokens=4096, max_tokens_exclusive=8192),
                             dict(min_tokens=16384, max_tokens_exclusive=24576)],
                    phases=[dict(id='multi-tenant', cohort='4096', tenants=4, concurrency=24,
                                 warmup_seconds=60, formal_seconds=300)])
    sources = report.workload_sources(dict(plan={}), dict(contract=contract), tmp_path)
    phases = sources['context']['phases']
    assert phases['multi-tenant-formal']['cohort'] == '4096'
    cohorts = sources['context']['cohorts']
    assert cohorts['4096']['datasets'] == ['Terminal-Bench 2.1']
    assert cohorts['16384']['datasets'] == ['SWE-bench Pro']
    assert cohorts['4096']['cases'][0]['task_id'] == 'terminal:dna-insert'
    assert 'unpublished body' not in json.dumps(sources)
    table = report.workload_source_table(sources)
    assert '四租户 C24</th><td>Terminal-Bench 2.1' in table
    assert 'terminal:dna-insert' in table
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='audited input digest'):
        report.workload_sources(dict(plan={}), dict(contract=contract), tmp_path)


def test_formal_context_rank_replays_sealed_whole_and_window_clocks(tmp_path):
    folder = tmp_path / 'service/acceptance'
    folder.mkdir(parents=True)
    requests = []
    for i, (start, ttft) in enumerate(((1, 1), (2, 2), (3, 3), (4, 10), (61, 5))):
        name = f'request-{i}.json'
        first = start + ttft
        (folder / name).write_text(json.dumps(dict(
            success=True, start=dict(monotonic=start), first_token=dict(monotonic=first),
            end=dict(monotonic=first + 2), usage=dict(completion_tokens=3), tenant='tenant-0')))
        requests.append(name)
    (folder / 'execution.json').write_text(json.dumps(dict(phases=[dict(
        name='32768-formal', start=dict(monotonic=0), submission_deadline_monotonic=120, request_files=requests)])))
    seal = {path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.iterdir()}
    (tmp_path / 'checksums.json').write_text(json.dumps(seal))
    def metrics():
        return dict(ttft_ms=dict(p95=1, p99=2), tpot_ms=dict(p95=3, p99=4))
    arm = dict(metrics=metrics(), per_tenant={'tenant-0': metrics()},
               windows=[dict(metrics=metrics()), dict(metrics=metrics())])
    context = dict(comparison=[{'phase': '32768-formal', 'B1': arm}],
                   arm_seals={'B1': hashlib.sha256((tmp_path / 'checksums.json').read_bytes()).hexdigest()})
    report.apply_v2_context_tails(context, {'B1': tmp_path})
    assert arm['metrics']['ttft_ms'] == {'p95': 10000, 'p99': 10000}
    assert arm['per_tenant']['tenant-0']['tpot_ms']['p99'] == 1000
    assert arm['windows'][0]['metrics']['ttft_ms']['p99'] == 10000
    assert arm['windows'][1]['metrics']['ttft_ms']['p99'] == 5000
    (folder / requests[0]).write_text('{}')
    with pytest.raises(ValueError, match='missing from the audited seal'):
        report.apply_v2_context_tails(context, {'B1': tmp_path})


def test_plot_only_failure_admits_sealed_context_but_other_failures_do_not(tmp_path):
    _, folder, arms, state, stored = fixture(tmp_path)
    state.update(status='failed', failure='An alternate renderer exited while formatting the graph')
    (folder/'execution.json').write_text(json.dumps(state))
    assert report.context_sources(folder, stored) == arms
    state['stages'][1].update(status='failed', returncode=1)
    (folder/'execution.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='before the plotting boundary'):
        report.context_sources(folder, stored)


def test_mod_catalog_is_selected_by_recorded_release_not_filename(tmp_path):
    file = tmp_path / 'active-stack-other-profile-name.json'
    release = 'full-answer-stage-renamed-profile'
    stack = dict(release=release, enabled_mods=[])
    file.write_text(json.dumps(stack))
    assert report.measured_stack(tmp_path, release) == (file.read_bytes(), stack)
    (tmp_path / 'active-stack-duplicate.json').write_bytes(file.read_bytes())
    with pytest.raises(ValueError, match='exactly one MOD catalog'):
        report.measured_stack(tmp_path, release)


def test_report_rebuilds_catalog_for_a_sealed_r10_client(tmp_path, monkeypatch):
    release = evaluation_client.R10_3072_REFERENCE_RELEASE
    measurement = dict(scope='Qwen3.5 BF16 TP2',
                       installed_files_canonical_sha256=evaluation_client.R10_3072_RUNTIME_SHA256)
    overlay = json.dumps(dict(release=release, measurement=measurement,
                              base_manifest_sha256=evaluation_client.R10_REFERENCE_OVERLAY_SHA256)).encode()
    microflow = b'reviewed fixture'
    monkeypatch.setattr(evaluation_client, 'R10_3072_OVERLAY_SHA256', evaluation_client.sha(overlay))
    monkeypatch.setattr(evaluation_client, 'R10_3072_MICROFLOW_SHA256', evaluation_client.sha(microflow))
    mods = ['microflow', 'intern-s2']
    parent = dict(schema='active-stack-test', release=evaluation_client.R10_REFERENCE_RELEASE, excluded=[],
                  enabled_mods=[dict(id=name, manifest=f'mods/{name}.json') for name in mods])
    files = {evaluation_client.R10_3072_OVERLAY: overlay,
             'overlays/b1-r10-3072/' + evaluation_client.R10_3072_MICROFLOW: microflow,
             'mods/active-stack-r10-stable.json': json.dumps(parent).encode(),
             **{f'mods/{name}.json': b'{}' for name in mods}}
    for name, raw in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    sealed = evaluation_client.inventory(tmp_path)
    manifest = json.dumps(dict(release=release, files=sealed)).encode()
    (tmp_path / 'CLIENT-MANIFEST.json').write_bytes(manifest)
    frozen = dict(release=release, version='2026.11.24.10', included_mods=mods,
                  runtime_geometry=dict(mixed_capture_tokens=3072),
                  measured_runtime=measurement, publication_scope=measurement['scope'])
    selection = dict(release=release, frozen=frozen, client=dict(
        path=str(tmp_path), release=release, manifest_sha256=evaluation_client.sha(manifest)))

    expected = dict(files)
    evaluation_client.r10_3072_report_catalog(expected, frozen)
    raw, stack = report.measured_stack(tmp_path / 'mods', release, selection=selection)
    assert raw == expected[evaluation_client.R10_3072_STACK]
    assert [mod['id'] for mod in stack['enabled_mods']] == ['microflow']
    assert evaluation_client.inventory(tmp_path) == sealed
    assert (tmp_path / 'CLIENT-MANIFEST.json').read_bytes() == manifest

    (tmp_path / evaluation_client.R10_3072_OVERLAY).write_bytes(b'changed')
    with pytest.raises(ValueError, match='evaluation client source inventory changed'):
        report.measured_stack(tmp_path / 'mods', release, selection=selection)
