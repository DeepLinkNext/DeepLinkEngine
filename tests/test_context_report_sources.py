"""Full-report consumers must follow sealed context arms outside the new run."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmark/one-click-kit/scripts')]
import full_report as report


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
    with patch('runtime_storage.restore_missing'), patch.object(report, 'kit_path', return_value=kit), \
            patch('third_party_report.inspect', return_value=observed), \
            patch('acceptance_suite.paired', side_effect=paired), \
            patch.object(report, 'audit_context', side_effect=audit), \
            patch.object(report, 'targets', return_value={}) as targets, patch.object(report, 'workload_pairs', return_value={}), \
            patch.object(report, 'workload_sources', return_value={}):
        result = report.collect(run, cost_path)
        expected = (cost_path or default) if actual_cost else None
        assert targets.call_args.args[-1] == expected
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


@pytest.mark.parametrize('mode', ['default', 'override'])
def test_sealed_report_picks_up_fee_file_and_respects_an_explicit_override(tmp_path, mode):
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
