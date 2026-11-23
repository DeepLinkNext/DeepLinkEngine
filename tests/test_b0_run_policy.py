"""Public entry measures only B0 stages that lack compatible sealed evidence."""
import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmark/one-click-kit/scripts')]
from benchmark import b0_run_policy as policy
from benchmark import one_click
import run_all


@pytest.fixture(autouse=True)
def default_policy():
    with patch.dict(os.environ):
        os.environ.pop(policy.ENVIRONMENT_KEY, None)
        yield


def test_default_without_a_source_allows_measurement_plan_before_launch(tmp_path):
    with patch('subprocess.run') as run:
        assert policy.public_arguments(['--model', '/model']) == ['--model', '/model']
        plan = {}
        policy.require_campaign_reuse([dict(name='continuous-B0', args=[])], {}, tmp_path, plan)
    run.assert_not_called()
    assert plan['baseline_execution_policy']['measure'] == ['continuous-B0']
    assert policy.mode() == 'auto'


def test_explicit_yes_is_required_and_false_overrides_inherited_environment(monkeypatch):
    assert policy.public_arguments(['--rerun-b0', '--model', '/model']) == ['--model', '/model']
    assert policy.enabled() is True
    policy.public_arguments(['--no-rerun-b0', '--reuse-completed-campaign', '/old'])
    assert policy.enabled() is False


@pytest.mark.parametrize('args', [
    ['--rerun-b0', '--no-rerun-b0'],
    ['--rerun-b0', '--reuse-completed-campaign', '/old'],
    ['--rerun-b0', '--reference-baseline-source=/old'],
])
def test_conflicting_choices_do_not_silently_override_old_measurements(args):
    with pytest.raises(ValueError):
        policy.public_arguments(args)


@pytest.mark.parametrize('value', ['false', 0, None, {}, []])
def test_config_requires_a_real_boolean(value):
    with pytest.raises(ValueError, match='true 或 false'):
        policy.boolean(value)


def test_prepare_never_requires_measurement_sources():
    policy.public_arguments(['--model', '/model'], prepare_only=True)
    assert policy.mode() == 'auto'


def test_incomplete_admitted_campaign_cannot_fall_back_to_b0(tmp_path):
    stages = [dict(name=name, args=[]) for name in
              ['continuous-B0', 'continuous-B1', 'mmlu-pair', 'hle-solver-B0', 'hle-judge-B0']]
    with pytest.raises(ValueError, match='continuous-B0.*mmlu-pair/B0.*hle-solver-B0'):
        policy.require_campaign_reuse(stages, {}, tmp_path, {})
    reused = {name: {} for name in ['continuous-B0', 'mmlu-pair', 'hle-solver-B0']}
    policy.require_campaign_reuse(stages, reused, tmp_path, {})


def test_mmlu_can_run_new_b1_with_a_bound_old_baseline(tmp_path):
    stages = [dict(name='mmlu-pair', args=['--reuse-baseline-arm', '/old'])]
    policy.require_campaign_reuse(stages, {}, tmp_path, {})


def test_native_and_context_missing_sources_fail_before_model_start():
    with pytest.raises(ValueError, match='原生计时'):
        policy.require_reference_reuse({})
    with pytest.raises(ValueError, match='上下文'):
        policy.require_context_reuse(SimpleNamespace(reuse_b0=None))
    policy.require_reference_reuse({'baseline_reuse': {'source': '/old'}})
    policy.require_context_reuse(SimpleNamespace(reuse_b0='/old'))


def test_explicit_rerun_allows_all_three_measurement_categories(tmp_path, monkeypatch):
    monkeypatch.setenv(policy.ENVIRONMENT_KEY, '1')
    policy.require_campaign_reuse([dict(name='continuous-B0', args=[])], {}, tmp_path, {})
    policy.require_reference_reuse({})
    policy.require_context_reuse(SimpleNamespace(reuse_b0=None))


def test_full_entry_requires_all_baseline_sources_before_environment_changes(tmp_path):
    assert policy.full_entry_policy({'out': str(tmp_path)}) == (None, None)
    with pytest.raises(ValueError, match='仅复用'):
        policy.full_entry_policy({'out': str(tmp_path), 'rerun_b0': False})
    old = tmp_path / 'old'
    (old / 'context-tenant/B0').mkdir(parents=True)
    (old / 'fixed-five/compute-reference/native-baseline-timing').mkdir(parents=True)
    assert policy.full_entry_policy({'out': '/new'}, prior=old) == (None, None)
    context, compute = policy.full_entry_policy({'out': '/new', 'rerun_b0': False}, prior=old)
    assert context == old / 'context-tenant/B0'
    assert compute == old / 'fixed-five/compute-reference/native-baseline-timing'
    assert policy.full_entry_policy({'rerun_b0': True}) == (None, None)


def test_saved_reuse_selection_keeps_original_external_baselines(tmp_path):
    (tmp_path / 'results.json').write_text(json.dumps({'reuse_selection': {
        'context_baseline_arm': '/original/context/B0', 'compute_baseline_arm': '/original/native/B0'}}))
    assert policy.full_entry_policy({'out': '/new', 'rerun_b0': False}, prior=tmp_path) == (
        Path('/original/context/B0'), Path('/original/native/B0'))


@pytest.mark.parametrize('configured,flag,expected', [
    (None, [], None), (False, [], '--no-rerun-b0'), (True, [], '--rerun-b0'),
    (False, ['--rerun-b0'], '--rerun-b0'), (True, ['--no-rerun-b0'], '--no-rerun-b0'),
])
def test_public_config_translates_switch_before_the_frozen_parser(tmp_path, configured, flag, expected):
    model = tmp_path / 'model'; model.mkdir()
    deployment = tmp_path / 'sdk'; deployment.mkdir()
    config = tmp_path / 'evaluation.json'
    config.write_text(json.dumps(dict(model=str(model), deployment=str(deployment),
        out=str(tmp_path/'out'), physical_devices=[1, 6], engine_profile='r6',
        **({'rerun_b0': configured} if configured is not None else {}))))
    arguments = one_click.configuration_arguments('run', ['--config', str(config), *flag])
    assert [arg for arg in arguments if arg in ('--rerun-b0', '--no-rerun-b0')] == ([expected] if expected else [])
    assert 'rerun_b0' not in arguments


def test_generated_guards_precede_each_model_launch_boundary():
    from test_context_baseline_reuse import combined
    files = combined()
    campaign = files['scripts/third_party.py'].decode()
    reference = files['scripts/automatic_reference.py'].decode()
    context = files['scripts/acceptance_suite.py'].decode()
    assert campaign.index('require_campaign_reuse(stages') < campaign.index('execute(args, out, frozen, receipt, stages)', campaign.index('require_campaign_reuse(stages'))
    assert reference.index('require_reference_reuse(plan)') < reference.index('for stage in plan["stages"]:')
    tree = ast.parse(context)
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'run')
    assert ast.unparse(function.body[1]) == 'require_context_reuse(args)'
    assert files['scripts/b0_run_policy.py'] == (ROOT / 'benchmark/b0_run_policy.py').read_bytes()


def test_public_policy_stays_out_of_the_original_frozen_config_schema(tmp_path):
    data = dict(model='/model', physical_devices=[1, 6], cache='/cache', out='/out', rerun_b0=True)
    result = run_all.evaluation_configuration(data, tmp_path/'installation')
    assert 'rerun_b0' not in result
    assert result['test_mode'] == 'full'


@pytest.mark.parametrize('kind,relative', [
    ('context', 'context-tenant/B0'),
    ('reference', 'fixed-five/compute-reference/native-baseline-timing'),
])
def test_auto_discovery_rejects_wrong_workload_then_reuses_an_older_match(
        tmp_path, monkeypatch, kind, relative):
    monkeypatch.setenv(policy.DISCOVERY_ROOT, str(tmp_path))
    policy.public_arguments([])
    old = tmp_path / 'old' / relative
    wrong = tmp_path / 'wrong' / relative
    for p in (old, wrong):
        p.mkdir(parents=True)
        (p/'checksums.json').write_text('{}')
    os.utime(tmp_path/'old', (1, 1))
    os.utime(tmp_path/'wrong', (2, 2))

    def admit(path):
        if path == wrong:
            raise ValueError('different input contract')
        return {'verified': True}

    selected, proof, audit = policy.select(tmp_path/'new', kind, admit)
    assert (selected, proof) == (old, {'verified': True})
    assert audit['action'] == 'reuse'
    assert audit['rejected'] == [{'source': str(wrong), 'reason': 'ValueError: different input contract'}]
    selected, _, audit = policy.select(tmp_path/'new', kind, lambda path: None)
    assert selected is None and audit['reason'] == 'no_matching_sealed_source'
    policy.public_arguments(['--rerun-b0'])
    assert policy.candidates(tmp_path/'new', kind) == []


def test_automatic_campaign_reuses_only_matching_b0_stages(tmp_path, monkeypatch):
    from test_context_baseline_reuse import combined
    files = combined()
    namespace = {'__file__': str(ROOT/'benchmark/one-click-kit/scripts/reuse_campaign.py')}
    exec(files['scripts/reuse_campaign.py'], namespace)
    monkeypatch.setitem(sys.modules, 'b0_run_policy', policy)
    policy.public_arguments([])
    old, current = tmp_path/'old', tmp_path/'new'
    old.mkdir()
    current.mkdir()
    model = tmp_path/'model'
    model.mkdir()
    shard = model/'model.safetensors'
    shard.write_bytes(b'weights')
    names = ['continuous-B0', 'continuous-B1', 'hle-solver-B0', 'hle-judge-B0']
    weight_record = {shard.name: dict(bytes=7, sha256=namespace['sha'](shard))}
    rows = [dict(name=name, status='completed') for name in names]
    for filename in ('campaign.json', 'campaign-checksums.json', 'frozen-execution-plan.json'):
        (old/filename).write_text('{}')
    for name in names:
        leaf = old/name
        leaf.mkdir()
        (leaf/'model-weights.json').write_text(json.dumps(weight_record))
    config = dict(model=str(model), baseline_deployment='/b0', baseline_source_map_sha256='map')
    namespace['source'] = lambda root: (root, config, {'stages': rows, 'status':'completed'}, {})
    namespace['identity'] = lambda plan, **_: plan
    namespace['compatible'] = lambda *_: None

    def audit(folder, name):
        if name == 'hle-solver-B0':
            raise ValueError('input differs')
        if name != 'continuous-B0':
            pytest.fail(f'Unexpected automatic audit of {name}')

    namespace['audit_stage'] = audit
    monkeypatch.setitem(sys.modules, 'device_reuse', SimpleNamespace(for_campaign=lambda *a: None))
    stages = [dict(name=name, args=[]) for name in names]
    assert set(namespace['prepare']([], current, config, stages, automatic_roots=[old])) == {'continuous-B0'}
    assert config['automatic_b0_campaign']['reused'] == ['continuous-B0']
    assert 'input differs' in config['automatic_b0_campaign']['rejected'][0]['reason']


@pytest.mark.parametrize('matched,first_role', [(True, 'B1'), (False, 'B0')])
def test_context_controller_runs_b0_only_on_a_miss(tmp_path, monkeypatch, matched, first_role):
    from test_context_baseline_reuse import combined
    ns = {'__file__': str(ROOT/'benchmark/one-click-kit/scripts/acceptance_suite.py')}
    exec(combined()['scripts/acceptance_suite.py'], ns)
    out = tmp_path/'new'
    contract = tmp_path/'input.json'
    contract.write_text('{}')
    candidate = tmp_path/'old'/'context-tenant'/'B0'
    candidate.mkdir(parents=True)
    (candidate/'checksums.json').write_text('{}')
    monkeypatch.setitem(sys.modules, 'b0_run_policy', policy)
    monkeypatch.setitem(sys.modules, 'acceptance_admission',
                        SimpleNamespace(prepare=lambda *_: {'contract': str(contract)}))
    policy.public_arguments([])

    def admit(path, role, args, requested):
        assert (path, role, requested) == (candidate, 'B0', contract)
        if not matched:
            raise ValueError('different device')
        return path, None, None

    ns['admit_reuse'] = admit
    ns['inspect'] = lambda path: {'execution_complete': path == candidate}
    launches = []

    def launch(argv, **kwargs):
        launches.append(argv[argv.index('--role')+1])
        return SimpleNamespace(returncode=2)

    ns['run_stage'] = launch
    args = SimpleNamespace(out=out, contract=contract, deployment=tmp_path/'sdk',
        baseline_deployment=tmp_path/'b0', model=tmp_path/'model', devices='0,1',
        physical_devices='2,3', report_python=None)
    assert ns['run'](args) == 2
    assert launches == [first_role]
    receipt = json.loads((out/'execution.json').read_bytes())
    assert receipt['baseline_execution_policy']['action'] == ('reuse' if matched else 'measure')
    if not matched:
        assert 'different device' in receipt['baseline_execution_policy']['rejected'][0]['reason']
