"""Public entry defaults must never launch fixed B0 as a fallback."""
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
import one_click
import run_all


@pytest.fixture(autouse=True)
def default_policy():
    with patch.dict(os.environ):
        os.environ.pop(policy.ENVIRONMENT_KEY, None)
        yield


def test_default_without_a_source_fails_before_launch():
    with patch('subprocess.run') as run:
        with pytest.raises(ValueError, match='默认不重跑'):
            policy.public_arguments(['--model', '/model'])
    run.assert_not_called()
    assert policy.enabled() is False


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
    assert not policy.enabled()


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
    with pytest.raises(ValueError, match='默认不重跑'):
        policy.full_entry_policy({'out': str(tmp_path)})
    old = tmp_path / 'old'
    (old / 'context-tenant/B0').mkdir(parents=True)
    (old / 'fixed-five/compute-reference/native-baseline-timing').mkdir(parents=True)
    context, compute = policy.full_entry_policy({'out': '/new'}, prior=old)
    assert context == old / 'context-tenant/B0'
    assert compute == old / 'fixed-five/compute-reference/native-baseline-timing'
    assert policy.full_entry_policy({'rerun_b0': True}) == (None, None)


def test_saved_reuse_selection_keeps_original_external_baselines(tmp_path):
    (tmp_path / 'results.json').write_text(json.dumps({'reuse_selection': {
        'context_baseline_arm': '/original/context/B0', 'compute_baseline_arm': '/original/native/B0'}}))
    assert policy.full_entry_policy({'out': '/new'}, prior=tmp_path) == (
        Path('/original/context/B0'), Path('/original/native/B0'))


@pytest.mark.parametrize('configured,flag,expected', [
    (False, [], False), (True, [], True), (False, ['--rerun-b0'], True),
    (True, ['--no-rerun-b0'], False),
])
def test_public_config_translates_switch_before_the_frozen_parser(tmp_path, configured, flag, expected):
    model = tmp_path / 'model'; model.mkdir()
    deployment = tmp_path / 'sdk'; deployment.mkdir()
    config = tmp_path / 'evaluation.json'
    config.write_text(json.dumps(dict(model=str(model), deployment=str(deployment),
        out=str(tmp_path/'out'), physical_devices=[1, 6], engine_profile='r6', rerun_b0=configured)))
    arguments = one_click.configuration_arguments('run', ['--config', str(config), *flag])
    assert ('--rerun-b0' in arguments) is expected
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
