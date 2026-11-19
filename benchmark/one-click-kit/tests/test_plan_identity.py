"""Historical run claims must never masquerade as current execution metadata."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plan_identity import current_b1_plan


def values():
    original = dict(schema='historical-r4', comparison='B0/B1 physical5/6',
                    physical_cards=[5, 6], criterion='No warmups;200 arrivals',
                    inputs='MMLU200', joint_order=['B0', 'B1'],
                    qualification=dict(old=True), host_budget_per_rank_bytes=8 * 2**30)
    current = dict(release='full-answer-b1-20261113-r6', logical_devices='0,1',
                   physical_cards=[1, 6], host_budget_per_rank_bytes=4 * 2**30)
    return original, current


def test_current_run_does_not_inherit_historical_comparison_or_protocol():
    original, current = values()
    untouched = copy.deepcopy(original)
    result = current_b1_plan(original, **current)
    assert result['schema'] == 'current-b1-deployment-plan/v2'
    assert result['engine_release'] == current['release']
    assert result['physical_cards'] == [1, 6]
    assert result['host_budget_per_rank_bytes'] == 4 * 2**30
    assert not any(name in result for name in ['comparison', 'criterion', 'joint_order', 'inputs'])
    assert result['historical_claims'] == original == untouched
    result['historical_claims']['qualification']['old'] = False
    result['physical_cards'][0] = 2
    assert original == untouched
    assert current['physical_cards'] == [1, 6]


@pytest.mark.parametrize('field,value', [
    ('release', ''), ('logical_devices', '0,0'), ('logical_devices', '0'),
    ('physical_cards', [1, 1]), ('physical_cards', [True, 6]),
    ('physical_cards', [-1, 6]), ('physical_cards', [1, 6, 7]),
    ('host_budget_per_rank_bytes', 8 * 2**30), ('host_budget_per_rank_bytes', True),
])
def test_reject_unbound_or_malformed_current_identity(field, value):
    original, current = values()
    current[field] = value
    with pytest.raises(ValueError):
        current_b1_plan(original, **current)


def test_historical_release_can_be_explicitly_selected_without_renaming_it():
    original, current = values()
    current['release'] = 'full-answer-b1-20261030-r4'
    assert current_b1_plan(original, **current)['engine_release'] == current['release']


@pytest.mark.parametrize('kind', ['joint', 'continuous', 'dataset'])
def test_actual_entry_writes_current_plan_and_preserves_service_dispatch(tmp_path, monkeypatch, kind):
    """All service, hardware and Manager calls below are explicit host fakes."""
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    spec = importlib.util.spec_from_file_location('metadata_entry_host_fixture', scripts / 'engine_entry.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setattr(entry.os, 'environ', dict(entry.os.environ))
    root, out, cache = tmp_path / 'deployment', tmp_path / 'output', tmp_path / 'cache'
    for path in [root / 'provenance/base-launch', root / 'runtime/ascend', out, cache]:
        path.mkdir(parents=True)
    original, current = values()
    raw_original = json.dumps(original).encode()
    (root / 'provenance/original-plan.json').write_bytes(raw_original)
    (root / 'provenance/installed-files.json').write_text('{}')
    (root / 'deployment.json').write_text(json.dumps(dict(
        release=current['release'], python=sys.executable,
        original_plan_sha256=hashlib.sha256(raw_original).hexdigest(),
        host_contract_qualification=dict(installed_metadata={'vllm-ascend': 'synthetic-version'}))))
    inputs = tmp_path / 'selected-inputs.json'
    inputs.write_text('{"synthetic": true}')
    selected = dict(service_role='B1', dataset='hle', stage='solver')
    manager_calls = []

    def fake_module(name, **fields):
        module = ModuleType(name)
        vars(module).update(fields)
        monkeypatch.setitem(sys.modules, name, module)

    fake_module('qualification', qualify=lambda *args: dict(status='synthetic-qualified'))
    fake_module('launch_boundary_b1')
    command = [sys.executable, 'synthetic-only', 'serve', '--limit-mm-per-prompt', '{"image":0,"video":0}']
    fake_module('launch_corrected_timing', prepare=lambda *args: (command.copy(),
        dict(VLLM_PLUGINS='ascend,hust_microflow_school028_fa', PYTHONPATH='/synthetic-b1-src',
             BETTERSCALE_STATE_CACHE_HOST_BYTES=str(4 * 2**30)), 'synthetic-base-launch'))
    fake_module('serve_b1_budget', configure_host_budget=lambda cmd, env, budget: (cmd, env))
    fake_module('resource_guard', cards=lambda: [1, 6])
    fake_module('serving_warmup', selected=lambda profile: True,
                protocol=lambda mode, *, capacity36=False: dict(synthetic=True, kind=mode, requests=194))
    fake_module('dataset_client', checked=lambda path: (selected.copy(), []))
    fake_module('judge_grammar', configure=lambda *args: None)
    fake_module('vllm_hust_ext')
    fake_module('vllm_hust_ext.cli', main=lambda args: manager_calls.append(args) or 17)
    monkeypatch.setattr(entry, 'check_completion_inbox', lambda *args: None)
    monkeypatch.setattr(entry, 'check_pinned_host_memory', lambda *args: None)
    monkeypatch.setenv('VLLMHUST_ENGINE_BOOTSTRAPPED', str(root))
    monkeypatch.setenv('VLLM_HUST_PHYSICAL_DEVICES', '1,6')
    argv = [str(scripts / 'engine_entry.py'), '--deployment', str(root), '--out', str(out),
            '--cache', str(cache), '--devices', '0,1', '--model', str(tmp_path / 'model'),
            '--phase', 'serve', '--kind', kind]
    if kind in ('joint', 'dataset'):
        argv += ['--qualified-continuous', str(tmp_path / 'qualified')]
    if kind == 'continuous':
        argv += ['--continuous-inputs', str(inputs)]
    elif kind == 'dataset':
        argv += ['--dataset-spec', str(inputs)]
    monkeypatch.setattr(sys, 'argv', argv)
    assert entry.main() == 17
    plan = json.loads((out / 'plan.json').read_bytes())
    assert plan['engine_release'] == current['release']
    assert plan['physical_cards'] == [1, 6]
    assert plan['serving_warmup']['requests'] == 194
    assert plan['historical_claims'] == original
    assert not any(key in plan for key in ('comparison', 'criterion', 'joint_order', 'inputs'))
    assert (root / 'provenance/original-plan.json').read_bytes() == raw_original
    assert plan['host_budget_per_rank_bytes'] == 4 * 2**30
    assert manager_calls[0][:5] == ['run', '--shutdown-grace-seconds', '90', '--', sys.executable]
    assert manager_calls[0][5] == str(root / 'runtime/entrypoints/serve_b1_child.py')
    assert manager_calls[0][6] == str(out / 'plan.json')
    assert '--worker-extension-cls' in manager_calls[0]
    if kind == 'dataset':
        assert 'quality_plan' not in plan and 'scorer' not in plan
        assert plan['input_contract']['sha256'] == hashlib.sha256(inputs.read_bytes()).hexdigest()
        assert plan['engine_release'] in plan['deployment_difference']
        assert '{"image": 1, "video": 0}' in manager_calls[0]
