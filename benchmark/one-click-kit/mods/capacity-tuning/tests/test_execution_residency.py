"""Preserve the real all-off point, geometry closure and version/budget boundaries."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('paired_residency_mod', ROOT / 'execution_residency.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def baseline():
    return json.loads((ROOT / 'fixtures/baseline-launch.json').read_text())


def test_baseline_retained_and_only_declared_geometry_changed():
    before = baseline()
    untouched = deepcopy(before)
    assert mod.configure_pair(before, 24, 36)['command'] == before['command']
    candidate = mod.configure_pair(before, 36, 54)
    assert before == untouched
    command = candidate['command']
    state = json.loads(command[command.index('--additional-config') + 1])
    graph = json.loads(command[command.index('--compilation-config') + 1])
    original_state = json.loads(before['command'][before['command'].index('--additional-config') + 1])
    assert state.pop('state_resident_seats') == 54
    assert state == {k:v for k,v in original_state.items() if k != 'state_resident_seats'}
    assert state['state_cache_host_bytes'] == 4 * 2**30
    assert set(graph['cudagraph_capture_sizes']) - set(mod.graph_sizes(24)) == {96, 108}
    assert command[command.index('--max-num-seqs') + 1] == '36'
    assert sum(a != b for a,b in zip(before['command'], command)) == 3
    assert not candidate['capacity_tuning']['inference_started']
    assert not candidate['capacity_tuning']['current_r7_combination_measured']


@pytest.mark.parametrize('pair', [(36, 36), (24, 54), (32, 54), (48, 72), (True, 36), (36, 54.0)])
def test_unmeasured_or_ambiguous_geometry_rejected(pair):
    with pytest.raises(ValueError):
        mod.configure_pair(baseline(), *pair)


def test_r7_requires_a_new_joint_admission_not_relabelled_r6_evidence():
    before = baseline()
    before['engine_release'] = 'full-answer-b1-20261116-r7'
    with pytest.raises(ValueError, match='r6'):
        mod.configure_pair(before, 36, 54)


def test_budget_and_mtp_drift_rejected_before_proposal():
    for flag, value in [('--additional-config', '{}'),
                        ('--speculative-config', '{"method":"mtp","num_speculative_tokens":3}')]:
        before = baseline()
        before['command'][before['command'].index(flag) + 1] = value
        with pytest.raises(ValueError):
            mod.configure_pair(before, 36, 54)


def test_both_full_pairs_and_four_trials_are_kept_in_evidence():
    result = mod.evidence()
    assert len(result['pairs']) == 2 and len(result['rows']) == 4
    assert all(row['recorded'] == row['planned'] == 200 for row in result['rows'])
    assert [pair['quality']['candidate_correct'] for pair in result['pairs']] == [144, 137]
