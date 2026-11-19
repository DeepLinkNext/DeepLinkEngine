"""Independent checks of reference state transitions and operation counting."""

import importlib.util
from pathlib import Path
import random
import sys

import pytest

KIT = Path(__file__).resolve().parents[1] / "third-party-test-kit"
spec = importlib.util.spec_from_file_location("gdn_reference_under_test", KIT / "gdn_reference.py")
gdn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gdn)


def test_import_does_not_load_runtime():
    import subprocess
    code = f"import runpy,sys;runpy.run_path({str(KIT / 'gdn_reference.py')!r});assert not any(n in sys.modules for n in ('torch','torch_npu','vllm'))"
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


@pytest.mark.parametrize("dk,dv", [(1, 1), (2, 3), (3, 2), (128, 128)])
def test_count_from_literal_scalar_loops(dk, dv):
    # Enumerate each operation site; a dot adds only after the first product.
    count = 0
    for _ in range(dk):
        count += 1  # query scale
    for _ in range(dk):
        for _ in range(dv):
            count += 1  # state decay
    for _ in range(dv):
        for i in range(dk):
            count += 1 + (i > 0)  # prediction dot
        count += 2  # value subtraction, beta multiply
    for _ in range(dk):
        for _ in range(dv):
            count += 2  # state multiply-add
    for _ in range(dv):
        for i in range(dk):
            count += 1 + (i > 0)  # output dot
    result = gdn.operation_counts(dk, dv)
    assert result["total_basic"] == count == 7 * dk * dv + dk
    assert result["special_functions"] == {"exp": 1}
    assert not result["complete_model_work"]


@pytest.mark.parametrize("dk,dv", [(True, 128), (0, 128), (128, -1), (2.0, 3)])
def test_invalid_count_dimensions(dk, dv):
    with pytest.raises(ValueError):
        gdn.operation_counts(dk, dv)


def test_hand_calculated_transition_and_state_ownership():
    state = [[2.0, 4.0]]
    output, updated = gdn.scalar_step([2], [3], [5, 7], 0, 0.5, state, 0.5)
    assert updated == [[0.5, -3.5]]
    assert output == [0.5, -3.5]
    assert state == [[2.0, 4.0]]


@pytest.mark.parametrize("g,beta", [(0, 0), (-2, 1), (-0.7, 0.3)])
def test_zero_key_and_zero_initial_state(g, beta):
    output, state = gdn.scalar_step([2, -1], [0, 0], [4], g, beta, [[0], [0]], 1)
    assert output == [0]
    assert state == [[0], [0]]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_invalid_scalar_values(bad):
    with pytest.raises(ValueError):
        gdn.scalar_step([bad], [1], [1], 0, 0.5, [[0]], 1)


def test_tensor_oracle_grouped_heads_and_multi_step_states():
    torch = pytest.importorskip("torch")
    rng = random.Random(17)
    n, hk, hv, dk, dv = 2, 2, 4, 3, 2
    state = torch.tensor([rng.uniform(-0.1, 0.1) for _ in range(n * hv * dk * dv)]).reshape(n, hv, dk, dv)
    scalar_states = state.tolist()
    for _ in range(3):
        q = torch.tensor([rng.uniform(-0.5, 0.5) for _ in range(n * hk * dk)]).reshape(n, hk, dk)
        k = q.flip(-1).clone()
        v = torch.tensor([rng.uniform(-0.5, 0.5) for _ in range(n * hv * dv)]).reshape(n, hv, dv)
        gate = torch.full((n, hv), -0.2)
        beta = torch.full((n, hv), 0.7)
        old = state.clone()
        input_state = state
        output, state = gdn.tensor_step(q, k, v, gate, beta, state, dk**-0.5)
        assert torch.equal(input_state, old)
        expected = []
        for batch in range(n):
            row = []
            for head in range(hv):
                value, updated = gdn.scalar_step(q[batch, head // 2].tolist(), k[batch, head // 2].tolist(), v[batch, head].tolist(), float(gate[batch, head]), float(beta[batch, head]), scalar_states[batch][head], dk**-0.5)
                scalar_states[batch][head] = updated
                row.append(value)
            expected.append(row)
        torch.testing.assert_close(output, torch.tensor(expected), atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(state, torch.tensor(scalar_states), atol=2e-7, rtol=2e-6)
        assert state.data_ptr() != old.data_ptr()


def test_historical_decode_arithmetic_matches_independent_reference():
    # Existing M01: 200 requests, 256 outputs, 30 GDN layers, 32 global heads.
    assert gdn.operation_counts(128, 128)["total_basic"] * 200 * 255 * 30 * 32 == 5621391360000
