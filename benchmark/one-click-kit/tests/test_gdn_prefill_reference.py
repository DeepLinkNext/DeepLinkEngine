"""Independent chunk algebra versus scalar recurrence, including state handoff."""

import importlib.util
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1] / "third-party-test-kit"


def load(name):
    spec = importlib.util.spec_from_file_location(name, KIT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prefill = load("gdn_prefill_reference")
decode = load("gdn_reference")


def literal_inverse_counts(n):
    """Count scalar source recurrences and structural triangular products."""
    recurrence = 0
    for base in range(0, n, 16):
        stop = min(n, base + 16)
        for i in range(base, stop):
            for j in range(base, i):
                for _ in range(j + 1, i):
                    recurrence += 2
    merges = 0
    for start, width in ((0, 16), (32, 16), (0, 32)):
        left = min(width, max(0, n - start))
        right = min(width, max(0, n - start - width))
        for i in range(right):
            for _ in range(left):
                for _ in range(i + 1):
                    merges += 2
        for _ in range(right):
            for j in range(left):
                for _ in range(j, left):
                    merges += 2
    return recurrence, merges


@pytest.mark.parametrize("n", [1, 2, 15, 16, 17, 31, 32, 33, 63, 64])
def test_inverse_catalog_against_literal_valid_scalar_dependencies(n):
    parts = prefill.chunk_arithmetic(n, True)["basic"]
    recurrence, merges = literal_inverse_counts(n)
    assert parts["inverse16_recurrence"] == recurrence
    assert parts["inverse_merge_triangular_dot"] == merges
    assert parts["inverse_negations"] == sum(range(n))


def test_zero_state_and_published_segment_boundaries_are_not_erased():
    zero = prefill.chunk_arithmetic(17, False)
    state = prefill.chunk_arithmetic(17, True)
    for name in ("h_wh_dot", "h_value_residual_subtract", "h_state_decay_and_add",
                 "o_qh_dot", "o_state_decay_and_add"):
        assert zero["basic"][name] == 0 < state["basic"][name]
    uncut = prefill.segment_arithmetic(66, False)
    cut = [prefill.segment_arithmetic(33, False), prefill.segment_arithmetic(33, True)]
    assert uncut["total_basic"] != sum(c["total_basic"] for c in cut)
    assert sum(c["length"] for c in uncut["chunks"]) == 66


@pytest.mark.parametrize("lengths", [[1], [16], [17], [64], [65], [33, 32], [1, 63, 2]])
@pytest.mark.parametrize("has_state", [False, True])
def test_segmented_chunk_matches_independent_scalar_states(lengths, has_state):
    torch = pytest.importorskip("torch")
    rng = torch.Generator().manual_seed(1701)
    n, hk, hv, k, v = sum(lengths), 2, 4, 3, 5
    q, keys = [torch.randn((n, hk, k), generator=rng) * .1 for _ in range(2)]
    values = torch.randn((n, hv, v), generator=rng) * .1
    gates = -torch.rand((n, hv), generator=rng) * .2
    beta = torch.rand((n, hv), generator=rng)
    state = torch.randn((hv, k, v), generator=rng) * .01 if has_state else torch.zeros((hv, k, v))
    before = state.clone()
    output, final = prefill.segmented(q, keys, values, gates, beta, state, k**-.5, lengths,
                                      initial_state=has_state)
    scalar_states = state.tolist()
    expected = []
    for i in range(n):
        row = []
        for head in range(hv):
            group = head // (hv // hk)
            o, scalar_states[head] = decode.scalar_step(
                q[i, group].tolist(), keys[i, group].tolist(), values[i, head].tolist(),
                float(gates[i, head]), float(beta[i, head]), scalar_states[head], k**-.5)
            row.append(o)
        expected.append(row)
    torch.testing.assert_close(output, torch.tensor(expected), atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(final, torch.tensor(scalar_states), atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(state, before, rtol=0, atol=0)


@pytest.mark.parametrize("n,state", [(0, False), (65, False), (True, False), (1, 0)])
def test_reject_padding_and_ambiguous_state(n, state):
    with pytest.raises(ValueError):
        prefill.chunk_arithmetic(n, state)


def test_final_fp32_pool_store_decays_the_declared_bf16_workspace():
    torch = pytest.importorskip("torch")
    q = torch.zeros((1, 1, 2))
    state = torch.full((1, 2, 2), .123456)
    gates = torch.tensor([[-.2]])
    _, stored = prefill.chunk(q, q, q, gates, torch.ones((1, 1)), state, 1.,
                              storage_dtype=torch.bfloat16)
    _, fp32 = prefill.chunk(q, q, q, gates, torch.ones((1, 1)), state, 1.)
    torch.testing.assert_close(stored, state.to(torch.bfloat16).float() * torch.exp(gates)[..., None], atol=0, rtol=0)
    assert not torch.equal(stored, fp32)
