import sys
from itertools import product
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from gdn_partition_bounds import bounds
from gdn_prefill_reference import chunk_arithmetic, segment_arithmetic


def compositions(n):
    if n == 0:
        yield []
    for first in range(1, min(64, n) + 1):
        for tail in compositions(n - first):
            yield [first, *tail]


def test_all_small_compositions_match_enumerated_extrema():
    for n in range(1, 10):
        oracle = [sum(chunk_arithmetic(c, i > 0)["total_basic"] for i, c in enumerate(parts))
                  for parts in compositions(n)]
        result = bounds([n])
        assert result["minimum_basic"] == min(oracle)
        assert result["maximum_basic"] == max(oracle)


def test_real_segment_grids_are_contained_and_witnesses_replay():
    result = bounds([1, 64, 65, 129, 256])
    for row in result["per_request_single_head_layer"]:
        n = row["length"]
        grids = [[n], [1] * n]
        if n > 1:
            grids += [[n // 2, n - n // 2], [n - 1, 1]]
        for grid in grids:
            count = sum(segment_arithmetic(length, i > 0)["total_basic"]
                        for i, length in enumerate(grid))
            assert row["minimum"]["total_basic"] <= count <= row["maximum"]["total_basic"]
        for endpoint in ("minimum", "maximum"):
            chunks = row[endpoint]["witness_chunk_lengths"]
            assert sum(chunks) == n
            assert all(1 <= c <= 64 for c in chunks)
    assert not result["historical_chunk_multiplicities_used"]
    assert not result["full_model_certificate"]


@pytest.mark.parametrize("values", [[], [0], [-1], [True], [1.5]])
def test_unknown_population_is_rejected(values):
    with pytest.raises(ValueError):
        bounds(values)


def test_original018_one_token_prompt_recurrence_is_included_at_both_state_origins():
    # Independently enumerate every partition and its WY/recurrent choices.
    # Tiny dimensions keep the scalar operation inventory transparent.
    k, v = 2, 3
    cold_recurrence = k + v + k * v + (2 * k - 1) * v
    stateful_recurrence = (k + k * v + (2 * k - 1) * v + v + v
                          + 2 * k * v + (2 * k - 1) * v)
    assert cold_recurrence == 20 and stateful_recurrence == 44
    for n in range(1, 7):
        totals = []
        for pieces in compositions(n):
            options = []
            for i, length in enumerate(pieces):
                options.append([chunk_arithmetic(length, i > 0, k, v)["total_basic"]]
                    + ([stateful_recurrence if i else cold_recurrence] if length == 1 else []))
            totals.extend(sum(values) for values in product(*options))
        result = bounds([n], key_dim=k, value_dim=v, allow_recurrent_prompt=True)
        assert result["minimum_basic"] == min(totals)
        assert result["maximum_basic"] == max(totals)
