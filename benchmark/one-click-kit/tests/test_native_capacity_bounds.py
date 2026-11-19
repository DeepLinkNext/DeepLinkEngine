import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from native_capacity_bounds import envelope


def compute(pool=11312, **changes):
    fields = dict(maximum_output=8192, resident_seats=54, execution_seats=36,
        block_size=128, pool_blocks=pool, in_flight_batches=2,
        speculative_tokens=2, lookahead_tokens=2)
    fields.update(changes)
    return envelope([925, 2502], **fields)


def test_full_output_budget_and_all_ownership_classes_fit_without_old_out256_count():
    result = compute()
    assert result["maximum_tokens_per_owned_prefix"] == 2502 + 8192 + 6 + 2
    assert result["maximum_pages_per_owned_prefix"] == 84
    assert result["required_pages"] == (54 + 2 + 1) * 84 + 1
    assert result["sufficient_page_capacity"]
    assert not result["no_preemption_certified"]
    assert not result["source_ownership_qualified"]


def test_inadequate_pool_is_explicit_and_not_clamped_to_pass():
    result = compute(pool=1000)
    assert not result["sufficient_page_capacity"]
    assert result["spare_pages"] < 0
    assert compute(in_flight_batches=4)["maximum_tokens_per_owned_prefix"] > compute()["maximum_tokens_per_owned_prefix"]


@pytest.mark.parametrize("change", [dict(resident_seats=35), dict(block_size=0),
    dict(in_flight_batches=True), dict(lookahead_tokens=-1)])
def test_unknown_or_invalid_capacity_is_rejected(change):
    with pytest.raises(ValueError):
        compute(**change)
