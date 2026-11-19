from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from baseline_capacity_bounds import envelope


def test_actual018_pool_fits_full8192_for_every_source_permitted_grouping():
    result = envelope([925, 2502], reported_tokens=473088)
    assert result["sufficient_page_capacity"]
    assert result["maximum_prefix_tokens"] == 10702
    assert not result["group_count_assumed"]
    assert not result["no_preemption_certified"]
    # Independently expand arbitrary group counts and mixed group block sizes.
    for row in result["admissible_minimum_block_sizes"]:
        b = row["minimum_block_size"]
        for groups in (1, 2, 7, 40):
            actual_pool = (473088 // b) * groups
            group_pages = [max((10702 + s - 1) // s, 4) for s in [b] + [b * 3] * (groups - 1)]
            allocated = 25 * sum(group_pages) + 1
            assert allocated <= actual_pool


def test_small_pool_and_larger_source_domain_cannot_silently_pass():
    assert not envelope([2502], reported_tokens=128)["sufficient_page_capacity"]
    assert not envelope([2502], reported_tokens=473088, maximum_block_size=65536)["sufficient_page_capacity"]


@pytest.mark.parametrize("arguments", [dict(reported_tokens=True), dict(reported_tokens=0),
    dict(reported_tokens=473088, speculative_tokens=-1)])
def test_invalid_capacity_domain_is_rejected(arguments):
    with pytest.raises(ValueError):
        envelope([2502], **arguments)
