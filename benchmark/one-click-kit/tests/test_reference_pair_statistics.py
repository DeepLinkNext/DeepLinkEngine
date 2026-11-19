from decimal import Decimal
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from reference_pair_statistics import arm, pair


def test_matched_unknown_f_cancels_without_mixing_independent_interval_endpoints():
    current = arm([10**15, 2 * 10**15], 1000, 500)
    assert current["achieved_tflops_per_card_interval"] == ["0.5", "1"]
    assert current["reference_tflops_per_card_upper"] == "2"
    assert current["efficiency_lower_percent"] == "50"
    assert Decimal(current["achieved_tflops_per_card_interval"][0]) / Decimal(current["reference_tflops_per_card_upper"]) * 100 == 25
    baseline = arm([10**15, 2 * 10**15], 2000, 500)
    result = pair(baseline, current)
    assert result["effective_compute_gain_percent_interval"] == ["0", "300"]
    assert result["milestone_met"] and not result["final_numeric_met"]
    assert not result["full_model_certificate"]


def test_calibrated_reference_ratio_is_not_clamped_to_hardware_peak():
    current = arm([10**15, 10**15], 100, 120)
    assert current["efficiency_lower_percent"] == "120"
    assert current["hardware_peak_utilization_percent"] is None


@pytest.mark.parametrize("time", [0, -1, True, "NaN", "Infinity"])
def test_invalid_matched_clock_cannot_produce_a_numeric_result(time):
    with pytest.raises(ValueError):
        arm([10**15, 10**15], time, 100)
