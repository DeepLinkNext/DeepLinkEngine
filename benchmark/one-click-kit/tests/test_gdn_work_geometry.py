"""Observed GDN splits, request identities, layer coverage and fail-closed joins."""

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
import gdn_work_geometry as geometry
from gdn_prefill_reference import segment_arithmetic


def wave(start, length, initial, ordinal):
    row = dict(request="a" * 64, start=start, count=length)
    group = dict(layers=["gdn.0", "gdn.1"], live=1, decode=False,
                 cumulative_lengths=[0, length], initial_state=[initial],
                 verify_indices=[], verify_count=0)
    return dict(wave=ordinal, rows=[row], gdn=[dict(requests=[row["request"]], groups=[group])])


def test_real_segment_boundaries_and_zero_initial_state_are_counted():
    data = [wave(0, 33, False, 0), wave(33, 32, True, 1)]
    requests = [dict(request="a" * 64, input_tokens=65, output_tokens=1)]
    result = geometry.target_prefill_work(data, requests, expected_layers=["gdn.0", "gdn.1"], global_heads=32)
    expected = 64 * (segment_arithmetic(33, False)["total_basic"] + segment_arithmetic(32, True)["total_basic"])
    assert result["total_basic"] == expected
    assert len(result["segments"]) == 2 and result["request_layer_coverage_verified"]
    assert result["full_model_certificate"] is False
    assert expected != 64 * segment_arithmetic(65, False)["total_basic"]


@pytest.mark.parametrize("mutation", ["count", "state", "duplicate_layer", "missing_layer", "unknown_request", "gap"])
def test_partial_or_inconsistent_native_metadata_cannot_be_counted(mutation):
    data = [wave(0, 33, False, 0), wave(33, 32, True, 1)]
    group = data[1]["gdn"][0]["groups"][0]
    if mutation == "count":
        group["cumulative_lengths"][1] -= 1
    elif mutation == "state":
        group["initial_state"] = [0]
    elif mutation == "duplicate_layer":
        group["layers"].append("gdn.0")
    elif mutation == "missing_layer":
        group["layers"].pop()
    elif mutation == "unknown_request":
        data[1]["gdn"][0]["requests"][0] = "b" * 64
    else:
        data[1]["rows"][0]["start"] += 1
    with pytest.raises(ValueError):
        geometry.target_prefill_work(data, [dict(request="a" * 64, input_tokens=65, output_tokens=1)],
                                    expected_layers=["gdn.0", "gdn.1"], global_heads=32)


def test_speculative_suffix_is_trimmed_without_changing_real_segment_start():
    data = [wave(0, 67, False, 0)]
    result = geometry.target_prefill_work(data, [dict(request="a" * 64, input_tokens=65, output_tokens=1)],
                                        expected_layers=["gdn.0", "gdn.1"], global_heads=32)
    assert result["total_basic"] == 64 * segment_arithmetic(65, False)["total_basic"]
    assert result["segments"][0]["observed_count"] == 67
    assert result["segments"][0]["necessary_count"] == 65


def test_recurrent_short_rows_cover_prefix_but_do_not_invent_wy_chunks():
    first = wave(0, 1, False, 0)
    data = wave(1, 2, True, 1)
    group = data["gdn"][0]["groups"][0]
    group["verify_indices"], group["verify_count"] = [0], 1
    result = geometry.target_prefill_work([first, data], [dict(request="a" * 64, input_tokens=1, output_tokens=3)],
                                        expected_layers=["gdn.0", "gdn.1"], global_heads=32)
    assert len(result["segments"]) == 1
    assert result["segments"][0]["necessary_count"] == 1
    assert result["recurrence_candidate"]["necessary_token_layers"] == 4
    assert result["recurrence_candidate"]["total_basic"] == 4 * 32 * (7 * 128 * 128 + 128)
    assert result["request_layer_coverage_verified"]


def test_missing_fresh_gdn_state_rule_is_not_filled_with_stateful_work():
    data = wave(0, 1, False, 0)
    group = data["gdn"][0]["groups"][0]
    group["verify_indices"], group["verify_count"] = [0], 1
    with pytest.raises(ValueError, match="zero-state arithmetic"):
        geometry.target_prefill_work([data], [dict(request="a" * 64, input_tokens=1, output_tokens=1)],
                                    expected_layers=["gdn.0", "gdn.1"], global_heads=32)
