import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from native_output_geometry import geometry


def outputs():
    return dict(request_ids=["first", "second"], T_seconds=999, start_ns=10, end_ns=100,
        public_output_frames=[[], [("first", False, 0, [(0, [10], None, None)])],
            [("first", True, 0, [(0, [11, 12], "stop", 12)]),
             ("second", False, 0, [(0, [], None, None)])],
            [("second", True, 0, [(0, [20], "stop", 20)])]])


def test_native_work_uses_own_prefixes_emissions_and_never_elapsed_time():
    raw = outputs()
    original = copy.deepcopy(raw)
    a = geometry(raw, [[1, 2, 3], [4, 5]])
    assert raw == original
    raw.update(T_seconds=0.00001, start_ns=-500, end_ns="not a timing")
    assert geometry(raw, [[1, 2, 3], [4, 5]]) == a
    assert a["necessary_target_prefix_rows"] == 7
    assert a["input_tokens"] == 5 and a["output_tokens"] == 4
    assert a["requests"][0]["positive_emission_lengths"] == [1, 2]
    assert a["requests"][1]["ordered_public_emissions"][0] == dict(public_step=2, emitted_tokens=0, finished=False)
    assert a["elapsed_time_used"] is False
    assert a["ordered_acceptance_source_binding_certified"] is False
    assert a["efficiency_percent"] is None


def test_coalesced_public_outputs_cannot_be_mistaken_for_mtp2_acceptance():
    raw = dict(request_ids=["one"], public_output_frames=[[("one", True, 0, [(0, [1, 2, 3, 4], "stop", 4)])]])
    a = geometry(raw, [[8]])
    assert a["output_tokens"] == 4
    assert not a["all_emissions_mtp_width_compatible"]
    assert not a["requests"][0]["first_emission_is_one"]
    assert not a["ordered_acceptance_source_binding_certified"]


def test_native_token_identity_and_cold_cache_remain_required():
    raw = outputs()
    bad = copy.deepcopy(raw)
    row = bad["public_output_frames"][1][0]
    bad["public_output_frames"][1][0] = (row[0], row[1], 1, row[3])
    with pytest.raises(ValueError, match="prefix-cached"):
        geometry(bad, [[1, 2, 3], [4, 5]])
    assert geometry(raw, [[1, 2, 3], [4, 5]])["geometry_sha256"] != geometry(raw, [[1, 2, 8], [4, 5]])["geometry_sha256"]
