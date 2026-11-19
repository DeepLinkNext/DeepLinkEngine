import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from canonical_protocol import SOURCES, current_plan, prepare, protocol


BINDINGS = {"/current/admitted/source.py": "a" * 64}


def test_current_dense_shapes_keep_original_five_by100_protocol_and_source(tmp_path):
    row, source = protocol("dense")
    cases = [dict(case=0, m=108, n=6144, k=2048)]
    receipt = prepare(tmp_path / "dense", "dense", cases, BINDINGS)
    plan = json.loads((tmp_path / "dense" / receipt["plan"]).read_bytes())
    assert (tmp_path / "dense" / receipt["sampler"]).read_bytes() == source.read_bytes()
    assert plan["repetitions"] == 5 and plan["timed_replays"] == 100
    assert plan["graph_warmups_per_repetition"] == 20
    assert plan["atol"] == .002 and plan["rtol"] == .02 and plan["seed"] == 137
    assert plan["cases"] == cases
    assert plan["current_source_sha256"] == BINDINGS
    assert not receipt["measurement_executed"] and not receipt["device_initialized"]


def test_current_norm_metadata_never_claims_old_target_source_hashes():
    plan, _ = current_plan("norm", [dict(case=0, kind="hidden_plain", rows=108, width=2048)], BINDINGS)
    assert plan["source_sha256"] == BINDINGS
    assert plan["repetitions"] == 3 and plan["timed_replays"] == 100
    assert plan["source_runtime_applicability_certified"] is False


@pytest.mark.parametrize("family", ["dense", "norm", "preprocess", "rope", "router", "swiglu", "vocab", "moe", "gdn_ho"])
def test_every_restored_sampler_has_safe_original_plan_and_output_locations(family):
    from canonical_protocol import locations
    row, source = protocol(family)
    assert locations(source.read_bytes())["PLAN"] == row["original_plan"]


def test_small_m_moe_cannot_silently_use_the_older_single_replay_protocol():
    for m in (1, 2):
        with pytest.raises(ValueError, match="M>=3"):
            current_plan("moe", [dict(case=0, rows=8*m, group_counts=[m]*8+[0]*248)], BINDINGS)


def test_moe_keeps_original_exact_expert_lists_and_three_single_replays():
    cases = [dict(case=0, rows=24, group_counts=[3]*8+[0]*248)]
    plan, _ = current_plan("moe", cases, BINDINGS)
    assert plan["cases"] == cases
    assert plan["repetitions"] == 3 and plan["timed_replays"] == 1
    assert plan["operators"] == [dict(name="w13", k=2048, n=512), dict(name="w2", k=256, n=2048)]
    assert plan["atol"] == .002 and plan["rtol"] == .02
    assert plan["historical_counts_or_timings_reused"] is False


@pytest.mark.parametrize("counts", [[True]*24+[0]*232, [4]*6+[0]*250, [3]*7+[0]*249, [3]*8])
def test_invalid_expert_lists_fail_before_device_launch(counts):
    with pytest.raises(ValueError, match="MoE"):
        current_plan("moe", [dict(case=0, rows=24, group_counts=counts)], BINDINGS)


def test_gdn_ho_keeps_original_library_and_statistical_protocol():
    row, _ = protocol("gdn_ho")
    cases = [dict(case=0, capacity=108, requests=36, lengths=[1, 65], initial_flags=[False, True])]
    plan, _ = current_plan("gdn_ho", cases, BINDINGS)
    assert plan["cases"] == cases
    assert (plan["key_heads"], plan["value_heads"], plan["head_dim"]) == (8, 16, 128)
    assert (plan["repetitions"], plan["timed_replays"], plan["warmups"]) == (3, 30, 3)
    assert plan["atol"] == plan["rtol"] == .01
    assert row["library_sha256"] == "1c522f04a701d3423798b4b7f157737ef41956be25cf46842b6ba33c7ec41bea"


@pytest.mark.parametrize("changes", [dict(capacity=32), dict(requests=1), dict(initial_flags=[False]), dict(lengths=[True, 65])])
def test_gdn_ho_rejects_invalid_capacity_and_state_metadata(changes):
    case = dict(case=0, capacity=108, requests=36, lengths=[1, 65], initial_flags=[False, True])
    case.update(changes)
    with pytest.raises(ValueError, match="GDN"):
        current_plan("gdn_ho", [case], BINDINGS)


def test_tampered_sampler_is_rejected_before_any_device_launch(tmp_path):
    shutil.copytree(SOURCES, tmp_path / "sources")
    row, _ = protocol("dense")
    (tmp_path / "sources" / row["sampler"]).write_text("# different operator\n")
    with pytest.raises(ValueError, match="sampler source changed"):
        protocol("dense", tmp_path / "sources")


def test_unknown_normalization_branch_cannot_silently_take_an_else_path():
    with pytest.raises(ValueError, match="operation branch"):
        current_plan("norm", [dict(case=0, kind="invented", rows=10, width=2048)], BINDINGS)
