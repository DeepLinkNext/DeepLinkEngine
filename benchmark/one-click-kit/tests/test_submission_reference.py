"""Submission capture regression tests; synthetic CPU records, no NPU scores."""

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "third-party-test-kit")]
from dataset_contract import sha
from reference_capture_profile import profile, validate_native_plan
from reference_final_stages import matched_pair, repetition_prefix
from reference_pair_statistics import arm, aggregate


def native_plan(name="submission-v2"):
    capture = profile(name)
    return dict(reference_capture_profile=name, max_tokens=capture["max_tokens"],
        formal_repetitions=capture["formal_repetitions"],
        sampling=dict(max_tokens=capture["max_tokens"], temperature=0, seed=0,
            ignore_eos=False, top_p=1.0, top_k=-1, presence_penalty=0.0, detokenize=False))


def test_submission_budget_is_independent_from_the_full_answer_budget():
    assert profile()["max_tokens"] == 256
    assert profile()["formal_repetitions"] == 3
    assert validate_native_plan(native_plan()) == profile()
    assert validate_native_plan(native_plan("full-answer-v1"))["max_tokens"] == 8192


@pytest.mark.parametrize("change", ["maximum", "sampling", "rounds", "bool", "profile"])
def test_a_different_budget_cannot_be_accepted_as_submission(change):
    plan = native_plan()
    if change == "maximum":
        plan["max_tokens"] = 8192
    elif change == "sampling":
        plan["sampling"]["max_tokens"] = 8192
    elif change == "rounds":
        plan["formal_repetitions"] = 1
    elif change == "bool":
        plan["sampling"]["temperature"] = False
    else:
        plan["reference_capture_profile"] = "arbitrary"
    with pytest.raises(ValueError):
        validate_native_plan(plan)


def test_repetition_aggregation_divides_sums_and_keeps_the_slow_round():
    rows = [dict(arm([10**15, 2 * 10**15], seconds, 10), repetition=index)
            for index, seconds in enumerate((10, 100, 1000), 1)]
    result = aggregate(rows, expected_repetitions=3)
    assert result["F_interval"] == [3 * 10**15, 6 * 10**15]
    assert result["T_seconds"] == "1110"
    assert result["reference_seconds_lower"] == "30"
    assert float(result["efficiency_lower_percent"]) == pytest.approx(100 * 30 / 1110)
    for bad in (rows[:2], rows + [rows[0]], [rows[0], rows[0], rows[2]], list(reversed(rows))):
        with pytest.raises(ValueError):
            aggregate(bad, expected_repetitions=3)


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def seal(root):
    write(root, "checksums.json", {str(p.relative_to(root)): sha(p)
        for p in root.rglob("*") if p.is_file() and p.name != "checksums.json"})


def matched_fixture(root):
    """Six distinct clocks with source/hash bindings, never native observations."""
    work_proofs, graph_proofs = {}, {}
    fences = [dict(rank=r, device=r) for r in (0, 1)]
    for role, directory in (("B0", "native-baseline-timing"), ("B1", "native-current-timing")):
        native = root / directory
        write(native, "receipt.json", dict(formal_repetitions=3))
        write(native, "service/native-plan.json", native_plan())
        write(native, "model-weights.json", dict(fixture_only=True))
        timings = []
        for repetition in (1, 2, 3):
            start = repetition * 10**12
            seconds = repetition * (10 if role == "B1" else 20)
            end = start + seconds * 10**9
            ids = [f"fixture-{role}-{repetition}-{i}" for i in range(200)]
            write(native, f"service/client/formal-{repetition}/public-output-geometry.json",
                dict(requests=[dict(input_ids_sha256=str(i)) for i in range(200)]))
            write(native, f"service/client/formal-{repetition}/raw-public-output-frames.json",
                dict(start_ns=start, end_ns=end, T_seconds=seconds,
                    start_fences=fences, end_fences=fences, request_ids=ids,
                    public_step_output_only=True, http_inside_timing=False,
                    tokenizer_inside_timing=False, per_step_engine_internals_read=False,
                    per_step_disk_writes=False))
            timing = dict(repetition=repetition, start_ns=start, end_ns=end, T_seconds=seconds,
                ordinary_public_outputs_retained=True,
                boundary=[dict(rank=r, sealed=True, failure=None) for r in (0, 1)])
            timings.append(timing)
            write(native, f"service/client/formal-{repetition}-timing.json", timing)
            write(native, f"service/client/formal-{repetition}-pre-audit.json",
                [dict(rank=r, native_per_step_observer=False) for r in (0, 1)])
        write(native, "service/client/timings.json", dict(repetitions=timings))
        seal(native)
        work_proofs[role], graph_proofs[role] = dict(repetitions=[]), dict(repetitions=[])
        for repetition in (1, 2, 3):
            prefix = repetition_prefix(role, repetition)
            write(root, prefix + "-necessary-work.json", dict(F_interval=[10**15, 2 * 10**15],
                native_checksums_sha256=sha(native / "checksums.json"), geometry_sha256=prefix,
                requests=200, necessary_work_certified=True, source_runtime_applicability_certified=True))
            write(root, prefix + "-dense-source.json", dict(fixture_only=True))
            dense_sha = sha(root / (prefix + "-dense-source.json"))
            write(root, prefix + "-reference-lower-graph.json", dict(geometry_sha256=prefix,
                reference_seconds_lower=str(repetition * 5),
                source_runtime_applicability_certified=True, source_proof_sha256=dense_sha))
            write(root, prefix + "-reference-graph-audit.json", dict(graph_integrity_passed=True, reference_model_lower_seconds=str(repetition * 5)))
            work_proofs[role]["repetitions"].append(dict(repetition=repetition,
                necessary_work_sha256=sha(root / (prefix + "-necessary-work.json")), dense_source_sha256=dense_sha))
            graph_proofs[role]["repetitions"].append(dict(repetition=repetition,
                graph_sha256=sha(root / (prefix + "-reference-lower-graph.json")),
                reference_seconds_lower=str(repetition * 5),
                independent_audit_sha256=sha(root / (prefix + "-reference-graph-audit.json"))))
    write(root, "necessary-work-audit.json", dict(completed=True, arms=work_proofs))
    from reference_statistics import canonical_sha
    calibration = dict(fixture_only=True)
    write(root, "qualified-dense-calibration.json", dict(result=calibration))
    write(root, "reference-calibration-freeze.json", dict(
        calibration_result_sha256=canonical_sha(calibration), frozen_monotonic_ns=100,
        target_times_or_efficiency_used=False))
    write(root, "reference-graph-audits.json", dict(completed=True, arms=graph_proofs,
        qualified_calibration_sha256=sha(root / "qualified-dense-calibration.json")))
    return dict(capture_profile=profile())


def test_pair_uses_six_matched_clocks_without_repeating_inference(tmp_path):
    plan = matched_fixture(tmp_path)
    result = matched_pair(plan, tmp_path)
    assert result["native_formal_repetitions_per_arm"] == 3
    assert result["capture_profile"]["max_tokens"] == 256
    assert result["arms"]["B0"]["T_seconds"] == "120"
    assert result["arms"]["B1"]["T_seconds"] == "60"
    assert result["status"] == "incomplete"
    assert result["efficiency_lower_observed_percent"] is None
    assert result["milestone_met"] is None and result["final_numeric_met"] is None
    assert result["arms"]["B1"]["partial_dense_diagnostic"]["efficiency_lower_percent"] == "50"


@pytest.mark.parametrize("change", ["missing", "clock", "duplicate", "native_plan", "work", "graph", "audit"])
def test_pair_rejects_missing_changed_or_reused_submission_round(tmp_path, change):
    plan = matched_fixture(tmp_path)
    native = tmp_path / "native-current-timing"
    if change == "missing":
        path = native / "service/client/formal-3-timing.json"
        path.unlink()
    elif change == "clock":
        path = native / "service/client/formal-2-timing.json"
        value = json.loads(path.read_bytes())
        value["T_seconds"] = 1
        path.write_text(json.dumps(value))
        seal(native)
    elif change == "duplicate":
        path = native / "service/client/timings.json"
        value = json.loads(path.read_bytes())
        value["repetitions"][2]["repetition"] = 2
        path.write_text(json.dumps(value))
        seal(native)
    elif change == "native_plan":
        write(native, "service/native-plan.json", native_plan("full-answer-v1"))
        seal(native)
    elif change == "work":
        write(tmp_path, "B1-r002-necessary-work.json", dict(F_interval=[1, 2]))
    elif change == "graph":
        write(tmp_path, "B1-r003-reference-lower-graph.json", dict(reference_seconds_lower="10000"))
    else:
        write(tmp_path, "B1-r003-reference-graph-audit.json", dict(graph_integrity_passed=False))
    with pytest.raises((ValueError, FileNotFoundError)):
        matched_pair(plan, tmp_path)


def test_failed_current_reference_never_borrows_a_historical_pass(tmp_path, monkeypatch):
    import project_evidence
    import reuse_campaign
    from third_party_report import inspect

    historical = dict(formal_targets={"3.4.1": dict(accepted=True, display_value="59.7573%", reason="historical")})
    monkeypatch.setattr(project_evidence, "ensure", lambda: tmp_path / "history")
    monkeypatch.setattr(project_evidence, "assess", lambda root: historical)
    monkeypatch.setattr(reuse_campaign, "load_report", lambda *args: ({}, {}))
    write(tmp_path, "campaign.json", dict(status="failed", stages=[],
        compute_reference=dict(status="blocked", blockers=["third repetition absent"])))
    write(tmp_path, "campaign-checksums.json", {"campaign.json": sha(tmp_path / "campaign.json")})
    report = inspect(tmp_path)
    assert report["existing_project_evidence"]["formal_targets"]["3.4.1"]["accepted"]
    assert report["formal_targets"]["3.4.1"]["accepted"] is False
    assert report["formal_targets"]["3.4.1"]["measured_value"] is None
    assert "third repetition absent" in report["formal_targets"]["3.4.1"]["reason"]


def test_post_target_calibration_cannot_qualify_a_submission_pair(tmp_path):
    plan = matched_fixture(tmp_path)
    path = tmp_path / "reference-calibration-freeze.json"
    freeze = json.loads(path.read_bytes())
    freeze["frozen_monotonic_ns"] = 100 * 10**12
    path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="predates"):
        matched_pair(plan, tmp_path)
