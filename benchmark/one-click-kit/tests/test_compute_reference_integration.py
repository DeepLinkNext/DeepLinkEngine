"""Real file/CLI admission checks; these do not supply NPU benchmark results."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("scripts", "third-party-test-kit")]
from compute_reference import binding, execute, prepare, reference_only, target  # noqa: E402
from dataset_contract import save, sha  # noqa: E402


def workload():
    return {
        "mmlu_inputs": {"quality_plan_sha256": "a" * 64},
        "model": "/models/real-model-required",
        "physical_cards": [1, 6],
        "engine_release": "full-answer-b1-20261113-r6",
        "datasets": ["mmlu"],
    }


def args(root):
    return SimpleNamespace(
        cache=root,
        compute_reference_plan=None,
        compute_reference_sha256=None,
        out=root / "new",
        reference_campaign=None,
    )


def test_automatic_reference_without_its_campaign_sdk_is_blocked(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    admission = prepare(args(tmp_path), out, workload(), force_live=True)
    assert admission["status"] == "blocked"
    assert "自动参考" in admission["blockers"][0]
    result = execute(out, admission)
    assert target(result)["accepted"] is False
    assert not (out / "compute-reference/native").exists()
    seals = json.loads((out / "compute-reference/checksums.json").read_bytes())
    assert seals["status.json"] == sha(out / "compute-reference/status.json")


def test_wrong_plan_digest_stops_before_any_collector(tmp_path):
    plan = tmp_path / "plan.json"
    save(plan, {})
    a = args(tmp_path)
    a.compute_reference_plan, a.compute_reference_sha256 = plan, "b" * 64
    out = tmp_path / "out"
    out.mkdir()
    admission = prepare(a, out, workload())
    assert admission["status"] == "blocked"
    assert "SHA256" in admission["blockers"][0]


def test_runtime_binding_mismatch_is_rejected(tmp_path):
    plan = tmp_path / "plan.json"
    save(
        plan,
        {"compute_reference_binding": dict(binding(workload()), physical_cards=[2, 3])},
    )
    a = args(tmp_path)
    a.compute_reference_plan, a.compute_reference_sha256 = plan, sha(plan)
    out = tmp_path / "out"
    out.mkdir()
    assert "身份不一致" in prepare(a, out, workload())["blockers"][0]


def test_archive_cannot_supply_live_reference(tmp_path):
    plan = tmp_path / "plan.json"
    save(
        plan,
        {
            "compute_reference_binding": binding(workload()),
            "files": {},
            "cases": [
                {
                    "kind": "compute",
                    "id": "mmlu",
                    "arms": {"B0": {"reuse": []}, "B1": {"reuse": []}},
                }
            ],
        },
    )
    a = args(tmp_path)
    a.compute_reference_plan, a.compute_reference_sha256 = plan, sha(plan)
    out = tmp_path / "out"
    out.mkdir()
    assert "禁止复用旧参考" in prepare(a, out, workload())["blockers"][0]


def test_reference_only_preserves_parent_and_does_not_repeat_performance(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    save(parent / "frozen-execution-plan.json", workload())
    save(parent / "campaign.json", {"status": "completed"})
    save(
        parent / "campaign-checksums.json",
        {
            name: sha(parent / name)
            for name in ["frozen-execution-plan.json", "campaign.json"]
        },
    )
    before = {p.name: p.read_bytes() for p in parent.iterdir()}
    a = args(tmp_path)
    a.reference_campaign = parent
    assert reference_only(a) == 2
    assert {p.name: p.read_bytes() for p in parent.iterdir()} == before
    assert not (a.out / "mmlu-pair").exists()
    assert (
        json.loads((a.out / "source-campaign.json").read_bytes())[
            "performance_inference_rerun"
        ]
        is False
    )
    with pytest.raises(FileExistsError):
        reference_only(a)


def test_modified_parent_is_rejected_before_new_run(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    save(parent / "campaign-checksums.json", {"frozen-execution-plan.json": "c" * 64})
    save(parent / "frozen-execution-plan.json", workload())
    a = args(tmp_path)
    a.reference_campaign = parent
    with pytest.raises(ValueError, match="绑定改变"):
        reference_only(a)
    assert not a.out.exists()


def test_public_entry_exposes_reference_workflow():
    result = subprocess.run(
        [sys.executable, str(ROOT / "one_click.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--compute-reference-plan" in result.stdout
    assert "--compute-reference-only" in result.stdout
    assert "--reference-campaign" in result.stdout


def test_copied_collector_preserves_native_module_path_identity(tmp_path):
    import shutil
    from compute_reference import KIT

    copied = tmp_path / "copied"
    copied.mkdir()
    for source in KIT.glob("*.py"):
        shutil.copyfile(source, copied / source.name)
    save(copied / "workload.json", {})
    files = {p.name: sha(p) for p in copied.iterdir()}
    profile = dict(
        workload="workload.json",
        resource_guard="resource_guard.py",
        target_backend="vllm_reference_backend.py",
        reference_backend="vllm_reference_backend.py",
        target_accounting="native_m01_accounting.py",
    )
    save(copied / "fixture.json", dict(files=files, profile=profile))
    code = (
        "import json,sys;from pathlib import Path;"
        "sys.path.insert(0,sys.argv[1]);import live_reference_capture as c;"
        'd=json.loads((Path(sys.argv[1])/"fixture.json").read_bytes());'
        'c.frozen_files(Path(sys.argv[1]),d["files"],d["profile"])'
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(copied)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "full M01 scoring needs verified arithmetic" in result.stderr
    assert "Freeze the actual shipped native module" not in result.stderr


def test_live_console_follows_reference_after_performance_completes(tmp_path):
    from live_console import Follow, progress
    import io

    save(
        tmp_path / "execution-checklist.json",
        dict(
            status="completed",
            stages=[dict(name="terminal-trace-B1", status="completed")],
        ),
    )
    reference = tmp_path / "compute-reference"
    reference.mkdir()
    save(reference / "status.json", dict(status="running"))
    phase = reference / "native/mmlu/r001/B1/timing"
    phase.mkdir(parents=True)
    (phase / "collector.log").write_text("real collector output is followed here\n")
    logdir = tmp_path / "logs"
    logdir.mkdir()
    stream = io.StringIO()
    follow = Follow(tmp_path, logdir, stream=stream, persist=False)
    follow.tick()
    assert progress(tmp_path)["stage"] == "compute-reference"
    assert progress(tmp_path)["phase"] == "running:timing"
    assert "real collector output is followed here" in stream.getvalue()


def test_default_cannot_substitute_historical_cache_for_missing_current_work(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    admission = prepare(args(tmp_path), out, workload())
    assert admission["status"] == "blocked"
    assert admission["binding"]["max_completion_tokens"] == 8192
    result = execute(out, admission)
    qualified = target(result)
    assert qualified["accepted"] is False
    assert qualified["measured_value"] is None
    assert admission["historical_substitution_allowed"] is False
    assert not (out / "compute-reference/native").exists()
    assert not (out / "compute-reference/collector.log").exists()


def test_changed_qualified_source_cannot_be_reused(tmp_path):
    source = tmp_path / "invalid-evidence"
    source.mkdir()
    (source / "source-readback.json").write_text("{}")
    a = args(tmp_path)
    a.project_evidence = source
    out = tmp_path / "out"
    out.mkdir()
    admission = prepare(a, out, workload())
    assert admission["status"] == "blocked"
    assert "自动参考" in admission["blockers"][0]
    assert target(admission)["accepted"] is False


def qualified_bound(reference_seconds="0.75"):
    from decimal import Decimal, localcontext
    from full_reference_reuse import arm_statistics
    from reference_pair_statistics import pair

    rows = [dict(repetition=i, F_interval=[10**12, 2*10**12], T_seconds="1") for i in range(1, 4)]
    def original(seconds):
        with localcontext() as context:
            context.prec = 40
            elapsed = Decimal(seconds) * 3
            upper = Decimal(3*10**12) / (2 * elapsed * 10**12)
        return dict(effective_flops_interval=[3*10**12, 3*10**12],
            sum_reference_seconds_lower=str(elapsed), reference_tflops_per_card_upper=str(upper),
            reference_repetitions=[dict(name="original-"+str(i)) for i in range(1, 4)])
    value = pair(arm_statistics(rows, original("0.5")), arm_statistics(rows, original(reference_seconds)))
    value.update(status="measured", full_model_certificate=True,
        certificate_kind="qualified_full_reference_denominator_reuse/v1",
        full_reference_denominator_qualified=True, original_reference_work_and_time_retained=True,
        original_efficiency_used=False, shared_work_cancellation_used=False,
        capture_profile=dict(max_tokens=256, formal_repetitions=3))
    return value


def test_qualified_report_labels_proven_bound_and_preserves_exact_unknown():
    from compute_reference import reference_report
    value = qualified_bound()
    assert target(value)["display_value"] == "≥75%"
    assert target(value)["value_kind"] == "certified_lower_bound"
    report = reference_report(value)
    assert "≥75%" in report and "不能据此判定" in report
    assert "matched-reference-pair.json" in report
    assert value["arms"]["B1"]["efficiency_exact_percent"] is None
    assert "置信区间" in report


def test_incomplete_or_unqualified_bound_never_displays_partial_efficiency():
    from compute_reference import reference_report
    for status, complete in [("blocked", True), ("measured", False)]:
        value = qualified_bound()
        value.update(status=status, full_model_certificate=complete, blockers=["<missing & unsafe>"])
        assert target(value)["measured_value"] is None
        report = reference_report(value)
        assert "<table>" not in report and "≥75%" not in report
        assert "&lt;missing &amp; unsafe&gt;" in report


def test_display_rounding_cannot_turn_a_bound_below_fifty_into_fifty():
    from compute_reference import reference_report, bound_display
    value = qualified_bound("0.49999999999999999999")
    # Binary float alone loses this threshold distinction.
    assert value["efficiency_lower_observed_percent"] == 50
    assert value["milestone_met"] is False
    displayed = target(value)
    assert displayed["display_value"] == "≥49.9999%" and displayed["accepted"] is False
    assert displayed["measured_value"] < 50
    assert "≥49.9999%" in reference_report(value)
    assert bound_display("1.00000001", upper=True) == "1.0001"


def test_incomplete_arm_statistics_cannot_claim_a_measured_reference():
    value = qualified_bound()
    del value["arms"]["B0"]
    assert target(value)["measured_value"] is None
    assert target(value)["accepted"] is False
    assert "完整B0/B1统计" in target(value)["reason"]
