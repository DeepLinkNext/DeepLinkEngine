import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import automatic_reference as automatic
import reference_model_reuse as reuse
from compute_reference import execute, prepare
from dataset_contract import sha
from test_reference_qualification_reuse import prior
from test_automatic_reference import fake_stage


def test_completed_sealed_model_stages_resume_all_four_providers_without_model_rerun(
    tmp_path, monkeypatch
):
    args, frozen, out, source = prior(tmp_path, monkeypatch)
    args.reference_resume_source = source
    # This is an isolated CPU orchestration fixture; the source/receipt loader
    # is real, and only the device lease operation is replaced.
    monkeypatch.setattr(
        reuse,
        "guarded_load",
        lambda s, pin, plan, root, guard: reuse.load(s, pin, plan, root),
    )
    admission = prepare(args, out, frozen)
    assert admission["status"] == "automatic_ready"
    calls = []

    def launch(command, **kwargs):
        calls.append(command)
        return fake_stage(command, **kwargs)

    monkeypatch.setattr(automatic, "run_stage", launch)
    result = execute(out, admission)
    assert result["status"] == "measured"
    assert len(calls) == 4
    assert all(
        any(part.endswith("reference_final_stages.py") for part in call)
        for call in calls
    )
    models = [row for row in result["stages"] if row["id"] in reuse.MODEL_STAGES]
    assert len(models) == 3 and all(row["reused_model_stage"] for row in models)
    assert result["new_reference_inference_executed"] is False


@pytest.mark.parametrize(
    "change", ["artifact", "clock_pin", "workload", "stage_controls"]
)
def test_changed_native_source_or_workload_never_reuses_a_clock(
    tmp_path, monkeypatch, change
):
    args, frozen, out, source = prior(tmp_path, monkeypatch)
    plan = json.loads((source / "automatic-plan.json").read_bytes())
    current = copy.deepcopy(plan)
    if change == "artifact":
        (source / "native-current-timing/receipt.json").write_text("{}")
    elif change == "clock_pin":
        key = next(
            name
            for name in current["files"]
            if name.endswith("scripts/native_public_client.py")
        )
        current["files"][key] = "0" * 64
    elif change == "workload":
        current["binding"]["quality_plan_sha256"] = "0" * 64
    else:
        next(
            row
            for row in current["stages"]
            if row["id"] == "current-native-synchronized-timing"
        )["scope"] = "Changed native warmup/control domain"
    with pytest.raises(ValueError):
        reuse.load(source, sha(source / "checksums.json"), current, automatic.ROOT)


def test_downstream_report_changes_do_not_force_a_native_model_rerun(
    tmp_path, monkeypatch
):
    args, frozen, out, source = prior(tmp_path, monkeypatch)
    current = json.loads((source / "automatic-plan.json").read_bytes())
    key = next(
        name
        for name in current["files"]
        if name.endswith("scripts/third_party_report.py")
    )
    current["files"][key] = "0" * 64
    admitted = reuse.load(
        source, sha(source / "checksums.json"), current, automatic.ROOT
    )
    assert admitted["source"] == str(source.resolve())


def test_closed_failed_http_needs_new_guarded_independent_derivation(
    tmp_path, monkeypatch
):
    args, frozen, out, source = prior(tmp_path, monkeypatch)
    from dataset_contract import save

    path = source / "http-current-work/receipt.json"
    receipt = json.loads(path.read_bytes())
    receipt.update(
        completed=False,
        quiescent=True,
        model_service_started=True,
        client_started=True,
        failure="ValueError: Draft consumer reused or mismatched its latest actual producer",
    )
    save(path, receipt)
    nested = path.parent / "checksums.json"
    seal = json.loads(nested.read_bytes())
    seal["receipt.json"] = sha(path)
    save(nested, seal)
    result = json.loads((source / "automatic-result.json").read_bytes())
    next(r for r in result["stages"] if r["id"] == "current-http-work-observation")[
        "status"
    ] = "failed"
    save(source / "automatic-result.json", result)
    save(
        source / "checksums.json",
        {
            str(p.relative_to(source)): sha(p)
            for p in source.rglob("*")
            if p.is_file() and p != source / "checksums.json"
        },
    )
    plan = json.loads((source / "automatic-plan.json").read_bytes())
    original = path.read_bytes()
    with pytest.raises(ValueError, match="independent guarded audit"):
        reuse.load(source, sha(source / "checksums.json"), plan, automatic.ROOT)
    import audit_http_reference

    calls = []

    def audit(origin, deployment, directory):
        calls.append((origin, deployment, directory))
        directory.mkdir()
        save(directory / "checksums.json", {})
        return dict(completed=True, original_completed=False)

    monkeypatch.setattr(audit_http_reference, "run", audit)
    recovery = tmp_path / "new-independent-audit"
    resumed = reuse.load(
        source, sha(source / "checksums.json"), plan, automatic.ROOT, recovery
    )
    assert len(calls) == 1
    row = resumed["stages"]["current-http-work-observation"]
    assert row["recovered_post_capture_cpu_derivation"] is True
    assert row["original_receipt_completed"] is False
    assert row["receipt_sha256"] == sha(path)
    assert path.read_bytes() == original
