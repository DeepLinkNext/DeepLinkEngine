"""Exercise default reference orchestration; these fixtures never load an NPU."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / name) for name in ("scripts", "third-party-test-kit")]
import automatic_reference as automatic
from compute_reference import prepare, execute, target
from dataset_contract import save, sha


def fixture(tmp_path):
    sdk = tmp_path / "sdk"
    source = sdk / "mods/betterscale/betterscale/models/qwen35/decode_kv.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Explicit CPU test fixture; never executed\n")
    (sdk / "provenance").mkdir()
    paths = [source, *automatic.prefill_sources(sdk).values()]
    base = sdk / "mods/betterscale/betterscale"
    paths += [base / "patches/qwen_gdn" / name for name in ("h_tiling.h", "o_tiling.h", "solve_tril.py")]
    paths += [base / "models/qwen35" / name for name in ("wy_head_major.py", "mixed_layout.py")]
    paths += [sdk / "runtime/ascend/vllm_ascend/ops/triton/fla" / name for name in ("solve_tril.py", "chunk_scaled_dot_kkt.py")]
    for p in paths[1:]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# Explicit CPU fixture; no native code\n")
    save(sdk / "provenance/installed-files.json", {str(p.relative_to(sdk)): sha(p) for p in paths})
    save(sdk / "deployment.json", dict(release="fixture-r8", python=sys.executable, pythonpath=[]))
    frozen = dict(optimized_deployment=str(sdk), baseline_deployment=str(tmp_path / "baseline-sdk"), engine_release="fixture-r8", model="/fixture/model", physical_cards=[1, 6], logical_devices="0,1", mmlu_inputs=dict(quality_plan_sha256="a" * 64), datasets=["mmlu"])
    args = SimpleNamespace(compute_reference_plan=None, compute_reference_sha256=None, performance_only=False)
    out = tmp_path / "run"
    out.mkdir()
    save(out / "frozen-stages.json", [dict(name="mmlu-pair", args=["--qualified-continuous", str(tmp_path / "fixture-continuous")])])
    return args, frozen, out


def fake_stage(command, **kwargs):
    if any(part.endswith("reference_final_stages.py") for part in command):
        folder = Path(command[command.index("--folder") + 1])
        phase = command[command.index("--stage") + 1]
        value = dict(orchestration_fixture_only=True)
        if phase == "calibration":
            from full_reference_reuse import DOMAIN_SHA256

            value.update(applicability_protocol_sha256=DOMAIN_SHA256,
                original_method_frozen_unix=1, target_times_used=False)
            save(folder / "qualified-full-reference-statistics.json", value)
        if phase == "matched":
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
            value.update(pair(arm_statistics(rows, original("0.5")), arm_statistics(rows, original("0.55"))))
            value.update(status="measured", full_model_certificate=True,
                certificate_kind="qualified_full_reference_denominator_reuse/v1",
                full_reference_denominator_qualified=True, original_reference_work_and_time_retained=True,
                original_efficiency_used=False, shared_work_cancellation_used=False,
                capture_profile=dict(max_tokens=256, formal_repetitions=3), current_efficiency=None)
        save(folder / ("final-stage-" + phase + ".json"), dict(completed=True, result=value))
        return SimpleNamespace(returncode=0)
    folder = Path(command[command.index("--out") + 1])
    folder.mkdir()
    if any(part.endswith("run_native_reference.py") for part in command):
        save(folder / "receipt.json", dict(completed=True, quiescent=True, preflight_only=False,
             role="B0" if "--baseline-deployment" in command else "B1", formal_repetitions=int(command[command.index("--formal-repetitions") + 1]), necessary_work_certified=False, efficiency_percent=None, fixture_only=True))
        save(folder / "device-mapping.json", dict(physical_cards=[1, 6], logical_cards=[0, 1]))
        save(folder / "checksums.json", {p.name: sha(p) for p in folder.iterdir() if p.is_file()})
        return SimpleNamespace(returncode=0)
    if "--reference-observation" in command:
        (folder / "service/raw-work").mkdir(parents=True)
        save(folder / "receipt.json", dict(role="B1", completed=True, quiescent=True, raw_observation=dict(completed=True)))
        save(folder / "service/raw-work/raw-partial-work.json", dict(
            effective_flops=None, worker_schedule_matches=True,
            moe_routes=dict(peer_routes_match=True),
            requests=[dict(covered=True, retained=1, output_tokens=1) for _ in range(200)]))
        save(folder / "device-mapping.json", dict(physical_cards=[1, 6], logical_cards=[0, 1]))
        save(folder / "checksums.json", {str(p.relative_to(folder)): sha(p) for p in folder.rglob("*") if p.is_file()})
        return SimpleNamespace(returncode=0)
    planfile = Path(command[command.index("--plan") + 1])
    plan = json.loads(planfile.read_bytes())
    if command[1].endswith("qualify_route_capture.py"):
        cases = [dict(case=case, physical_card=card, passed=True, disabled_untouched=True,
                      overflow_bounded=True, input_unchanged=True)
                 for card in plan["physical_cards"] for case in plan["route_cases"]]
        save(folder / "receipt.json", dict(complete=True, plan_sha256=sha(planfile), cases=cases, fixture_only=True))
        return SimpleNamespace(returncode=0)
    key = "prefill_cases" if command[1].endswith("qualify_gdn_prefill_reference.py") else "cases"
    cases = []
    for card in plan["physical_cards"]:
        for ordinal, case in enumerate(plan[key]):
            samples = [dict(measurement_id=f"fixture-{key}-{card}-{ordinal}-{i}",
                physical_card=card, start_ns=i * 10, end_ns=i * 10 + 1,
                group="calibration" if i < plan["calibration_repetitions"] else "held_out_validation")
                for i in range(plan["calibration_repetitions"] + plan["validation_repetitions"])]
            cases.append(dict(case=case, physical_card=card, numerical_passed=True,
                              output_max_abs=0.0, state_max_abs=0.0, samples=samples))
    save(folder / "receipt.json", dict(complete=True, plan_sha256=sha(planfile), cases=cases, fixture_only=True))
    return SimpleNamespace(returncode=0)


def test_default_builds_and_executes_automatic_stages_without_manual_plan(tmp_path, monkeypatch):
    args, frozen, out = fixture(tmp_path)
    admitted = prepare(args, out, frozen)
    assert admitted["status"] == "automatic_ready"
    assert admitted["binding"]["max_completion_tokens"] == 8192
    assert admitted["historical_substitution_allowed"] is False
    planned = json.loads((out / admitted["plan"]).read_bytes())
    assert planned["capture_profile"]["id"] == "submission-v2"
    assert planned["capture_profile"]["max_tokens"] == 256
    assert planned["capture_profile"]["formal_repetitions"] == 3
    assert planned["reference_policy"]["certified_lower_bound_allowed"] is True
    assert planned["reference_policy"]["separate_executable_whole_chain_required"] is False
    assert planned["reference_policy"]["unknown_duration_replacement_by_zero_allowed"] is False
    assert "independent-whole-chain-validation" not in [stage["id"] for stage in planned["stages"]]
    called = []
    def stage(command, **kwargs):
        called.append(command)
        return fake_stage(command, **kwargs)
    monkeypatch.setattr(automatic, "run_stage", stage)
    import engine_entry
    monkeypatch.setattr(engine_entry, "environment", lambda *args: {})
    result = execute(out, admitted)
    assert len(called) == 10 and called[0][1].endswith("qualify_gdn_reference.py")
    assert called[1][1].endswith("qualify_gdn_prefill_reference.py")
    assert result["status"] == "measured"
    assert result["stages"][0]["status"] == "completed"
    assert result["stages"][1]["status"] == "completed"
    stages = {row["id"]: row for row in result["stages"]}
    assert stages["current-native-synchronized-timing"]["numerator_pending_necessary_work_correspondence"] is True
    native = [command for command in called if any(part.endswith("run_native_reference.py") for part in command)]
    assert len(native) == 2
    assert "--baseline-deployment" not in native[0] and "--baseline-deployment" in native[1]
    assert all(command[command.index("--formal-repetitions") + 1] == "3" for command in native)
    assert result["blockers"] == []
    assert [call[call.index("--stage") + 1] for call in called if "--stage" in call] == ["calibration", "necessary", "graphs", "matched"]
    assert target(result)["accepted"]
    assert target(result)["display_value"] == "≥55%"
    assert target(result)["value_kind"] == "certified_lower_bound"
    assert result["stages"][-1]["status"] == "completed"
    assert result["operator_reference_executed"] is True
    assert result["new_reference_inference_executed"] is True
    assert (out / "compute-reference/automatic-result.json").is_file()


@pytest.mark.parametrize("provider_started", [False, True])
def test_missing_source_certificate_after_successful_model_clocks_cannot_produce_efficiency(
    tmp_path, monkeypatch, provider_started
):
    args, frozen, out = fixture(tmp_path)
    args.reference_profile = "full-answer-v1"
    def stage(command, **kwargs):
        if "--stage" in command:
            if provider_started:
                # Fail the real provider's first check without taking a device lease.
                Path(command[command.index("--plan") + 1]).write_text("{}")
                return subprocess.run(command, **kwargs)
            return SimpleNamespace(returncode=2)
        return fake_stage(command, **kwargs)
    monkeypatch.setattr(automatic, "run_stage", stage)
    import engine_entry
    monkeypatch.setattr(engine_entry, "environment", lambda *args: {})
    result = execute(out, prepare(args, out, frozen))
    assert all(row["status"] == "completed" for row in result["stages"][:6])
    assert result["status"] == "blocked"
    assert not result["full_model_certificate"]
    assert result["efficiency_lower_observed_percent"] is None
    assert not target(result)["accepted"]
    failed, = [row for row in result["stages"] if row["status"] == "failed"]
    assert failed["id"] == "complete-current-necessary-work"
    assert sha(out / failed["log"]) == failed["log_sha256"]
    if provider_started:
        assert sha(out / failed["receipt"]) == failed["receipt_sha256"]
        assert "Final reference provider plan differs from the automatic admission" in result["blockers"][0]
    else:
        assert "exit 2" in result["blockers"][0]


def test_performance_only_does_not_schedule_operator_reference(tmp_path):
    args, frozen, out = fixture(tmp_path)
    args.performance_only = True
    admitted = prepare(args, out, frozen)
    assert admitted["status"] == "not_requested"
    assert not (out / "plans/compute-reference/automatic-plan.json").exists()


def test_route_recorder_failure_stops_before_reference_model_start(tmp_path, monkeypatch):
    args, frozen, out = fixture(tmp_path)
    admitted = prepare(args, out, frozen)
    called = []
    def stage(command, **kwargs):
        called.append(command)
        result = fake_stage(command, **kwargs)
        if command[1].endswith("qualify_route_capture.py"):
            path = Path(command[command.index("--out") + 1]) / "receipt.json"
            receipt = json.loads(path.read_bytes())
            receipt["cases"][0]["overflow_bounded"] = False
            save(path, receipt)
        return result
    monkeypatch.setattr(automatic, "run_stage", stage)
    import engine_entry
    monkeypatch.setattr(engine_entry, "environment", lambda *args: {})
    assert execute(out, admitted)["status"] == "blocked"
    assert len(called) == 3 and all("--reference-observation" not in call for call in called)


@pytest.mark.parametrize("change", ["runtime", "target", "plan"])
def test_changed_bound_input_stops_before_runtime_environment(tmp_path, monkeypatch, change):
    args, frozen, out = fixture(tmp_path)
    admitted = prepare(args, out, frozen)
    planfile = out / admitted["plan"]
    plan = json.loads(planfile.read_bytes())
    if change == "runtime":
        Path(frozen["optimized_deployment"], "deployment.json").write_text("{}")
    elif change == "target":
        Path(plan["target"]).write_text("# changed source\n")
    else:
        planfile.write_text("{}")
    import engine_entry
    monkeypatch.setattr(engine_entry, "environment", lambda *args: pytest.fail("NPU environment must not start"))
    assert execute(out, admitted)["status"] == "blocked"


def test_duplicate_operator_cases_cannot_claim_success(tmp_path, monkeypatch):
    args, frozen, out = fixture(tmp_path)
    admitted = prepare(args, out, frozen)
    def duplicate(command, **kwargs):
        result = fake_stage(command, **kwargs)
        path = Path(command[command.index("--out") + 1]) / "receipt.json"
        receipt = json.loads(path.read_bytes())
        receipt["cases"][-1] = receipt["cases"][0]
        save(path, receipt)
        return result
    monkeypatch.setattr(automatic, "run_stage", duplicate)
    import engine_entry
    monkeypatch.setattr(engine_entry, "environment", lambda *args: {})
    assert execute(out, admitted)["status"] == "blocked"


def test_unknown_release_or_unbound_target_rejected_during_prepare(tmp_path):
    args, frozen, out = fixture(tmp_path)
    frozen["engine_release"] = "another-release"
    assert prepare(args, out, frozen)["status"] == "blocked"


@pytest.mark.parametrize("mutation", ["missing_sample", "reused_id", "negative_duration", "wrong_card", "nan_error"])
def test_invalid_operator_samples_cannot_complete_a_reference_stage(tmp_path, monkeypatch, mutation):
    args, frozen, out = fixture(tmp_path)
    admitted = prepare(args, out, frozen)
    def invalid(command, **kwargs):
        result = fake_stage(command, **kwargs)
        path = Path(command[command.index("--out") + 1]) / "receipt.json"
        receipt = json.loads(path.read_bytes())
        samples = receipt["cases"][0]["samples"]
        if mutation == "missing_sample":
            samples.pop()
        elif mutation == "reused_id":
            samples[0]["measurement_id"] = samples[1]["measurement_id"]
        elif mutation == "negative_duration":
            samples[0]["end_ns"] = samples[0]["start_ns"] - 1
        elif mutation == "wrong_card":
            samples[0]["physical_card"] = 99
        else:
            receipt["cases"][0]["output_max_abs"] = float("nan")
        save(path, receipt)
        return result
    monkeypatch.setattr(automatic, "run_stage", invalid)
    import engine_entry
    monkeypatch.setattr(engine_entry, "environment", lambda *args: {})
    assert execute(out, admitted)["status"] == "blocked"


def test_instrumented_capture_failure_keeps_both_completed_native_arms(tmp_path, monkeypatch):
    args, frozen, out = fixture(tmp_path)
    admitted = prepare(args, out, frozen)
    def stage(command, **kwargs):
        value = fake_stage(command, **kwargs)
        if any(str(part).endswith("run_b1_r4.py") for part in command):
            return SimpleNamespace(returncode=2)
        return value
    monkeypatch.setattr(automatic, "run_stage", stage)
    import engine_entry
    monkeypatch.setattr(engine_entry, "environment", lambda *args: {})
    result = execute(out, admitted)
    stages = {stage["id"]: stage for stage in result["stages"]}
    assert result["status"] == "blocked"
    assert stages["current-native-synchronized-timing"]["status"] == "completed"
    assert stages["baseline-native-synchronized-timing"]["status"] == "completed"
    assert stages["current-http-work-observation"]["status"] == "failed"
    assert result["current_efficiency"] is None


def test_single_round_child_cannot_complete_a_three_round_submission(tmp_path, monkeypatch):
    args, frozen, out = fixture(tmp_path)
    def incomplete(command, **kwargs):
        result = fake_stage(command, **kwargs)
        if any(part.endswith("run_native_reference.py") for part in command):
            folder = Path(command[command.index("--out") + 1])
            receipt = json.loads((folder / "receipt.json").read_bytes())
            receipt["formal_repetitions"] = 1
            (folder / "receipt.json").write_text(json.dumps(receipt))
        return result
    monkeypatch.setattr(automatic, "run_stage", incomplete)
    import engine_entry
    monkeypatch.setattr(engine_entry, "environment", lambda *a: {})
    result = execute(out, prepare(args, out, frozen))
    assert result["status"] == "blocked"
    assert result["full_model_certificate"] is False
    assert result["stages"][-1]["status"] == "failed"


def test_submission_freezes_independent_statistics_before_formal_native_clocks(tmp_path, monkeypatch):
    args, frozen, out = fixture(tmp_path)
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        if any(part.endswith("run_native_reference.py") for part in command):
            assert (out / "compute-reference/reference-calibration-freeze.json").exists()
        return fake_stage(command, **kwargs)
    monkeypatch.setattr(automatic, "run_stage", launch)
    import engine_entry
    monkeypatch.setattr(engine_entry, "environment", lambda *a: {})
    result = execute(out, prepare(args, out, frozen))
    assert result["status"] == "measured"
    calibrated = next(i for i, c in enumerate(calls) if "--stage" in c and c[c.index("--stage") + 1] == "calibration")
    measured = [i for i, c in enumerate(calls) if any(p.endswith("run_native_reference.py") for p in c)]
    assert len(measured) == 2 and all(calibrated < i for i in measured)


def test_controller_rejects_retired_certificate_even_if_child_claims_complete(tmp_path, monkeypatch):
    args, frozen, out = fixture(tmp_path)
    def launch(command, **kwargs):
        execution = fake_stage(command, **kwargs)
        if "--stage" in command and command[command.index("--stage") + 1] == "matched":
            folder = Path(command[command.index("--folder") + 1])
            path = folder / "final-stage-matched.json"
            value = json.loads(path.read_bytes())
            value["result"].update(status="measured", full_model_certificate=True,
                certificate_kind="matched_complete_necessary_work_and_full_reference_lower_bound")
            save(path, value)
        return execution
    monkeypatch.setattr(automatic, "run_stage", launch)
    import engine_entry
    monkeypatch.setattr(engine_entry, "environment", lambda *a: {})
    result = execute(out, prepare(args, out, frozen))
    assert result["status"] == "blocked"
    assert result["full_model_certificate"] is False
    assert result["efficiency_lower_observed_percent"] is None
    assert "Withdrawn" in result["blockers"][0]
