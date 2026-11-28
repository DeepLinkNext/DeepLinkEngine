"""Plan and run shipped reference-certification stages from the one-click entry.

Incomplete families remain explicit blockers; a passed primitive never creates
a full-model efficiency. No user-authored live-reference plan is required.
"""

import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from child_runner import run_stage
from dataset_contract import save, sha

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "third-party-test-kit"
PROTOCOL = ROOT / "protocols/automatic-compute-reference-v1.json"


def prefill_cases():
    cases = [dict(segments=[lengths], initial=[initial], decay=decay)
             for lengths, initial, decay in (
                 ([1], False, "normal"), ([17], False, "normal"),
                 ([64], False, "normal"), ([65], False, "normal"),
                 ([33, 32], False, "normal"), ([17, 65, 47], True, "normal"),
                 ([64], True, "strong"), ([1, 63, 1], True, "near_zero"))]
    cases.append(dict(segments=[[17], [65], [129]], initial=[False, True, False], decay="normal"))
    return cases


def prefill_sources(deployment):
    base = deployment / "mods/betterscale/betterscale"
    return dict(
        prefill_target=base / "models/qwen35/chunk_layout.py",
        prefill_runtime=base / "patches/qwen_gdn/runtime.py",
        gdn_kernel=deployment / "runtime/betterscale-distribution/betterscale/patches/qwen_gdn/libbs_gdn.so",
        gdn_host=deployment / "native-libs/libbs_gdn_host_cann91.so")


def verify_qualification(receipt, plan, plan_path, cases_key):
    expected = sorted((card, json.dumps(case, sort_keys=True)) for card in plan["physical_cards"] for case in plan[cases_key])
    actual = sorted((row["physical_card"], json.dumps(row["case"], sort_keys=True)) for row in receipt["cases"])
    if receipt["complete"] is not True or receipt["plan_sha256"] != sha(plan_path) or actual != expected:
        raise ValueError("Incomplete, duplicate or foreign automatic reference cases")
    seen = set()
    for row in receipt["cases"]:
        if (row["numerical_passed"] is not True
                or any(type(row[key]) not in (float, int) or not math.isfinite(row[key]) or row[key] < 0
                       for key in ("output_max_abs", "state_max_abs"))):
            raise ValueError("Invalid automatic reference numerical receipt")
        groups = dict(calibration=0, held_out_validation=0)
        for sample in row["samples"]:
            identity = sample["measurement_id"]
            if (not isinstance(identity, str) or not identity or identity in seen
                    or sample["physical_card"] != row["physical_card"]
                    or sample["group"] not in groups
                    or type(sample["start_ns"]) is not int or sample["start_ns"] < 0
                    or type(sample["end_ns"]) is not int or sample["end_ns"] <= sample["start_ns"]):
                raise ValueError("Invalid or reused automatic reference timing sample")
            seen.add(identity)
            groups[sample["group"]] += 1
        if groups != dict(calibration=plan["calibration_repetitions"], held_out_validation=plan["validation_repetitions"]):
            raise ValueError("Automatic reference calibration/held-out sample population differs")


def verify_route_qualification(receipt, plan, plan_path):
    expected = sorted((card, json.dumps(case, sort_keys=True)) for card in plan["physical_cards"] for case in plan["route_cases"])
    actual = sorted((case["physical_card"], json.dumps(case["case"], sort_keys=True)) for case in receipt["cases"])
    if (receipt["complete"] is not True or receipt["plan_sha256"] != sha(plan_path)
            or actual != expected or any(not all(case.get(key) is True for key in
                ("passed", "disabled_untouched", "overflow_bounded", "input_unchanged")) for case in receipt["cases"])):
        raise ValueError("Raw route-recorder graph-replay qualification failed")


def prepare(args, out, frozen, result):
    result = dict(result)
    folder = out / "plans/compute-reference"
    try:
        deployment = Path(frozen["optimized_deployment"]).resolve()
        profile = json.loads((deployment / "deployment.json").read_bytes())
        if profile["release"] != frozen["engine_release"]:
            raise ValueError("Automatic reference SDK differs from this campaign")
        target = deployment / "mods/betterscale/betterscale/models/qwen35/decode_kv.py"
        installed = json.loads((deployment / "provenance/installed-files.json").read_bytes())
        relative = str(target.relative_to(deployment))
        # Bind the actual deployed source, rather than trusting a directory name.
        if not isinstance(installed, dict) or installed.get(relative) != sha(target):
            raise ValueError("Automatic reference target lacks its frozen installed source binding")
        prefill = prefill_sources(deployment)
        extra = list(prefill.values())
        base = deployment / "mods/betterscale/betterscale"
        extra += [base / "patches/qwen_gdn" / name for name in ("h_tiling.h", "o_tiling.h", "solve_tril.py")]
        extra += [base / "models/qwen35" / name for name in ("wy_head_major.py", "mixed_layout.py")]
        extra += [deployment / "runtime/ascend/vllm_ascend/ops/triton/fla" / name for name in ("solve_tril.py", "chunk_scaled_dot_kkt.py")]
        for source in extra:
            if installed.get(str(source.relative_to(deployment))) != sha(source):
                raise ValueError("Automatic prefill source lacks its frozen installed binding: " + str(source))
        protocol = json.loads(PROTOCOL.read_bytes())
        from reference_capture_profile import profile as capture_profile

        capture = capture_profile(getattr(args, "reference_profile", "submission-v2"))
        policy_source = ROOT / protocol["reference_policy"]["authority_snapshot"]
        if sha(policy_source) != protocol["reference_policy"]["authority_snapshot_sha256"]:
            raise ValueError("Confirmed M01 reference-method source changed")
        files = {
            str(path.resolve()): sha(path)
            for path in [*KIT.glob("*.py"), KIT / "semantic-work-origins.json", *(KIT / "proc_capture").rglob("*"), *(KIT / "gdn-contract-sources").glob("*"), *(KIT / "canonical-reference-sources").glob("*"),
                         *(ROOT / "scripts").glob("*.py"), target, PROTOCOL, ROOT / "protocols/compute-capture-profiles-v2.json", ROOT / "protocols/async-http-device-input-source-domain-v1.json", policy_source, ROOT / "protocols/native-public-client-source-domain-v1.json", ROOT / "protocols/m01-full-reference-reuse-domain-v1.json", *extra]
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        stages = list(protocol["stages"])
        if capture["id"] == "submission-v2":
            calibration_stage = next(row for row in stages if row["id"] == "shape-matched-independent-reference-dag")
            stages.remove(calibration_stage)
            stages.insert(3, calibration_stage)
        plan = dict(
            schema="automatic-current-reference-plan/v1",
            binding=result["binding"],
            capture_profile=capture,
            reference_capture_binding=dict(result["binding"], max_completion_tokens=capture["max_tokens"], formal_repetitions=capture["formal_repetitions"]),
            measurement_domain=dict(frontend="native_inproc", dtype="BF16", tp=2, max_completion_tokens=capture["max_tokens"], formal_repetitions=capture["formal_repetitions"], profiles=["mmlu-fixed200-v1"]),
            observation_domain_only=False,
            reference_policy=protocol["reference_policy"],
            required_formal_timing_domain="native; HTTP and tokenizer outside synchronized model interval",
            deployment=str(deployment),
            baseline_deployment=str(Path(frozen["baseline_deployment"]).resolve()),
            deployment_sha256=sha(deployment / "deployment.json"),
            python=profile["python"],
            target=str(target),
            reference=str(KIT / "gdn_reference.py"),
            prefill_reference=str(KIT / "gdn_prefill_reference.py"),
            **{name: str(path) for name, path in prefill.items()},
            resource_guard=str(ROOT / "scripts/resource_guard.py"),
            physical_cards=frozen["physical_cards"],
            logical_devices=frozen.get("logical_devices", getattr(args, "devices", "0,1")),
            seed=17,
            cases=[dict(batch=n, width=w, ragged=ragged) for n, w, ragged in ((1, 1, False), (8, 1, False), (36, 1, False), (1, 3, False), (8, 3, False), (36, 3, True))],
            prefill_cases=prefill_cases(),
            route_cases=[dict(rows=rows, dtype=dtype) for rows in (1, 36, 4096) for dtype in ("int32", "int64")],
            prefill_output_tolerance=dict(atol=.0003, rtol=.025),
            prefill_state_tolerance=dict(atol=.0003, rtol=.025),
            output_tolerance=dict(atol=0.0002, rtol=0.01),
            state_tolerance=dict(atol=0.000003, rtol=0.0003),
            warmup_repetitions=3,
            calibration_repetitions=7,
            validation_repetitions=3,
            resource_wait_seconds=86400,
            files=files,
            stages=stages,
            protocol_sha256=sha(PROTOCOL),
            original_performance_rerun=False,
            source_campaign=str(Path(args.reference_campaign).resolve()) if getattr(args, "reference_campaign", None) else None,
            full_model_certificate=False,
        )
        resume_source = getattr(args, "reference_resume_source", None)
        resume = None
        if resume_source:
            from reference_model_reuse import guarded_load as load_models

            resume_source = Path(resume_source).resolve()
            plan["stage_resume"] = dict(source=str(resume_source), checksums_sha256=sha(resume_source / "checksums.json"))
            resume = load_models(resume_source, plan["stage_resume"]["checksums_sha256"], plan, ROOT,
                folder / "model-resume-source-guard")
        canonical_source = getattr(args, "reference_canonical_source", None)
        if canonical_source:
            canonical_source = Path(canonical_source).resolve()
            plan["canonical_reuse"] = dict(source=str(canonical_source), checksums_sha256=sha(canonical_source / "checksums.json"))
        reuse_source = getattr(args, "reference_qualification_source", None)
        observation_source = getattr(args, "reference_observation_source", None)
        if observation_source:
            if resume_source:
                raise ValueError("Choose a completed model-stage source or a separate HTTP observation source")
            from reference_observation_reuse import guarded_load as load_observation

            observation_source = Path(observation_source).resolve()
            plan["observation_reuse"] = dict(source=str(observation_source), checksums_sha256=sha(observation_source / "checksums.json"))
            load_observation(observation_source, plan["observation_reuse"]["checksums_sha256"], plan, ROOT,
                folder / "observation-reuse-source-guard")
        if not reuse_source and resume:
            identity = resume["operator_qualification_source"]
            reuse_source = Path(identity["source"])
            if sha(reuse_source / "checksums.json") != identity["checksums_sha256"]:
                raise ValueError("Original model run's operator qualification source changed")
        if reuse_source:
            from reference_qualification_reuse import guarded_load as load

            reuse_source = Path(reuse_source).resolve()
            plan["qualification_reuse"] = dict(source=str(reuse_source), checksums_sha256=sha(reuse_source / "checksums.json"))
            load(reuse_source, plan["qualification_reuse"]["checksums_sha256"], plan, ROOT,
                 folder / "operator-reuse-source-guard")
        save(folder / "automatic-plan.json", plan)
        result.update(status="automatic_ready", mode="automatic_current_certification", capture_profile=capture, plan="plans/compute-reference/automatic-plan.json", plan_sha256=sha(folder / "automatic-plan.json"), blockers=[], current_efficiency=None, milestone_met=None, final_numeric_met=None, historical_substitution_allowed=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status="blocked", mode="automatic_current_certification", blockers=["自动参考准备未通过: " + str(error)])
    save(folder / "admission.json", result)
    print("COMPUTE_REFERENCE_PREFLIGHT", result["status"], json.dumps(result["blockers"], ensure_ascii=False), flush=True)
    return result


def execute(out, admission):
    folder = out / "compute-reference"
    result = dict(admission, status="running", stages=[])
    save(folder / "status.json", result)
    try:
        path = out / admission["plan"]
        if sha(path) != admission["plan_sha256"]:
            raise ValueError("Automatic reference plan changed after preparation")
        plan = json.loads(path.read_bytes())
        if sha(Path(plan["deployment"]) / "deployment.json") != plan["deployment_sha256"]:
            raise ValueError("Automatic reference deployment changed after preparation")
        for name, expected in plan["files"].items():
            if sha(Path(name)) != expected:
                raise ValueError("Automatic reference source changed: " + name)
        # Keep a self-contained plan beside the actual operator run. The SDK's
        # source remains external and hash-bound; no production files are edited.
        copied = folder / "automatic-plan.json"
        shutil.copyfile(path, copied)
        if sha(copied) != admission["plan_sha256"]:
            raise ValueError("Automatic reference copied plan differs from its admission")
        reuse = None
        if plan.get("qualification_reuse"):
            from reference_qualification_reuse import guarded_load as load

            identity = plan["qualification_reuse"]
            reuse = load(identity["source"], identity["checksums_sha256"], plan, ROOT,
                         folder / "operator-reuse-execution-guard")
        resumed_models = None
        if plan.get("stage_resume"):
            from reference_model_reuse import guarded_load as load_models

            identity = plan["stage_resume"]
            resumed_models = load_models(identity["source"], identity["checksums_sha256"], plan, ROOT,
                folder / "model-resume-execution-guard")
        from engine_entry import environment

        profile = json.loads((Path(plan["deployment"]) / "deployment.json").read_bytes())
        runtime = folder / "operator-runtime"
        runtime.mkdir()
        env = environment(Path(plan["deployment"]), profile, runtime, folder / "operator-cache", plan["logical_devices"])
        env["ASCEND_PROCESS_LOG_PATH"] = str(folder / "cann-logs")
        env["TRITON_CACHE_DIR"] = str(folder / "operator-cache/triton")
        Path(env["ASCEND_PROCESS_LOG_PATH"]).mkdir()
        for stage in plan["stages"]:
            row = dict(stage)
            result["stages"].append(row)
            if stage["implementation"] is None:
                row.update(status="incomplete", reason=stage["missing"])
                continue
            if resumed_models and stage["id"] in resumed_models["stages"]:
                prior = resumed_models["stages"][stage["id"]]
                recovery = {key: value for key, value in prior.items() if key in (
                    "recovered_post_capture_cpu_derivation", "original_receipt_completed", "original_failure",
                    "independent_recovery_source", "independent_recovery_checksums_sha256")}
                row.update(status="completed", returncode=0, reused_model_stage=True,
                    original_stage_source=prior["source"], receipt_sha256=prior["receipt_sha256"],
                    original_plan_sha256=prior["original_plan_sha256"], full_model_certificate=False, **recovery)
                save(folder / "status.json", result)
                print("THIRD_PARTY_REFERENCE_MODEL_RESUMED", stage["id"], flush=True)
                continue
            final_phases = {"complete-current-necessary-work": "necessary",
                "shape-matched-independent-reference-dag": "calibration",
                "independent-reference-dag-audit": "graphs",
                "matched-current-numerator-and-recomputation": "matched"}
            if stage["id"] in final_phases:
                if stage["implementation"] != "scripts/reference_final_stages.py":
                    raise ValueError("Unknown complete reference provider")
                phase = final_phases[stage["id"]]
                row["status"] = "running"
                save(folder / "status.json", result)
                command = [plan["python"], "-B", str(ROOT / "scripts/reference_final_stages.py"),
                    "--stage", phase, "--plan", str(copied), "--sha256", admission["plan_sha256"],
                    "--folder", str(folder)]
                if resumed_models:
                    command += ["--model-source", resumed_models["source"]]
                    if "current-http-work-observation" not in resumed_models["stages"]:
                        command += ["--http-source", str(folder / "http-current-work")]
                if phase == "calibration" and plan.get("canonical_reuse"):
                    identity = plan["canonical_reuse"]
                    if sha(Path(identity["source"]) / "checksums.json") != identity["checksums_sha256"]:
                        raise ValueError("Canonical continuation source seal changed")
                    command += ["--calibration-source", identity["source"]]
                if plan.get("observation_reuse"):
                    command += ["--http-source", plan["observation_reuse"]["source"]]
                print("THIRD_PARTY_REFERENCE", stage["id"], flush=True)
                log_path = folder / ("final-stage-" + phase + ".log")
                with log_path.open("x") as log:
                    execution = run_stage(command, stdout=log, stderr=subprocess.STDOUT)
                receipt = folder / ("final-stage-" + phase + ".json")
                if execution.returncode:
                    row.update(returncode=execution.returncode,
                        log=str(log_path.relative_to(out)), log_sha256=sha(log_path))
                    detail = f"exit {execution.returncode}; see {row['log']}"
                    try:
                        value = json.loads(receipt.read_bytes())
                    except (OSError, ValueError):
                        value = None
                    if (isinstance(value, dict) and value.get("completed") is False
                            and value.get("stage") == phase and value.get("plan_sha256") == admission["plan_sha256"]
                            and isinstance(value.get("failure_type"), str) and isinstance(value.get("failure"), str)):
                        row.update(receipt=str(receipt.relative_to(out)), receipt_sha256=sha(receipt))
                        detail = f"{value['failure_type']}: {value['failure']}"
                    raise ValueError(f"Complete reference stage failed ({stage['id']}): {detail}")
                value = json.loads(receipt.read_bytes())
                if value["completed"] is not True:
                    raise ValueError("Complete reference provider did not close: " + stage["id"])
                row.update(status="completed", receipt_sha256=sha(receipt),
                    full_model_certificate=False)
                if phase == "calibration" and plan.get("capture_profile", {}).get("id") == "submission-v2":
                    from full_reference_reuse import DOMAIN_SHA256

                    full_reference = folder / "qualified-full-reference-statistics.json"
                    if not full_reference.is_file():
                        raise ValueError("Complete original reference statistics missing before native clocks")
                    qualified = json.loads(full_reference.read_bytes())
                    if (qualified != value["result"]
                            or qualified["applicability_protocol_sha256"] != DOMAIN_SHA256
                            or qualified["target_times_used"] is not False):
                        raise ValueError("Original complete-reference provider result changed")
                    frozen = dict(schema="pre-existing-complete-reference-statistics/v1",
                        qualified_full_reference_statistics_sha256=sha(full_reference),
                        original_method_frozen_unix=qualified["original_method_frozen_unix"],
                        target_times_or_efficiency_used=False)
                    save(folder / "reference-calibration-freeze.json", frozen)
                    row["pre_target_calibration_freeze_sha256"] = sha(folder / "reference-calibration-freeze.json")
                if phase == "matched":
                    result.update(value["result"])
                    if result.get("certificate_kind") == "matched_complete_necessary_work_and_full_reference_lower_bound":
                        raise ValueError("Withdrawn dense-tail certificate cannot certify the complete reference")
                    if result.get("full_model_certificate") is not True:
                        row.update(status="incomplete", reason="；".join(result["blockers"]))
                    else:
                        row["full_model_certificate"] = True
                continue
            if reuse and stage["id"] in reuse["stages"]:
                from reference_qualification_reuse import STAGES

                prior = reuse["stages"][stage["id"]]
                if stage["id"] == "raw-route-recorder-qualification":
                    verify_route_qualification(prior["receipt"], reuse["plan"], reuse["plan_path"])
                else:
                    verify_qualification(prior["receipt"], reuse["plan"], reuse["plan_path"], STAGES[stage["id"]][1])
                row.update(status="completed", returncode=0, reused_qualification=True,
                    original_receipt=str(prior["path"]), receipt_sha256=sha(prior["path"]),
                    original_plan_sha256=sha(reuse["plan_path"]), full_model_certificate=False)
                save(folder / "status.json", result)
                print("THIRD_PARTY_REFERENCE_REUSED", stage["id"], flush=True)
                continue
            if stage["id"] == "raw-route-recorder-qualification":
                if stage["implementation"] != "third-party-test-kit/qualify_route_capture.py":
                    raise ValueError("Unknown raw route-recorder implementation")
                row["status"] = "running"
                save(folder / "status.json", result)
                directory = folder / "route-recorder"
                command = [plan["python"], str(KIT / "qualify_route_capture.py"), "--plan", str(copied), "--out", str(directory)]
                with (folder / "route-recorder.log").open("x") as log:
                    run = run_stage(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                row["returncode"] = run.returncode
                receipt = json.loads((directory / "receipt.json").read_bytes())
                if run.returncode:
                    raise ValueError("Raw route-recorder graph-replay qualification failed")
                verify_route_qualification(receipt, plan, copied)
                row.update(status="completed", receipt_sha256=sha(directory / "receipt.json"),
                           observer_only=True, full_model_certificate=False)
                continue
            if stage["id"] == "current-http-work-observation":
                if stage["implementation"] != "scripts/reference_observation.py":
                    raise ValueError("Unknown HTTP observation implementation")
                if plan.get("observation_reuse"):
                    from reference_observation_reuse import guarded_load as load_observation

                    identity = plan["observation_reuse"]
                    observed = load_observation(identity["source"], identity["checksums_sha256"], plan, ROOT,
                        folder / "observation-reuse-execution-guard")
                    row.update(status="completed", returncode=0, reused_observation=True,
                        original_stage_source=observed["source"], receipt_sha256=observed["receipt_sha256"],
                        old_counts_or_times_used_for_native_work=False, full_model_certificate=False)
                    save(folder / "status.json", result)
                    print("THIRD_PARTY_REFERENCE_HTTP_SOURCE_REUSED", stage["id"], flush=True)
                    continue
                origin = Path(plan["source_campaign"]) if plan.get("source_campaign") else out
                stages_path = origin / "frozen-stages.json"
                if plan.get("source_campaign"):
                    seal = json.loads((origin / "campaign-checksums.json").read_bytes())
                    if sha(stages_path) != seal.get("frozen-stages.json"):
                        raise ValueError("Source campaign stage plan changed")
                stages = json.loads(stages_path.read_bytes())
                paired = [s for s in stages if s["name"] == "mmlu-pair"]
                if len(paired) != 1:
                    raise ValueError("One current fixed200 pair qualification is required")
                argv = paired[0]["args"]
                qualified = Path(argv[argv.index("--qualified-continuous") + 1])
                row["status"] = "running"
                save(folder / "status.json", result)
                directory = folder / "http-current-work"
                print("THIRD_PARTY_REFERENCE", stage["id"], flush=True)
                command = [sys.executable, "-B", str(ROOT / "scripts/run_b1_r4.py"),
                    "--deployment", plan["deployment"], "--out", str(directory),
                    "--model", plan["binding"]["model"], "--devices", plan["logical_devices"],
                    "--physical-devices", ",".join(map(str, plan["physical_cards"])),
                    "--role", "B1", "--kind", "joint", "--qualified-continuous", str(qualified),
                    "--reference-observation", "--reference-profile", plan["capture_profile"]["id"], "--lease-timeout", "86400"]
                with (folder / "http-current-work.log").open("x") as log:
                    run = run_stage(command, stdout=log, stderr=subprocess.STDOUT)
                row["returncode"] = run.returncode
                receipt_path = directory / "receipt.json"
                receipt = json.loads(receipt_path.read_bytes())
                result["new_reference_inference_executed"] = bool(receipt.get("warmup_started") or receipt.get("client_started"))
                result["current_work_observation_requests_started"] = bool(receipt.get("client_started"))
                if run.returncode:
                    row["status"] = "failed"
                    raise ValueError("Current HTTP observation failed: " + str(receipt.get("failure")))
                partial = json.loads((directory / "service/raw-work/raw-partial-work.json").read_bytes())
                if (run.returncode or not receipt["completed"] or not receipt["quiescent"]
                        or not receipt["raw_observation"]["completed"] or partial["effective_flops"] is not None
                        or len(partial["requests"]) != 200 or not partial["worker_schedule_matches"]
                        or partial.get("moe_routes", {}).get("peer_routes_match") is not True
                        or any(not r["covered"] or r["retained"] != r["output_tokens"] for r in partial["requests"])):
                    raise ValueError("Current HTTP raw-work capture is incomplete")
                row.update(status="completed", full_model_certificate=False,
                           receipt_sha256=sha(receipt_path), observed_requests=200,
                           timing_eligible_for_performance_comparison=False)
                result["new_reference_inference_executed"] = True
                continue
            if stage["id"] in ("current-native-synchronized-timing", "baseline-native-synchronized-timing"):
                baseline_native = stage["id"] == "baseline-native-synchronized-timing"
                if stage["implementation"] != "scripts/run_native_reference.py":
                    raise ValueError("Unknown native timing implementation")
                origin = Path(plan["source_campaign"]) if plan.get("source_campaign") else out
                stages_path = origin / "frozen-stages.json"
                if plan.get("source_campaign"):
                    seal = json.loads((origin / "campaign-checksums.json").read_bytes())
                    if sha(stages_path) != seal.get("frozen-stages.json"):
                        raise ValueError("Source campaign native stage plan changed")
                paired = [s for s in json.loads(stages_path.read_bytes()) if s["name"] == "mmlu-pair"]
                if len(paired) != 1:
                    raise ValueError("Native timing requires one current MMLU pair binding")
                argv = paired[0]["args"]
                qualified = Path(argv[argv.index("--qualified-continuous") + 1])
                row["status"] = "running"
                save(folder / "status.json", result)
                directory = folder / ("native-baseline-timing" if baseline_native else "native-current-timing")
                command = [sys.executable, "-B", str(ROOT / "scripts/run_native_reference.py"),
                           "--deployment", plan["deployment"], "--model", plan["binding"]["model"],
                           "--out", str(directory), "--qualified-continuous", str(qualified),
                           "--devices", plan["logical_devices"], "--physical-devices", ",".join(map(str, plan["physical_cards"])),
                           "--formal-repetitions", str(plan["capture_profile"]["formal_repetitions"]),
                           "--reference-profile", plan["capture_profile"]["id"], "--lease-timeout", "86400"]
                if baseline_native:
                    command += ["--baseline-deployment", plan["baseline_deployment"]]
                print("THIRD_PARTY_REFERENCE", stage["id"], flush=True)
                with directory.with_suffix(".log").open("x") as log:
                    run = run_stage(command, stdout=log, stderr=subprocess.STDOUT)
                row["returncode"] = run.returncode
                receipt_path = directory / "receipt.json"
                receipt = json.loads(receipt_path.read_bytes())
                if (run.returncode or receipt["completed"] is not True or receipt["quiescent"] is not True
                        or receipt["role"] != ("B0" if baseline_native else "B1")
                        or receipt["preflight_only"] is not False or receipt["formal_repetitions"] != plan["capture_profile"]["formal_repetitions"]
                        or receipt["necessary_work_certified"] is not False or receipt["efficiency_percent"] is not None):
                    raise ValueError("Current native timing did not complete: " + str(receipt.get("failure")))
                row.update(status="completed", receipt_sha256=sha(receipt_path),
                           full_model_certificate=False, numerator_pending_necessary_work_correspondence=True)
                result["new_reference_inference_executed"] = True
                continue
            implementations = {
                "gdn-decode-independent-reference": ("qualify_gdn_reference.py", "gdn-decode", "cases"),
                "gdn-prefill-independent-reference": ("qualify_gdn_prefill_reference.py", "gdn-prefill", "prefill_cases"),
            }
            if stage["id"] not in implementations:
                raise ValueError("Unknown automatic reference stage")
            driver, directory, cases_key = implementations[stage["id"]]
            if stage["implementation"] != "third-party-test-kit/" + driver:
                raise ValueError("Automatic reference implementation differs from its required stage")
            row["status"] = "running"
            save(folder / "status.json", result)
            print("THIRD_PARTY_REFERENCE", stage["id"], flush=True)
            command = [plan["python"], str(KIT / driver), "--plan", str(copied), "--out", str(folder / directory)]
            logfile = "collector.log" if cases_key == "cases" else directory + ".log"
            with (folder / logfile).open("x") as log:
                run = run_stage(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            row["returncode"] = run.returncode
            receipt_path = folder / directory / "receipt.json"
            receipt = json.loads(receipt_path.read_bytes())
            if run.returncode:
                raise ValueError("Automatic GDN reference qualification failed: " + directory)
            verify_qualification(receipt, plan, copied, cases_key)
            row.update(status="completed", receipt_sha256=sha(receipt_path), full_model_certificate=False)
        missing = [row["reason"] for row in result["stages"] if row["status"] == "incomplete"]
        if missing:
            result.update(status="incomplete", blockers=missing, current_efficiency=None,
                efficiency_lower_observed_percent=None, milestone_met=None,
                final_numeric_met=None, full_model_certificate=False)
        elif result.get("status") != "measured" or result.get("full_model_certificate") is not True:
            raise ValueError("All stages closed without a matched complete reference certificate")
        else:
            result["blockers"] = []
        result.update(new_reference_inference_executed=result.get("new_reference_inference_executed", False),
            operator_reference_executed=True)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        for row in result["stages"]:
            if row.get("status") == "running":
                row.update(status="failed", reason=str(error))
        result.update(status="blocked", blockers=[str(error)], current_efficiency=None,
            efficiency_lower_observed_percent=None, milestone_met=None,
            final_numeric_met=None, full_model_certificate=False)
    save(folder / "automatic-result.json", result)
    return result
