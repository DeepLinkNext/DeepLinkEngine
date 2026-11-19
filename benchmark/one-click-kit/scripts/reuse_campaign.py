"""Reuse explicitly selected sealed stages on the same host without generating requests."""

import json
from pathlib import Path
import time

from dataset_contract import sha, save


def read(path):
    return json.loads(path.read_bytes())


def identity(plan):
    result = {
        key: plan[key]
        for key in (
            "model",
            "physical_cards",
            "baseline_source_map_sha256",
            "optimized_deployment",
            "optimized_dependencies",
        )
    }
    result["engine_release"] = plan.get("engine_release", "full-answer-b1-20261030-r4")
    result["serving_warmup"] = plan.get("serving_warmup")
    return result


def source(root):
    from third_party_report import verify_campaign_bindings

    root = root.resolve()
    bindings = verify_campaign_bindings(root)
    plan, receipt = (
        read(root / "frozen-execution-plan.json"),
        read(root / "campaign.json"),
    )
    if sha(root / "frozen-execution-plan.json") != receipt["prepared_plan_sha256"]:
        raise ValueError("Reuse source prospective plan changed")
    captured = {
        str(p.relative_to(root / "execution-source")): sha(p)
        for p in (root / "execution-source").rglob("*")
        if p.is_file()
    }
    if captured != plan["sources"]:
        raise ValueError("Reuse source execution code changed")
    return root, plan, receipt, bindings


def verify_baseline_serve_args(stage, plan_path):
    """A B0 startup-only adaptation must leave the effective argv unchanged.

    The sealed plan is the only per-run record of what the old entry started;
    the fixed profile is the only record of what the new entry must start.
    """
    from baseline_profile import expected

    plan_path = Path(plan_path)
    if not plan_path.is_file():
        raise ValueError("Reuse baseline service plan is missing: " + str(plan_path))
    plan = read(plan_path)
    argv = list(plan["serve_args"])
    appended = ["--worker-extension-cls", "compile_boundary.WorkerExtension"]
    if argv[-2:] == appended:
        del argv[-2:]
    kind = "continuous" if stage.startswith("continuous-") else "dataset"
    if argv != expected(plan.get("baseline_variant"), kind):
        raise ValueError("Reuse baseline effective serve args differ: " + stage)


def compatible(stage, previous, current, root):
    if (not stage.startswith("continuous-") and
            previous.get("task_serving_warmup") != current.get("task_serving_warmup")):
        raise ValueError("Reuse task warmup protocol differs: " + stage)
    if identity(previous) != identity(current):
        # A B1-only startup adaptation cannot invalidate an already sealed B0
        # measurement under the exact same shared warmup policy. Keep the old
        # receipt's real runtime identity; never relabel it as the new B1.
        baseline_stages = {"continuous-B0", "mmlu-pair/B0", "hle-solver-B0",
            "frontier-solver-B0", "swe-trace-B0", "terminal-trace-B0"}
        if stage not in baseline_stages:
            raise ValueError("Reuse runtime/model/device identity differs: " + stage)
        keys = ("model", "physical_cards", "logical_devices", "baseline_deployment",
                "baseline_source_map_sha256", "serving_warmup")
        if (not previous.get("serving_warmup") or
                any(previous.get(k) != current.get(k) for k in keys)):
            raise ValueError("Reuse baseline runtime/model/device identity differs: " + stage)
        if stage == "continuous-B0":
            old_recipe = read(root / stage / "receipt.json")["original_input_recipe"]["recipe_sha256"]
            new_recipe = Path(current["optimized_deployment"]) / "runtime/entrypoints/prepare_continuous.py"
            if sha(new_recipe) != old_recipe:
                raise ValueError("Reuse baseline C8 input recipe differs")
        relative = "scripts/serving_warmup.py"
        old_source = (root / "execution-source" / relative).read_text()
        new_source = (Path(__file__).parent / "serving_warmup.py").read_text()
        expected = old_source.replace(
            'return profile["release"] == "full-answer-b1-20261113-r5"',
            'return profile["release"] in ("full-answer-b1-20261113-r5", "full-answer-b1-20261113-r6")')
        if new_source != expected:
            compat = read(Path(__file__).resolve().parents[1] / "protocols/task-warmup-v3-c8-compatibility.json")
            if (sha(root / "execution-source" / relative) not in compat["previous_sha256"]
                    or sha(Path(__file__).parent / "serving_warmup.py") != compat["current_sha256"]
                    or previous["serving_warmup"] != __import__("serving_warmup").protocol("continuous")):
                raise ValueError("Reuse baseline shared warmup implementation differs")
        # B0 startup, compile guards and request generation must be byte-identical.
        for relative in ("scripts/compile_boundary.py",
                         "scripts/baseline_entry.py", "scripts/baseline_admission.py",
                         "scripts/continuous_inputs.py"):
            if previous["sources"].get(relative) != current["sources"].get(relative):
                compat = read(Path(__file__).resolve().parents[1] / "protocols/task-warmup-v3-c8-compatibility.json")
                adaptation = compat.get("unchanged_continuous_source_adaptations", {}).get(relative, {})
                if (previous["sources"].get(relative) not in adaptation.get("previous_sha256", [])
                        or current["sources"].get(relative) != adaptation.get("current_sha256")):
                    raise ValueError("Reuse baseline execution source differs: " + relative)
                if relative == "scripts/baseline_entry.py":
                    verify_baseline_serve_args(stage, root / stage / "service/plan.json")
        if stage not in ("continuous-B0", "mmlu-pair/B0"):
            for relative in ("scripts/dataset_client.py", "scripts/dataset_contract.py", "scripts/agent_trace.py"):
                if previous["sources"].get(relative) != current["sources"].get(relative):
                    raise ValueError("Reuse baseline task client source differs: " + relative)
    if stage in ("mmlu-pair", "mmlu-pair/B0"):
        for key in ("quality_plan_sha256", "scorer_sha256"):
            if previous["mmlu_inputs"][key] != current["mmlu_inputs"][key]:
                raise ValueError("Reuse MMLU inputs or scorer differ")
    elif not stage.startswith("continuous-"):
        dataset = stage.split("-")[0]
        if previous.get("smoke_evaluators") or current.get("smoke_evaluators"):
            raise ValueError("Diagnostic stages cannot supply standard reuse")
        before = previous["prepared_datasets"][dataset]
        after = current["prepared_datasets"][dataset]
        if (
            before["contract_sha256"] != after["contract_sha256"]
            or before["tasks"] != after["tasks"]
        ):
            raise ValueError("Reuse frozen dataset scope/budget differs: " + stage)
        spec = read(root / stage / "dataset-spec.json")
        if spec["contract_sha256"] != after["contract_sha256"] or spec["diagnostic"]:
            raise ValueError("Reuse actual dataset descriptor differs: " + stage)
        if stage.startswith("hle-judge-"):
            from judge_grammar import requested, selected

            if requested(spec) != selected(current):
                raise ValueError("Reuse HLE judge whitespace policy differs: " + stage)


def audit_stage(root, name):
    stage = root / name
    if name == "mmlu-pair":
        from paired_execution_report import inspect_pair

        return inspect_pair(stage)
    receipt = read(stage / "receipt.json")
    if not (
        receipt.get("completed") is True
        and receipt.get("quiescent") is True
        and receipt.get("status") == "completed"
    ):
        raise ValueError("Reuse requires completed, released child evidence: " + name)
    if name.startswith("continuous-"):
        from report_execution import inspect_run

        result = inspect_run(stage)
        if result["failed_requests"] or not result["recorded_requests"]:
            raise ValueError("Reuse C8 has failed or absent requests")
        if name == "continuous-B1" and not result.get("continuous_qualification"):
            raise ValueError("Reuse B1 qualification failed")
        return result
    from report_dataset import inspect

    result = inspect(stage)
    if (
        result.get("unreturned")
        or result.get("failed")
        or result.get("recorded") != result.get("planned")
    ):
        raise ValueError(
            "Reuse requires all planned requests; errors remain in source: " + name
        )
    # Invalid judges and length-ended answers remain observed outcomes, never retried.
    return result


def verify_weights(stages):
    shared = None
    for entry in stages.values():
        stage = Path(entry["source_root"]) / entry["stage"]
        for child in (
            [stage / "B0", stage / "B1"] if entry["stage"] == "mmlu-pair" else [stage]
        ):
            weights = read(child / "model-weights.json")
            values = {
                n: {k: row[k] for k in ("sha256", "bytes")}
                for n, row in weights.items()
            }
            if shared is not None and shared != values:
                raise ValueError("Reused stages used different model weight bytes")
            shared = values
    return shared


def prepare(roots, out, plan, stages):
    if not roots:
        return {}
    wanted = {s["name"] for s in stages}
    entries, origins, paired_arms = {}, [], {}
    for index, path in enumerate(roots):
        root, previous, receipt, bindings = source(path)
        if root == out or root.is_relative_to(out) or out.is_relative_to(root):
            raise ValueError("Reuse source and new output must be independent")
        origin = {
            "source_root": str(root),
            "campaign_sha256": sha(root / "campaign.json"),
            "campaign_seal_sha256": sha(root / "campaign-checksums.json"),
            "plan_sha256": sha(root / "frozen-execution-plan.json"),
            "parent_status": receipt["status"],
            "stages": [],
        }
        for row in receipt["stages"]:
            name = row["name"]
            if name not in wanted or row["status"] != "completed":
                continue
            selected_args = next(stage.get("args", []) for stage in stages if stage["name"] == name)
            if name == "mmlu-pair" and any(flag in selected_args for flag in
                    ("--reuse-baseline-arm", "--reuse-optimized-arm")):
                origin.setdefault("incompatible_stages", []).append(dict(stage=name,
                    reason="Explicit arm selection takes precedence over whole-pair reuse"))
                continue
            if identity(previous) != identity(plan) and name == "mmlu-pair":
                compatible("mmlu-pair/B0", previous, plan, root)
                stage_args = next(stage["args"] for stage in stages if stage["name"] == name)
                if "--reuse-baseline-arm" not in stage_args:
                    if "B0" in paired_arms:
                        raise ValueError("Ambiguous completed MMLU baseline arm; select one source")
                    from arm_reuse import admit as admit_arm, expected as expected_arm

                    arm = root / "mmlu-pair/B0"
                    conditions = expected_arm(plan["optimized_deployment"], plan["baseline_deployment"],
                        plan["model"], plan["physical_cards"], plan["logical_devices"],
                        read(arm / "model-weights.json"))
                    binding = admit_arm(arm, "B0", conditions)
                    paired_arms["B0"] = dict(stage="mmlu-pair/B0", source_root=str(root),
                        binding=binding, source_stage_receipt=row)
                    origin["paired_arms"] = ["B0"]
                origin.setdefault("incompatible_stages", []).append(
                    dict(stage=name, reason="B1 version differs; reuse only its independently admitted B0 arm"))
                continue
            if identity(previous) != identity(plan) and not name.endswith("-B0"):
                origin.setdefault("incompatible_stages", []).append(
                    dict(stage=name, reason="Optimized service version differs"))
                continue
            if identity(previous) != identity(plan) and "-judge-" in name:
                origin.setdefault("incompatible_stages", []).append(
                    dict(stage=name, reason="Judge service is B1 even when grading B0 answers"))
                continue
            if name.startswith("hle-judge-"):
                from judge_grammar import selected

                if selected(previous) != selected(plan):
                    origin.setdefault("incompatible_stages", []).append(
                        {"stage": name, "reason": "HLE judge whitespace policy changed"}
                    )
                    continue
            if name in entries:
                raise ValueError(
                    "Ambiguous completed reuse candidates; select one source: " + name
                )
            compatible(name, previous, plan, root)
            audit_stage(root, name)
            entries[name] = {
                "stage": name,
                "source_root": str(root),
                "source_index": index,
                "source_stage_receipt": row,
                "verified_at": time.time(),
            }
            origin["stages"].append(name)
        origins.append(origin)
    explicitly_selected_arms = any(stage["name"] == "mmlu-pair" and any(
        flag in stage.get("args", []) for flag in ("--reuse-baseline-arm", "--reuse-optimized-arm"))
        for stage in stages)
    if not entries and not paired_arms and not explicitly_selected_arms:
        raise ValueError(
            "Explicit reuse sources contain no compatible completed stages"
        )
    weight_entries = dict(entries)
    if paired_arms:
        weight_entries["mmlu-pair/B0"] = paired_arms["B0"]
    weights = verify_weights(weight_entries)
    if weights is not None and set(entries) != wanted:
        # Mixed reuse/new inference must bind the current weight bytes before launching.
        for name, expected in weights.items():
            path = Path(plan["model"]) / name
            if (
                path.stat().st_size != expected["bytes"]
                or sha(path) != expected["sha256"]
            ):
                raise ValueError("Current model differs from reused model: " + name)
    data = {
        "schema": "vllm-hust-sealed-stage-reuse/v1",
        "prepared_before_new_inference": time.time(),
        "scope": "Same-host composition of earlier sealed native stages; not one new uninterrupted inference run.",
        "origins": origins,
        "stages": entries,
        "paired_arms": paired_arms,
        "weights": weights,
        "new_inference_stages": sorted(wanted - set(entries)),
    }
    save(out / "reused-stages.json", data)
    print(
        "THIRD_PARTY_REUSE_ADMITTED",
        len(entries),
        "new stages",
        len(wanted - set(entries)),
        flush=True,
    )
    return entries


def load_report(root, plan):
    path = root / "reused-stages.json"
    if not path.exists():
        return {}, {}
    data = read(path)
    audited, stage_roots, checked_arms = {}, {}, set()
    for index, origin in enumerate(data["origins"]):
        source_root, previous, receipt, bindings = source(Path(origin["source_root"]))
        for name, key in (
            ("campaign.json", "campaign_sha256"),
            ("campaign-checksums.json", "campaign_seal_sha256"),
            ("frozen-execution-plan.json", "plan_sha256"),
        ):
            if sha(source_root / name) != origin[key]:
                raise ValueError("Reused source changed after admission: " + name)
        rows = {r["name"]: r for r in receipt["stages"]}
        for role in origin.get("paired_arms", []):
            from arm_reuse import recheck

            entry = data["paired_arms"][role]
            if (entry["source_root"] != str(source_root)
                    or rows["mmlu-pair"] != entry["source_stage_receipt"]):
                raise ValueError("Reused MMLU arm parent provenance differs")
            compatible("mmlu-pair/" + role, previous, plan, source_root)
            if recheck(entry["binding"]) != source_root / "mmlu-pair" / role:
                raise ValueError("Reused MMLU arm path differs")
            checked_arms.add(role)
        for name in origin["stages"]:
            entry = data["stages"][name]
            if (
                entry["source_index"] != index
                or entry["source_root"] != str(source_root)
                or rows[name] != entry["source_stage_receipt"]
                or rows[name]["status"] != "completed"
            ):
                raise ValueError("Reused stage provenance differs")
            compatible(name, previous, plan, source_root)
            audited[name] = audit_stage(source_root, name)
            stage_roots[name] = source_root / name
    weight_entries = dict(data["stages"])
    weight_entries.update({"mmlu-pair/" + role:entry for role, entry in data.get("paired_arms", {}).items()})
    verify_weights(weight_entries)
    if checked_arms != set(data.get("paired_arms", {})):
        raise ValueError("Reused MMLU arm membership differs")
    if set(audited) != set(data["stages"]):
        raise ValueError("Reused stage membership differs")
    return stage_roots, audited


def wire_dependencies(out, stages, reused):
    """Freeze original source paths for missing stages that depend on reused stages."""
    for stage in stages:
        args = stage["args"]
        if stage["name"] == "mmlu-pair" and (out / "reused-stages.json").exists():
            arm = read(out / "reused-stages.json").get("paired_arms", {}).get("B0")
            if arm and "--reuse-baseline-arm" not in args:
                args += ["--reuse-baseline-arm", arm["binding"]["run"],
                    "--reuse-baseline-seal", arm["binding"]["seal_sha256"]]
        if "continuous-B1" in reused and "--qualified-continuous" in args:
            index = args.index("--qualified-continuous") + 1
            entry = reused["continuous-B1"]
            args[index] = str(Path(entry["source_root"]) / "continuous-B1")
        if "--dataset-spec" not in args:
            continue
        spec = Path(args[args.index("--dataset-spec") + 1])
        data = read(spec)
        if data["stage"] == "judge":
            key = data["dataset"] + "-solver-" + data["solver_role"]
            if key in reused:
                data["responses"] = str(
                    Path(reused[key]["source_root"]) / key / "service/evaluation"
                )
        elif data["stage"] == "trace" and data["service_role"] == "B1":
            key = data["dataset"] + "-trace-B0"
            if key in reused:
                data["baseline_trace_run"] = str(Path(reused[key]["source_root"]) / key)
        save(spec, data)
    save(out / "frozen-stages.json", stages)
