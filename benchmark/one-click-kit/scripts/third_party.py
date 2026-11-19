"""Restore fixed runtimes, freeze quality subsets and public trace samples, run and plot."""

import argparse
import fcntl
import os
import json
import shutil
from pathlib import Path
import subprocess
import sys
import time
import uuid

from dataset_contract import prepare as prepare_dataset, client, save, sha
from project_evidence import assess, ensure
from resource_guard import cards, verify_mapping
from restore_b1_r4 import restore as restore_b1, verify_installed_sources
from baseline_admission import admit
from child_runner import cancellation_handlers, run_stage
from restore_b0_image import restore as restore_image
from restore_b0_018 import prepare as prepare_baseline
from mmlu_scope import verify as verify_mmlu_scope, description as mmlu_description
from judge_disclosure import NOTICE_ZH, NOTICE_EN, policy as judge_policy

ROOT = Path(__file__).resolve().parents[1]


def cached_runtime(cache, name, explicit, build):
    if explicit is not None:
        if not explicit.is_dir():
            raise ValueError(
                "Explicit runtime directory does not exist: " + str(explicit)
            )
        return explicit.resolve()
    # Publish only a fully prepared runtime. Interrupted extraction is retained
    # under its unique attempt path; it cannot masquerade as a completed cache.
    with (cache / (name + ".lock")).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        final = cache / name
        if final.exists():
            return final.resolve()
        attempt = cache / (name + "-attempt-" + uuid.uuid4().hex)
        build(attempt)
        final.symlink_to(attempt.name, target_is_directory=True)
        return attempt


def script_sources():
    paths = []
    if (ROOT / "one_click.py").is_file():
        paths.append(ROOT / "one_click.py")
    for folder in (
        "scripts",
        "baseline-sources",
        "evaluation-clients",
        "protocols",
        "engines",
        "overlays",
        "abi-sources",
        "third-party-test-kit",
    ):
        paths += [
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        ]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def descriptor(
    dataset, plan, role, stage, solver_role, out, diagnostic, responses=None
):
    module = client(dataset)
    _, tasks = module.load_plan(plan)
    selected = tasks
    if diagnostic:
        selected = []
        for track in (
            ("gold", "revision") if dataset == "hle" else ("olympiad", "research")
        ):
            cohort = [task for task in tasks if task["track"] == track]
            if dataset == "hle":
                selected += [
                    next(task for task in cohort if not task["image"]),
                    next(task for task in cohort if task["image"]),
                ]
            else:
                selected.append(cohort[0])
    data = {
        "schema": "vllm-hust-fixed-dataset-phase/v1",
        "dataset": dataset,
        "plan": str(plan.resolve()),
        "contract_sha256": sha(plan / "contract.json"),
        "case_ids": [t["case_id"] for t in selected],
        "stage": stage,
        "service_role": role,
        "solver_role": None if dataset in ("swe", "terminal") else solver_role,
        "diagnostic": diagnostic,
        "responses": str(responses.resolve()) if responses else None,
    }
    save(out, data)
    return data


def prepare(args, out):
    os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = args.physical_devices
    mapping = subprocess.run(
        ["npu-smi", "info", "-m"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout
    if not args.devices:
        found = {}
        for line in mapping.splitlines():
            words = line.split()
            if (
                len(words) >= 3
                and words[0].isdigit()
                and words[1] == "0"
                and words[2].isdigit()
            ):
                found[int(words[0])] = int(words[2])
        if any(card not in found for card in cards()):
            raise ValueError(
                "Selected physical cards are absent from the actual NPU mapping"
            )
        args.devices = ",".join(str(found[card]) for card in cards())
    save(out / "device-mapping.json", verify_mapping(args.devices))
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)

    def build_baseline(target):
        restore_image(cache / "b0-oci", target)
        prepare_baseline(target)

    baseline = cached_runtime(
        cache, "original018-sdk", args.baseline_deployment, build_baseline
    )
    b0 = admit(baseline, out / "baseline-source-admission.json")

    def build_optimized(target):
        if args.b1_python is None:
            raise ValueError(
                "Provide --b1-python for the fixed Torch2.13/torch-npu2.13rc1/Transformers5.14.1 SDK"
            )
        restore_b1(cache / "b1-original-assets", target, args.b1_python, args.env_file)
        if getattr(args, "engine_profile", "r4") in ("r5", "r6", "r7", "r8"):
            from runtime_overlay import apply, RELEASES

            apply(target, RELEASES[args.engine_profile])

    deployment = cached_runtime(
        cache,
        "b1-" + getattr(args, "engine_profile", "r4") + "-deployment",
        args.deployment,
        build_optimized,
    )
    verify_installed_sources(deployment)
    actual_profile = json.loads((deployment / "deployment.json").read_bytes())
    from runtime_overlay import RELEASES

    expected_release = RELEASES[getattr(args, "engine_profile", "r4")]
    if actual_profile["release"] != expected_release:
        raise ValueError("Explicit Engine profile differs from admitted runtime")
    save(out / "mmlu-task-scope.json", verify_mmlu_scope(deployment))
    print("MMLU_SCOPE", json.dumps(mmlu_description(), ensure_ascii=False), flush=True)
    from dependency_admission import admit_dependencies

    b1 = admit_dependencies(deployment)
    save(out / "optimized-dependency-admission.json", b1)
    evidence_root = ensure(args.project_evidence, env_file=args.env_file)
    os.environ["VLLM_HUST_PROJECT_EVIDENCE"] = str(evidence_root)
    reference = assess(evidence_root)
    save(out / "existing-project-evidence.json", reference)
    plans = out / "plans"
    plans.mkdir()
    datasets = args.datasets.split(",")
    if len(datasets) != len(set(datasets)) or set(datasets) - {
        "mmlu",
        "hle",
        "frontier",
        "swe",
        "terminal",
    }:
        raise ValueError(
            "Supported explicit scope is mmlu,hle,frontier,swe,terminal; SWE/Terminal use published request traces"
        )
    prepared = {}
    for name in datasets:
        if name != "mmlu":
            assets = (
                args.dataset_assets / name
                if args.dataset_assets
                else cache / "datasets" / name
            )
            if name in ("swe", "terminal"):
                assets = (args.dataset_assets or cache) / "public-agent-traces"
            prepared[name] = prepare_dataset(
                name,
                assets,
                plans / name,
                args.model,
                hle_profile=args.hle_profile,
            )
    sources = script_sources()
    for name in sources:
        target = out / "execution-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    quality_plan = deployment / "inputs/mmlu-quality018-r2/quality-request-plan.json"
    quality = json.loads(quality_plan.read_bytes())
    if (
        len(quality["requests"]) != 200
        or quality["max_completion_tokens"] != 8192
        or quality["temperature"] != 0
        or quality["seed"] != 0
    ):
        raise ValueError("Original MMLU200 protocol differs")
    frozen = {
        "schema": "vllm-hust-third-party-datasets/v3",
        "prepared_before_model_start": time.time(),
        "datasets": datasets,
        "outside_execution_profile": [
            name
            for key, name in (
                ("mmlu", "MMLU-Pro"),
                ("hle", "HLE-Verified"),
                ("frontier", "FrontierScience"),
                ("swe", "SWE-bench-Pro"),
                ("terminal", "TerminalBench2.1"),
            )
            if key not in datasets
        ],
        "mmlu_inputs": {
            "requests": 200,
            "quality_plan_sha256": sha(quality_plan),
            "scorer_sha256": sha(deployment / "scorers/mmlu_pro_pyramidkv.py"),
            "scope_file": "mmlu-task-scope.json",
            "scope_file_sha256": sha(out / "mmlu-task-scope.json"),
            **mmlu_description(),
        },
        "model": str(args.model.resolve()),
        "physical_cards": cards(),
        "logical_devices": args.devices,
        "baseline_deployment": str(baseline.resolve()),
        "baseline_source_map_sha256": b0["source_map_sha256"],
        "baseline_profile": {
            "path": "engines/b0-vllm-v018.json",
            "sha256": sha(ROOT / "engines/b0-vllm-v018.json"),
            "variant_rule": "multimodal for the HLE solver stage, text for every other stage",
        },
        "optimized_deployment": str(deployment.resolve()),
        "engine_release": actual_profile["release"],
        "serving_warmup": __import__("serving_warmup").protocol(capacity36=actual_profile["release"] == "full-answer-b1-20261117-r8")
        if __import__("serving_warmup").selected(actual_profile)
        else None,
        "task_serving_warmup": __import__("serving_warmup").protocol("task", capacity36=actual_profile["release"] == "full-answer-b1-20261117-r8")
        if __import__("serving_warmup").selected(actual_profile)
        else None,
        "optimized_dependencies": b1,
        "prepared_datasets": prepared,
        "hle_profile": args.hle_profile,
        "hle_judge_whitespace": getattr(args, "hle_judge_whitespace", "bounded-v2"),
        "execution_source_snapshots": True,
        "sources": sources,
        "smoke_evaluators": args.smoke_evaluators,
        "mmlu_scope": "Original fixed200/max8192, one simultaneous batch, original scorer; no diagnostic length reduction",
        "judge_scope": "Same fixed B1 Qwen substitute judge for both solver arms, original separate-track parsing; judge traffic excluded from solver metrics",
        "quality_scoring_policy": judge_policy(),
        "failure_policy": "Retain failed/unreturned cases; no inference retries, replacement tasks, automatic budget changes or eager fallback",
        "public_trace_scope": "SWE/Terminal are fixed samples of published requests, measured with real model generation. No tool execution or solve-rate claim; original timestamps are not replay arrival times. Quality subsets and trace performance are reported separately.",
        "formal_complete": False,
    }
    from compute_reference import prepare as prepare_reference

    frozen["compute_reference"] = prepare_reference(args, out, frozen)
    save(out / "frozen-execution-plan.json", frozen)
    return frozen


def build_stages(args, out, frozen):
    deployment = frozen["optimized_deployment"]
    baseline = frozen["baseline_deployment"]
    common = [
        "--deployment",
        deployment,
        "--devices",
        args.devices,
        "--physical-devices",
        args.physical_devices,
        "--model",
        str(args.model.resolve()),
    ]
    if args.report_python:
        common += ["--report-python", str(args.report_python.absolute())]
    continuous = args.continuous_evidence
    stages = []
    if continuous is None:
        for role in ("B0", "B1"):
            stages.append(
                {
                    "name": "continuous-" + role,
                    "script": "run_b1_r4.py",
                    "args": common
                    + ["--role", role, "--kind", "continuous"]
                    + (["--baseline-deployment", baseline] if role == "B0" else []),
                }
            )
        continuous = out / "continuous-B1"
    else:
        from qualification import qualify

        qualified = qualify(continuous, Path(deployment))
        if qualified["physical_cards"] != cards():
            raise ValueError(
                "Supplied continuous evidence belongs to another physical pair"
            )
        save(out / "reused-local-continuous.json", qualified)
    if "mmlu" in frozen["datasets"]:
        stages.append(
            {
                "name": "mmlu-pair",
                "script": "run_paired.py",
                "args": common
                + [
                    "--baseline-deployment",
                    baseline,
                    "--qualified-continuous",
                    str(continuous.resolve()),
                ],
            }
        )
        for attribute, flag in (("reuse_mmlu_baseline_arm", "--reuse-baseline-arm"),
                                ("reuse_mmlu_optimized_arm", "--reuse-optimized-arm")):
            source = getattr(args, attribute, None)
            if source is not None:
                stages[-1]["args"] += [flag, str(source.resolve()),
                    flag.replace("-arm", "-seal"), sha(source.resolve() / "checksums.json")]
    for name in ("frontier", "hle", "swe", "terminal"):
        if name not in frozen["datasets"]:
            continue
        for role in ("B0", "B1"):
            trace = name in ("swe", "terminal")
            stage_name = name + ("-trace-" if trace else "-solver-") + role
            spec = out / "plans" / (stage_name + ".json")
            specification = descriptor(
                name,
                out / "plans" / name,
                role,
                "trace" if trace else "solver",
                role,
                spec,
                args.smoke_evaluators and not trace,
            )
            if trace and role == "B1":
                specification["baseline_trace_run"] = str(
                    (out / (name + "-trace-B0")).resolve()
                )
                save(spec, specification)
            stages.append(
                {
                    "name": stage_name,
                    "script": "run_b1_r4.py",
                    "args": common
                    + [
                        "--role",
                        role,
                        "--kind",
                        "dataset",
                        "--dataset-spec",
                        str(spec),
                        "--qualified-continuous",
                        str(continuous.resolve()),
                    ]
                    + (["--baseline-deployment", baseline] if role == "B0" else []),
                }
            )
        if name in ("swe", "terminal"):
            continue
        for solver_role in ("B0", "B1"):
            stage_name = name + "-judge-" + solver_role
            spec = out / "plans" / (stage_name + ".json")
            responses = out / (name + "-solver-" + solver_role) / "service/evaluation"
            specification = descriptor(
                name,
                out / "plans" / name,
                "B1",
                "judge",
                solver_role,
                spec,
                args.smoke_evaluators,
                responses,
            )
            if name == "hle":
                from judge_grammar import selected

                specification["structured_output_policy"] = selected(frozen)
                save(spec, specification)
            stages.append(
                {
                    "name": stage_name,
                    "script": "run_b1_r4.py",
                    "args": common
                    + [
                        "--role",
                        "B1",
                        "--kind",
                        "dataset",
                        "--dataset-spec",
                        str(spec),
                        "--qualified-continuous",
                        str(continuous.resolve()),
                    ],
                }
            )
    save(out / "frozen-stages.json", stages)
    return stages


def execute(args, out, frozen, receipt, stages):
    reused = (
        json.loads((out / "reused-stages.json").read_bytes())["stages"]
        if (out / "reused-stages.json").exists()
        else {}
    )
    for stage, row in zip(stages, receipt["stages"], strict=True):
        if (
            (out / "reused-stages.json").exists()
            and sha(out / "reused-stages.json") != frozen["reuse_manifest_sha256"]
        ):
            raise ValueError("Reused stage map changed after prospective freeze")
        if sha(out / "frozen-stages.json") != frozen["stages_sha256"]:
            raise ValueError("Stage plan changed after prospective freeze")
        for name, expected in frozen["plan_files"].items():
            if sha(out / name) != expected:
                raise ValueError(
                    "Dataset task/config changed after prospective freeze: " + name
                )
        if script_sources() != frozen["sources"]:
            raise ValueError("Execution source changed after prospective freeze")
        if stage["name"] in reused:
            entry = reused[stage["name"]]
            row.update(status="reused_verified", source=entry)
            save(out / "execution-checklist.json", receipt)
            print("THIRD_PARTY_REUSE", stage["name"], flush=True)
            continue
        command = [
            sys.executable,
            str(ROOT / "scripts" / stage["script"]),
            *stage["args"],
            "--out",
            str(out / stage["name"]),
        ]
        row.update(status="running", started=time.time(), command=command)
        save(out / "execution-checklist.json", receipt)
        print("THIRD_PARTY_STAGE", stage["name"], flush=True)
        result = run_stage(command)
        row.update(
            status="completed" if result.returncode == 0 else "failed",
            returncode=result.returncode,
            ended=time.time(),
        )
        save(out / "execution-checklist.json", receipt)
        if result.returncode:
            raise RuntimeError(
                "Stage failed; all remaining stages retained as unexecuted: "
                + stage["name"]
            )
    for name in ("hle", "frontier"):
        if name not in frozen["datasets"]:
            continue
        module = client(name)
        for role in ("B0", "B1"):
            quality = out / (name + "-quality-" + role)
            module.summary(
                argparse.Namespace(
                    plan=str(out / "plans" / name),
                    responses=str(
                        Path(reused[name + "-solver-" + role]["source_root"])
                        / (name + "-solver-" + role)
                        / "service/evaluation"
                        if name + "-solver-" + role in reused
                        else out / (name + "-solver-" + role) / "service/evaluation"
                    ),
                    grades=str(
                        Path(reused[name + "-judge-" + role]["source_root"])
                        / (name + "-judge-" + role)
                        / "service/evaluation"
                        if name + "-judge-" + role in reused
                        else out / (name + "-judge-" + role) / "service/evaluation"
                    ),
                    output=str(quality),
                )
            )
            save(
                quality / "checksums.json",
                {p.name: sha(p) for p in quality.iterdir() if p.is_file()},
            )
    receipt.update(
        status="completed",
        selected_execution_complete=True,
        selected_scope_execution_complete=True,
        selected_full_dataset_execution_complete=not (
            (
                args.smoke_evaluators
                and any(name in frozen["datasets"] for name in ("hle", "frontier"))
            )
            or ("hle" in frozen["datasets"] and args.hle_profile != "full1811")
            or any(name in frozen["datasets"] for name in ("swe", "terminal"))
            or "mmlu" in frozen["datasets"]
        ),
    )


def run(args):
    if getattr(args, "compute_reference_only", False):
        from compute_reference import reference_only

        return reference_only(args)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "vllm-hust-third-party-execution/v1",
        "status": "preparing",
        "started": time.time(),
        "selected_execution_complete": False,
        "formal_complete": False,
        "stages": [],
        "failure": None,
    }
    result = 2
    try:
        usage = shutil.disk_usage(out)
        save(
            out / "output-volume-admission.json",
            {
                "output": str(out),
                "total_bytes": usage.total,
                "free_bytes": usage.free,
                "required_startup_reserve_bytes": 4 * 2**30,
                "prepare_only": args.prepare_only,
            },
        )
        if not args.prepare_only and usage.free < 4 * 2**30:
            raise RuntimeError(
                "Output filesystem has less than the original 4 GiB startup reserve; "
                "select another --out volume before SDK verification or model startup"
            )
        frozen = prepare(args, out)
        receipt["compute_reference"] = frozen["compute_reference"]
        if (
            not args.prepare_only
            and not getattr(args, "performance_only", False)
            and frozen["compute_reference"]["status"] not in ("ready", "automatic_ready")
        ):
            raise RuntimeError(
                "MMLU200有效算力参考核验未通过，未启动性能模型；详见plans/compute-reference/admission.json。"
                "仅性能执行须显式--performance-only，不能据此宣称四目标完整验收。"
            )
        stages = build_stages(args, out, frozen)
        receipt.update(
            evidence_mode="new_native_execution",
            reused_inference_stages=0,
            new_inference_stages=len(stages),
        )
        if getattr(args, "reuse_completed_campaign", []) and not args.prepare_only:
            from reuse_campaign import prepare as prepare_reuse, wire_dependencies

            reused = prepare_reuse(args.reuse_completed_campaign, out, frozen, stages)
            wire_dependencies(out, stages, reused)
            frozen["reuse_manifest_sha256"] = sha(out / "reused-stages.json")
            receipt.update(
                evidence_mode="composition_of_sealed_native_runs",
                reused_inference_stages=len(reused),
                new_inference_stages=len(stages) - len(reused),
            )
        composed_pairs = [stage for stage in stages if stage["name"] == "mmlu-pair"
            and all(flag in stage["args"] for flag in ("--reuse-baseline-arm", "--reuse-optimized-arm"))]
        if any(stage["name"] == "mmlu-pair" and any(flag in stage["args"] for flag in
                ("--reuse-baseline-arm", "--reuse-optimized-arm")) for stage in stages):
            receipt["evidence_mode"] = "composition_of_sealed_native_runs"
        receipt["new_comparison_stages"] = len(composed_pairs)
        receipt["new_inference_stages"] -= len(composed_pairs)
        frozen["stages_sha256"] = sha(out / "frozen-stages.json")
        frozen["plan_files"] = {
            str(p.relative_to(out)): sha(p)
            for p in (out / "plans").rglob("*")
            if p.is_file()
        }
        save(out / "frozen-execution-plan.json", frozen)
        receipt["stages"] = [
            {"name": stage["name"], "status": "planned"} for stage in stages
        ]
        save(out / "execution-checklist.json", receipt)
        receipt["prepared_plan_sha256"] = sha(out / "frozen-execution-plan.json")
        if args.prepare_only:
            receipt["status"] = "prepared_only"
            result = 0
        else:
            execute(args, out, frozen, receipt, stages)
            from compute_reference import execute as execute_reference
            from compute_reference import target as reference_target

            receipt["compute_reference"] = execute_reference(
                out, frozen["compute_reference"]
            )
            result = (
                0
                if (
                    getattr(args, "performance_only", False)
                    or reference_target(receipt["compute_reference"])["status"] == "current_workload_measured"
                )
                else 2
            )
    except BaseException as error:
        receipt.update(
            status="failed", failure=type(error).__name__ + ": " + str(error)
        )
        print("THIRD_PARTY_FAILED", receipt["failure"], flush=True)
    finally:
        receipt["ended"] = time.time()
        save(out / "campaign.json", receipt)
        save(out / "execution-checklist.json", receipt)
        bound = [
            p
            for p in out.iterdir()
            if p.is_file() and p.name != "campaign-checksums.json"
        ]
        bound += list((out / "plans").rglob("*")) if (out / "plans").exists() else []
        bound += [
            p for p in out.rglob("checksums.json") if "compiler-cache" not in str(p)
        ]
        bound += list(out.rglob("pair-checksums.json"))
        bound += [
            path for path in (out / "execution-source").rglob("*") if path.is_file()
        ]
        save(
            out / "campaign-checksums.json",
            {str(p.relative_to(out)): sha(p) for p in bound if p.is_file()},
        )
        try:
            from report_environment import report_python

            python, env = report_python(args.report_python)
            subprocess.run(
                [
                    str(python),
                    str(ROOT / "scripts/third_party_report.py"),
                    "--run",
                    str(out),
                    "--out",
                    str(out.parent / (out.name + "-report")),
                ],
                env=env,
                check=True,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            print("REPORT_FAILED", type(error).__name__ + ": " + str(error), flush=True)
            result = 2
    return result


def main():
    parser = argparse.ArgumentParser(
        prog="one_click.py",
        description=__doc__,
        epilog="MMLU-Pro fixed200; HLE stratified200(23images,source1811), --hle-profile full1811 for full source; Frontier160. SWE/Terminal default to published trace fixed32-task samples, at most64requests each, max2048/concurrency8, no task solve-rate grading. Frozen protocols are in protocols/. "
        + NOTICE_ZH
        + " "
        + NOTICE_EN
        + " 验收采用一个固定数据集满足同一项目标的全部条件，不要求五项均达标；五项结果披露与专项要求不变。 One fixed profile meeting all conditions of a target is sufficient; never combine metrics from different profiles. Full result disclosure and context-specific requirements remain unchanged.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine-profile", choices=["r4", "r5", "r6", "r7", "r8"], default="r6")
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--b1-python", type=Path)
    parser.add_argument("--baseline-deployment", type=Path)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".cache/vllm-hust-engine/third-party",
    )
    parser.add_argument("--physical-devices", default="5,6")
    parser.add_argument("--devices")
    parser.add_argument("--datasets", default="mmlu,hle,frontier,swe,terminal")
    parser.add_argument(
        "--hle-judge-whitespace",
        choices=("bounded-v2", "bounded-v1", "unrestricted"),
        default="bounded-v2",
        help="HLE judge defaults to bounded structural whitespace and integer0..100 confidence, matching the unchanged strict parser. Strings/prompts/budgets remain unchanged. bounded-v1 and unrestricted reproduce prior grammar revisions; policy changes rerun judges only.",
    )
    parser.add_argument("--dataset-assets", type=Path)
    parser.add_argument(
        "--hle-profile",
        choices=("fixed200-v1", "full1811"),
        default="fixed200-v1",
        help="Versioned HLE scope: default Gold text64/image10, Revision text113/image13; original generation/scoring budgets unchanged",
    )
    parser.add_argument("--project-evidence", type=Path)
    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="Explicitly run only the selected performance/quality profiles, without effective-compute certification. Default execution automatically prepares and runs the shipped current-workload reference-certification stages; incomplete certification produces a report and nonzero exit.",
    )
    parser.add_argument(
        "--compute-reference-plan",
        type=Path,
        help="Frozen live MMLU200 independent-reference plan; automatically runs calibration, validation, target observation and synchronized timing after performance. Optional expert override. Without this option, the one-click entry automatically prepares and runs shipped reference-certification stages; it never substitutes historical efficiency.",
    )
    parser.add_argument(
        "--compute-reference-sha256",
        help="Independently frozen SHA256 required when explicitly requesting a new live reference plan",
    )
    parser.add_argument(
        "--compute-reference-only",
        action="store_true",
        help="Append live MMLU200 reference capture to a sealed campaign without rerunning any performance stage",
    )
    parser.add_argument(
        "--reference-campaign",
        type=Path,
        help="Completed sealed performance campaign bound by --compute-reference-only",
    )
    parser.add_argument("--reference-profile", choices=("submission-v2", "full-answer-v1"), default="submission-v2",
        help="Compute capture: submission-v2 fixes MMLU200/out256, three native repetitions per arm; full-answer-v1 reproduces the earlier out8192 single-repetition development protocol. Quality/performance budgets stay unchanged.")
    parser.add_argument("--reference-qualification-source", type=Path,
        help="Explicit sealed compute-reference directory; reuse unchanged passed operator qualifications after checking every source, control, receipt and artifact. Model timings are never reused by this option.")
    parser.add_argument("--reference-resume-source", type=Path,
        help="Explicit sealed compute-reference directory with all three completed/quiescent unchanged model stages; continue arithmetic/reference certification without rerunning those model services.")
    parser.add_argument("--reference-canonical-source", type=Path,
        help="Explicit sealed exact canonical dense calibration; independently re-audit all original samples, device mapping, source shapes and statistical controls before reuse.")
    parser.add_argument("--reference-observation-source", type=Path,
        help="Explicit complete sealed HTTP observation directory for the same frozen SDK/devices; re-audit source and producer applicability without reusing its counts or time for native arithmetic.")
    parser.add_argument(
        "--continuous-evidence",
        type=Path,
        help="Explicit already sealed local qualification; new third-party machines should omit and rerun C8",
    )
    parser.add_argument("--reuse-mmlu-baseline-arm", type=Path,
        help="Explicit sealed original200 B0 arm; preserve its original runtime/date and validate current matched task/warmup conditions.")
    parser.add_argument("--reuse-mmlu-optimized-arm", type=Path,
        help="Explicit sealed original200 B1 arm of the exact current deployment, such as an already completed integrated validation.")
    parser.add_argument("--report-python", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--reuse-completed-campaign",
        type=Path,
        action="append",
        default=[],
        help="Explicit same-host sealed campaign; repeat for disjoint completed stages. Re-audit original streams and runtime/input identities; never rerun admitted stages.",
    )
    parser.add_argument(
        "--smoke-evaluators",
        action="store_true",
        help="Explicit HLE4/Frontier2 diagnostic; MMLU, if selected, remains the original full200",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    with cancellation_handlers():
        raise SystemExit(main())
