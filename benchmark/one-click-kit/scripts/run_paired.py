#!/usr/bin/env python3
"""Freeze and execute a new original MMLU200 B0/B1 pair without retries."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from baseline_admission import admit
from child_runner import cancellation_handlers, run_stage
from dependency_admission import admit_dependencies
from qualification import qualify
from restore_b1_r4 import verify_installed_sources
from mmlu_scope import verify as verify_mmlu_scope, description as mmlu_description


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--baseline-deployment", type=Path, required=True)
    parser.add_argument("--physical-devices", default="5,6")
    parser.add_argument("--qualified-continuous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--model", type=Path, default=Path("/models/Qwen3.5-35B-A3B"))
    parser.add_argument("--report-python", type=Path)
    parser.add_argument("--reuse-baseline-arm", type=Path)
    parser.add_argument("--reuse-optimized-arm", type=Path)
    parser.add_argument("--reuse-baseline-seal")
    parser.add_argument("--reuse-optimized-seal")
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    try:
        return execute_pair(args, root)
    except BaseException as error:
        if not (root / "receipt.json").is_file():
            save(
                root / "receipt.json",
                {
                    "status": "failed",
                    "completed": False,
                    "formal_complete": False,
                    "planned": ["B0", "B1"],
                    "attempts": [],
                    "model_service_started": False,
                    "failure": type(error).__name__ + ": " + str(error),
                    "ended": time.time(),
                    "quiescent": True,
                },
            )
            save(
                root / "pair-checksums.json",
                {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in root.iterdir()
                    if p.is_file() and p.name != "pair-checksums.json"
                },
            )
        print(
            "PAIRED_EXECUTION_FAILED",
            type(error).__name__ + ": " + str(error),
            flush=True,
        )
        return 2


def execute_pair(args, root):
    import os

    os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = args.physical_devices
    deployment = args.deployment.resolve()
    verify_installed_sources(deployment)
    save(root / "mmlu-task-scope.json", verify_mmlu_scope(deployment))
    baseline = admit(args.baseline_deployment, root / "baseline-admission.json")
    optimized = admit_dependencies(deployment)
    qualification = qualify(args.qualified_continuous, deployment)
    from arm_reuse import admit as admit_arm, expected as expected_arm

    reused = {}
    reuse_paths = {"B0": getattr(args, "reuse_baseline_arm", None),
                "B1": getattr(args, "reuse_optimized_arm", None)}
    if any(reuse_paths.values()):
        conditions = expected_arm(deployment, args.baseline_deployment, args.model,
            [int(x) for x in args.physical_devices.split(",")], args.devices,
            qualification["weights"])
        for role, source in reuse_paths.items():
            if source is None:
                continue
            source = source.resolve()
            if source == root or source.is_relative_to(root) or root.is_relative_to(source):
                raise ValueError("Reused arm and new pair must be independent directories")
            pin = getattr(args, "reuse_baseline_seal" if role == "B0" else "reuse_optimized_seal", None)
            if pin and hashlib.sha256((source / "checksums.json").read_bytes()).hexdigest() != pin:
                raise ValueError("Selected original arm seal changed after campaign preparation")
            reused[role] = admit_arm(source, role, conditions)
    scripts = Path(__file__).parent
    native_names = sorted(path.name for path in scripts.glob("*.py"))
    native = {
        name: hashlib.sha256((scripts / name).read_bytes()).hexdigest()
        for name in native_names
    }
    frozen = {
        "schema": "vllm-hust-prospective-original-pair/v1",
        "frozen_before_first_arm": time.time(),
        "roles": ["B0", "B1"],
        "model": str(args.model.resolve()),
        "physical_devices": [int(x) for x in args.physical_devices.split(",")],
        "logical_devices": args.devices,
        "task_plan_sha256": hashlib.sha256(
            (
                deployment / "inputs/mmlu-quality018-r2/quality-request-plan.json"
            ).read_bytes()
        ).hexdigest(),
        "scorer_sha256": hashlib.sha256(
            (deployment / "scorers/mmlu_pro_pyramidkv.py").read_bytes()
        ).hexdigest(),
        "requests": 200,
        "mmlu_task_scope": mmlu_description(),
        "max_completion_tokens": 8192,
        "temperature": 0,
        "request_seed": 0,
        "arrival": "one simultaneous200 batch",
        "eos_enabled": True,
        "additional_task_warmups": 0,
        "failure_policy": "No retries, replacements, length reductions or hidden fallback; retain all200 denominators",
        "baseline": {
            "python": baseline["python"],
            "versions": baseline["versions"],
            "source_map_sha256": baseline["source_map_sha256"],
            "adaptations": baseline["adaptations"],
            "historical_sdk_byte_equivalence": False,
        },
        "optimized": optimized,
        "qualified_continuous": {
            "run": qualification["run"],
            "seal_sha256": qualification["seal_sha256"],
        },
        "native_controller_sources": native,
        "cache_policy": "New services and compiler caches; same checkpoint-only clean-cache hints and raw cgroup reserves. No model writes or global cache flush.",
        "measurement": "Original first-submission through response-drain clocks; actual successful output tokens; per-request equal-weight TTFT/TPOT",
        "scope": "New full-answer MMLU200 pair; other datasets and existing M01 reference are separate evidence, not silently supplied by this pair",
        "optimized_deployment_sha256": hashlib.sha256((deployment / "deployment.json").read_bytes()).hexdigest(),
        "optimized_installed_files_sha256": hashlib.sha256((deployment / "provenance/installed-files.json").read_bytes()).hexdigest(),
    }
    from serving_warmup import selected, protocol

    profile = json.loads((deployment / "deployment.json").read_bytes())
    if selected(profile):
        frozen["schema"] = "vllm-hust-prospective-original-pair/v2"
        frozen["serving_warmup"] = protocol(capacity36=profile["release"] == "full-answer-b1-20261117-r8")
        frozen["engine_release"] = profile["release"]
    if reused:
        frozen.update(schema="vllm-hust-prospective-original-pair/v3",
            reused_arms=reused, evidence_mode="composition_of_sealed_arms",
            cache_policy="Earlier arms keep their original services/caches/dates; newly executed arms use fresh caches under the unchanged warmup protocol.",
            scope="Prospectively compose the explicitly selected sealed original200 arms with any new arms; never relabel an old run as new inference.")
    save(root / "frozen-pair-plan.json", frozen)
    receipt = {
        "status": "running",
        "completed": False,
        "attempts": [],
        "planned": frozen["roles"],
        "formal_complete": False,
        "evidence_mode": "composition_of_sealed_arms" if reused else "new_native_execution",
        "new_inference_arms": len(frozen["roles"]) - len(reused),
        "reused_arms": sorted(reused),
    }
    save(root / "receipt.json", receipt)
    result = 2
    try:
        for role in frozen["roles"]:
            for name, digest in native.items():
                if hashlib.sha256((scripts / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(
                        "Native controller changed after prospective freeze: " + name
                    )
            if role in reused:
                from arm_reuse import recheck

                recheck(reused[role])
                receipt["attempts"].append(dict(role=role, reused_verified=True,
                    source_seal_sha256=reused[role]["seal_sha256"]))
                save(root / "receipt.json", receipt)
                print("PAIRED_ARM_REUSED", role, reused[role]["run"], flush=True)
                continue
            command = [
                sys.executable,
                str(scripts / "run_b1_r4.py"),
                "--deployment",
                str(deployment),
                "--role",
                role,
                "--out",
                str(root / role),
                "--devices",
                args.devices,
                "--model",
                str(args.model.resolve()),
                "--kind",
                "joint",
                "--qualified-continuous",
                str(args.qualified_continuous.resolve()),
            ]
            command += ["--physical-devices", args.physical_devices]
            if role == "B0":
                command += [
                    "--baseline-deployment",
                    str(args.baseline_deployment.resolve()),
                ]
            if args.report_python:
                command += ["--report-python", str(args.report_python.absolute())]
            save(
                root / (role + "-command.json"),
                {
                    "argv": command,
                    "pair_plan_sha256": hashlib.sha256(
                        (root / "frozen-pair-plan.json").read_bytes()
                    ).hexdigest(),
                },
            )
            child = run_stage(command)
            receipt["attempts"].append({"role": role, "returncode": child.returncode})
            save(root / "receipt.json", receipt)
            if child.returncode:
                receipt.update(
                    status="failed",
                    failure="Original arm failed; no retry or replacement",
                )
                return 2
        receipt.update(status="completed", completed=True)
        result = 0
    except BaseException as error:
        receipt.update(
            status="failed", failure=type(error).__name__ + ": " + str(error)
        )
        raise
    finally:
        receipt["ended"] = time.time()
        save(root / "receipt.json", receipt)
        bound = [
            root / "frozen-pair-plan.json",
            root / "receipt.json",
            root / "baseline-admission.json",
            root / "mmlu-task-scope.json",
        ]
        bound += [root / role / "checksums.json" for role in frozen["roles"] if role not in reused]
        save(
            root / "pair-checksums.json",
            {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in bound
                if p.is_file()
            },
        )
        print("PAIRED_EXECUTION", root / "receipt.json", flush=True)
    if result == 0:
        from report_environment import report_python

        python, env = report_python(args.report_python)
        aggregate = subprocess.run(
            [
                str(python),
                str(scripts / "paired_execution_report.py"),
                "--run",
                str(root),
                "--out",
                str(root.parent / (root.name + "-report")),
            ],
            env=env,
            check=False,
        )
        if aggregate.returncode:
            return 2
    return result


if __name__ == "__main__":
    with cancellation_handlers():
        raise SystemExit(main())
