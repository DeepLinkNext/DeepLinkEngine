"""Run a sealed B0/B1 context/tenant supplement without rerunning fixed-five tasks."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from acceptance_client import load
from acceptance_report import inspect
from agent_trace import save, sha
from child_runner import cancellation_handlers, run_stage

ROOT = Path(__file__).resolve().parents[1]


def provenance():
    return {
        str(p.relative_to(ROOT)): sha(p)
        for p in sorted((ROOT / "scripts").iterdir())
        if p.is_file()
    }


def admit_reuse(prior, role, args, contract):
    """Only reuse a released, complete, identically bound supplementary arm."""
    prior = Path(prior).resolve()
    result = inspect(prior)
    receipt = json.loads((prior / "receipt.json").read_bytes())
    admission = json.loads((prior / "source-admission.json").read_bytes())
    requested, _ = load(contract)
    if (
        not result["execution_complete"]
        or result["role"] != role
        or result["contract"] != requested
    ):
        raise ValueError("Reuse arm does not prove this exact complete supplement")
    if admission["deployment"] != str(args.deployment.resolve()):
        raise ValueError("Reuse uses a different deployment")
    if json.loads((prior / "controller-source.json").read_bytes()) != provenance():
        raise ValueError("Reuse controller code differs; no silent adaptation")
    cards = json.loads((prior / "resource-admission.json").read_bytes())[
        "physical_cards"
    ]
    if list(map(int, args.physical_devices.split(","))) != cards:
        raise ValueError("Reuse uses another physical card pair")
    weights = json.loads((prior / "model-weights.json").read_bytes())
    for name, frozen in weights.items():
        path = args.model / name
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (
            frozen["bytes"],
            frozen["mtime_ns"],
            frozen["inode"],
        ) or sha(path) != frozen["sha256"]:
            raise ValueError("Reuse checkpoint identity changed")
    if role == "B0":
        from baseline_admission import admit

        target = admit(
            args.baseline_deployment, args.out / "reuse-baseline-source-admission.json"
        )
        captured = json.loads((prior / "baseline-source-admission.json").read_bytes())
        if target != captured:
            raise ValueError("Reuse baseline identity differs")
    return prior, result, receipt


def paired(arms):
    reports = {role: inspect(path) for role, path in arms.items()}
    if set(reports) != {"B0", "B1"} or not all(
        r["execution_complete"] for r in reports.values()
    ):
        raise ValueError("A complete sealed B0/B1 supplement is required")
    if any(report["role"] != role for role, report in reports.items()):
        raise ValueError("Actual supplementary arm role differs")
    plan = json.loads((arms["B1"] / "service/plan.json").read_bytes())
    if plan["acceptance_suite"]["baseline_run"] != str(arms["B0"].resolve()):
        raise ValueError("B1 was not admitted against this exact B0 supplement")
    for name in ("model-weights.json", "resource-admission.json"):
        left = json.loads((arms["B0"] / name).read_bytes())
        right = json.loads((arms["B1"] / name).read_bytes())
        if name == "resource-admission.json":
            left, right = left["physical_cards"], right["physical_cards"]
        if left != right:
            raise ValueError("Paired cards or model weights differ")
    if reports["B0"]["contract"] != reports["B1"]["contract"]:
        raise ValueError("Paired supplement contracts differ")
    comparison = []
    for a, b in zip(reports["B0"]["phases"], reports["B1"]["phases"], strict=True):
        if a["phase"] != b["phase"]:
            raise ValueError("Paired phase order differs")
        if not a["phase"].endswith("-formal"):
            continue
        ma, mb = a["metrics"], b["metrics"]
        comparison.append(
            dict(
                phase=a["phase"],
                throughput_ratio=mb["output_tokens_per_second"]
                / ma["output_tokens_per_second"],
                equal_rate_same_card_count_unit_cost_reduction_percent=100
                * (1 - ma["output_tokens_per_second"] / mb["output_tokens_per_second"]),
                mean_ttft_reduction_percent=100
                * (1 - mb["ttft_ms"]["mean"] / ma["ttft_ms"]["mean"]),
                mean_tpot_reduction_percent=100
                * (1 - mb["tpot_ms"]["mean"] / ma["tpot_ms"]["mean"]),
                B0=a,
                B1=b,
            )
        )
    return dict(
        schema="vllm-hust-paired-context-tenant-supplement/v1",
        execution_complete=True,
        arms={role: str(path) for role, path in arms.items()},
        arm_seals={role: sha(path / "checksums.json") for role, path in arms.items()},
        contract=reports["B0"]["contract"],
        comparison=comparison,
        fixed_five_datasets_rerun=False,
        quality_measured=False,
        effective_compute_reference_included=False,
        acceptance="Measurements collected; assess the original qualitative stability/utilization criteria from all curves. No new numeric pass cutoff or quality equivalence is invented.",
    )


def run(args, checkpoint=None):
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(
        schema="vllm-hust-acceptance-supplement-execution/v1",
        status="preparing",
        planned_roles=["B0", "B1"],
        started=time.time(),
        stages=[],
        sources=provenance(),
    )
    save(output / "execution.json", state)
    environment = dict(
        os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1"
    )
    try:
        contract = args.contract
        if contract is None:
            command = [
                str(args.b1_python),
                str(ROOT / "scripts/acceptance_inputs.py"),
                "--model",
                str(args.model),
                "--cache",
                str(args.cache),
                "--out",
                str(output / "inputs"),
            ]
            with (output / "input-preparation.log").open("xb") as stream:
                result = run_stage(
                    command, env=environment, stdout=stream, stderr=stream
                )
            if result.returncode:
                raise RuntimeError("Real input freezing failed; original log retained")
            contract = output / "inputs/contract.json"
        from acceptance_admission import prepare

        inputs = prepare(contract, output)
        contract = Path(inputs["contract"])
        state.update(
            status="running", input_admission=inputs, contract_sha256=sha(contract)
        )
        save(output / "execution.json", state)
        arms = {}
        for role in ["B0", "B1"]:
            prior = getattr(args, "reuse_" + role.lower(), None)
            if prior:
                root, _, _ = admit_reuse(prior, role, args, contract)
                stage = dict(role=role, run=str(root), reused=True, returncode=0)
            else:
                root = output / role
                command = [
                    sys.executable,
                    str(ROOT / "scripts/run_b1_r4.py"),
                    "--deployment",
                    str(args.deployment),
                    "--role",
                    role,
                    "--baseline-deployment",
                    str(args.baseline_deployment),
                    "--out",
                    str(root),
                    "--devices",
                    args.devices,
                    "--physical-devices",
                    args.physical_devices,
                    "--model",
                    str(args.model),
                    "--kind",
                    "continuous",
                    "--acceptance-contract",
                    str(contract),
                ]
                if role == "B1":
                    command += ["--acceptance-baseline", str(arms["B0"])]
                if args.report_python:
                    command += ["--report-python", str(args.report_python)]
                stage = dict(
                    role=role,
                    run=str(root),
                    reused=False,
                    command=command,
                    status="running",
                )
                state["stages"].append(stage)
                save(output / "execution.json", state)
                print("ACCEPTANCE_ARM_START", role, root, flush=True)
                with (output / (role + "-controller.log")).open("xb") as stream:
                    result = run_stage(
                        command, env=environment, stdout=stream, stderr=stream
                    )
                stage.update(
                    returncode=result.returncode,
                    status="completed" if result.returncode == 0 else "failed",
                )
                save(output / "execution.json", state)
                if checkpoint is not None:
                    checkpoint(role, root, stage)
                if result.returncode:
                    raise RuntimeError(role + " supplement failed; no automatic retry")
            if prior:
                state["stages"].append(stage)
            verified = inspect(root)
            if not verified["execution_complete"]:
                raise ValueError("Arm is not a complete validated supplement")
            stage.update(status="completed", seal_sha256=sha(root / "checksums.json"))
            arms[role] = root
            save(output / "execution.json", state)
            print("ACCEPTANCE_ARM_COMPLETE", role, root, flush=True)
        save(output / "paired-summary.json", paired(arms))
        from report_environment import report_python

        python, env = report_python(args.report_python)
        command = [
            str(python),
            str(ROOT / "scripts/acceptance_report.py"),
            "--paired-summary",
            str(output / "paired-summary.json"),
            "--out",
            str(output / "paired-report"),
        ]
        with (output / "paired-report.log").open("xb") as stream:
            rendered = run_stage(command, env=env, stdout=stream, stderr=stream)
        if rendered.returncode:
            raise RuntimeError(
                "Paired plotting failed; completed arm evidence retained for CPU-only recomputation"
            )
        state["status"] = "completed"
        return 0
    except BaseException as error:
        state.update(status="failed", failure=type(error).__name__ + ": " + str(error))
        print("ACCEPTANCE_SUPPLEMENT_FAILED", state["failure"], flush=True)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return 2
    finally:
        state["ended"] = time.time()
        save(output / "execution.json", state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deployment", type=Path, required=True)
    p.add_argument("--baseline-deployment", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--b1-python", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--devices", required=True)
    p.add_argument("--physical-devices", required=True)
    p.add_argument(
        "--contract",
        type=Path,
        help="Already frozen real inputs; otherwise prepared automatically",
    )
    p.add_argument("--reuse-b0", type=Path)
    p.add_argument("--reuse-b1", type=Path)
    p.add_argument("--report-python", type=Path)
    with cancellation_handlers():
        return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
