"""Install frozen r8, run fixed-five and context/tenant tests, and retain reports."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent
PATH_FIELDS = {"model", "cache", "out", "baseline_deployment", "dataset_assets", "env_file"}
FIELDS = PATH_FIELDS | {"physical_devices", "devices"}


def configuration(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or set(data) - FIELDS:
        raise ValueError("Use the fields in benchmark/run-all.example.json")
    for name in ("model", "physical_devices", "cache", "out"):
        if name not in data:
            raise ValueError("Missing configuration field: " + name)
    for name in PATH_FIELDS & data.keys():
        if not isinstance(data[name], str) or not data[name].strip():
            raise ValueError("Expected a path: " + name)
        value = Path(data[name]).expanduser()
        data[name] = str((path.parent / value).resolve())
    for name in ("physical_devices", "devices"):
        if name not in data:
            continue
        pair = data[name]
        if (not isinstance(pair, list) or len(pair) != 2
            or any(type(x) is not int or x < 0 for x in pair) or pair[0] == pair[1]):
            raise ValueError(name + " requires two distinct non-negative integers")
    if not Path(data["model"]).is_dir():
        raise ValueError("Model directory does not exist: " + data["model"])
    if os.path.lexists(data["out"]):
        raise ValueError("Choose a new output directory: " + data["out"])
    return data


def run(data):
    out = Path(data["out"])
    out.mkdir(parents=True, exist_ok=False)
    installation = out / "installation"
    cli = str(Path(sysconfig.get_path("scripts")) / "vllm-hust-engine")
    receipt = {"status": "running", "stages": [], "third_party_verification": "pending"}

    def save():
        (out / "results.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")

    def stage(name, args):
        record = {"name": name, "status": "running"}
        receipt["stages"].append(record)
        save()
        print(name, flush=True)
        try:
            result = run_owned_stage(args)
        except BaseException as error:
            record.update(status="failed", failure=str(error) or type(error).__name__)
            save()
            raise
        record.update(returncode=result.returncode,
                      status="completed" if result.returncode == 0 else "failed")
        save()
        if result.returncode:
            raise RuntimeError(name + " failed; original results retained, no automatic retry")

    try:
        from one_click import verify_kit
        verify_kit()
        sys.path.insert(0, str(ROOT / "benchmark/one-click-kit/scripts"))
        from child_runner import run_stage as run_owned_stage
        stage("install-package", [sys.executable, "-m", "pip", "install", "--no-deps",
              "-r", str(ROOT / "engine/r8-pip-requirements-deeplinkengine.txt")])
        stage("verify-engine", [cli, "verify"])
        stage("install-engine", [cli, "install", "--out", str(installation)])
        evaluation = dict(data, engine_profile="r8", test_mode="full", reference_profile="submission-v2",
                          deployment=str(installation / "sdk"), out=str(out / "fixed-five"))
        config = out / "evaluation.json"
        config.write_text(json.dumps(evaluation, indent=2) + "\n")
        # The frozen distribution retains its original evaluator. Route new
        # assessments through the independently versioned public repair kit.
        evaluate = [sys.executable, str(ROOT / "one_click.py")]
        stage("prepare-fixed-five", [*evaluate, "prepare", "--config", str(config)])
        stage("run-fixed-five", [*evaluate, "run", "--config", str(config)])
        campaign = json.loads((out / "fixed-five/campaign.json").read_text())
        sys.path.insert(0, str(ROOT / "benchmark/one-click-kit/scripts"))
        from compute_reference import target
        reference = campaign.get("compute_reference", {})
        reference_target = target(reference)
        receipt.update(compute_reference=reference, compute_reference_target=reference_target)
        save()
        if (reference.get("certificate_kind") != "qualified_full_reference_denominator_reuse/v1"
                or reference_target["status"] != "current_workload_measured"):
            raise RuntimeError("Current-workload effective-compute reference remains incomplete")
        plan = json.loads((out / "fixed-five/frozen-execution-plan.json").read_text())
        supplement = [*evaluate, "--context-tenant-only", "--deployment", plan["optimized_deployment"],
                      "--baseline-deployment", plan["baseline_deployment"],
                      "--b1-python", sys.executable, "--model", data["model"],
                      "--cache", data["cache"], "--out", str(out / "context-tenant"),
                      "--physical-devices", ",".join(map(str, data["physical_devices"])),
                      "--devices", plan["logical_devices"]]
        stage("run-context-tenant", supplement)
        reports = {"fixed_five": "fixed-five-report/report.html",
                   "context_tenant": "context-tenant/paired-report/report.html"}
        for path in reports.values():
            if not (out / path).is_file():
                raise RuntimeError("Missing generated report: " + path)
        receipt.update(status="completed", reports=reports,
                       fixed_five_execution=campaign.get("selected_execution_complete"),
                       compute_reference=reference, compute_reference_target=reference_target)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        receipt.update(status="failed", error=str(error) or type(error).__name__)
        print(receipt["error"], file=sys.stderr)
        return 1
    finally:
        save()
        print("RESULTS", out / "results.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        data = configuration(args.config)
        if sys.version_info[:2] != (3, 12) or platform.system() != "Linux" or platform.machine() != "aarch64":
            raise ValueError("Requires CPython 3.12, Linux aarch64 and the documented Ascend runtime")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return run(data)


if __name__ == "__main__":
    raise SystemExit(main())
