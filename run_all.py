"""Install frozen r8, run fixed-five and context/tenant tests, and retain reports."""
import argparse
import json
import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
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
        data[name] = (os.path.abspath(path.parent / value) if name == "out"
                      else str((path.parent / value).resolve()))
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


def preflight():
    if (sys.version_info[:2] != (3, 12) or platform.python_implementation() != "CPython"
        or platform.system() != "Linux" or platform.machine() != "aarch64"):
        raise ValueError("Requires CPython 3.12, Linux aarch64 and the documented Ascend runtime")
    lock = json.loads((ROOT / "engine/r8-lock.json").read_text())
    differences = []
    for name, expected in lock["required_python_packages"].items():
        try:
            observed = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed = "missing"
        if observed != expected:
            differences.append(f"{name}: expected {expected}, found {observed}")
    if differences:
        raise ValueError("Prepare the required environment before installation: " + "; ".join(differences))


def completed_evidence(out):
    campaign = json.loads((out / "fixed-five/campaign.json").read_text())
    execution = json.loads((out / "context-tenant/execution.json").read_text())
    pair = json.loads((out / "context-tenant/paired-summary.json").read_text())
    phases = [row["phase"] for row in pair["comparison"]]
    if (campaign["status"] != "completed" or campaign["selected_execution_complete"] is not True
        or execution["status"] != "completed" or pair["execution_complete"] is not True
        or phases != ["4096-formal", "16384-formal", "32768-formal", "multi-tenant-formal"]):
        raise ValueError("Campaign evidence does not show all planned tests completed")
    reports = {"fixed_five": "fixed-five-report/report.html",
               "context_tenant": "context-tenant/paired-report/report.html"}
    for path in reports.values():
        if not (out / path).is_file():
            raise ValueError("Missing generated report: " + path)
    records = ["installation/installation.json", "fixed-five/campaign.json",
               "fixed-five/frozen-execution-plan.json", "context-tenant/execution.json",
               "context-tenant/paired-summary.json", *reports.values()]
    hashes = {}
    for name in records:
        with (out / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return campaign, reports, hashes


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
            result = subprocess.run(args, cwd=ROOT, check=False)
        except BaseException as error:
            record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "launch_failed",
                          error=type(error).__name__ + ": " + str(error))
            save()
            raise
        record.update(returncode=result.returncode,
                      status="completed" if result.returncode == 0 else "failed")
        save()
        if result.returncode:
            raise RuntimeError(name + " failed; original results retained, no automatic retry")

    try:
        stage("install-package", [sys.executable, "-m", "pip", "install", "--no-deps",
              "-r", str(ROOT / "engine/r8-pip-requirements-deeplinkengine.txt")])
        stage("verify-engine", [cli, "verify"])
        stage("install-engine", [cli, "install", "--out", str(installation)])
        evaluation = dict(data, engine_profile="r8", test_mode="full",
                          deployment=str(installation / "sdk"), out=str(out / "fixed-five"))
        config = out / "evaluation.json"
        config.write_text(json.dumps(evaluation, indent=2) + "\n")
        evaluate = [cli, "evaluate", "--installation", str(installation), "--"]
        stage("prepare-fixed-five", [*evaluate, "prepare", "--config", str(config)])
        stage("run-fixed-five", [*evaluate, "run", "--config", str(config)])
        plan = json.loads((out / "fixed-five/frozen-execution-plan.json").read_text())
        # The frozen package evaluate() injects --engine-profile, which this
        # separate frozen controller does not accept. Verify its source first.
        stage("verify-installed-tools", [sys.executable, "-c",
              "from pathlib import Path; from vllm_hust_engine.cli import verify_tools; "
              "import sys; verify_tools(Path(sys.argv[1]))", str(installation / "tools")])
        supplement = [sys.executable, str(installation / "tools/one_click.py"), "--context-tenant-only", "--deployment", plan["optimized_deployment"],
                      "--baseline-deployment", plan["baseline_deployment"],
                      "--b1-python", sys.executable, "--model", data["model"],
                      "--cache", data["cache"], "--out", str(out / "context-tenant"),
                      "--physical-devices", ",".join(map(str, data["physical_devices"])),
                      "--devices", plan["logical_devices"]]
        stage("run-context-tenant", supplement)
        campaign, reports, hashes = completed_evidence(out)
        receipt.update(status="completed", reports=reports, evidence_sha256=hashes,
                       fixed_five_execution=campaign["selected_execution_complete"],
                       compute_reference=campaign.get("compute_reference"))
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
        preflight()
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return run(data)


if __name__ == "__main__":
    raise SystemExit(main())
