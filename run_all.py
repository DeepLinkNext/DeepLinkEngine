"""Install frozen r8, run fixed-five and context/tenant tests, and retain reports."""
import argparse
import json
import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import sys
import sysconfig
import subprocess
import uuid

ROOT = Path(__file__).resolve().parent
PATH_FIELDS = {"model", "cache", "out", "baseline_deployment", "dataset_assets", "env_file"}
FIELDS = PATH_FIELDS | {"physical_devices", "devices", "environment"}


def configuration(path, *, allow_existing=False):
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
    if "environment" in data:
        from environment_setup import OPTIONS
        options = data["environment"]
        if not isinstance(options, dict) or set(options) - OPTIONS:
            raise ValueError("environment accepts only python, wheelhouse and cann_home")
        for name, value in options.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError("environment." + name + " requires a path")
            if name != "python" or "/" in value or "\\" in value:
                options[name] = str((path.parent / Path(value).expanduser()).resolve())
    if not Path(data["model"]).is_dir():
        raise ValueError("Model directory does not exist: " + data["model"])
    if os.path.lexists(data["out"]) and not allow_existing:
        raise ValueError("Choose a new output directory: " + data["out"])
    return data


def evaluation_configuration(data, installation):
    return dict({k: v for k, v in data.items() if k != "environment"}, engine_profile="r8", test_mode="full",
                reference_profile="submission-v2", deployment=str(installation / "sdk"),
                out=str(Path(data["out"]) / "fixed-five"))


def bootstrap_resume(data):
    """Only recover a failed preflight before any inference campaign exists."""
    out = Path(data["out"])
    if out.is_symlink() or not out.is_dir():
        raise ValueError("Bootstrap recovery requires an existing result directory")
    prior = json.loads((out / "results.json").read_text())
    stages = prior.get("stages", [])
    expected = ["prepare-datasets", "install-package", "verify-engine", "install-engine", "prepare-fixed-five"]
    if (prior.get("status") != "failed" or [r.get("name") for r in stages] != expected
            or any(r.get("status") != "completed" for r in stages[:4])
            or stages[-1].get("status") not in ("failed", "launch_failed", "interrupted")):
        raise ValueError("Bootstrap recovery requires four completed installation stages and a failed fixed-five preflight")
    if any(os.path.lexists(out / name) for name in ("fixed-five", "context-tenant")):
        raise ValueError("An inference campaign already exists; bootstrap recovery will not repeat it")
    stored = json.loads((out / "evaluation.json").read_text())
    if stored != evaluation_configuration(data, out / "installation"):
        raise ValueError("Recovery configuration differs from the original full-test configuration")
    if not (out / "installation/installation.json").is_file():
        raise ValueError("The prior runtime installation is incomplete")
    return prior


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


def qualified_reference(campaign):
    sys.path.insert(0, str(ROOT / "benchmark/one-click-kit/scripts"))
    from compute_reference import target
    reference = campaign.get("compute_reference", {})
    if reference.get("certificate_kind") != "qualified_full_reference_denominator_reuse/v1":
        raise ValueError("Current-workload effective-compute reference remains incomplete")
    result = target(reference)
    if result["status"] != "current_workload_measured":
        raise ValueError("Current-workload effective-compute reference remains incomplete")
    return result


def run(data, *, resume_bootstrap=False):
    out = Path(data["out"])
    prior = bootstrap_resume(data) if resume_bootstrap else None
    out.mkdir(parents=True, exist_ok=resume_bootstrap)
    installation = out / "installation"
    cli = str(Path(sysconfig.get_path("scripts")) / "vllm-hust-engine")
    receipt = {"status": "running", "stages": [], "third_party_verification": "pending"}
    if prior is not None:
        history = out / "attempts"
        history.mkdir(exist_ok=True)
        archived = history / ("bootstrap-failure-" + uuid.uuid4().hex + ".json")
        raw = (out / "results.json").read_bytes()
        archived.write_bytes(raw)
        receipt["previous_attempts"] = [*prior.get("previous_attempts", []),
            {"path": str(archived.relative_to(out)), "sha256": hashlib.sha256(raw).hexdigest()}]

    environment_report = os.environ.get("DEEPLINKENGINE_ENVIRONMENT_REPORT")
    if environment_report:
        import shutil
        shutil.copyfile(environment_report, out / "environment.json")
        receipt["environment"] = {"path": "environment.json", "sha256": hashlib.sha256((out / "environment.json").read_bytes()).hexdigest()}

    def save():
        (out / "results.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")

    def stage(name, args, *, mode="execute"):
        record = {"name": name, "status": "running", "command": args, "mode": mode}
        receipt["stages"].append(record)
        save()
        print(name, flush=True)
        try:
            result = run_owned_stage(args)
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
        from one_click import verify_kit
        verify_kit()
        sys.path.insert(0, str(ROOT / "benchmark/one-click-kit/scripts"))
        from child_runner import run_stage as run_owned_stage
        data_command = [sys.executable, str(ROOT / "tools/prepare_datasets.py"), "--cache", data["cache"], "--context"]
        if data.get("dataset_assets"):
            data_command += ["--assets", data["dataset_assets"]]
        stage("prepare-datasets", data_command)
        install_command = [sys.executable, str(ROOT / "engine_install.py"), "--cache", data["cache"]]
        if data.get("environment", {}).get("wheelhouse"):
            install_command += ["--wheelhouse", data["environment"]["wheelhouse"]]
        if resume_bootstrap:
            stage("install-package", [sys.executable, "-c",
                  "import importlib.metadata as m,sys; assert m.version('vllm-hust-engine') == sys.argv[1]",
                  json.loads((ROOT / "engine/r8-lock.json").read_text())["public_distribution"].split("==")[1]],
                  mode="verify_existing")
        else:
            stage("install-package", install_command)
        stage("verify-engine", [cli, "verify"])
        if resume_bootstrap:
            verify_install = """import json,sys
from pathlib import Path
from vllm_hust_engine.cli import verify,audit_install,verify_tools
root=Path(sys.argv[1]);receipt=json.loads((root/'installation.json').read_text())
locked=verify();assert receipt['release']==locked['release'] and receipt['package_version']==locked['version']
profile=json.loads((root/'sdk/deployment.json').read_text());assert profile['python']==sys.executable
assert Path(receipt['deployment']).resolve()==(root/'sdk').resolve()
verify_tools(root/'tools');result=audit_install(root/'sdk',root/'tools')
assert result['verified_runtime_files']==locked['runtime_files']
print('EXISTING_INSTALLATION_VERIFIED',result['verified_runtime_files'])
"""
            stage("install-engine", [sys.executable, "-c", verify_install, str(installation)], mode="verify_existing")
        else:
            stage("install-engine", [cli, "install", "--out", str(installation)])
        evaluation = evaluation_configuration(data, installation)
        config = out / "evaluation.json"
        config.write_text(json.dumps(evaluation, indent=2) + "\n")
        evaluate = [sys.executable, str(ROOT / "one_click.py")]
        prepare_command = [*evaluate, "prepare", "--config", str(config)]
        if resume_bootstrap:
            prepare_command += ["--out", str(out / ("preflight-" + uuid.uuid4().hex))]
        stage("prepare-fixed-five", prepare_command)
        stage("run-fixed-five", [*evaluate, "run", "--config", str(config)])
        current = json.loads((out / "fixed-five/campaign.json").read_text())
        reference_target = qualified_reference(current)
        receipt.update(compute_reference=current["compute_reference"], compute_reference_target=reference_target)
        save()
        plan = json.loads((out / "fixed-five/frozen-execution-plan.json").read_text())
        # The frozen package evaluate() injects --engine-profile, which this
        # separate frozen controller does not accept. Verify its source first.
        stage("verify-installed-tools", [sys.executable, "-c",
              "from pathlib import Path; from vllm_hust_engine.cli import verify_tools; "
              "import sys; verify_tools(Path(sys.argv[1]))", str(installation / "tools")])
        supplement = [*evaluate, "--context-tenant-only", "--deployment", plan["optimized_deployment"],
                      "--baseline-deployment", plan["baseline_deployment"],
                      "--b1-python", sys.executable, "--model", data["model"],
                      "--cache", data["cache"], "--out", str(out / "context-tenant"),
                      "--physical-devices", ",".join(map(str, data["physical_devices"])),
                      "--devices", plan["logical_devices"]]
        stage("run-context-tenant", supplement)
        campaign, reports, hashes = completed_evidence(out)
        reference_target = qualified_reference(campaign)
        receipt.update(status="completed", reports=reports, evidence_sha256=hashes,
                       fixed_five_execution=campaign["selected_execution_complete"],
                       compute_reference=campaign["compute_reference"], compute_reference_target=reference_target)
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
    parser.add_argument("--prepare-environment", action="store_true", help="Repair and verify the environment without starting model tests")
    parser.add_argument("--resume-bootstrap", action="store_true", help="Recover a failed preflight using its verified installation; refuses existing inference campaigns")
    args = parser.parse_args()
    try:
        data = configuration(args.config, allow_existing=args.resume_bootstrap)
        if args.resume_bootstrap:
            bootstrap_resume(data)
        from environment_setup import ensure_environment
        prepared = ensure_environment(data)
        if prepared is not None:
            os.execve(prepared["python"], [prepared["python"], str(Path(__file__).resolve()), *sys.argv[1:]], prepared["env"])
            raise ValueError("Could not hand off to the prepared Python environment")
        preflight()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    if args.prepare_environment:
        print("ENVIRONMENT verified; no model tests started", flush=True)
        return 0
    return run(data, resume_bootstrap=args.resume_bootstrap)


if __name__ == "__main__":
    raise SystemExit(main())
