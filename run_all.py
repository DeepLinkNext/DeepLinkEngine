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
    if (prior.get("status") != "failed" or len(stages) not in (4,5)
            or [r.get("name") for r in stages] != expected[:len(stages)]
            or any(r.get("status") != "completed" for r in stages[:-1])
            or stages[-1].get("status") not in ("failed", "launch_failed", "interrupted")):
        raise ValueError("Bootstrap recovery requires failed installation verification or a failed fixed-five preflight before inference")
    if any(os.path.lexists(out / name) for name in ("fixed-five", "context-tenant")):
        raise ValueError("An inference campaign already exists; bootstrap recovery will not repeat it")
    if (out / "evaluation.json").is_file():
        stored = json.loads((out / "evaluation.json").read_text())
        unchanged = stored == evaluation_configuration(data, out / "installation")
    else:
        unchanged = prior.get("configuration") == data
    if not unchanged:
        raise ValueError("Recovery configuration differs from the original full-test configuration")
    if not (out / "installation/installation.json").is_file() and not (out / "environment.json").is_file():
        raise ValueError("The prior runtime installation or recorded private environment is incomplete")
    return prior


def before_model_resume(data, *, allow_model_startup=False):
    """Recover only a sealed first-stage failure that never started a model."""
    out = Path(data["out"])
    if out.is_symlink() or not out.is_dir():
        raise ValueError("Pre-model recovery requires an existing result directory")
    prior = json.loads((out / "results.json").read_text())
    stages = prior.get("stages", [])
    names = ["prepare-datasets", "install-package", "verify-engine", "install-engine", "prepare-fixed-five", "run-fixed-five"]
    if (prior.get("status") != "failed" or [r.get("name") for r in stages] != names
            or any(r.get("status") != "completed" for r in stages[:5])
            or stages[-1].get("status") != "failed"):
        raise ValueError("Pre-model recovery requires a failed first campaign stage")
    campaign = out / "fixed-five"
    ledger = json.loads((campaign / "campaign.json").read_text())
    rows = ledger.get("stages", [])
    first = json.loads((campaign / "continuous-B0/receipt.json").read_text())
    if (ledger.get("status") != "failed" or ledger.get("selected_execution_complete") is not False
            or not rows or rows[0].get("name") != "continuous-B0" or rows[0].get("status") != "failed"
            or any(r.get("status") != "planned" for r in rows[1:])
            or (not allow_model_startup and first.get("model_service_started") is not False)
            or first.get("client_started") is not False
            or first.get("quiescent") is not True or first.get("completed") is not False
            or (not allow_model_startup and any(campaign.rglob("service-process.json")))
            or (out / "context-tenant").exists()):
        raise ValueError("A model may have started; pre-model recovery refuses to repeat inference")
    if allow_model_startup:
        arm=campaign / "continuous-B0"
        if (type(first.get("model_service_started")) is not bool or first.get("role") != "B0"
            or first.get("kind") != "continuous" or (arm / "client.log").exists()
            or (arm / "service/continuous").exists() or any(arm.rglob("*.sse"))):
            raise ValueError("Pre-inference recovery refuses any possible formal client execution")
        sys.path.insert(0,str(ROOT / "benchmark/one-click-kit/scripts"))
        from qualification import sealed_files
        from third_party_report import verify_campaign_bindings
        sealed_files(arm)
        verify_campaign_bindings(campaign)
    if json.loads((out / "evaluation.json").read_text()) != evaluation_configuration(data, out / "installation"):
        raise ValueError("Recovery configuration differs from the original full-test configuration")
    return prior


def recovery_environment(data):
    """Reuse the installation's private interpreter after fresh version/file probes."""
    from environment_setup import cann_environment, probe, differences, compatible_python, vendor_integrity, runtime_imports
    out = Path(data["out"])
    if (out / "installation/installation.json").is_file():
        info = json.loads((out / "installation/installation.json").read_text())
        python = Path(info["python"]).absolute()
    else:
        info = json.loads((out / "environment.json").read_text())
        python = Path(info["after"]["python"]).absolute()
    from python_storage import admitted, ensure as restore_python
    if not admitted(python, data["cache"]):
        raise ValueError("Recovery requires the installation's private interpreter")
    restore_python(python.parent.parent)
    if not (python.parent.parent / "pyvenv.cfg").is_file():
        from environment_setup import ensure_environment
        return ensure_environment(data)
    lock = json.loads((ROOT / "engine/r8-lock.json").read_text())
    env, _ = cann_environment(data.get("environment", {}))
    observed = probe(python, lock["required_python_packages"], env)
    if not compatible_python(observed) or differences(observed, lock["required_python_packages"]):
        raise ValueError("Recovery interpreter versions differ from the current runtime lock")
    integrity = vendor_integrity(python, lock["required_python_packages"], env)
    if not integrity["verified"]:
        raise ValueError("Recovery interpreter vendor files differ")
    if str(python) != sys.executable:
        return dict(python=str(python), env=env)
    imports = runtime_imports(python, lock["required_python_packages"], env)
    (Path(data["out"]) / "environment-recovery.json").write_text(json.dumps(
        dict(verified=True, observed=observed, vendor_integrity=integrity, runtime_imports=imports), indent=2) + "\n")
    os.environ.update(env)
    return None


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
    fixed_five_plan(out)
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


def fixed_five_plan(out):
    plan = json.loads((out / "fixed-five/frozen-execution-plan.json").read_text())
    if plan.get("datasets") != ["mmlu", "hle", "frontier", "swe", "terminal"]:
        raise ValueError("The full suite must contain all five fixed datasets in canonical order")
    return plan


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


def full_report_command(out, destination=None):
    command="""import subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'benchmark/one-click-kit/scripts'))
from report_environment import report_python
python,env=report_python()
raise SystemExit(subprocess.run([str(python),'-B',str(root/'full_report.py'),'--run',sys.argv[2],'--out',sys.argv[3]],env=env).returncode)
"""
    return [sys.executable,'-c',command,str(ROOT),str(out),str(destination or out/'full-report')]


def run(data, *, resume_bootstrap=False, resume_before_model=False, resume_before_inference=False):
    out = Path(data["out"])
    pre_inference = resume_before_model or resume_before_inference
    prior = before_model_resume(data, allow_model_startup=resume_before_inference) if pre_inference else bootstrap_resume(data) if resume_bootstrap else None
    reuse_installation = resume_bootstrap or pre_inference
    out.mkdir(parents=True, exist_ok=reuse_installation)
    installation = out / "installation"
    cli = str(Path(sysconfig.get_path("scripts")) / "vllm-hust-engine")
    receipt = {"status": "running", "stages": [], "configuration": data, "automated_acceptance_complete": False}
    if prior is not None:
        history = out / "attempts"
        history.mkdir(exist_ok=True)
        archived = history / ("bootstrap-failure-" + uuid.uuid4().hex + ".json")
        raw = (out / "results.json").read_bytes()
        archived.write_bytes(raw)
        receipt["previous_attempts"] = [*prior.get("previous_attempts", []),
            {"path": str(archived.relative_to(out)), "sha256": hashlib.sha256(raw).hexdigest()}]
        if pre_inference:
            original = history / ("pre-model-failure-" + uuid.uuid4().hex)
            original.mkdir()
            for name in ("fixed-five", "fixed-five-logs", "fixed-five-report"):
                if (out / name).exists():
                    (out / name).rename(original / name)
            receipt["previous_attempts"][-1]["campaign_directory"] = str(original.relative_to(out))

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
        if reuse_installation:
            stage("install-package", [sys.executable, str(ROOT / "engine_install.py"), "--cache", data["cache"], "--repair-dependencies"],
                  mode="verify_existing")
        else:
            stage("install-package", install_command)
        dependency_report = Path(data["cache"]) / "engine-install/runtime-dependencies.json"
        if dependency_report.is_file():
            import shutil
            shutil.copyfile(dependency_report, out / "runtime-dependencies.json")
            receipt["runtime_dependencies"] = dict(path="runtime-dependencies.json",
                sha256=hashlib.sha256((out / "runtime-dependencies.json").read_bytes()).hexdigest())
            save()
        from python_storage import admitted, ensure as local_python
        if admitted(Path(sys.executable),data["cache"]):
            managed=local_python(Path(sys.executable).parent.parent,allow_current=True)
            from python_storage import marker as python_storage_marker
            if python_storage_marker(managed).is_file():
                receipt["python_storage"] = json.loads(python_storage_marker(managed).read_text())
                save()
        stage("verify-engine", [cli, "verify"])
        runtime_command = [sys.executable, str(ROOT / "runtime_storage.py"), "--out", str(installation)]
        if reuse_installation:
            runtime_command.append("--reuse")
        stage("install-engine", runtime_command, mode="verify_existing" if reuse_installation else "execute")
        evaluation = evaluation_configuration(data, installation)
        config = out / "evaluation.json"
        config.write_text(json.dumps(evaluation, indent=2) + "\n")
        evaluate = [sys.executable, str(ROOT / "one_click.py")]
        prepare_command = [*evaluate, "prepare", "--config", str(config)]
        if reuse_installation:
            prepare_command += ["--out", str(out / ("preflight-" + uuid.uuid4().hex))]
        if pre_inference:
            receipt["reused_preflight"] = prior["stages"][4]
            stage("prepare-fixed-five", [*evaluate, "verify"], mode="verify_existing_preflight")
        else:
            stage("prepare-fixed-five", prepare_command)
        stage("run-fixed-five", [*evaluate, "run", "--config", str(config)])
        current = json.loads((out / "fixed-five/campaign.json").read_text())
        reference_target = qualified_reference(current)
        receipt.update(compute_reference=current["compute_reference"], compute_reference_target=reference_target)
        save()
        plan = fixed_five_plan(out)
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
        stage("render-full-report", full_report_command(out))
        summary_path = out / "full-report/summary.json"
        overview = json.loads(summary_path.read_text())
        if (overview.get("schema") != "deeplinkengine-full-acceptance-report/v1"
            or overview.get("execution_complete") is not True
            or set(overview.get("targets", {})) != {"4.1", "4.2", "4.3", "4.4"}
            or any(type(goal.get(level, {}).get("met")) is not bool
                   or goal[level].get("status") != ("达成目标" if goal[level]["met"] else "未达成目标")
                   for goal in overview.get("targets", {}).values() for level in ("milestone", "final"))):
            raise ValueError("The full report does not contain four completed acceptance decisions")
        reports["overview"] = "full-report/report.html"
        for name in ("full-report/summary.json", "full-report/report.html", "full-report/fixed-five.svg", "full-report/fixed-five.png"):
            with (out/name).open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream,"sha256").hexdigest()
        receipt.update(status="completed", automated_acceptance_complete=True, targets=overview["targets"], reports=reports, evidence_sha256=hashes,
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
    parser.add_argument("--resume-before-model", action="store_true", help="Recover a sealed first-stage failure only when no service or inference ever started")
    parser.add_argument("--resume-before-inference", action="store_true", help="Recover a sealed failed model startup only if no formal client ever started; preserves startup originals")
    parser.add_argument("--report-only", action="store_true", help="Re-audit completed originals and regenerate all four decisions without repeating inference")
    args = parser.parse_args()
    try:
        data = configuration(args.config, allow_existing=args.resume_bootstrap or args.resume_before_model or args.resume_before_inference or args.report_only)
        if args.report_only:
            if args.resume_bootstrap or args.resume_before_model or args.resume_before_inference or args.prepare_environment:
                raise ValueError("Report-only cannot be combined with execution or recovery modes")
            out = Path(data["out"])
            destination = out / ("full-report-" + uuid.uuid4().hex)
            return subprocess.run(full_report_command(out,destination)).returncode
        if sum((args.resume_bootstrap,args.resume_before_model,args.resume_before_inference))>1:
            raise ValueError("Choose only one recovery mode")
        if args.resume_before_model or args.resume_before_inference:
            before_model_resume(data,allow_model_startup=args.resume_before_inference)
        if args.resume_bootstrap:
            bootstrap_resume(data)
        from environment_setup import ensure_environment
        prepared = recovery_environment(data) if (args.resume_before_model or args.resume_before_inference or args.resume_bootstrap) else ensure_environment(data)
        if prepared is not None:
            os.execve(prepared["python"], [prepared["python"], str(Path(__file__).resolve()), *sys.argv[1:]], prepared["env"])
            raise ValueError("Could not hand off to the prepared Python environment")
        preflight()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    if args.prepare_environment:
        print("ENVIRONMENT verified; no model tests started", flush=True)
        return 0
    return run(data, resume_bootstrap=args.resume_bootstrap, resume_before_model=args.resume_before_model, resume_before_inference=args.resume_before_inference)


if __name__ == "__main__":
    raise SystemExit(main())
