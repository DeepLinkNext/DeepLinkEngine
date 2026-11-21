from release_selection import kit_path, runtime_lock, profile as selected_profile
"""Install a selected stable engine and run fixed-five and context/tenant tests."""
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
PATH_FIELDS = {"model", "cache", "out", "baseline_deployment", "dataset_assets", "env_file", "actual_cost"}
FIELDS = PATH_FIELDS | {"physical_devices", "devices", "environment", "engine_version", "rerun_b0", "allow_device_change"}


def configuration(path, *, allow_existing=False, report_only=False):
    path = Path(path).resolve()
    data = json.loads(path.read_bytes())
    if not isinstance(data, dict) or (not report_only and set(data) - FIELDS):
        raise ValueError("Use the fields in benchmark/run-all.example.json")
    if report_only:
        data = {k: data[k] for k in ("cache", "out", "actual_cost") if k in data}
    required = ("cache", "out") if report_only else ("model", "physical_devices", "cache", "out")
    for name in required:
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
            raise ValueError("environment accepts only python, wheelhouse, cann_home and official_sdk")
        for name, value in options.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError("environment." + name + " requires a path")
            if name != "python" or "/" in value or "\\" in value:
                options[name] = str((path.parent / Path(value).expanduser()).resolve())
    if "rerun_b0" in data:
        from b0_run_policy import boolean
        boolean(data["rerun_b0"])
    if "allow_device_change" in data and type(data["allow_device_change"]) is not bool:
        raise ValueError("allow_device_change requires a boolean")
    if not report_only and not Path(data["model"]).is_dir():
        raise ValueError("Model directory does not exist: " + data["model"])
    if os.path.lexists(data["out"]) and not allow_existing:
        raise ValueError("Choose a new output directory: " + data["out"])
    return data


def evaluation_configuration(data, installation):
    return dict({k: v for k, v in data.items() if k not in ("environment", "engine_version", "rerun_b0", "allow_device_change", "actual_cost")}, engine_profile=selected_profile(), test_mode="full",
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
        sys.path.insert(0,str(kit_path() / "scripts"))
        from qualification import sealed_files
        from third_party_report import verify_campaign_bindings
        sealed_files(arm)
        verify_campaign_bindings(campaign)
    if json.loads((out / "evaluation.json").read_text()) != evaluation_configuration(data, out / "installation"):
        raise ValueError("Recovery configuration differs from the original full-test configuration")
    return prior


def recovery_environment(data, *, source=None, source_cache=None):
    """Reuse the installation's private interpreter after fresh version/file probes."""
    from environment_setup import cann_environment, probe, differences, compatible_python, vendor_integrity, runtime_imports
    out = Path(source or data["out"])
    if (out / "installation/installation.json").is_file():
        info = json.loads((out / "installation/installation.json").read_text())
        python = Path(info["python"]).absolute()
    else:
        info = json.loads((out / "environment.json").read_text())
        python = Path(info["after"]["python"]).absolute()
    from python_storage import admitted, ensure as restore_python
    if not admitted(python, source_cache or data["cache"]):
        raise ValueError("Recovery requires the installation's private interpreter")
    restore_python(python.parent.parent)
    if not (python.parent.parent / "pyvenv.cfg").is_file():
        from environment_setup import ensure_environment
        return ensure_environment(data)
    lock = runtime_lock(ROOT)
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
    if source is None:
        (out / "environment-recovery.json").write_text(json.dumps(
            dict(verified=True, observed=observed, vendor_integrity=integrity, runtime_imports=imports), indent=2) + "\n")
    os.environ.update(env)
    return None


def preflight():
    if (sys.version_info[:2] != (3, 12) or platform.python_implementation() != "CPython"
        or platform.system() != "Linux" or platform.machine() != "aarch64"):
        raise ValueError("Requires CPython 3.12, Linux aarch64 and the documented Ascend runtime")
    lock = runtime_lock(ROOT)
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
    from full_report import completed_context_after_error, context_sources
    report_failed = completed_context_after_error(execution, pair)
    phases = [row["phase"] for row in pair["comparison"]]
    if (campaign["status"] != "completed" or campaign["selected_execution_complete"] is not True
        or (execution["status"] != "completed" and not report_failed) or pair["execution_complete"] is not True
        or phases != ["4096-formal", "16384-formal", "32768-formal", "multi-tenant-formal"]):
        raise ValueError("Campaign evidence does not show all planned tests completed")
    reports = {"fixed_five": "fixed-five-report/report.html"}
    if report_failed:
        context_sources(out / "context-tenant", pair)
    else:
        reports["context_tenant"] = "context-tenant/paired-report/report.html"
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
    sys.path.insert(0, str(kit_path() / "scripts"))
    from compute_reference import target
    reference = campaign.get("compute_reference", {})
    if reference.get("certificate_kind") != "qualified_full_reference_denominator_reuse/v1":
        raise ValueError("Current-workload effective-compute reference remains incomplete")
    result = target(reference)
    if result["status"] != "current_workload_measured":
        raise ValueError("Current-workload effective-compute reference remains incomplete")
    return result


def full_report_command(out, destination=None, actual_cost=None):
    command="""import subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(__import__('release_selection').kit_path()/'scripts'))
from report_environment import report_python
python,env=report_python()
raise SystemExit(subprocess.run([str(python),'-B',str(root/'full_report.py'),'--run',sys.argv[2],'--out',sys.argv[3],*sys.argv[4:]],env=env).returncode)
"""
    args = [sys.executable, '-c', command, str(ROOT), str(out), str(destination or out/'full-report')]
    if actual_cost is not None:
        args.extend(['--actual-cost', str(Path(actual_cost).resolve())])
    return args


def completed_run_source(data, source):
    """Audit a sealed prior run before preparing a fresh, independent output."""
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    source = Path(source).resolve()
    out = Path(data["out"]).resolve()
    if source == out or source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError("Reuse source and new output must be independent")
    prior = json.loads((source / "results.json").read_text())
    comparable = lambda value: {k: v for k, v in value.items() if k not in ("out", "cache", "engine_version", "rerun_b0", "allow_device_change")}
    before, after = comparable(prior.get("configuration", {})), comparable(data)
    changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    device_change = changed == {"physical_devices"} and data.get("allow_device_change") is True
    if before != after and not device_change:
        detail = ", ".join(sorted(changed))
        if "physical_devices" in changed:
            detail += f" (original {before.get('physical_devices')}, current {after.get('physical_devices')}; explicit device-change admission is required)"
        raise ValueError("Reuse workload or runtime configuration differs: " + detail)
    sys.path.insert(0, str(kit_path() / "scripts"))
    from recovery_source import stopped
    interruption = stopped(source, prior)
    from reuse_campaign import source as audit_source, audit_stage
    print("RECOVERY_AUDIT campaign", source / "fixed-five", flush=True)
    _, previous_plan, campaign, bindings = audit_source(source / "fixed-five")
    if campaign.get("status") not in ("failed", "completed"):
        raise ValueError("Reuse campaign is not sealed and stopped")
    names = [row["name"] for row in campaign["stages"] if row["status"] == "completed"]
    from release_selection import active
    selected = active()
    previous_selection = prior.get("engine_selection") or {}
    if (source / "engine-selection.json").is_file():
        previous_selection = json.loads((source / "engine-selection.json").read_bytes())
    previous_release = previous_plan.get("engine_release") or previous_selection.get("release")
    changed_target = selected and (selected["release"] != previous_release or
        (previous_selection.get("version") and selected["version"] != previous_selection["version"]))
    if changed_target:
        # The campaign admits a whole MMLU pair's B0 arm independently.
        # B1 services and their judge stages cannot follow a changed target.
        names = [name for name in names
                 if name == "mmlu-pair" or (name.endswith("-B0") and "-judge-" not in name)]
    if not names:
        raise ValueError("No compatible completed stages are available to reuse")
    for name in names:
        if changed_target and name == "mmlu-pair":
            continue
        print("RECOVERY_AUDIT stage", name, flush=True)
        audit_stage(source / "fixed-five", name)
        print("RECOVERY_AUDIT verified", name, flush=True)
    result = dict(root=str(source), completed_stages=names,
                  results_sha256=hashlib.sha256((source / "results.json").read_bytes()).hexdigest())
    reference = source / "fixed-five/compute-reference"
    native_seal = "compute-reference/native-current-timing/checksums.json"
    if (not changed_target and campaign.get("selected_execution_complete") is True
            and "compute-reference/checksums.json" in bindings and native_seal in bindings
            and (reference / "automatic-result.json").is_file()):
        measured = json.loads((reference / "automatic-result.json").read_bytes())
        if any(row.get("id") == "current-native-synchronized-timing"
               and row.get("status") == "completed" for row in measured["stages"]):
            result["compute_resume_source"] = str(reference)
    if device_change:
        result["physical_device_change"] = dict(original=before["physical_devices"], current=after["physical_devices"])
    if interruption:
        result["interruption_recovery"] = interruption
    return result


def run(data, *, prepare_protocol_only=False, prepare_official_protocol_only=False, resume_bootstrap=False, resume_before_model=False, resume_before_inference=False,
        reuse_completed_run=None, reuse_completed_campaigns=(), reuse_mmlu_baseline_arm=None, reuse_context_baseline_arm=None, reuse_compute_baseline_arm=None,
        reuse_compute_observation=None,
        actual_cost=None):
    out = Path(data["out"])
    reused_run = completed_run_source(data, reuse_completed_run) if reuse_completed_run else None
    if reused_run and os.path.lexists(out):
        raise ValueError("Completed-run reuse requires a new output directory")
    pre_inference = resume_before_model or resume_before_inference
    prior = before_model_resume(data, allow_model_startup=resume_before_inference) if pre_inference else bootstrap_resume(data) if resume_bootstrap else None
    if reused_run is None and prior and prior.get("reused_run"):
        # A failed preflight must not discard the explicit no-repeat choice.
        reused_run = completed_run_source(data, prior["reused_run"]["root"])
    reuse_selection = dict(campaigns=[str(Path(p).resolve()) for p in reuse_completed_campaigns],
                           mmlu_baseline_arm=str(Path(reuse_mmlu_baseline_arm).resolve()) if reuse_mmlu_baseline_arm else None,
                           context_baseline_arm=str(Path(reuse_context_baseline_arm).resolve()) if reuse_context_baseline_arm else None,
                           compute_baseline_arm=str(Path(reuse_compute_baseline_arm).resolve()) if reuse_compute_baseline_arm else None)
    if reuse_compute_observation:
        reuse_selection["compute_observation"] = str(Path(reuse_compute_observation).resolve())
    if prior and prior.get('reuse_selection'):
        previous_selection = dict(prior['reuse_selection'])
        previous_selection.setdefault('context_baseline_arm', None)
        previous_selection.setdefault('compute_baseline_arm', None)
        if any(reuse_selection.values()) and reuse_selection != previous_selection:
            raise ValueError('Recovery reuse selection differs from the original invocation')
        reuse_selection = previous_selection
    if any(reuse_selection.values()) and not reused_run:
        raise ValueError('Additional sealed sources require explicit completed-run recovery')
    reuse_installation = resume_bootstrap or pre_inference or reused_run is not None
    out.mkdir(parents=True, exist_ok=reuse_installation)
    installation = out / "installation"
    cli = str(Path(sysconfig.get_path("scripts")) / "vllm-hust-engine")
    from recovery_source import process_identity
    receipt = {"status": "running", "stages": [], "configuration": data, "automated_acceptance_complete": False, "baseline_policy": {"rerun_b0": data.get("rerun_b0", False)}, "controller_owner": process_identity()}
    if reused_run:
        receipt["reused_run"] = reused_run
        receipt["reuse_selection"] = reuse_selection
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

    from release_selection import selected_client
    if selected_client():
        receipt['assessment_client'] = selected_client()
    if os.environ.get('DEEPLINKENGINE_SDK_RECOVERY'):
        receipt['sdk_bytecode_recovery'] = json.loads(os.environ['DEEPLINKENGINE_SDK_RECOVERY'])
    from native_sdk_controller import ENV as native_marker
    if os.environ.get(native_marker):receipt['native_controller_handoff']=json.loads(os.environ[native_marker])
    environment_report = os.environ.get("DEEPLINKENGINE_ENVIRONMENT_REPORT")
    if environment_report:
        import shutil
        shutil.copyfile(environment_report, out / "environment.json")
        receipt["environment"] = {"path": "environment.json", "sha256": hashlib.sha256((out / "environment.json").read_bytes()).hexdigest()}

    from release_selection import save as save_selection, active as active_selection
    save_selection(out)
    receipt["engine_selection"] = active_selection()

    def save():
        temporary = out / (".results-" + uuid.uuid4().hex + ".tmp")
        with temporary.open("w") as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(out / "results.json")

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
        if result.returncode > 0 and name == "run-context-tenant":
            from full_report import completed_context_after_error
            folder = out / "context-tenant"
            path, summary = folder / "execution.json", folder / "paired-summary.json"
            if path.is_file() and summary.is_file():
                execution = json.loads(path.read_bytes())
                if completed_context_after_error(execution, json.loads(summary.read_bytes())):
                    record.update(status="sealed_with_report_failure", report_failure=execution.get("failure"))
                    save()
                    return
        if result.returncode:
            raise RuntimeError(name + " failed; original results retained, no automatic retry")

    try:
        from one_click import verify_kit
        verify_kit()
        sys.path.insert(0, str(kit_path() / "scripts"))
        from child_runner import run_stage as run_owned_stage
        data_command = [sys.executable, str(ROOT / "tools/prepare_datasets.py"), "--cache", data["cache"], "--context"]
        if data.get("dataset_assets"):
            data_command += ["--assets", data["dataset_assets"]]
        stage("prepare-datasets", data_command)
        install_command = [sys.executable, str(ROOT / "engine_install.py"), "--cache", data["cache"]]
        if data.get("environment", {}).get("wheelhouse"):
            install_command += ["--wheelhouse", data["environment"]["wheelhouse"]]
        if reuse_installation:
            stage("install-package", [*install_command, "--repair-dependencies"],
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
        if reused_run and prior is None:
            previous_installation = Path(reused_run['root']) / 'installation'
            previous_runtime = json.loads((previous_installation / 'installation.json').read_text())
            if previous_runtime['package_version'] == active_selection()['version']:
                runtime_command += ['--reuse-from', str(previous_installation)]
        stage("install-engine", runtime_command, mode="verify_existing" if reuse_installation else "execute")
        evaluation = evaluation_configuration(data, installation)
        config = out / "evaluation.json"
        config.write_text(json.dumps(evaluation, indent=2) + "\n")
        evaluate = [sys.executable, str(ROOT / "one_click.py")]
        prepare_command = ([sys.executable, '-B', str(ROOT / 'official_protocol_data.py'), '--config', str(config),
                            '--anchor-sdk', data['environment']['official_sdk']]
                           if prepare_official_protocol_only else [*evaluate, "prepare", "--config", str(config)])
        if data.get("rerun_b0", False) and not prepare_official_protocol_only:
            prepare_command += ["--rerun-b0"]
        if reuse_installation:
            prepare_command += ["--out", str(out / ("preflight-" + uuid.uuid4().hex))]
        if pre_inference:
            receipt["reused_preflight"] = prior["stages"][4]
            stage("prepare-fixed-five", [*evaluate, "verify"], mode="verify_existing_preflight")
        else:
            stage("prepare-fixed-five", prepare_command)
        if prepare_protocol_only or prepare_official_protocol_only:
            receipt.update(status='prepared', protocol_preparation_complete=True,
                           model_service_started=False, formal_client_started=False,
                           protocol_run=str(out / 'fixed-five'),
                           protocol_deployment=str(installation / 'sdk'),
                           protocol_python=sys.executable)
            if prepare_official_protocol_only:receipt['protocol_preparation_mode']='official-inputs-only'
            return 0
        run_command = [*evaluate, "run", "--config", str(config)]
        if reused_run:
            # Translate the public configuration before adding a controller-only
            # option. The frozen subcommand parser intentionally rejects extras.
            from one_click import configuration_arguments
            run_command = [*evaluate, *configuration_arguments("run", ["--config", str(config)]),
                           "--reuse-completed-campaign", str(Path(reused_run["root"]) / "fixed-five")]
        if reused_run:
            for campaign in reuse_selection['campaigns']:
                run_command += ['--reuse-completed-campaign', campaign]
            if reuse_selection['mmlu_baseline_arm']:
                run_command += ['--reuse-mmlu-baseline-arm', reuse_selection['mmlu_baseline_arm']]
        if reuse_selection.get("compute_observation"):
            run_command += ["--reference-observation-source", reuse_selection["compute_observation"]]
        if reused_run and reused_run.get("compute_resume_source"):
            run_command += ["--reference-resume-source", reused_run["compute_resume_source"]]
        if reuse_selection.get("compute_baseline_arm"):
            run_command += ["--reference-baseline-source", reuse_selection["compute_baseline_arm"]]
        if data.get("rerun_b0", False):
            run_command += ["--rerun-b0"]
        if data.get("allow_device_change"):
            run_command += ["--allow-device-change"]
        stage("run-fixed-five", run_command)
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
        if reuse_selection.get("context_baseline_arm"):
            supplement += ["--reuse-b0", reuse_selection["context_baseline_arm"]]
        if data.get("rerun_b0", False):
            supplement += ["--rerun-b0"]
        if data.get("allow_device_change"):
            supplement += ["--allow-device-change"]
        stage("run-context-tenant", supplement)
        campaign, reports, hashes = completed_evidence(out)
        reference_target = qualified_reference(campaign)
        stage("render-full-report", full_report_command(out, actual_cost=actual_cost))
        summary_path = out / "full-report/summary.json"
        overview = json.loads(summary_path.read_bytes())
        if (overview.get("schema") != "deeplinkengine-full-acceptance-report/v1"
            or overview.get("execution_complete") is not True
            or set(overview.get("targets", {})) != {"4.1", "4.2", "4.3", "4.4"}
            or any(type(goal.get(level, {}).get("met")) is not bool
                   or goal[level].get("status") != ("达成目标" if goal[level]["met"] else "未达成目标")
                   for goal in overview.get("targets", {}).values() for level in ("milestone", "final"))):
            raise ValueError("The full report does not contain four completed acceptance decisions")
        reports["overview"] = "full-report/report.html"
        if "context_tenant" not in reports:
            reports["context_tenant"] = "full-report/report.html#context-results"
        for name in ("full-report/summary.json", "full-report/report.html", "full-report/fixed-five.svg", "full-report/fixed-five.png"):
            with (out/name).open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream,"sha256").hexdigest()
        template = out / "full-report/actual-cost.template.json"
        if template.is_file():
            relative = "full-report/actual-cost.template.json"
            reports["actual_cost_template"] = relative
            with template.open("rb") as stream:
                hashes[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
        cost = overview["targets"]["4.4"]
        cost_complete = all(cost[level].get("evidence_status") in {"published_rental", "verified"}
                            for level in ("milestone", "final"))
        receipt.update(status="completed", automated_acceptance_complete=cost_complete,
                       targets=overview["targets"], reports=reports, evidence_sha256=hashes,
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


def historical_engine(source):
    def read(name):
        path = source / name
        return json.loads(path.read_bytes()) if path.is_file() else {}

    prior = read("results.json")
    selection = prior.get("engine_selection") or {}
    installation = read("installation/installation.json")
    plan = read("fixed-five/frozen-execution-plan.json")
    versions = {value for value in (
        selection.get("version"), installation.get("package_version"),
        (prior.get("configuration") or {}).get("engine_version"),
    ) if value}
    releases = {value for value in (
        selection.get("release"), installation.get("release"), plan.get("engine_release"),
    ) if value}
    if len(versions) > 1 or len(releases) > 1:
        raise ValueError("Historical engine identity differs across preserved receipts")
    release = next(iter(releases), None)
    version = next(iter(versions), None)
    if version is None:
        legacy = json.loads((ROOT / "engine/r8-lock.json").read_bytes())
        if release == legacy["upstream_release"]:
            version = legacy["public_distribution"].split("==", 1)[1]
    if version is None:
        raise ValueError("Historical engine version is missing from preserved receipts")
    return version, release


def select_run_engine(data, *, recovery=False, reuse_completed_run=None):
    """Freeze a new target once; in-place recovery retains its original target."""
    from release_selection import select, bind
    saved = Path(data["out"]) / "engine-selection.json"
    if recovery or (reuse_completed_run and not data.get("engine_version")):
        if reuse_completed_run:
            saved = Path(reuse_completed_run).resolve() / "engine-selection.json"
        if saved.is_file():
            record = bind(saved)
        else:
            version, release = historical_engine(saved.parent)
            record = select(data["cache"], version=version)
            if release is not None and record["release"] != release:
                raise ValueError("Selected engine release differs from the preserved historical receipt")
        if data.get("engine_version") and record["version"] != data["engine_version"]:
            raise ValueError("In-place recovery cannot change the frozen engine version")
        return record
    # With a fresh output an explicit target may reuse sealed B0 only. The
    # caller audits every completed stage before environment/model operations;
    # the campaign additionally checks B0 source, argv, weights and warmup.
    return select(data["cache"], version=data.get("engine_version"))


def main():
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if '--official-baselines' in sys.argv[1:]:
        from official_baselines import main as official_main
        return official_main([value for value in sys.argv[1:] if value != '--official-baselines'])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prepare-environment", action="store_true", help="Repair and verify the environment without starting model tests")
    parser.add_argument('--prepare-protocol-only', action='store_true', help='Prepare datasets, SDK and fixed-five inputs; stop before any model service or evaluation client')
    parser.add_argument('--prepare-official-protocol-only', action='store_true', help='Prepare original data and R8 judge for official solvers, without installing or claiming the legacy0.18 SDK')
    parser.add_argument("--resume-bootstrap", action="store_true", help="Recover a failed preflight using its verified installation; refuses existing inference campaigns")
    parser.add_argument("--resume-before-model", action="store_true", help="Recover a sealed first-stage failure only when no service or inference ever started")
    parser.add_argument("--resume-before-inference", action="store_true", help="Recover a sealed failed model startup only if no formal client ever started; preserves startup originals")
    parser.add_argument("--report-only", action="store_true", help="Re-audit completed originals and regenerate all four decisions without repeating inference")
    parser.add_argument("--actual-cost", type=Path, help="V2 actual card cost manifest for the full report")
    parser.add_argument("--reuse-completed-run", type=Path, help="Continue in a NEW output directory; audit and reuse completed sealed stages from this prior run, retaining all failed originals")
    parser.add_argument('--reuse-completed-campaign', type=Path, action='append', default=[], help='Additional explicitly selected sealed campaign; all scope, runtime, raw and provenance checks remain required')
    parser.add_argument('--reuse-mmlu-baseline-arm', type=Path, help='Explicit sealed fixed-B0 original200 arm; retain its original answers, dates and identity')
    parser.add_argument('--reuse-context-baseline-arm', type=Path, help='Explicit sealed fixed-B0 context/four-tenant arm; verify identical inputs, budgets, warmup, model, image, devices and reviewed controller sources')
    parser.add_argument("--reuse-compute-baseline-arm", type=Path, help="Reuse sealed fixed-B0 native timing repetitions; retain original clocks and sources instead of starting B0")
    parser.add_argument("--reuse-compute-observation", type=Path,
        help="Reuse a complete sealed HTTP work observation with the same engine and devices")
    parser.add_argument("--rerun-b0", action=argparse.BooleanOptionalAction, default=None, help="是否重跑固定 B0；默认否，复用已封存结果；首次测量显式开启")
    args = parser.parse_args()
    if (args.prepare_protocol_only or args.prepare_official_protocol_only) and (args.prepare_environment or args.resume_bootstrap or args.resume_before_model or args.resume_before_inference or args.report_only or args.reuse_completed_run or args.reuse_completed_campaign or args.reuse_mmlu_baseline_arm or args.reuse_context_baseline_arm or args.reuse_compute_baseline_arm or args.reuse_compute_observation or (args.prepare_protocol_only and args.prepare_official_protocol_only)):
        parser.error('Protocol preparation requires a fresh run and cannot combine with execution/recovery modes')
    try:
        data = configuration(args.config,
                             allow_existing=args.resume_bootstrap or args.resume_before_model or args.resume_before_inference or args.report_only,
                             report_only=args.report_only)
        configured_cost = data.pop("actual_cost", None)
        cost_path = args.actual_cost or configured_cost
        actual_cost = Path(cost_path).expanduser().resolve() if cost_path is not None else None
        if actual_cost is not None:
            if not actual_cost.is_file():
                raise ValueError("Actual cost manifest is not a file: " + str(actual_cost))
            if not args.report_only and not args.prepare_environment and not args.prepare_protocol_only and not args.prepare_official_protocol_only:
                from full_report import actual_cost_inputs
                actual_cost_inputs(actual_cost)
        if args.prepare_official_protocol_only and not data.get('environment',{}).get('official_sdk'):
            raise ValueError('Official input preparation requires the publicly prepared environment.official_sdk anchor')
        from official_execution import maybe_enter_config, needed
        if data.get('environment', {}).get('official_sdk') and not args.report_only:
            from official_sdk_bytecode import verify_or_repair
            recovery = verify_or_repair(data['environment']['official_sdk'], data['cache'])
            if recovery:
                os.environ['DEEPLINKENGINE_SDK_RECOVERY'] = json.dumps(recovery)
        if data.get('environment', {}).get('official_sdk') and not args.report_only:
            from native_sdk_controller import maybe_enter as native_controller
            native_controller(data['environment']['official_sdk'],ROOT/'run_all.py',sys.argv[1:],cache=data['cache'])
        if data.get('environment', {}).get('official_sdk') and needed():
            from transport_trust import configure
            configure(data['cache'], data['environment']['official_sdk'])
        maybe_enter_config(args.config, sys.argv[1:])
        if args.rerun_b0 is not None:
            data["rerun_b0"] = args.rerun_b0
        os.environ["XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME") or str(Path(data["cache"]) / "xdg")
        os.environ["MPLCONFIGDIR"] = os.environ.get("MPLCONFIGDIR") or str(Path(os.environ["XDG_CACHE_HOME"]) / "matplotlib")
        if args.report_only:
            if args.resume_bootstrap or args.resume_before_model or args.resume_before_inference or args.prepare_environment or args.reuse_completed_run or args.reuse_completed_campaign or args.reuse_mmlu_baseline_arm or args.reuse_context_baseline_arm or args.reuse_compute_baseline_arm or args.reuse_compute_observation:
                raise ValueError("Report-only cannot be combined with execution or recovery modes")
            out = Path(data["out"])
            from release_selection import bind
            if (out / 'engine-selection.json').is_file():
                bind(out / 'engine-selection.json')
            destination = out / ("full-report-" + uuid.uuid4().hex)
            return subprocess.run(full_report_command(out, destination, actual_cost=actual_cost)).returncode
        recovery = args.resume_bootstrap or args.resume_before_model or args.resume_before_inference
        if (args.reuse_completed_campaign or args.reuse_mmlu_baseline_arm or args.reuse_context_baseline_arm or args.reuse_compute_baseline_arm or args.reuse_compute_observation) and not (args.reuse_completed_run or recovery):
            raise ValueError('Additional sealed sources require explicit completed-run recovery')
        if not args.prepare_environment and not args.prepare_protocol_only and not args.prepare_official_protocol_only:
            from b0_run_policy import full_entry_policy
            args.reuse_context_baseline_arm, args.reuse_compute_baseline_arm = full_entry_policy(
                data, prior=args.reuse_completed_run, context=args.reuse_context_baseline_arm,
                compute=args.reuse_compute_baseline_arm, recovery=recovery)
        select_run_engine(data, recovery=recovery, reuse_completed_run=args.reuse_completed_run)
        if not args.prepare_environment:
            from release_selection import ensure_metadata_client
            ensure_metadata_client(data['cache'])
        if sum((args.resume_bootstrap,args.resume_before_model,args.resume_before_inference))>1:
            raise ValueError("Choose only one recovery mode")
        if args.reuse_completed_run and (args.resume_bootstrap or args.resume_before_model or args.resume_before_inference):
            raise ValueError("Completed-run reuse requires a new output, separate from in-place startup recovery")
        if data.get("allow_device_change") and (data.get("rerun_b0") or not (args.reuse_completed_run or recovery)):
            raise ValueError("Device-change admission requires a sealed continuation with fixed B0 reuse")
        if args.reuse_completed_run:
            completed_run_source(data, args.reuse_completed_run)
        if args.resume_before_model or args.resume_before_inference:
            before_model_resume(data,allow_model_startup=args.resume_before_inference)
        if args.resume_bootstrap:
            bootstrap_resume(data)
        from environment_setup import ensure_environment
        if args.reuse_completed_run:
            source = args.reuse_completed_run.resolve()
            saved_configuration = json.loads((source / "results.json").read_bytes()).get("configuration", {})
            prepared = recovery_environment(data, source=source, source_cache=saved_configuration.get("cache"))
        elif recovery:
            prepared = recovery_environment(data)
        else:
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
    return run(data, prepare_protocol_only=args.prepare_protocol_only, prepare_official_protocol_only=args.prepare_official_protocol_only, resume_bootstrap=args.resume_bootstrap, resume_before_model=args.resume_before_model, resume_before_inference=args.resume_before_inference,
               reuse_completed_run=args.reuse_completed_run, reuse_completed_campaigns=args.reuse_completed_campaign,
               reuse_mmlu_baseline_arm=args.reuse_mmlu_baseline_arm, reuse_context_baseline_arm=args.reuse_context_baseline_arm, reuse_compute_baseline_arm=args.reuse_compute_baseline_arm, actual_cost=actual_cost,
               reuse_compute_observation=args.reuse_compute_observation)


if __name__ == "__main__":
    raise SystemExit(main())
