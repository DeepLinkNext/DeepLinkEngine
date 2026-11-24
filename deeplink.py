from engine.release_selection import kit_path, runtime_lock, profile as selected_profile
"""Install DeepLinkEngine and start an inference service without evaluation workloads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parent


def installation(path):
    from engine.runtime_storage import restore_missing
    restore_missing(Path(path).expanduser())
    path = Path(path).expanduser().resolve()
    receipt = json.loads((path / "installation.json").read_bytes())
    from engine.release_selection import release_name
    if receipt['release'] != release_name(receipt['package_version']):
        raise ValueError('Installation release/version identity differs')
    if json.loads((path / 'sdk/deployment.json').read_bytes())['release'] != receipt['release']:
        raise ValueError('Installation SDK belongs to a different release')
    if Path(receipt["deployment"]).resolve() != path / "sdk":
        raise ValueError("Installation receipt points outside the selected SDK")
    return path, receipt


def reuse_installation(args, target, cache):
    from engine.environment_setup import cann_environment, run_repair_stage
    from engine.runtime_storage import marker

    source = Path(os.path.abspath(Path(args.reuse_runtime_from).expanduser()))
    saved = json.loads(marker(source).read_bytes())
    receipt = json.loads((source / "installation.json").read_bytes())
    if (saved.get("schema") != "deeplinkengine-runtime-storage/v1" or saved.get("verified") is not True
            or saved.get("logical") != str(source) or saved.get("physical") != str(source.resolve())
            or any(saved.get(key) != receipt.get(key) for key in ("package_version", "release", "python"))):
        raise ValueError("Runtime reuse requires a verified managed installation")
    python = Path(receipt["python"])
    if args.engine_version and args.engine_version != receipt["package_version"]:
        raise ValueError("Requested version differs from the installation to reuse")
    for override in (args.python, args.use_environment):
        if override and Path(override).expanduser().absolute() != python:
            raise ValueError("Runtime reuse requires the original Python environment")
    if not python.is_absolute() or not (python.parent.parent / "pyvenv.cfg").is_file():
        raise ValueError("Runtime reuse requires an available private Python environment")
    options = {"cann_home": args.cann_home} if args.cann_home else {}
    env, _ = cann_environment(options)
    env["XDG_CACHE_HOME"] = str(cache)
    command = [str(python), "-B", str(ROOT / "engine/runtime_storage.py"),
               "--out", str(target), "--reuse-from", str(source)]
    result = run_repair_stage(command, env, None)
    if result.returncode == 0:
        print("INSTALLED", target, flush=True)
    return result.returncode


def install(args):
    from engine.environment_setup import ensure_environment, cann_environment, probe, differences, compatible_python, vendor_integrity
    target = Path(os.path.abspath(Path(args.out).expanduser()))
    cache = Path(args.cache or target.parent / ".cache/deeplinkengine").expanduser().resolve()
    if os.path.lexists(target):
        raise ValueError("Choose a new installation directory: " + str(target))
    if getattr(args, "reuse_runtime_from", None):
        return reuse_installation(args, target, cache)
    for name, folder in {"TMPDIR": cache / "tmp", "PIP_CACHE_DIR": cache / "pip"}.items():
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(folder)
    from engine.release_selection import select
    select(cache, version=getattr(args,"engine_version",None), wheelhouse=args.wheelhouse)
    options = {key: value for key, value in {"python": args.python, "wheelhouse": args.wheelhouse, "cann_home": args.cann_home}.items() if value}
    if args.use_environment:
        python = Path(args.use_environment).expanduser().absolute()
        if not (python.parent.parent / "pyvenv.cfg").is_file():
            raise ValueError("--use-environment must point to a private venv Python")
        lock = runtime_lock(ROOT)
        env, _ = cann_environment(options)
        observed = probe(python, lock["required_python_packages"], env)
        if not compatible_python(observed) or differences(observed, lock["required_python_packages"]):
            raise ValueError("The selected private environment does not meet the runtime dependency versions")
        if not vendor_integrity(python, lock["required_python_packages"], env)["verified"]:
            raise ValueError("The selected private environment has overwritten Ascend Triton files; "
                             "omit --use-environment to prepare a clean private runtime")
        if Path(sys.executable).absolute() != python:
            os.execve(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], env)
            raise RuntimeError("Environment handoff returned unexpectedly")
        os.environ.update(env)
    else:
        prepared = ensure_environment({"cache": str(cache), "out": str(target), "environment": options})
        if prepared is not None:
            os.execve(prepared["python"], [prepared["python"], str(Path(__file__).resolve()), *sys.argv[1:]], prepared["env"])
            raise RuntimeError("Environment handoff returned unexpectedly")
    from engine.environment_setup import run_repair_stage
    package_command = [sys.executable, str(ROOT / "engine/engine_install.py"), "--cache", str(cache)]
    if args.wheelhouse:
        package_command += ["--wheelhouse", args.wheelhouse]
    cli = str(Path(sysconfig.get_path("scripts")) / "vllm-hust-engine")
    commands = [package_command, [cli, "verify"], [sys.executable, str(ROOT / "engine/runtime_storage.py"), "--out", str(target)]]
    for name, command in zip(("install-package", "verify-package", "install-runtime"), commands):
        if name == "verify-package":
            from engine.python_storage import admitted, ensure as local_python
            if admitted(Path(sys.executable),cache):
                local_python(Path(sys.executable).parent.parent,allow_current=True)
        print("INSTALL", name, flush=True)
        result = run_repair_stage(command, dict(os.environ), None)
        if result.returncode:
            return result.returncode
    from engine.release_selection import save
    save(target)
    print("INSTALLED", target, flush=True)
    return 0


def serve(args):
    from engine.environment_setup import cann_environment
    root, receipt = installation(args.installation)
    python = receipt["python"]
    env, _ = cann_environment({})
    if args.aot_cache_format == "graph":
        env["VLLM_USE_AOT_COMPILE"] = "1"
        env["VLLM_USE_MEGA_AOT_ARTIFACT"] = "0"
    command = [python, str(ROOT / "engine/service.py"), "--installation", str(root),
               "--model", str(Path(args.model).expanduser().resolve()), "--physical-devices", args.physical_devices,
               "--host", args.host, "--port", str(args.port), "--served-model-name", args.served_model_name,
               "--cache", str(Path(args.cache or root.parent / ".cache/deeplinkengine/serving").expanduser().resolve()), "--lease-timeout", str(args.lease_timeout)]
    if args.work_dir:
        command += ["--work-dir", os.path.abspath(Path(args.work_dir).expanduser())]
    if args.cache_from:
        command += ["--cache-from", str(Path(args.cache_from).expanduser().resolve())]
    if args.devices:
        command += ["--devices", args.devices]
    if args.dry_run:
        command.append("--dry-run")
    os.execve(python, command, env)
    raise RuntimeError("Service handoff returned unexpectedly")


def controller_campaign(args, parser):
    checkout = os.environ.get("VLLM_HUST_CONTROLLER_ROOT")
    if not checkout:
        if not args or args == ["--help"] or args == ["-h"]:
            print("Usage: python deeplink.py control-plane MODE [OPTIONS]")
            print("Set VLLM_HUST_CONTROLLER_ROOT to the request-throttling-controller checkout.")
            print("Run control-plane --help with the checkout selected to see all campaign options.")
            return 0
        parser.error("Set VLLM_HUST_CONTROLLER_ROOT to the request-throttling-controller checkout")
    entry = (Path(checkout).expanduser().resolve()
             / "experiments/vllm_hust_benchmark/run_vllm_hust_benchmark.py")
    if not entry.is_file():
        parser.error(f"Controller one-click entry not found: {entry}")
    return subprocess.run([sys.executable, str(entry), *args]).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("control-plane", add_help=False, help="Run the shared Controller campaign, including native §4.1")
    if sys.argv[1:2] == ["control-plane"]:
        return controller_campaign(sys.argv[2:], parser)
    setup = sub.add_parser("install", help="Prepare dependencies and install the runtime; no model or dataset required")
    setup.add_argument("--out", required=True)
    setup.add_argument("--engine-version", help="Pin an explicit stable version; otherwise use latest stable")
    setup.add_argument("--cache", help="Cache directory; defaults beside the installation on its target filesystem")
    setup.add_argument("--python")
    setup.add_argument("--wheelhouse")
    setup.add_argument("--cann-home")
    setup.add_argument("--use-environment", help="Reuse an explicitly selected qualified private venv")
    setup.add_argument("--reuse-runtime-from", help="Audit and reuse an existing installation at its original SDK and Python paths")
    run = sub.add_parser("serve", help="Start an OpenAI-compatible inference service")
    run.add_argument("--installation", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--physical-devices", required=True, help="Two physical NPU IDs, e.g. 0,7")
    run.add_argument("--devices", help="Logical IDs; otherwise detected from npu-smi")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)
    run.add_argument("--served-model-name", default="deeplink")
    run.add_argument("--cache", help="Service state and caches; defaults beside the installation")
    run.add_argument("--cache-from", help="Also look for compatible compiled artifacts in this trusted serving cache pool")
    run.add_argument("--aot-cache-format", choices=("graph",),
                     help="Save the vLLM AOT graph without bundling standalone backend artifacts")
    run.add_argument("--work-dir", help="New per-service directory for logs and runtime state")
    run.add_argument("--lease-timeout", type=float, default=0)
    run.add_argument("--dry-run", action="store_true", help="Build and verify the launch plan without loading a model or reserving NPUs")
    from benchmark import unified_run
    unified_run.arguments(sub.add_parser("benchmark", help="Run one frozen experiment group"))
    review = sub.add_parser("compare", help="Compare sealed MMLU group outputs offline")
    review.add_argument("--deployment", type=Path, required=True, help="SDK with original inputs and scorer")
    review.add_argument("groups", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        if args.action == "compare":
            from benchmark.compare_groups import compare
            print(json.dumps(compare(args.groups, args.deployment), indent=2))
            return 0
        return {"install": install, "serve": serve, "benchmark": unified_run.run}[args.action](args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(2, "DEEPLINK_ERROR: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
