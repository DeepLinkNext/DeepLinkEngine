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
    from packaging.version import Version
    v = Version(receipt['package_version'])
    a, b, c, number = v.release
    if receipt['release'] != f'full-answer-b1-{a:04d}{b:02d}{c:02d}-r{number}':
        raise ValueError('Installation release/version identity differs')
    if json.loads((path / 'sdk/deployment.json').read_bytes())['release'] != receipt['release']:
        raise ValueError('Installation SDK belongs to a different release')
    if Path(receipt["deployment"]).resolve() != path / "sdk":
        raise ValueError("Installation receipt points outside the selected SDK")
    return path, receipt


def install(args):
    from engine.environment_setup import ensure_environment, cann_environment, probe, differences, compatible_python, vendor_integrity
    target = Path(os.path.abspath(Path(args.out).expanduser()))
    cache = Path(args.cache or target.parent / ".cache/deeplinkengine").expanduser().resolve()
    if os.path.lexists(target):
        raise ValueError("Choose a new installation directory: " + str(target))
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
    command = [python, str(ROOT / "engine/service.py"), "--installation", str(root),
               "--model", str(Path(args.model).expanduser().resolve()), "--physical-devices", args.physical_devices,
               "--host", args.host, "--port", str(args.port), "--served-model-name", args.served_model_name,
               "--cache", str(Path(args.cache or root.parent / ".cache/deeplinkengine/serving").expanduser().resolve()), "--lease-timeout", str(args.lease_timeout)]
    if args.work_dir:
        command += ["--work-dir", os.path.abspath(Path(args.work_dir).expanduser())]
    if args.devices:
        command += ["--devices", args.devices]
    if args.dry_run:
        command.append("--dry-run")
    os.execve(python, command, env)
    raise RuntimeError("Service handoff returned unexpectedly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    setup = sub.add_parser("install", help="Prepare dependencies and install the runtime; no model or dataset required")
    setup.add_argument("--out", required=True)
    setup.add_argument("--engine-version", help="Pin an explicit stable version; otherwise use latest stable")
    setup.add_argument("--cache", help="Cache directory; defaults beside the installation on its target filesystem")
    setup.add_argument("--python")
    setup.add_argument("--wheelhouse")
    setup.add_argument("--cann-home")
    setup.add_argument("--use-environment", help="Reuse an explicitly selected qualified private venv")
    run = sub.add_parser("serve", help="Start an OpenAI-compatible inference service")
    run.add_argument("--installation", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--physical-devices", required=True, help="Two physical NPU IDs, e.g. 0,7")
    run.add_argument("--devices", help="Logical IDs; otherwise detected from npu-smi")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)
    run.add_argument("--served-model-name", default="deeplink")
    run.add_argument("--cache", help="Service state and caches; defaults beside the installation")
    run.add_argument("--work-dir", help="New per-service directory for logs and runtime state")
    run.add_argument("--lease-timeout", type=float, default=0)
    run.add_argument("--dry-run", action="store_true", help="Build and verify the launch plan without loading a model or reserving NPUs")
    args = parser.parse_args()
    try:
        return install(args) if args.action == "install" else serve(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(2, "DEEPLINK_ERROR: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
