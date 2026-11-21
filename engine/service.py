"""Version-bound service adapter; launches only inference, with no evaluation campaign."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pair(value):
    try:
        values = [int(x) for x in value.split(",")]
    except ValueError as error:
        raise ValueError("Device IDs must be non-negative integers") from error
    if len(values) != 2 or len(set(values)) != 2 or any(x < 0 for x in values):
        raise ValueError("Two distinct device IDs are required")
    return values


def set_option(command, flag, value):
    if command.count(flag) != 1:
        raise ValueError("Ambiguous runtime flag: " + flag)
    command[command.index(flag) + 1] = str(value)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--installation", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--physical-devices", required=True)
    p.add_argument("--devices")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--served-model-name", default="deeplink")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--work-dir", type=Path)
    p.add_argument("--lease-timeout", type=float, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


def build(root, profile, model, out, cache, logical, host, port, name):
    import launch_boundary_b1
    launch_host = root / "launch-host"
    launch_host.mkdir(exist_ok=True)
    for link_name, target in {"runtime": root / "runtime/ascend", "mmlu-b1-microflow-fa-on-r10": root / "provenance/base-launch"}.items():
        link = launch_host / link_name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        if link.resolve() != target.resolve():
            raise ValueError("Unexpected runtime source alias")
    launch_boundary_b1.ROOT = launch_host
    launch_boundary_b1.MODEL = model
    command, env, _ = launch_boundary_b1.prepare_command(out, cache)
    if command[:4] != [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"]:
        raise ValueError("Unexpected packaged serving entry")
    if command[-2:] != ["--worker-extension-cls", "synchronized_boundary.WorkerBoundary"]:
        raise ValueError("Unexpected packaged worker extension")
    command = [profile["python"], str(root / "mods/correction/corrected_service_entry.py"), *command[3:-2],
               "--per-request-spec-decode-metrics", "detailed"]
    env.update(HUST_M01_CORRECTED_SERVING="1", HUST_M01_HOST_PREFILL_AGGREGATE="1")
    env["PYTHONPATH"] = str(root / "mods/correction/correction-plugin-installed-r1") + os.pathsep + os.environ["PYTHONPATH"]
    from serve_b1_budget import configure_host_budget
    command, env = configure_host_budget(command, env, 4 * 2**30)
    from runtime_geometry import apply
    geometry = apply(profile, command, env)
    ascend_config = root / "runtime/ascend/vllm_ascend/ascend_config.py"
    if ascend_config.is_file() and "ascend_log_path" in ascend_config.read_text():
        at = command.index("--additional-config") + 1
        additional = json.loads(command[at])
        additional["ascend_log_path"] = str(out / "vllm-ascend-logs")
        Path(additional["ascend_log_path"]).mkdir(exist_ok=True)
        command[at] = json.dumps(additional)
    for flag, value in [("--host", host), ("--port", port), ("--served-model-name", name)]:
        set_option(command, flag, value)
    command += ["--reasoning-parser", "qwen3"]
    env.update(ASCEND_RT_VISIBLE_DEVICES=logical, HUST_JOINT_PACKED_HOST="1", ASCEND_CACHE_PATH=str(cache / "ascend"),
               HCCL_IF_BASE_PORT=str(20000 + (port % 2000) * 16), VLLM_SERVER_DEV_MODE="0")
    env["VLLM_PLUGINS"] = ",".join(dict.fromkeys(env["VLLM_PLUGINS"].split(",") + ["hust_joint_packed"]))
    Path(env["ASCEND_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
    plan = dict(purpose="inference-service", deployment=str(root), model=str(model), output=str(out),
                cache=str(cache), logical_devices=logical, runtime_geometry=geometry,
                expected_versions={"vllm": "0.28.1.post1.dev143+gf18cf803c", "vllm_ascend": profile["host_contract_qualification"]["installed_metadata"]["vllm-ascend"]})
    plan_path = out / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    command = [profile["python"], str(root / "runtime/entrypoints/serve_b1_child.py"), str(plan_path), *command[2:]]
    import service_logging
    logging_plan = service_logging.prepare(root, out, env)
    logging_plan.update(child=command[1], child_sha256=service_logging.digest(command[1]),
                        entry_sha256=service_logging.digest(ROOT / 'logging_entry.py'))
    logging_path = out / 'service-logging-plan.json'
    service_logging.save(logging_path, logging_plan)
    command = [command[0], str(ROOT / 'logging_entry.py'), str(logging_path), '--', *command[1:]]
    return command, env


def main():
    args = parser().parse_args()
    from deeplink import installation
    installed, receipt = installation(args.installation)
    if receipt['release'] != 'full-answer-b1-20261117-r8':
        # The installed producer owns model admission, worker selection and
        # geometry. Keep the original adapter for explicit historical R8 SDKs.
        if args.devices:
            logical = ','.join(map(str, pair(args.devices)))
        else:
            sample = subprocess.run(['npu-smi', 'info', '-m'], capture_output=True,
                                    text=True, check=True, timeout=20).stdout
            mapping = {}
            for line in sample.splitlines():
                words = line.split()
                if len(words) >= 3 and words[0].isdigit() and words[1] == '0' and words[2].isdigit():
                    mapping[int(words[0])] = int(words[2])
            logical = ','.join(str(mapping[x]) for x in pair(args.physical_devices))
        out = args.work_dir or args.cache / ('service-' + uuid.uuid4().hex)
        if os.path.lexists(out):
            raise ValueError('Choose a new service work directory')
        forwarded = ['--model', str(args.model.resolve()), '--out', str(out.absolute()),
                     '--cache', str(args.cache.resolve()), '--devices', logical,
                     '--physical-devices', args.physical_devices, '--host', args.host,
                     '--port', str(args.port), '--served-model-name', args.served_model_name,
                     '--lease-timeout', str(args.lease_timeout)]
        if args.dry_run:
            forwarded.append('--dry-run')
        python = receipt['python']
        entry = 'from vllm_hust_engine.cli import main; raise SystemExit(main())'
        os.execve(python, [python, '-c', entry, 'serve',
                          '--installation', str(installed), '--', *forwarded], dict(os.environ))
        raise RuntimeError('Producer service handoff returned unexpectedly')
    root = installed / "sdk"
    model = args.model.resolve()
    if not 1 <= args.port <= 65535 or not args.host or not args.served_model_name or not math.isfinite(args.lease_timeout) or args.lease_timeout < 0:
        raise ValueError("Invalid service address or wait timeout")
    physical = pair(args.physical_devices)
    model_config = json.loads((model / "config.json").read_bytes())
    text = model_config.get("text_config", model_config)
    if (text.get("model_type"), text.get("num_hidden_layers"), text.get("hidden_size")) != ("qwen3_5_moe_text", 40, 2048):
        raise ValueError("The installed backend currently supports Qwen3.5-35B-A3B only")
    from one_click import verify_kit
    verify_kit()
    sys.path.insert(0, str(ROOT / "benchmark/one-click-kit/scripts"))
    import engine_entry
    import resource_guard
    os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = ",".join(map(str, physical))
    if not os.environ.get("DEEPLINK_SERVICE_BOOTSTRAPPED"):
        print("SERVICE verifying runtime and dependencies", flush=True)
        from restore_b1_r4 import verify_installed_sources
        from dependency_admission import admit_dependencies
        verify_installed_sources(root)
        admit_dependencies(root)
        from environment_setup import probe, differences, vendor_integrity
        profile = json.loads((root / "deployment.json").read_bytes())
        required = json.loads((ROOT / "engine/r8-lock.json").read_bytes())["required_python_packages"]
        if differences(probe(profile["python"], required, os.environ), required):
            raise ValueError("Installed runtime dependency versions differ; prepare a clean installation")
        if not vendor_integrity(profile["python"], required, os.environ)["verified"]:
            raise ValueError("Installed Ascend Triton files are overwritten; prepare a clean installation")
        if args.devices:
            logical = ",".join(map(str, pair(args.devices)))
        else:
            sample = subprocess.run(["npu-smi", "info", "-m"], capture_output=True, text=True, check=True, timeout=20).stdout
            mapping = {}
            for line in sample.splitlines():
                words = line.split()
                if len(words) >= 3 and words[0].isdigit() and words[1] == "0" and words[2].isdigit():
                    mapping[int(words[0])] = int(words[2])
            logical = ",".join(str(mapping[x]) for x in physical)
        if not args.dry_run:
            resource_guard.verify_mapping(logical)
        out = (args.work_dir or args.cache / ("service-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])).absolute()
        if os.path.lexists(out):
            raise ValueError("Choose a new service work directory")
        out.mkdir(parents=True)
        print("SERVICE_WORK_DIR", out, flush=True)
        cache = out / "runtime-cache"
        cache.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            resource_guard.acquire(out, args.lease_timeout)
        profile = json.loads((root / "deployment.json").read_bytes())
        env = engine_entry.environment(root, profile, out, cache, logical)
        env["DEEPLINK_SERVICE_BOOTSTRAPPED"] = str(out)
        command = [profile["python"], str(Path(__file__).resolve()), *sys.argv[1:], "--work-dir", str(out), "--devices", logical]
        os.execve(profile["python"], command, env)
        raise RuntimeError("Runtime bootstrap returned unexpectedly")
    out = Path(os.environ["DEEPLINK_SERVICE_BOOTSTRAPPED"]).resolve()
    if out != args.work_dir.resolve():
        raise ValueError("Service bootstrap state differs")
    profile = json.loads((root / "deployment.json").read_bytes())
    cache = out / "runtime-cache"
    leases = None if args.dry_run else resource_guard.inherited()
    if not args.dry_run and leases is None:
        raise ValueError("Serving requires a valid inherited device lease")
    try:
        command, env = build(root, profile, model, out, cache, args.devices, args.host, args.port, args.served_model_name)
        (out / "launch-command.json").write_text(json.dumps(command, indent=2) + "\n")
        if args.dry_run:
            if {"torch", "torch_npu", "vllm", "vllm_ascend"} & set(sys.modules):
                raise ValueError("Dry-run unexpectedly imported a model framework")
            print("SERVICE_PLAN", out / "launch-command.json", flush=True)
            return 0
        engine_entry.check_completion_inbox(root, out, env)
        engine_entry.check_pinned_host_memory(out, env)
        os.environ.update(env)
        from vllm_hust_ext.cli import main as manager
        return manager(["run", "--shutdown-grace-seconds", "90", "--", *command])
    finally:
        if leases is not None:
            resource_guard.release(leases)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print("SERVICE_ERROR:", error, file=sys.stderr)
        raise SystemExit(2)
