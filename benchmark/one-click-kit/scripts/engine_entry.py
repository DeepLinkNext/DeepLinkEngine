#!/usr/bin/env python3
"""Run preserved B1 components using the verified current deployment identity."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def ipc_directory(out):
    record = out / "ipc-directory.json"
    if record.exists():
        path = Path(json.loads(record.read_text())["path"])
        if (
            path.parent != Path("/tmp")
            or not path.name.startswith("vhe-ipc-")
            or not path.is_dir()
        ):
            raise ValueError("Unexpected private IPC directory")
        return path
    path = Path(tempfile.mkdtemp(prefix="vhe-ipc-", dir="/tmp"))
    write(
        record,
        {
            "path": str(path),
            "purpose": "Short per-run ZMQ socket paths; separate from compiler caches",
        },
    )
    return path


def temporary_directory(out):
    path = ipc_directory(out) / "tmp"
    path.mkdir(mode=0o700, exist_ok=True)
    return path


def check_completion_inbox(root, out, env):
    """Exercise the preserved BetterScale datagram socket before model imports."""
    source = root / "mods/betterscale/betterscale/models/qwen35/cache_engine.py"
    code = """
import hashlib,importlib.util,json,sys
from pathlib import Path
path=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('engine_completion_socket_probe',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
inbox=module.CompletionInbox(lambda: None)
try:
    with module.socket.socket(module.socket.AF_UNIX,module.socket.SOCK_DGRAM) as channel:
        channel.sendto(b'{"event":"deployment-socket-probe"}',inbox.endpoint)
    assert inbox.messages.get(timeout=5)=={'event':'deployment-socket-probe'}
    record={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'endpoint':inbox.endpoint,'endpoint_bytes':len(inbox.endpoint.encode()),
            'real_bind_and_message':True,'model_loaded':False}
finally:
    inbox.close()
assert not Path(record['endpoint']).exists()
record['socket_released']=True
print(json.dumps(record))
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(source)],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode:
        raise RuntimeError(
            "BetterScale completion socket preflight failed: " + result.stderr[-2000:]
        )
    write(out / "completion-socket-preflight.json", json.loads(result.stdout))


def check_pinned_host_memory(out, env):
    result_file = out / "pinned-memory-preflight.json"
    if result_file.exists():
        raise FileExistsError("Preserve the previous pinned-memory admission receipt")
    code = """
import json,resource,sys,torch,torch_npu
results=[]
for device in (0,1):
    torch_npu.npu.set_device(device)
    host=torch.empty(32*2**20,dtype=torch.uint8,device='cpu',pin_memory=True)
    host.fill_(device+1)
    assert host.is_pinned()
    target=host.to('npu:'+str(device),non_blocking=True)
    torch_npu.npu.synchronize()
    assert torch.equal(target.cpu(),host)
    results.append({'device':device,'bytes':host.numel(),'pinned':True,'h2d_d2h_equal':True})
    del target,host
record={'devices':results,'memlock':resource.getrlimit(resource.RLIMIT_MEMLOCK),'model_loaded':False}
with open(sys.argv[1],'x') as stream:
    json.dump(record,stream,indent=2)
    stream.write('\\n')
print('HOST_PIN_ADMISSION_FILE '+sys.argv[1])
"""
    result = subprocess.run(
        [
            sys.executable,
            *(["-S"] if env.get("VLLMHUST_NO_SITE") == "1" else []),
            "-c",
            code,
            str(result_file),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (out / "pinned-memory-preflight.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(
            "Pinned host memory/transfer admission failed before model loading; see pinned-memory-preflight.log"
        )
    if not result_file.is_file():
        raise ValueError("Pinned host admission result file missing")
    record = json.loads(result_file.read_bytes())
    if (record["model_loaded"] is not False or len(record["devices"]) != 2
            or {r["device"] for r in record["devices"]} != {0, 1}
            or any(r["pinned"] is not True or r["h2d_d2h_equal"] is not True
                   or r["bytes"] != 32 * 2**20 for r in record["devices"])):
        raise ValueError("Pinned host admission file does not cover both devices")


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def environment(root, profile, out, cache, devices):
    env = dict(os.environ)
    for key in list(env):
        if key in {
            "LD_PRELOAD",
            "PYTHONPATH",
            "OPENAI_API_KEY",
            "GH_TOKEN",
            "GITHUB_TOKEN",
        } or key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            env.pop(key)
    command = "set -e\nsource /usr/local/Ascend/cann-9.1.0/set_env.sh >/dev/null\nsource /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1 >/dev/null\nexec env -0"
    result = subprocess.run(
        ["bash", "-c", command], env=env, capture_output=True, check=True, timeout=30
    )
    env = dict(
        item.split("=", 1) for item in result.stdout.decode().split("\0") if item
    )
    env["PYTHONPATH"] = ":".join(
        profile["pythonpath"]
        + [
            "/usr/local/Ascend/cann-9.1.0/python/site-packages",
            "/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe",
        ]
    )
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "4",
            "TASK_QUEUE_ENABLE": "0",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
            "BETTERSCALE_FIA_METADATA": "owned",
            "BETTERSCALE_STATE_CACHE_HOST_BYTES": str(8 * 2**30),
            "BETTERSCALE_GDN_LIBRARY": str(
                root
                / "runtime/betterscale-distribution/betterscale/patches/qwen_gdn/libbs_gdn.so"
            ),
            "BETTERSCALE_GDN_HOST_LIBRARY": str(
                root / "native-libs/libbs_gdn_host_cann91.so"
            ),
            "BETTERSCALE_CP_LIBRARY": str(
                root / "native-libs/context-parallel/libbs_fia_cp.so"
            ),
            "VLLM_HUST_EXT_CONFIG": str(root / "config/manager.json"),
            "VLLM_KV_ENABLE_SCHOOL_028": "1",
            "VLLM_KV_POLICY_MODE": "controller",
            "HUST_MICROFLOW_ARM": "on",
            "HUST_TOKEN_OBSERVER": "0",
            "MTP_PROFILE": "0",
            "VLLM_PLUGINS": "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,kv_materialization,hust_microflow_school028_fa,hust_joint_packed",
            "MTP_TOKENS": "2",
            "MTP_GDN_LAYOUT_FUSION": "1",
            "BETTERSCALE_MTP_GREEDY": "1",
            "BETTERSCALE_GDN_SMALL_COPIES": "1",
            "ASCEND_RT_VISIBLE_DEVICES": devices,
            "TMPDIR": str(temporary_directory(out)),
            "XDG_CACHE_HOME": str(cache / "import-cache"),
            "ASCEND_PROCESS_LOG_PATH": str(out / "cann-logs"),
            "KVMAT_RUN_DIR": str(out),
            "VLLMHUST_ENGINE_BOOTSTRAPPED": str(root),
            "VLLM_RPC_BASE_PATH": str(ipc_directory(out)),
        }
    )
    for key in ("TMPDIR", "XDG_CACHE_HOME", "ASCEND_PROCESS_LOG_PATH"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--devices",
        required=True,
        help="Two container logical indices; physical cards come from VLLM_HUST_PHYSICAL_DEVICES",
    )
    parser.add_argument("--model", type=Path, default=Path("/models/Qwen3.5-35B-A3B"))
    parser.add_argument(
        "--phase", choices=["preflight", "serve", "warmup", "client"], required=True
    )
    parser.add_argument("--lease-timeout", type=float, default=1800)
    parser.add_argument(
        "--kind", choices=["joint", "continuous", "dataset"], default="joint"
    )
    parser.add_argument("--dataset-spec", type=Path)
    parser.add_argument("--continuous-inputs", type=Path)
    parser.add_argument("--acceptance-contract", type=Path)
    parser.add_argument("--acceptance-baseline", type=Path)
    parser.add_argument("--qualified-continuous", type=Path)
    parser.add_argument("--reference-observation", action="store_true")
    parser.add_argument("--reference-profile", choices=("submission-v2", "full-answer-v1"), default="full-answer-v1")
    args = parser.parse_args()
    root, out, cache = (
        args.deployment.resolve(),
        args.out.resolve(),
        args.cache.resolve(),
    )
    profile = json.loads((root / "deployment.json").read_text())
    qualification = None
    if args.kind in ("joint", "dataset") and args.phase in {"serve", "client"}:
        if args.qualified_continuous is None:
            raise ValueError(
                "Joint inference requires a current --qualified-continuous run"
            )
        from qualification import qualify

        qualification = qualify(args.qualified_continuous, root)
    if os.environ.get("VLLMHUST_ENGINE_BOOTSTRAPPED") != str(root):
        from restore_b1_r4 import verify_installed_sources

        verify_installed_sources(root)
        from dependency_admission import admit_dependencies

        dependency_admission = admit_dependencies(root)
        if args.phase not in ("client", "warmup"):
            out.mkdir(parents=True, exist_ok=False)
            write(out / "dependency-admission.json", dependency_admission)
            cache.mkdir(parents=True, exist_ok=False)
            from resource_guard import acquire

            acquire(out, args.lease_timeout)
        env = environment(root, profile, out, cache, args.devices)
        os.execve(
            profile["python"],
            [profile["python"], str(Path(__file__).resolve()), *sys.argv[1:]],
            env,
        )
    if args.phase == "warmup":
        if args.acceptance_contract:
            from completion_warmup import run
        else:
            from serving_warmup import run

        return run(
            out, args.dataset_spec, args.continuous_inputs, args.qualified_continuous
        )
    if args.phase == "client":
        if args.acceptance_contract:
            from acceptance_client import run as run_acceptance

            return run_acceptance(args.acceptance_contract, out)
        if args.kind == "dataset":
            from dataset_client import run

            return run(args.dataset_spec, out)
        if args.kind == "continuous":
            client = load(
                "engine_continuous_client",
                root / "runtime/entrypoints/continuous_client.py",
            )
            client.TASK = args.continuous_inputs.resolve().parent
        else:
            client = load(
                "engine_joint_client", root / "mods/packed-host/joint_client.py"
            )
            client.ROOT = root / "inputs"
            if args.reference_observation:
                from reference_http_client import run as run_reference_client

                return asyncio.run(run_reference_client(client, out, args.reference_profile))
        return asyncio.run(client.main(out))
    import launch_boundary_b1

    host = root / "launch-host"
    host.mkdir(exist_ok=True)
    for name, target in {
        "runtime": root / "runtime/ascend",
        "mmlu-b1-microflow-fa-on-r10": root / "provenance/base-launch",
    }.items():
        link = host / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        if link.resolve() != target.resolve():
            raise ValueError("Unexpected launch-host alias: " + name)
    launch_boundary_b1.ROOT = host
    launch_boundary_b1.MODEL = args.model.resolve()
    from launch_corrected_timing import prepare
    from serve_b1_budget import configure_host_budget

    command, env, original = prepare(out, cache, out.name)
    command, env = configure_host_budget(command, env, 4 * 2**30)
    ascend_config = root / "runtime/ascend/vllm_ascend/ascend_config.py"
    if ascend_config.is_file() and "ascend_log_path" in ascend_config.read_text():
        # This is the SDK's supported file-logging option, separate from CANN's
        # ASCEND_PROCESS_LOG_PATH. Do not relocate HOME or alter runtime code.
        index = command.index("--additional-config") + 1
        additional = json.loads(command[index])
        additional["ascend_log_path"] = str(out / "vllm-ascend-logs")
        Path(additional["ascend_log_path"]).mkdir()
        command[index] = json.dumps(additional)
    from runtime_geometry import apply as apply_geometry

    geometry = apply_geometry(profile, command, env)
    env["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
    env["VLLM_PLUGINS"] += ",hust_joint_packed"
    env.update(
        HUST_JOINT_PACKED_HOST="1",
        TMPDIR=str(temporary_directory(out)),
        ASCEND_CACHE_PATH=str(cache / "ascend"),
    )
    for key in ("TMPDIR", "ASCEND_CACHE_PATH"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    original_plan = json.loads((root / "provenance/original-plan.json").read_bytes())
    from plan_identity import current_b1_plan
    from resource_guard import cards

    plan = current_b1_plan(
        original_plan, release=profile["release"], logical_devices=args.devices,
        physical_cards=cards(), host_budget_per_rank_bytes=4 * 2**30,
    )
    plan.update(
        qualification=qualification
        or {"status": "pending", "historical_result_reused": False},
        activation_audit="Pending re-audit of this run's raw events and both ranks",
        cache_policy="Fresh per-arm service/compiler caches. Read-only POSIX_FADV_DONTNEED hints release this checkpoint's clean pages before startup and after both ranks load; 12GiB raw admission is required before model allocation, then the original SDK4GiB conservative working-set reserve and OOM guard apply. No global drop_caches, no checkpoint writes, no imported answers/state. Same admission is required for compared arms.",
        source_bindings_sha256=hashlib.sha256(
            (root / "provenance/installed-files.json").read_bytes()
        ).hexdigest(),
        kind=args.kind,
        output=str(out),
        cache=str(cache),
        model_view=str(args.model.resolve()),
        logical_devices=args.devices,
        physical_cards=cards(),
        quality_plan=str(root / "inputs/mmlu-quality018-r2/quality-request-plan.json"),
        scorer=str(root / "scorers/mmlu_pro_pyramidkv.py"),
        pricing_basis=str(root / "config/pricing-basis.json"),
        expected_versions={
            "vllm": "0.28.1.post1.dev143+gf18cf803c",
            "vllm_ascend": profile["host_contract_qualification"]["installed_metadata"][
                "vllm-ascend"
            ],
        },
        original_plan_sha256=profile["original_plan_sha256"],
        deployment=str(root),
        deployment_changes=str(root / "provenance/deployment-changes.json"),
    )
    if geometry is not None:
        plan["runtime_geometry"] = geometry
    if args.kind == "continuous":
        if args.continuous_inputs is None:
            raise ValueError("Continuous mode requires prospectively frozen inputs")
        plan["input_sha256"] = hashlib.sha256(
            args.continuous_inputs.read_bytes()
        ).hexdigest()
        plan["input_contract"] = {
            "kind": "continuous", "path": str(args.continuous_inputs.resolve()),
            "sha256": plan["input_sha256"],
        }
        command += ["--reasoning-parser", "qwen3"]
    if args.kind == "dataset":
        from dataset_client import checked

        spec, _ = checked(args.dataset_spec)
        if spec["service_role"] != "B1":
            raise ValueError("B1 serving role differs from dataset plan")
        maximum_images = (
            1 if spec["dataset"] == "hle" and spec["stage"] == "solver" else 0
        )
        index = command.index("--limit-mm-per-prompt") + 1
        command[index] = json.dumps({"image": maximum_images, "video": 0})
        plan["dataset_spec_sha256"] = hashlib.sha256(
            args.dataset_spec.read_bytes()
        ).hexdigest()
        plan["input_contract"] = {
            "kind": "dataset", "path": str(args.dataset_spec.resolve()),
            "sha256": plan["dataset_spec_sha256"],
        }
        # Dataset-specific solver/judge/trace scoring is bound by its checked
        # execution specification; do not label it as the joint MMLU scorer.
        plan.pop("quality_plan")
        plan.pop("scorer")
        plan["dataset_modality"] = "all" if maximum_images else "text"
        plan["deployment_difference"] = (
            "Enable one problem image for HLE solver when selected; preserve the verified "
            + profile["release"] + " composition and4GiB/rank State budget"
        )
        from judge_grammar import configure

        configure(spec, command, plan)
    from serving_warmup import selected, protocol

    if selected(profile):
        command += ["--worker-extension-cls", "compile_boundary.WorkerExtension"]
        env["VLLM_SERVER_DEV_MODE"] = "1"
        env["PYTHONPATH"] = str(Path(__file__).parent) + ":" + env["PYTHONPATH"]
        plan["serving_warmup"] = protocol(args.kind, capacity36=profile["release"] == "full-answer-b1-20261117-r8")
        plan["engine_release"] = profile["release"]
    if args.acceptance_contract:
        from acceptance_admission import apply as apply_acceptance

        apply_acceptance(
            plan, command, args.acceptance_contract, args.acceptance_baseline
        )
        if "serving_warmup" in plan:
            from completion_warmup import protocol as completion_protocol

            plan["serving_warmup"] = completion_protocol(args.kind, acceptance=True, capacity36=profile["release"] == "full-answer-b1-20261117-r8")
    if args.reference_observation:
        if args.kind != "joint" or not selected(profile):
            raise ValueError("Reference observation requires the observed-warmup MMLU service")
        from reference_observation import configure

        configure(command, env, out)
        plan.update(reference_observation=True, timing_eligible_for_performance_comparison=False)
    write(out / "plan.json", plan)
    if args.phase == "preflight":
        command = [
            profile["python"],
            str(root / "mods/correction/corrected_service_entry.py"),
            "--preflight-receipt",
            str(out / "component-activation.json"),
            *command[2:],
        ]
    else:
        command = [
            profile["python"],
            str(root / "runtime/entrypoints/serve_b1_child.py"),
            str(out / "plan.json"),
            *command[2:],
        ]
    write(
        out / "launch-command.json",
        {
            "command": command,
            "phase": args.phase,
            "deployment": str(root),
            "original_base_launch": str(original),
        },
    )
    audit_env = {
        k: v
        for k, v in env.items()
        if (
            k.startswith(
                ("ASCEND", "ATB", "BETTERSCALE", "HUST", "VLLM", "MTP", "HCCL")
            )
            or k in {"PYTHONPATH", "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "TMPDIR"}
        )
        and not k.endswith(("_TOKEN", "_API_KEY", "_PASSWORD", "_SECRET"))
        and k != "VLLMHUST_ENGINE_LOCKS"
    }
    if any(
        "gennova" in v.lower()
        for k, v in audit_env.items()
        if k == "PYTHONPATH" or k.startswith("BETTERSCALE")
    ):
        raise ValueError("Legacy project path leaked into active deployment")
    write(out / "runtime-environment.json", audit_env)
    check_completion_inbox(root, out, env)
    check_pinned_host_memory(out, env)
    os.environ.update(env)
    from vllm_hust_ext.cli import main as manager

    return manager(["run", "--shutdown-grace-seconds", "90", "--", *command])


if __name__ == "__main__":
    raise SystemExit(main())
