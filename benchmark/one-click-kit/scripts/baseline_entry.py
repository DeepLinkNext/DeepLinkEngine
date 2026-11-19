#!/usr/bin/env python3
"""Serve original018/CANN8.5.1 and delegate clients to the shared frozen protocol."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

from baseline_admission import admit
from baseline_image_sdk import environment
from baseline_profile import HLE_PROFILE_SHA256
from baseline_profile import save as save_profile
from baseline_profile import verify as verify_profile
from engine_entry import (
    write,
    check_pinned_host_memory,
    temporary_directory,
    ipc_directory,
)
from resource_guard import acquire, cards
from restore_b1_r4 import ORIGINAL_PLAN, verify_installed_sources


def serving_arguments(root, kind, multimodal=False):
    if multimodal:
        path = (
            Path(__file__).resolve().parents[1]
            / "baseline-sources/hle-original018-serving.json"
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != HLE_PROFILE_SHA256:
            raise ValueError("Original018 HLE serving profile changed")
        return json.loads(path.read_bytes())["argv"]
    source = root / "runtime/entrypoints/prepare_retrieval_b0.py"
    nodes = [
        kw.value
        for node in ast.walk(ast.parse(source.read_bytes()))
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "serve_args"
    ]
    if len(nodes) != 1:
        raise ValueError("Ambiguous original018 serving arguments")
    args = ast.literal_eval(nodes[0])
    if kind != "continuous":
        index = args.index("--reasoning-parser")
        if args[index + 1] != "qwen3":
            raise ValueError("Original reasoning parser differs")
        del args[index : index + 2]
    if "--enforce-eager" in args:
        raise ValueError("Formal baseline cannot use eager mode")
    return args


def baseline_environment(root, out, cache, devices, variant="text"):
    env = environment(root, variant)
    selected = env.get("VLLM_HUST_PHYSICAL_DEVICES")
    for key in list(env):
        if key.startswith(
            ("BETTERSCALE", "HUST", "MTP", "KVMAT", "VLLM_HUST", "VLLM_KV")
        ):
            env.pop(key)
    if selected:
        env["VLLM_HUST_PHYSICAL_DEVICES"] = selected
    for key in ("GH_TOKEN", "GITHUB_TOKEN", "OPENAI_API_KEY"):
        env.pop(key, None)
    env.update(
        VLLM_PLUGINS="ascend,ascend_model_loader,ascend_kv_connector",
        VLLMHUST_BASELINE_BOOTSTRAPPED=str(out),
        ASCEND_CACHE_PATH=str(cache / "ascend"),
        ASCEND_RT_VISIBLE_DEVICES=devices,
        VLLM_HUST_TEST_OUTPUT=str(out),
        TMPDIR=str(temporary_directory(out)),
        VLLM_RPC_BASE_PATH=str(ipc_directory(out)),
        XDG_CACHE_HOME=str(cache / "xdg"),
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        NUMBA_CACHE_DIR=str(cache / "numba"),
        ASCEND_PROCESS_LOG_PATH=str(out / "cann-logs"),
    )
    for key in (
        "ASCEND_CACHE_PATH",
        "XDG_CACHE_HOME",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "NUMBA_CACHE_DIR",
        "ASCEND_PROCESS_LOG_PATH",
    ):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def model_view(model, cache, out, multimodal):
    if multimodal:
        return model.resolve()
    view = cache / "model-text-only"
    view.mkdir(exist_ok=False)
    config = json.loads((model / "config.json").read_bytes())["text_config"]
    config["architectures"] = ["Qwen3_5MoeForCausalLM"]
    # The draft ModelConfig is built independently and can inspect these keys
    # before the target architecture's original text-only verifier removes them.
    # Normalize the generated text view once, so both models use 1-D positions.
    rope = config.get("rope_parameters", {})
    removed = {
        key: rope.pop(key)
        for key in ("mrope_section", "mrope_interleaved")
        if key in rope
    }
    (view / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "model-config-view.json").write_bytes((view / "config.json").read_bytes())
    write(
        out / "model-config-normalization.json",
        {
            "source_config_sha256": hashlib.sha256(
                (model / "config.json").read_bytes()
            ).hexdigest(),
            "generated_config_sha256": hashlib.sha256(
                (view / "config.json").read_bytes()
            ).hexdigest(),
            "removed_text_only_rope_fields": removed,
            "reason": "Target and independently loaded MTP draft must both use one-dimensional text positions",
            "weights_modified": False,
        },
    )
    for path in model.iterdir():
        if path.is_file() and path.name != "config.json":
            (view / path.name).symlink_to(path.resolve())
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("deployment", "out", "cache", "baseline-deployment", "model"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument(
        "--kind", choices=["joint", "continuous", "dataset"], required=True
    )
    parser.add_argument("--phase", choices=["serve", "warmup", "client"], required=True)
    parser.add_argument("--dataset-spec", type=Path)
    parser.add_argument("--qualified-continuous", type=Path)
    parser.add_argument("--continuous-inputs", type=Path)
    parser.add_argument("--acceptance-contract", type=Path)
    parser.add_argument("--acceptance-baseline", type=Path)
    parser.add_argument("--lease-timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.phase in ("client", "warmup"):
        argv = list(sys.argv[1:])
        index = argv.index("--baseline-deployment")
        del argv[index : index + 2]
        os.execv(
            sys.executable,
            [sys.executable, str(Path(__file__).with_name("engine_entry.py")), *argv],
        )
    root, out, cache = (
        args.deployment.resolve(),
        args.out.resolve(),
        args.cache.resolve(),
    )
    multimodal = False
    if args.kind == "dataset":
        from dataset_client import checked

        spec, _ = checked(args.dataset_spec)
        multimodal = spec["dataset"] == "hle" and spec["stage"] == "solver"
    variant = "multimodal" if multimodal else "text"
    if os.environ.get("VLLMHUST_BASELINE_BOOTSTRAPPED") != str(out):
        verify_installed_sources(root)
        out.mkdir(parents=True, exist_ok=False)
        cache.mkdir(parents=True, exist_ok=False)
        receipt = admit(
            args.baseline_deployment, out / "baseline-source-admission.json", variant
        )
        acquire(out, args.lease_timeout)
        env = baseline_environment(
            args.baseline_deployment, out, cache, args.devices, variant
        )
        executable = receipt["python"]
        os.execve(
            executable,
            [executable, "-S", str(Path(__file__).resolve()), *sys.argv[1:]],
            env,
        )
    check_pinned_host_memory(out, dict(os.environ))
    view = model_view(args.model.resolve(), cache, out, multimodal)
    original = root / "provenance/original-plan.json"
    frozen_entry = root / "runtime/entrypoints/serve_b0_r2.py"
    entry_sha256 = hashlib.sha256(frozen_entry.read_bytes()).hexdigest()
    text_source = root / "runtime/entrypoints/prepare_retrieval_b0.py"
    plan = {
        "role": "B0",
        "kind": args.kind,
        "output": str(out),
        "cache": str(cache),
        "model_view": str(view),
        "serve_args": serving_arguments(root, args.kind, multimodal),
        "baseline_variant": variant,
        "baseline_deployment": str(args.baseline_deployment.resolve()),
        "deployment": str(root),
        "frozen_serving_entry_sha256": entry_sha256,
        "expected_versions": {"vllm": "0.18.0", "vllm_ascend": "0.18.0"},
        "quality_plan": str(
            root / "inputs/mmlu-quality018-r2/quality-request-plan.json"
        ),
        "scorer": str(root / "scorers/mmlu_pro_pyramidkv.py"),
        "pricing_basis": str(root / "config/pricing-basis.json"),
        "original_plan_sha256": ORIGINAL_PLAN,
        "original_plan_file_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "physical_cards": cards(),
        "logical_devices": args.devices,
        "scope": "Original pinned018 image SDK/CANN8.5.1; original text-only or full multimodal recipe selected explicitly per dataset. Host OS/driver/compiler observed separately.",
        "cache_policy": "Same checkpoint-only clean-cache hints and raw cgroup reserve; fresh writable service/compiler caches",
    }
    admission = verify_profile(
        plan["serve_args"],
        variant,
        args.kind,
        deployment=root,
        source_sha256=hashlib.sha256(text_source.read_bytes()).hexdigest()
        if variant == "text"
        else None,
        entry_sha256=entry_sha256,
    )
    save_profile(out, admission)
    plan["b0_profile"] = {
        "profile": admission["profile"],
        "profile_sha256": admission["profile_sha256"],
        "variant": admission["variant"],
        "argv_sha256": admission["argv_sha256"],
        "admission": "b0-profile-admission.json",
    }
    if args.kind == "continuous":
        plan["input_sha256"] = hashlib.sha256(
            args.continuous_inputs.read_bytes()
        ).hexdigest()
    if args.kind == "dataset":
        plan["dataset_spec_sha256"] = hashlib.sha256(
            args.dataset_spec.read_bytes()
        ).hexdigest()
    from serving_warmup import selected, protocol

    profile = json.loads((root / "deployment.json").read_bytes())
    if selected(profile):
        plan["serve_args"] += [
            "--worker-extension-cls",
            "compile_boundary.WorkerExtension",
        ]
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"
        os.environ["PYTHONPATH"] = (
            str(Path(__file__).parent) + ":" + os.environ["PYTHONPATH"]
        )
        plan["serving_warmup"] = protocol(args.kind, capacity36=profile["release"] == "full-answer-b1-20261117-r8")
        plan["engine_release"] = profile["release"]
    if args.acceptance_contract:
        from acceptance_admission import apply as apply_acceptance

        apply_acceptance(
            plan, plan["serve_args"], args.acceptance_contract, args.acceptance_baseline
        )
        if "serving_warmup" in plan:
            from completion_warmup import protocol as completion_protocol

            plan["serving_warmup"] = completion_protocol(args.kind, acceptance=True, capacity36=profile["release"] == "full-answer-b1-20261117-r8")
    write(out / "plan.json", plan)
    command = [
        sys.executable,
        "-S",
        str(Path(__file__).with_name("baseline_serve.py")),
        str(out / "plan.json"),
    ]
    write(
        out / "launch-command.json",
        {"command": command, "role": "B0", "variant": variant},
    )
    audit = {
        k: v
        for k, v in os.environ.items()
        if (
            k.startswith(("ASCEND", "ATB", "VLLM", "HCCL"))
            or k
            in (
                "PYTHONPATH",
                "PYTHONHOME",
                "TMPDIR",
                "LD_LIBRARY_PATH",
                "OMP_NUM_THREADS",
            )
        )
        and k != "VLLMHUST_ENGINE_LOCKS"
        and not k.endswith(("_TOKEN", "_API_KEY", "_SECRET", "_PASSWORD"))
    }
    write(out / "runtime-environment.json", audit)
    os.execv(command[0], command)


if __name__ == "__main__":
    raise SystemExit(main())
