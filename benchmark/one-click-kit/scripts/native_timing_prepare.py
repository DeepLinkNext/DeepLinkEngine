"""Build a native plan from the frozen serving launcher inside admitted SDK env."""

import argparse
import json
import os
from pathlib import Path
import sys

from dataset_contract import save, sha
from engine_entry import check_completion_inbox, check_pinned_host_memory
from resource_guard import acquire
from runtime_geometry import apply as apply_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--qualified-continuous", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--formal-repetitions", type=int, default=1)
    parser.add_argument("--reference-profile", choices=("submission-v2", "full-answer-v1"), default="full-answer-v1")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    from reference_capture_profile import profile as capture_profile

    capture = capture_profile(args.reference_profile)
    if args.formal_repetitions != capture["formal_repetitions"]:
        raise ValueError("Native repetitions differ from the selected frozen reference profile")
    root, out, cache = args.deployment.resolve(), args.out.resolve(), args.cache.resolve()
    profile = json.loads((root / "deployment.json").read_bytes())
    if profile["release"] != "full-answer-b1-20261117-r8":
        raise ValueError("Current native timing requires the frozen r8 configuration")
    if os.environ.get("VLLMHUST_ENGINE_BOOTSTRAPPED") != str(root):
        raise ValueError("Native preparation requires the parent-admitted SDK environment")
    acquire(out, 0)  # Validate inherited descriptors before any SDK/model import.
    if not args.preflight:
        check_completion_inbox(root, out, os.environ)
        check_pinned_host_memory(out, os.environ)
    import launch_boundary_b1
    from launch_corrected_timing import prepare
    from serve_b1_budget import configure_host_budget

    host = root / "launch-host"
    for name, expected in {"runtime": root / "runtime/ascend", "mmlu-b1-microflow-fa-on-r10": root / "provenance/base-launch"}.items():
        if (host / name).resolve() != expected.resolve():
            raise ValueError("Frozen launcher alias has not been admitted")
    launch_boundary_b1.ROOT = host
    launch_boundary_b1.MODEL = args.model.resolve()
    command, env, original = prepare(out, cache, out.name)
    command, env = configure_host_budget(command, env, 4 * 2**30)
    geometry = apply_geometry(profile, command, env)
    if geometry is None:
        raise ValueError("Native timing has no current r8 geometry")
    additional_index = command.index("--additional-config") + 1
    additional = json.loads(command[additional_index])
    additional["ascend_log_path"] = str(out / "vllm-ascend-logs")
    Path(additional["ascend_log_path"]).mkdir()
    command[additional_index] = json.dumps(additional)
    command += ["--worker-extension-cls", "native_timing_worker.NativeTimingWorker"]
    script_root = Path(__file__).resolve().parent
    env.update(ASCEND_RT_VISIBLE_DEVICES=args.devices, HUST_JOINT_PACKED_HOST="1",
               VLLM_HUST_TEST_OUTPUT=str(out), VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1",
               PYTHONPATH=str(script_root) + ":" + env["PYTHONPATH"],
               ASCEND_CACHE_PATH=str(cache / "ascend"))
    env["VLLM_PLUGINS"] += ",hust_joint_packed"
    Path(env["ASCEND_CACHE_PATH"]).mkdir(parents=True)
    from native_timing_inputs import prepare as prepare_inputs

    workload_path = out / "token-workload.json"
    prepare_inputs(root, args.model, args.qualified_continuous, workload_path)
    source_rows = {
        "vllm_kv_materialization.school_vllm_028": ("kvmat", "mods/kv-materialization/vllm_kv_materialization/school_vllm_028.py"),
        "hust_prefill_microflow.runtime.school028_attention": ("attention", "mods/microflow/hust_prefill_microflow/runtime/school028_attention.py"),
        "hust_prefill_microflow.school028.worker": ("worker", "mods/microflow/hust_prefill_microflow/school028/worker.py"),
    }
    observer_sources = {name: dict(kind=kind, path=str(root / relative), sha256=sha(root / relative))
                        for name, (kind, relative) in source_rows.items()}
    from native_observer_removal import transform

    for row in observer_sources.values():
        transform(Path(row["path"]).read_bytes(), row["kind"])
    native_files = [script_root / name for name in ("native_timing_prepare.py", "native_timing_entry.py", "native_timing_inputs.py",
                    "native_public_client.py", "native_output_geometry.py", "native_timing_session.py", "native_state_cache.py", "native_observer_removal.py", "native_timing_worker.py",
                    "reference_capture_profile.py", "native_plan_values.py", "compile_boundary.py", "runtime_identity.py", "runtime_geometry.py")]
    files = {str(path): sha(path) for path in [*native_files, workload_path, root / "deployment.json", root / "provenance/installed-files.json", original]}
    plan = dict(schema="current-native-m01-timing/v1", deployment=str(root), model=str(args.model.resolve()),
                engine_release=profile["release"], output=str(out / "client"), admission_root=str(out.parent),
                workload=str(workload_path), files=files, observer_sources=observer_sources,
                serve_arguments=command[2:], serving_launcher_geometry=geometry,
                expected_versions=dict(vllm="0.28.1.post1.dev143+gf18cf803c", vllm_ascend=profile["host_contract_qualification"]["installed_metadata"]["vllm-ascend"]),
                formal_repetitions=args.formal_repetitions, reference_capture_profile=capture["id"], max_tokens=capture["max_tokens"],
                sampling=dict(max_tokens=capture["max_tokens"], temperature=0, seed=0, ignore_eos=capture["ignore_eos"],
                              top_p=1.0, top_k=-1, presence_penalty=0.0, detokenize=False),
                statistic="Preserve every predeclared full fixed200 repetition; use sum of matched reference times / sum of native times. No best-run selection.",
                intentional_frontend_changes=["Public native token-prompt client; HTTP and chat formatting outside T", "Public DELTA token outputs, detokenize=False; no text/tokenizer inside T", "disable_log_stats=True; per_request_spec_decode_metrics=none", "Source-pinned removal of only three pure observer families; SDK files unedited"],
                necessary_work_certified=False, efficiency_percent=None)
    plan_path = out / "native-plan.json"
    save(plan_path, plan)
    env["VLLMHUST_NATIVE_OBSERVER_PLAN"] = str(plan_path)
    native_command = [profile["python"], str(script_root / "native_timing_entry.py"), "--config", str(plan_path),
                      "--additional-config", json.dumps(additional)]
    if args.preflight:
        native_command += ["--preflight"]
    save(out / "launch-command.json", dict(command=native_command, preserved_serving_command=command,
                                           environment={key: value for key, value in env.items() if key.startswith(("VLLM", "HUST", "BETTERSCALE", "ASCEND", "MTP")) or key == "PYTHONPATH"}))
    os.environ.update(env)
    from vllm_hust_ext.cli import main as manager

    return manager(["run", "--shutdown-grace-seconds", "90", "--", *native_command])


if __name__ == "__main__":
    raise SystemExit(main())
