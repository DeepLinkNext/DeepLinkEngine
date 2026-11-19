"""Spawn-safe original018 public native client; no model load on import."""


def validate_config(config, *, no_site):
    from reference_capture_profile import validate_native_plan

    if (config["schema"] != "baseline-native-m01-timing/v1" or config["role"] != "B0"
            or no_site is not True):
        raise ValueError("Original018 native plan or isolated interpreter differs")
    return validate_native_plan(config)


def reset_prefix_cache(engine, output):
    from dataset_contract import save

    if engine.has_unfinished_requests():
        raise ValueError("Cannot reset original018 cache with unfinished requests")
    if not engine.reset_prefix_cache():
        raise ValueError("Original018 prefix-cache reset failed")
    save(output, dict(schema="baseline-native-prefix-reset/v1", reset=True,
                     b1_state_cache_present=False, shared_request_cache_salts_unique=True))


def main():
    import argparse
    import dataclasses
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    import sys

    from baseline_admission import validate_identity
    from dataset_contract import save, sha
    from native_plan_values import encode

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    validate_config(config, no_site=bool(sys.flags.no_site))
    for name, expected in config["files"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Frozen baseline native source changed: " + name)
    if (os.environ["VLLM_PLUGINS"] != "ascend,ascend_model_loader,ascend_kv_connector"
            or os.environ.get("VLLMHUST_NATIVE_OBSERVER_PLAN")
            or any(os.environ.get(key) == "1" for key in
                   ("VLLMHUST_M01_PROC_CAPTURE", "MTP_PROFILE", "HUST_TOKEN_OBSERVER"))):
        raise ValueError("Baseline plugins or native observation environment differs")
    output = Path(config["output"])
    output.mkdir(exist_ok=False)
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="1", VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1")
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    import torch
    import vllm
    import vllm_ascend
    from vllm import envs

    identity = dict(role="B0", vllm_version=vllm.__version__,
        ascend_version=importlib.metadata.version("vllm_ascend"), vllm_source=vllm.__file__,
        ascend_source=vllm_ascend.__file__, cann_home=os.environ["ASCEND_HOME_PATH"])
    admitted = json.loads(Path(config["baseline_admission"]).read_bytes())
    validate_identity(identity, admitted)
    if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
        raise ValueError("Native baseline requires safe named worker RPCs")
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.serial_utils import MsgpackEncoder

    parser = FlexibleArgumentParser()
    ServeSubcommand().subparser_init(parser.add_subparsers(dest="subparser"))
    cli = parser.parse_args(config["serve_arguments"])
    if getattr(cli, "model_tag", None):
        cli.model = cli.model_tag
    engine_args = EngineArgs.from_cli_args(cli)
    if (engine_args.max_num_seqs != 24 or engine_args.tensor_parallel_size != 2
            or engine_args.max_num_batched_tokens != 4096 or engine_args.kv_cache_memory_bytes != 20 * 2**30
            or engine_args.worker_extension_cls != "baseline_native_worker.BaselineNativeWorker"):
        raise ValueError("Original018 native geometry differs from the frozen profile")
    engine_args.disable_log_stats = True
    save(output / "effective-engine-args.json", encode({f.name: getattr(engine_args, f.name)
         for f in dataclasses.fields(engine_args)}))
    encoder = MsgpackEncoder()
    for method in ("hust_native_observer_audit", "hust_runtime_synchronize", "hust_compile_prepare",
                   "hust_compile_seal", "hust_compile_status"):
        encoder.encode((0, 1, "collective_rpc", (method, 120, (), None)))
    save(output / "composition.json", dict(preflight_completed=True, model_loaded=False,
        npu_context_initialized=torch.npu.is_initialized(), engine_core_mode="native_offline_multiprocess",
        identity=identity, config_sha256=sha(args.config), plugins=os.environ["VLLM_PLUGINS"],
        named_rpc_serialization_verified=True, efficiency_percent=None, necessary_work_certified=False))
    if args.preflight:
        print("BASELINE_NATIVE_M01_COMPOSITION_PASSED", flush=True)
        return
    from native_timing_session import run

    run(config, engine_args, reset_prefix_cache)


if __name__ == "__main__":
    main()
