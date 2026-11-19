"""Manager-owned clean native client, with a separately frozen timing plan."""

import os

# multiprocessing.spawn imports this main module before constructing Core or
# Worker. Install the same pinned import transform in every process, before any
# serving plugin can import its target modules.
if os.environ.get("VLLMHUST_NATIVE_OBSERVER_PLAN"):
    from native_observer_removal import install

    install(os.environ["VLLMHUST_NATIVE_OBSERVER_PLAN"])


def validate_config(config):
    from reference_capture_profile import validate_native_plan

    if config["schema"] != "current-native-m01-timing/v1":
        raise ValueError("Native timing plan schema differs")
    return validate_native_plan(config)


def main():
    import argparse
    import dataclasses
    import importlib.metadata
    import json
    from pathlib import Path
    import sys

    from dataset_contract import save, sha
    from native_state_cache import reset_state_cache

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--additional-config", type=json.loads, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    validate_config(config)
    if str(args.config.resolve()) != os.environ.get("VLLMHUST_NATIVE_OBSERVER_PLAN"):
        raise ValueError("Every native process must use the same observer-removal plan")
    for name, expected in config["files"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Frozen native timing input/source changed: " + name)
    output = Path(config["output"])
    output.mkdir(exist_ok=False)
    if "vllm" in sys.modules:
        raise ValueError("Frozen Manager composition must precede vLLM imports")
    from corrected_service_entry import compose_environment

    os.environ.update(compose_environment(os.environ))
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="1", VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1")
    if any(os.environ.get(name) == "1" for name in ("VLLMHUST_M01_PROC_CAPTURE", "MTP_PROFILE", "HUST_TOKEN_OBSERVER")):
        raise ValueError("Raw accounting/profiling is enabled in the native clock")
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    import betterscale.qwen35_worker as actual_worker
    import corrected_serving_runtime
    import torch
    import vllm

    if (actual_worker.Worker is not corrected_serving_runtime.CorrectedWorker
            or vllm.__version__ != config["expected_versions"]["vllm"]
            or importlib.metadata.version("vllm_ascend") != config["expected_versions"]["vllm_ascend"]):
        raise ValueError("Native runtime composition differs from the frozen current engine")
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.engine.arg_utils import EngineArgs

    cli = FlexibleArgumentParser()
    ServeSubcommand().subparser_init(cli.add_subparsers(dest="subparser"))
    serve_args = cli.parse_args(config["serve_arguments"])
    # ServeSubcommand normalizes the positional model tag before EngineArgs.
    if getattr(serve_args, "model_tag", None):
        serve_args.model = serve_args.model_tag
    engine_args = EngineArgs.from_cli_args(serve_args)
    if (engine_args.additional_config != args.additional_config
            or engine_args.max_num_seqs != 36 or engine_args.tensor_parallel_size != 2
            or engine_args.worker_extension_cls != "native_timing_worker.NativeTimingWorker"):
        raise ValueError("Parsed native EngineArgs differ from r8/current source arguments")
    engine_args.disable_log_stats = True
    engine_args.per_request_spec_decode_metrics = "none"
    from native_plan_values import encode

    # asdict reconstructs Mapping subclasses and changes Counter semantics.
    # Read fields without mutating or reconstructing any effective value.
    save(output / "effective-engine-args.json", encode({field.name: getattr(engine_args, field.name)
                                                       for field in dataclasses.fields(engine_args)}))
    from betterscale.models.qwen35 import cache_engine
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.serial_utils import MsgpackEncoder
    from vllm import envs

    cache_engine.install()
    if not getattr(EngineCoreProc, "_betterscale_cache_wakeup", False) or envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
        raise ValueError("Native timing requires production Proc State maintenance and named safe RPCs")
    encoder = MsgpackEncoder()
    for method in ("hust_native_observer_audit", "hust_runtime_synchronize", "hust_compile_prepare", "hust_compile_seal", "hust_compile_status"):
        encoder.encode((0, 1, "collective_rpc", (method, 120, (), None)))
    save(output / "composition.json", dict(preflight_completed=True, model_loaded=False,
         npu_context_initialized=torch.npu.is_initialized(), engine_core_mode="native_offline_multiprocess",
         versions=config["expected_versions"], config_sha256=sha(args.config),
         plugins=os.environ["VLLM_PLUGINS"], named_rpc_serialization_verified=True,
         efficiency_percent=None, necessary_work_certified=False))
    if args.preflight:
        print("NATIVE_M01_COMPOSITION_PASSED", flush=True)
        return
    from native_timing_session import run

    run(config, engine_args, reset_state_cache)


if __name__ == "__main__":
    main()
