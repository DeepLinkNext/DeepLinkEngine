"""Official-image adapter to the unchanged shared native timing protocol."""
import dataclasses
from importlib import metadata
import json
from pathlib import Path
import sys


def main():
    config = json.loads(Path(sys.argv[1]).read_bytes())
    sys.path.insert(0, str(Path(config['client_tools']) / 'scripts'))
    from dataset_contract import save, sha
    from reference_capture_profile import validate_native_plan
    validate_native_plan(config)
    if not sys.flags.no_site or not Path(sys.prefix).resolve().is_relative_to(Path(config['sdk']).resolve()):
        raise ValueError('Official native interpreter differs')
    for name, expected in config['files'].items():
        if sha(Path(name)) != expected:
            raise ValueError('Official native source/input changed')
    actual = {name: metadata.version(name) for name in config['expected_versions']}
    if actual != config['expected_versions']:
        raise ValueError('Official native runtime package versions differ')
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    import vllm_ascend
    ascend_config = Path(vllm_ascend.__file__).resolve().parent / 'ascend_config.py'
    if ascend_config.is_file() and 'ascend_log_path' in ascend_config.read_text():
        logs = Path(config['output']).parent / 'vllm-ascend-logs'
        logs.mkdir()
        config['serve_arguments'] += ['--additional-config', json.dumps(dict(ascend_log_path=str(logs)))]
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.engine.arg_utils import EngineArgs
    from native_plan_values import encode
    parser = FlexibleArgumentParser()
    ServeSubcommand().subparser_init(parser.add_subparsers(dest='subparser'))
    cli = parser.parse_args(config['serve_arguments'])
    if getattr(cli, 'model_tag', None):
        cli.model = cli.model_tag
    args = EngineArgs.from_cli_args(cli)
    if args.tensor_parallel_size != 2 or args.max_num_seqs != 24 or args.max_num_batched_tokens != 4096 or args.kv_cache_memory_bytes != 20 * 1024**3:
        raise ValueError('Official native TP/batch/KV geometry differs')
    args.disable_log_stats = True
    output = Path(config['output'])
    output.mkdir(exist_ok=False)
    import os
    from engine_entry import check_pinned_host_memory
    check_pinned_host_memory(output, dict(os.environ))
    save(output / 'effective-engine-args.json', encode({field.name: getattr(args, field.name) for field in dataclasses.fields(args)}))
    save(output / 'actual-identity.json', dict(versions=actual, python=sys.executable,
         baseline_id=config['baseline_id'], official_sources_patched=False,
         no_optimized_mods=True, native_per_step_observer=False))
    from baseline_native_entry import reset_prefix_cache
    from native_timing_session import run
    run(config, args, reset_prefix_cache)


if __name__ == '__main__':
    main()
