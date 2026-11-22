"""Run unmodified upstream serving code with an explicitly observed worker."""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    plan = json.loads(Path(sys.argv[1]).read_bytes())
    root = Path(plan['sdk']).resolve()
    actual = {name: metadata.version(name) for name in ('vllm', 'vllm-ascend', 'torch', 'torch-npu')}
    if actual != plan['runtime_versions'] or not Path(sys.prefix).resolve().is_relative_to(root):
        raise ValueError('Official serving interpreter/package identity differs')
    for name, expected in plan['worker_sources'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Observed-warmup source differs')
    from official_runtime_identity import runtime_identity
    observed_runtime = runtime_identity(root)
    if observed_runtime != plan['runtime_identity']:
        raise ValueError('Serving runtime source/build identity differs')
    if any(key.startswith(('BETTERSCALE', 'HUST', 'MTP', 'KVMAT')) for key in os.environ):
        raise ValueError('Optimized-engine settings leaked into official serving')
    from engine import service_logging
    output = Path(plan['output'])
    logging_plan = service_logging.prepare(root, output, os.environ)
    service_logging.install(logging_plan)
    import torch
    import torch_npu
    import vllm
    import vllm_ascend
    modules = dict(torch=torch, torch_npu=torch_npu, vllm=vllm, vllm_ascend=vllm_ascend)
    origins = {}
    for name, module in modules.items():
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(root):
            raise ValueError('Official runtime module loaded outside image')
        origins[name] = dict(path=str(origin), sha256=hashlib.sha256(origin.read_bytes()).hexdigest())
    (output / 'actual-package-imports.json').write_text(json.dumps(dict(
        python=sys.executable, prefix=sys.prefix, no_site=bool(sys.flags.no_site),
        versions=actual, modules=origins, official_baseline=plan['baseline_id'],
        runtime_identity=observed_runtime, upstream_source_patched=False,
        source_patched=False, optimized_mods_enabled=False,
        logging_adapter_plan=str(output / 'service-logging-plan.json'),
        logging_adapter_receipt=logging_plan['receipt']), indent=2) + '\n')
    from engine_entry import check_pinned_host_memory
    check_pinned_host_memory(output, dict(os.environ))
    # Use only the vendor-supported log option when present in this release.
    ascend_config = Path(vllm_ascend.__file__).resolve().parent / 'ascend_config.py'
    if ascend_config.is_file() and 'ascend_log_path' in ascend_config.read_text():
        logs = output / 'vllm-ascend-logs'
        logs.mkdir(exist_ok=True)
        plan['serve_arguments'] += ['--additional-config', json.dumps(dict(ascend_log_path=str(logs)))]
    (output / 'effective-serving-arguments.json').write_text(json.dumps(plan['serve_arguments'], indent=2) + '\n')
    sys.argv = ['vllm', 'serve', plan['model'], *plan['serve_arguments']]
    runpy.run_module('vllm.entrypoints.cli.main', run_name='__main__')


if __name__ == '__main__':
    main()
