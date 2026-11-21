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
    if any(key.startswith(('BETTERSCALE', 'HUST', 'MTP', 'KVMAT')) for key in os.environ):
        raise ValueError('Optimized-engine settings leaked into official serving')
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
    output = Path(plan['output'])
    (output / 'actual-package-imports.json').write_text(json.dumps(dict(
        python=sys.executable, prefix=sys.prefix, no_site=bool(sys.flags.no_site),
        versions=actual, modules=origins, official_baseline=plan['baseline_id'],
        source_patched=False, optimized_mods_enabled=False), indent=2) + '\n')
    # Use only the vendor-supported log option when present in this release.
    ascend_config = Path(vllm_ascend.__file__).resolve().parent / 'ascend_config.py'
    if ascend_config.is_file() and 'ascend_log_path' in ascend_config.read_text():
        logs = output / 'vllm-ascend-logs'
        logs.mkdir()
        plan['serve_arguments'] += ['--additional-config', json.dumps(dict(ascend_log_path=str(logs)))]
    (output / 'effective-serving-arguments.json').write_text(json.dumps(plan['serve_arguments'], indent=2) + '\n')
    sys.argv = ['vllm', 'serve', plan['model'], *plan['serve_arguments']]
    runpy.run_module('vllm.entrypoints.cli.main', run_name='__main__')


if __name__ == '__main__':
    main()
