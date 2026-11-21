"""Explicit adapters to the unchanged, pinned HTTP evaluation clients."""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import subprocess


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=('warmup', 'client', 'prepare-context', 'prepare-native', 'summarize', 'metrics', 'plots'), required=True)
    args = parser.parse_args()
    plan = json.loads(args.config.read_bytes())
    tools = Path(plan['client_tools'])
    sys.path.insert(0, str(tools / 'scripts'))
    from dataset_contract import sha
    for name, expected in plan['client_sources'].items():
        if sha(Path(name)) != expected:
            raise ValueError('Pinned client source changed')
    sdk, output = Path(plan['protocol_deployment']), Path(plan['output'])
    if args.phase == 'plots':
        from report_environment import report_python
        python, env = report_python(plan.get('report_python'))
        return subprocess.run([str(python), '-B', str(Path(__file__).with_name('official_plots.py')),
                               str(output / 'report.json')], env=env).returncode
    if args.phase == 'summarize':
        from types import SimpleNamespace
        from dataset_client import checked
        data, original = checked(Path(plan['dataset_spec']))
        original.summary(SimpleNamespace(plan=data['plan'], responses=data['responses'],
                         grades=str(output / 'evaluation'), output=str(output / 'quality-summary')))
        return 0
    if args.phase == 'metrics':
        if plan['kind'] == 'context':
            from acceptance_report import inspect
        else:
            from report_dataset import inspect
        result = inspect(output.parent)
        print('OFFICIAL_METRICS_JSON ' + json.dumps(result, ensure_ascii=False))
        return 0
    if args.phase == 'prepare-context':
        from acceptance_inputs import prepare
        prepare(Path(plan['model']), Path(plan['data_cache']), Path(plan['context_output']))
        return 0
    if args.phase == 'prepare-native':
        from native_timing_inputs import prepare
        prepare(sdk, Path(plan['model']), Path(plan['qualified_continuous']), Path(plan['native_workload']))
        return 0
    if args.phase == 'warmup':
        from serving_warmup import run
        if plan['kind'] == 'context':
            from completion_warmup import run
        return run(output, plan.get('dataset_spec'), plan['continuous_inputs'], None)
    if plan['kind'] == 'dataset':
        from dataset_client import run
        return run(Path(plan['dataset_spec']), output)
    if plan['kind'] == 'context':
        from acceptance_client import run
        return run(Path(plan['acceptance_contract']), output)
    if plan['kind'] == 'continuous':
        client = module(sdk / 'runtime/entrypoints/continuous_client.py', 'official_continuous_http')
        client.TASK = Path(plan['continuous_inputs']).parent
    else:
        client = module(sdk / 'mods/packed-host/joint_client.py', 'official_mmlu_http')
        client.ROOT = sdk / 'inputs'
    return asyncio.run(client.main(output)) or 0


if __name__ == '__main__':
    raise SystemExit(main())
