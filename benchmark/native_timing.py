"""Run only the fixed200 native timing stage of the installed stable R10."""

import argparse
import json
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile

from benchmark import evaluation_client
from benchmark.native_timing_contract import MEASURED_RUNTIME, RELEASE, SDIST_SHA, VERSION, workload

ROOT = Path(__file__).resolve().parents[1]


def derive(source):
    once = evaluation_client.once
    controller = source['scripts/run_native_reference.py'].decode()
    controller = once(controller, 'from resource_guard import acquire, release, cards, verify_mapping',
        'from resource_guard import release, cards, verify_mapping\n'
        'from official.official_resource_compat import acquire\n'
        'from official.official_memory import admit_startup, memory_guard as host_memory_guard\n'
        'from native_timing_contract import model_weights, runtime, selected_processes, workload')
    start = controller.index('        qualification = qualify(')
    end = controller.index('        for path in (deployment / "inputs/model-metadata")', start)
    controller = controller[:start] + (
        '        _, request_identity = workload(args.fixed_workload)\n'
        '        runtime_identity = runtime(profile, deployment, bindings)\n'
        '        qualification = dict(weights=model_weights(model, args.model_lock))\n'
        '        manifest = Path(__file__).resolve().parents[1] / "CLIENT-MANIFEST.json"\n'
        '        selection = deployment.parent / "engine-selection.json"\n'
        '        save(root / "fixed-input-admission.json", dict(runtime=runtime_identity, workload=request_identity,\n'
        '             client_manifest_sha256=sha(manifest), engine_selection_sha256=sha(selection),\n'
        '             model_lock_sha256=sha(args.model_lock)))\n'
    ) + controller[end:]
    controller = once(controller, '            spec.loader.exec_module(memory_guard)\n',
        '            spec.loader.exec_module(memory_guard)\n'
        '            memory_guard = host_memory_guard(memory_guard)\n')
    controller = once(controller, 'initial = admit_startup_memory(memory_guard, model, weights, root)',
        'initial = admit_startup(memory_guard, model, weights, root)')
    controller = once(controller, '"--cache", str(cache), "--qualified-continuous", str(args.qualified_continuous.resolve()),',
        '"--cache", str(cache), "--fixed-workload", str(args.fixed_workload.resolve()),')
    controller = once(controller, '    parser.add_argument("--qualified-continuous", type=Path, required=True)',
        '    parser.add_argument("--fixed-workload", type=Path, required=True)\n'
        '    parser.add_argument("--model-lock", type=Path, required=True)')
    start = controller.index('                            for line in sample.stdout.splitlines():')
    end = controller.index('                            last_sample = now', start)
    controller = controller[:start] + (
        '                            for pid in selected_processes(sample.stdout, cards()):\n'
        '                                if pid not in known and Path(f"/proc/{pid}").exists():\n'
        '                                    known = {row["pid"] for row in owned.scan()}\n'
        '                                    if pid not in known and Path(f"/proc/{pid}").exists():\n'
        '                                        raise RuntimeError(f"Foreign selected-device process {pid}")\n'
    ) + controller[end:]
    launch = source['scripts/native_timing_prepare.py'].decode()
    launch = once(launch, '    parser.add_argument("--qualified-continuous", type=Path, required=True)',
        '    parser.add_argument("--fixed-workload", type=Path, required=True)')
    launch = once(launch, '    from native_timing_inputs import prepare as prepare_inputs\n',
        '    from native_timing_contract import workload as admit_workload\n'
        '    from model_adapter import configure as select_model\n'
        '    model_adapter = select_model(profile, args.model, command, env, event_dir=out / "mod-events")\n')
    launch = once(launch, '    prepare_inputs(root, args.model, args.qualified_continuous, workload_path)',
        '    fixed, fixed_identity = admit_workload(args.fixed_workload)\n'
        '    input_sources = [args.fixed_workload.resolve(), *(root / "inputs/mmlu-quality018-r2" / name\n'
        '                     for name in ("quality-request-plan.json", "quality-tokenize-preflight.json"))]\n'
        '    save(workload_path, dict(request_token_ids=fixed["request_token_ids"], warmup_cohorts=fixed["warmup_cohorts"],\n'
        '                             formal_questions_used_for_warmup=False, admission=fixed_identity,\n'
        '                             source_files={str(path): sha(path) for path in input_sources}))')
    launch = once(launch, '                workload=str(workload_path), files=files, observer_sources=observer_sources,',
        '                workload=str(workload_path), files=files, observer_sources=observer_sources,\n'
        '                model_adapter=model_adapter, fixed_workload_identity=fixed_identity,')
    launch = once(launch, '"reference_capture_profile.py", "native_plan_values.py",',
                  '"reference_capture_profile.py", "native_timing_contract.py", "native_plan_values.py",')
    return {'scripts/run_native_reference.py': controller.encode(),
            'scripts/native_timing_prepare.py': launch.encode(),
            'scripts/native_timing_contract.py': Path(__file__).with_name('native_timing_contract.py').read_bytes()}


def prepare(cache, record, frozen):
    parent = evaluation_client.verify(record)
    lock = json.loads((ROOT / 'benchmark/one-click-lock.json').read_bytes())
    if (parent['producer_manifest_sha256'] != frozen['tools_manifest_sha256']
            or parent['assessment_manifest_sha256'] != lock['tools']['manifest_sha256']):
        raise ValueError('Installed timing client does not match the producer and assessment locks')
    source = Path(record['path'])
    inputs = {name: (source / name).read_bytes() for name in parent['files']}
    updates = derive(inputs)
    inputs.update(updates)
    manifest = dict(parent, native_timing_parent=record['manifest_sha256'],
                    native_timing_sources={name: evaluation_client.sha(data) for name, data in updates.items()},
                    files={name: evaluation_client.sha(data) for name, data in inputs.items()})
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    digest = evaluation_client.sha(raw)
    folder = Path(cache) / 'native-timing-clients'
    folder.mkdir(parents=True, exist_ok=True)
    root = folder / digest
    selected = dict(path=str(root), manifest_sha256=digest, release=record['release'])
    if not root.exists():
        with tempfile.TemporaryDirectory(prefix='native-timing-', dir=folder) as temporary:
            attempt = Path(temporary) / 'client'
            attempt.mkdir()
            for name, body in inputs.items():
                destination = attempt / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(body)
            (attempt / 'CLIENT-MANIFEST.json').write_bytes(raw)
            try:
                attempt.rename(root)
            except FileExistsError:
                pass
    evaluation_client.verify(selected)
    return selected


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installation', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--fixed-workload', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--devices', required=True)
    parser.add_argument('--physical-devices', required=True)
    parser.add_argument('--lease-timeout', type=float, default=0)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(arguments)
    if not math.isfinite(args.lease_timeout) or args.lease_timeout < 0:
        raise ValueError('Invalid native device lease timeout')
    from deeplink import installation
    from engine.release_selection import bind

    installed, receipt = installation(args.installation)
    record = bind(installed / 'engine-selection.json')
    if (record['release'] != receipt['release'] or receipt['release'] != RELEASE
            or record['version'] != receipt['package_version'] or receipt['package_version'] != VERSION
            or record['sdist']['sha256'] != SDIST_SHA
            or record['frozen']['measured_runtime'] != MEASURED_RUNTIME):
        raise ValueError('Native timing requires the pinned installed stable R10')
    if Path(sys.executable).absolute() != Path(receipt['python']).absolute():
        os.execv(receipt['python'], [receipt['python'], '-B', str(ROOT / 'benchmark/one_click.py'), 'native-timing', *arguments])
    from vllm_hust_engine.cli import verify, verify_tools

    if verify() != record['frozen']:
        raise ValueError('The active producer package differs from the selected stable R10')
    verify_tools(installed / 'tools')
    workload(args.fixed_workload)
    deployment = installed / 'sdk'
    client = prepare(args.cache.resolve(), record['client'], record['frozen'])
    scripts = Path(client['path']) / 'scripts'
    source_identity = dict(engine_selection_sha256=evaluation_client.sha((installed / 'engine-selection.json').read_bytes()),
                           client=client, issuer=RELEASE)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print('Native timing source admission ' + json.dumps(source_identity, sort_keys=True), flush=True)
    command = ['--deployment', str(deployment), '--model', str(args.model.resolve()), '--out', str(args.out.resolve()),
               '--fixed-workload', str(args.fixed_workload.resolve()), '--model-lock', str(ROOT / 'engine/models/qwen3.5-35b-a3b.json'),
               '--devices', args.devices, '--physical-devices', args.physical_devices, '--lease-timeout', str(args.lease_timeout),
               '--reference-profile', 'submission-v2', '--formal-repetitions', '3']
    if args.preflight:
        command.append('--preflight')
    script = scripts / 'run_native_reference.py'
    from official import enable_module_imports

    enable_module_imports()
    sys.path.insert(0, str(scripts))
    sys.argv = [str(script), *command]
    runpy.run_path(str(script), run_name='__main__')
