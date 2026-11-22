"""Score all official solver versions with the same admitted reference judge."""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from official_baselines import save
from official_image import digest
from official_judge_runtime import judge_command
from official_memory import admit_startup, memory_guard
from official_suite import ROOT, clean_client_environment, load, monitor, run_client, service_cache


def run(config, dataset, output, tools, sdk, sources, leases, weights):
    from owned_processes import Owned
    from run_b1_r4 import check_memory, listener_owned
    output.mkdir(exist_ok=False)
    client = output / 'service'
    client.mkdir()
    runtime = output / 'judge-runtime'
    installation = Path(config.get('judge_installation', str(sdk.parent))).resolve()
    installed = json.loads((installation / 'installation.json').read_bytes())
    if Path(installed['deployment']).resolve() != sdk or installed['release'] != json.loads((sdk / 'deployment.json').read_bytes())['release']:
        raise ValueError('Fixed judge must be the admitted protocol SDK; never latest auto-selection')
    original_spec = Path(config['protocol_run']) / 'plans' / (dataset + '-judge-B0.json')
    spec = json.loads(original_spec.read_bytes())
    spec['responses'] = str(output.parent / dataset / 'service/evaluation')
    save(client / 'judge-spec.json', spec)
    plan = dict(kind='dataset', output=str(client), model=config['model'],
                protocol_deployment=str(sdk), client_tools=str(tools), client_sources=sources,
                dataset_spec=str(client / 'judge-spec.json'), fixed_judge_identity=installed,
                original_spec_sha256=digest(original_spec), solver_version=str(output.parent.name),
                performance_measurement=False)
    save(client / 'plan.json', plan)
    guard = memory_guard(load(sdk / 'mods/packed-host/memory_guard.py', 'official_fixed_judge_memory_guard'))
    initial = admit_startup(guard, Path(config['model']), weights, output)
    env = clean_client_environment()
    env.update(VLLMHUST_ENGINE_LOCKS=json.dumps(leases),
               VLLM_HUST_PHYSICAL_DEVICES=','.join(map(str, config['physical_devices'])))
    command = [config['protocol_python'], '-B', str(ROOT / 'deeplink.py'), 'serve',
               '--installation', str(installation), '--model', config['model'],
               '--physical-devices', ','.join(map(str, config['physical_devices'])), '--devices', '0,1',
               '--host', '127.0.0.1', '--port', '18390', '--served-model-name', 'qwen3.5-35b',
               '--cache', str(service_cache(config, output)), '--work-dir', str(runtime), '--lease-timeout', '0']
    command = judge_command(command, config, output)
    owned = Owned(runtime)
    processes = []
    receipt = dict(schema='deeplinkengine-official-fixed-judge/v1', dataset=dataset,
                   status='starting', fixed_judge=installed['release'], started=time.time(), completed=False,
                   client_started=False, solver_performance_includes_judge=False,
                   original_judge_spec_sha256=digest(original_spec))
    try:
        from official_endpoint import available
        available()
        with (output / 'judge-service.log').open('xb') as stream:
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       pass_fds=tuple(v['fd'] for v in leases), start_new_session=True)
        processes.append(process)
        owned.add(process.pid)
        deadline = time.monotonic() + 2400
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            owned.scan()
            if process.poll() is not None:
                raise RuntimeError('Fixed reference judge failed before readiness')
            check_memory(guard, initial, output)
            try:
                if listener_owned(owned, 18390):
                    with opener.open('http://127.0.0.1:18390/v1/models', timeout=3) as response:
                        data = json.load(response)
                    if any(row.get('id') == 'qwen3.5-35b' for row in data.get('data', [])):
                        save(output / 'ready-models.json', data)
                        break
            except (OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Fixed judge startup exceeded 2400 seconds')
            time.sleep(3)
        # The separately installed common judge is not a solver performance arm.
        # Existing clients retain every planned answer and every invalid grade.
        receipt['client_started'] = True
        save(output / 'receipt.json', receipt)
        client_process = run_client(config, client / 'plan.json', 'client', leases, timeout=14400)
        processes.append(client_process)
        owned.add(client_process.pid)
        monitor(client_process, owned, guard, initial, output, time.monotonic() + 14400, server=process)
        summary = run_client(config, client / 'plan.json', 'summarize', leases, timeout=600)
        processes.append(summary)
        owned.add(summary.pid)
        monitor(summary, owned, guard, initial, output, time.monotonic() + 600, server=process)
        receipt.update(status='completed', completed=True)
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        try:
            receipt['cleanup_actions'] = owned.stop()
            receipt['quiescent'] = not owned.scan()
            for process in processes:
                process.wait(timeout=5)
        finally:
            receipt['ended'] = time.time()
            save(output / 'receipt.json', receipt)
            from official_report import seal
            seal(output)
    return receipt
