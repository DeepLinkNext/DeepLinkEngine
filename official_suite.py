"""Supervise official services through the existing frozen evaluation clients."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from official_baselines import save
from official_image import digest, environment

ROOT = Path(__file__).resolve().parent
PHASES = ('continuous', 'mmlu', 'frontier', 'hle', 'swe', 'terminal', 'frontier-judge', 'hle-judge', 'context', 'native-timing')


def validate_configuration(config):
    for name in ('protocol_run', 'protocol_deployment', 'protocol_python'):
        if name not in config:
            raise ValueError('Full official suite requires ' + name)
    campaign = Path(config['protocol_run'])
    frozen = json.loads((campaign / 'frozen-execution-plan.json').read_bytes())
    if frozen['datasets'] != ['mmlu', 'hle', 'frontier', 'swe', 'terminal'] or frozen.get('smoke_evaluators'):
        raise ValueError('Full fixed-five source protocol required')
    if Path(frozen['model']).resolve() != Path(config['model']).resolve():
        raise ValueError('Protocol checkpoint differs')
    sdk = Path(config['protocol_deployment'])
    if not Path(config['protocol_python']).is_file() or not (sdk / 'provenance/installed-files.json').is_file():
        raise ValueError('An admitted protocol SDK/interpreter is required')
    if not list((campaign / 'continuous-B0/inputs').glob('continuous-inputs-r1.json')):
        raise ValueError('Original continuous inputs missing')
    for dataset in ('frontier', 'hle', 'swe', 'terminal'):
        suffix = 'trace' if dataset in ('swe', 'terminal') else 'solver'
        spec = json.loads((campaign / 'plans' / (dataset + '-' + suffix + '-B0.json')).read_bytes())
        expected = 200 if dataset == 'hle' else 160 if dataset == 'frontier' else 64 if dataset == 'swe' else 48
        if spec['diagnostic'] or len(spec['case_ids']) != expected:
            raise ValueError('Original full fixed-profile denominator differs: ' + dataset)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_client_environment():
    return {k: v for k, v in os.environ.items()
            if not k.endswith(('_TOKEN', '_API_KEY', '_PASSWORD', '_SECRET'))
            and k not in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'SSH_AUTH_SOCK', 'GIT_ASKPASS')}


def public_arguments(kind):
    args = ['--host', '127.0.0.1', '--port', '18390', '--served-model-name', 'qwen3.5-35b',
            '--tensor-parallel-size', '2', '--dtype', 'bfloat16', '--max-model-len', '65536',
            '--max-num-batched-tokens', '4096', '--max-num-seqs', '24',
            '--kv-cache-memory-bytes', str(20 * 1024**3), '--no-enable-prefix-caching',
            '--speculative-config', json.dumps(dict(method='mtp', num_speculative_tokens=2)),
            '--compilation-config', json.dumps(dict(cudagraph_mode='FULL')),
            '--limit-mm-per-prompt', json.dumps(dict(image=1, video=0)),
            '--worker-extension-cls', 'official_worker.OfficialWorker']
    if kind == 'continuous':
        args += ['--reasoning-parser', 'qwen3']
    return args


def source_bindings(tools, sdk):
    files = {str(p.resolve()): digest(p) for p in (tools / 'scripts').glob('*.py')}
    admitted = json.loads((sdk / 'provenance/installed-files.json').read_bytes())
    for name in ('mods/packed-host/joint_client.py', 'runtime/entrypoints/continuous_client.py',
                 'mods/packed-host/memory_guard.py'):
        path = sdk / name
        if digest(path) != admitted[name]:
            raise ValueError('Original protocol SDK source changed: ' + name)
        files[str(path.resolve())] = admitted[name]
    return files


def run_client(config, plan, phase, leases, *, timeout):
    command = [config['protocol_python'], '-B', str(ROOT / 'official_http.py'), '--config',
               str(plan), '--phase', phase]
    with (plan.parent / (phase + '.log')).open('xb') as log:
        process = subprocess.Popen(command, env=clean_client_environment(), stdout=log,
                                   stderr=subprocess.STDOUT, pass_fds=tuple(v['fd'] for v in leases))
        return process


def monitor(process, owned, guard, initial, out, deadline, *, server=None):
    from run_b1_r4 import check_memory
    next_sample = 0
    while process.poll() is None:
        owned.scan()
        if server is not None and server.poll() is not None:
            raise RuntimeError('Official model service exited during client execution')
        if time.monotonic() > deadline:
            raise TimeoutError('Predeclared official phase deadline reached')
        check_memory(guard, initial, out)
        if time.monotonic() >= next_sample:
            sample = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=15)
            with (out / 'utilization.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(at=time.time(), raw=sample.stdout, returncode=sample.returncode)) + '\n')
            next_sample = time.monotonic() + 5
        time.sleep(2)
    if process.returncode:
        raise RuntimeError('Official ' + str(out.name) + ' client exited ' + str(process.returncode))


def phase(config, image, phase_name, output, tools, sdk, sources, leases, continuous, weights):
    from owned_processes import Owned
    from run_b1_r4 import admit_startup_memory, listener_owned, check_memory
    output.mkdir(exist_ok=False)
    service = output / 'service'
    service.mkdir()
    cache = output / 'cache'
    guard = load(sdk / 'mods/packed-host/memory_guard.py', 'official_memory_guard')
    save(output / 'model-weights.json', weights)
    initial = admit_startup_memory(guard, Path(config['model']), weights, output)
    kind = 'dataset' if phase_name in ('frontier', 'hle', 'swe', 'terminal') else 'context' if phase_name == 'context' else 'continuous' if phase_name == 'continuous' else 'joint'
    installed = json.loads((image / 'installation.json').read_bytes())
    lock = json.loads((image / 'image-lock.json').read_bytes())
    plan = dict(schema='deeplinkengine-official-service-plan/v1', baseline_id=lock['id'], role='B0',
                kind=kind, sdk=str(image), model=config['model'], output=str(service),
                protocol_deployment=str(sdk), client_tools=str(tools), client_sources=sources,
                continuous_inputs=str(continuous), input_sha256=digest(continuous),
                runtime_versions={n: installed['metadata']['versions'][n] for n in ('vllm', 'vllm-ascend', 'torch', 'torch-npu')},
                worker_sources={str(tools / 'scripts' / name): digest(tools / 'scripts' / name)
                                for name in ('compile_boundary.py', 'runtime_identity.py')},
                quality_plan=str(sdk / 'inputs/mmlu-quality018-r2/quality-request-plan.json'),
                scorer=str(sdk / 'scorers/mmlu_pro_pyramidkv.py'),
                serve_arguments=public_arguments(kind), engine_release=lock['id'],
                original_baseline_018_relabelled=False, necessary_work_certified=False)
    if kind == 'dataset':
        suffix = 'trace' if phase_name in ('swe', 'terminal') else 'solver'
        plan['dataset_spec'] = str(Path(config['protocol_run']) / 'plans' / (phase_name + '-' + suffix + '-B0.json'))
    if kind == 'context':
        plan['acceptance_contract'] = str(output.parent / 'context-inputs/contract.json')
        plan['acceptance_suite'] = dict(baseline_run=None)
        merged_inputs = json.loads(continuous.read_bytes())
        from acceptance_client import load as load_context
        _, actual_inputs = load_context(Path(plan['acceptance_contract']))
        merged_inputs['32768'] = actual_inputs['32768']
        save(service / 'context-warmup-inputs.json', merged_inputs)
        plan['continuous_inputs'] = str(service / 'context-warmup-inputs.json')
    save(service / 'plan.json', plan)
    python, env = environment(image, cache, devices='0,1')
    env.update(PYTHONPATH=str(ROOT) + ':' + str(tools / 'scripts') + ':' + env['PYTHONPATH'],
               VLLM_HUST_TEST_OUTPUT=str(service), VLLM_HUST_PHYSICAL_DEVICES=','.join(map(str, config['physical_devices'])),
               VLLMHUST_ENGINE_LOCKS=json.dumps(leases), VLLM_SERVER_DEV_MODE='1')
    from engine_entry import ipc_directory, temporary_directory
    env['VLLM_RPC_BASE_PATH'] = str(ipc_directory(service))
    env['TMPDIR'] = str(temporary_directory(service))
    logs = output / 'cann-logs'
    logs.mkdir()
    env['ASCEND_PROCESS_LOG_PATH'] = str(logs)
    owned = Owned(service)
    receipt = dict(schema='deeplinkengine-official-arm/v1', baseline_id=lock['id'], phase=phase_name,
                   status='starting', started=time.time(), model_started=False, client_started=False,
                   completed=False, input_plan_sha256=digest(Path(config['protocol_run']) / 'frozen-execution-plan.json'))
    save(output / 'receipt.json', receipt)
    processes = []
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 18390))
        command = [str(python), '-S', str(ROOT / 'official_serve.py'), str(service / 'plan.json')]
        with (output / 'service.log').open('xb') as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      pass_fds=tuple(v['fd'] for v in leases), start_new_session=True)
        processes.append(server)
        owned.add(server.pid)
        receipt.update(model_started=True, server_pid=server.pid, command=command)
        save(output / 'receipt.json', receipt)
        deadline = time.monotonic() + 2400
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            owned.scan()
            if server.poll() is not None:
                raise RuntimeError('Official service failed before readiness; inspect original service.log')
            check_memory(guard, initial, output)
            try:
                if listener_owned(owned, 18390):
                    with opener.open('http://127.0.0.1:18390/v1/models', timeout=3) as response:
                        models = json.load(response)
                    if any(row.get('id') == 'qwen3.5-35b' for row in models.get('data', [])):
                        save(service / 'ready-models.json', models)
                        break
            except (OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Official service startup exceeded 2400 seconds')
            time.sleep(3)
        from checkpoint_cache import release_loaded_checkpoint_cache
        release_loaded_checkpoint_cache(Path(config['model']), weights, output / 'checkpoint-cache-after-load.json', guard.sample)
        for step, seconds in [('warmup', 2400), ('client', 7200)]:
            receipt.update(status=step, client_started=step == 'client')
            save(output / 'receipt.json', receipt)
            process = run_client(config, service / 'plan.json', step, leases, timeout=seconds)
            processes.append(process)
            owned.add(process.pid)
            monitor(process, owned, guard, initial, output, time.monotonic() + seconds, server=server)
        from serving_warmup import verify_after
        # The unchanged named worker RPC verifies the current live boundary.
        verify_after(service)
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
    return receipt


def run(config, image, output):
    validate_configuration(config)
    output.mkdir(parents=True, exist_ok=False)
    sdk = Path(config['protocol_deployment']).resolve()
    tools = sdk.parent / 'tools'
    sys.path.insert(0, str(tools / 'scripts'))
    sources = source_bindings(tools, sdk)
    from resource_guard import acquire, release, verify_mapping
    physical = ','.join(map(str, config['physical_devices']))
    os.environ['VLLM_HUST_PHYSICAL_DEVICES'] = physical
    mapping = verify_mapping('0,1')
    sources.update({str(ROOT / name): digest(ROOT / name) for name in
                    ('official_suite.py', 'official_http.py', 'official_serve.py', 'official_worker.py',
                     'official_judge.py', 'official_native.py', 'official_native_entry.py', 'official_report.py')})
    frozen = dict(schema='deeplinkengine-official-full-suite/v1', baseline_image=json.loads((image / 'image-lock.json').read_bytes()),
                  status='preparing', prepared_before_model_start=True, started=time.time(),
                  sources=sources, physical_mapping=mapping,
                  stages=[dict(name=name, status='planned') for name in PHASES],
                  fixed_five_denominators=dict(mmlu=200, hle=200, frontier=160, swe=64, terminal=48),
                  no_retries_or_eager_fallback=True, original_results_modified=False,
                  qualification=dict(independent_third_party=False, complete=False))
    save(output / 'campaign.json', frozen)
    continuous = Path(config['protocol_run']) / 'continuous-B0/inputs/continuous-inputs-r1.json'
    frozen['continuous_input_sha256'] = digest(continuous)
    leases = []
    try:
        weights = {}
        for path in sorted(Path(config['model']).glob('*.safetensors')):
            before = path.stat()
            value = digest(path)
            after = path.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('Checkpoint changed during hashing')
            weights[path.name] = dict(bytes=after.st_size, sha256=value, mtime_ns=after.st_mtime_ns, inode=after.st_ino)
        if len(weights) != 14 or sum(v['bytes'] for v in weights.values()) != 71903878016:
            raise ValueError('Fixed BF16 14-shard Qwen3.5-35B-A3B checkpoint required')
        save(output / 'model-weights.json', weights)
        if config.get('context_contract'):
            raise ValueError('Explicit external context contracts require a dedicated identity adapter')
        preparation = output / 'input-preparation'
        preparation.mkdir()
        plan = dict(client_tools=str(tools), client_sources=sources, model=config['model'],
                    protocol_deployment=str(sdk), output=str(preparation),
                    data_cache=str(Path(config['protocol_run']).parents[1] / 'cache'),
                    context_output=str(output / 'context-inputs'))
        save(preparation / 'plan.json', plan)
        with (preparation / 'prepare-context.log').open('xb') as stream:
            result = subprocess.run([config['protocol_python'], '-B', str(ROOT / 'official_http.py'),
                                     '--config', str(preparation / 'plan.json'), '--phase', 'prepare-context'],
                                    env=clean_client_environment(), stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
        if result.returncode:
            raise RuntimeError('Original real-history context input preparation failed')
        leases = acquire(output, 1800)
        frozen['status'] = 'running'
        save(output / 'campaign.json', frozen)
        for row in frozen['stages']:
            row.update(status='running', started=time.time())
            save(output / 'campaign.json', frozen)
            try:
                if row['name'] == 'native-timing':
                    from official_native import run as native_run
                    result = native_run(config, image, output / row['name'], tools, sdk, sources, leases, output / 'continuous')
                elif row['name'].endswith('-judge'):
                    from official_judge import run as judge_run
                    result = judge_run(config, row['name'].split('-')[0], output / row['name'], tools, sdk, sources, leases, weights)
                else:
                    result = phase(config, image, row['name'], output / row['name'], tools, sdk, sources, leases, continuous, weights)
                row.update(status='completed', receipt=result, ended=time.time())
            except BaseException as error:
                row.update(status='failed', error=type(error).__name__ + ': ' + str(error), ended=time.time())
                for remaining in frozen['stages']:
                    if remaining['status'] == 'planned':
                        remaining.update(status='not_attempted', reason='Preceding workflow stage failed; no automatic retry')
                raise
            finally:
                save(output / 'campaign.json', frozen)
        frozen['status'] = 'measurements_collected'
        return dict(status='incomplete', reason='Native F counting and fixed-judge verification must be qualified before full completion', campaign=str(output / 'campaign.json'))
    except BaseException as error:
        frozen.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        if leases:
            release(leases)
            os.environ.pop('VLLMHUST_ENGINE_LOCKS', None)
        frozen['ended'] = time.time()
        save(output / 'campaign.json', frozen)
        from official_report import report
        report(output)
