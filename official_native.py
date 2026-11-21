"""Supervise official native timing without reusing 0.18 identities or clocks."""
import json
from pathlib import Path
import subprocess
import shutil
import time

from official_baselines import save
from official_image import digest, environment
from official_suite import ROOT, clean_client_environment, load, public_arguments, service_cache, kernel_cache, runtime_environment


def run(config, image, output, tools, sdk, sources, leases, continuous, *, native_source=None):
    from owned_processes import Owned
    from run_b1_r4 import admit_startup_memory, check_memory
    from checkpoint_cache import release_loaded_checkpoint_cache
    from reference_capture_profile import profile
    output.mkdir(exist_ok=False)
    preparation = output / 'prepare-inputs'
    preparation.mkdir()
    workload = output / 'token-workload.json'
    prepare = dict(client_tools=str(tools), client_sources=sources, model=config['model'],
                   protocol_deployment=str(sdk), output=str(preparation),
                   qualified_continuous=str(continuous), native_workload=str(workload))
    if native_source is None:
        save(preparation / 'plan.json', prepare)
        with (preparation / 'prepare-native.log').open('xb') as log:
            result = subprocess.run([config['protocol_python'], '-B', str(ROOT / 'official_http.py'),
                                     '--config', str(preparation / 'plan.json'), '--phase', 'prepare-native'],
                                    env=clean_client_environment(), stdout=log, stderr=subprocess.STDOUT,
                                    timeout=600, pass_fds=tuple(v['fd'] for v in leases))
        if result.returncode:
            raise RuntimeError('Original MMLU200 native token preparation failed')
    else:
        # The caller verifies the closed native seal first. Do not prepare
        # another prompt population or repeat any native timing repetition.
        shutil.copyfile(Path(native_source) / 'token-workload.json', workload)
        save(preparation / 'retained-native-input.json', dict(source=str(native_source),
             workload_sha256=digest(workload), native_timing_repeated=False))
    capture = profile('submission-v2')
    identity = json.loads((image / 'installation.json').read_bytes())
    lock = json.loads((image / 'image-lock.json').read_bytes())
    plan = dict(schema='deeplinkengine-official-native-timing/v1', baseline_id=lock['id'], role='B0',
                client_tools=str(tools), sdk=str(image), model=config['model'],
                output=str(output / 'client'), admission_root=str(output), workload=str(workload),
                files=dict(sources, **{str(workload): digest(workload)}),
                expected_versions={n: identity['metadata']['versions'][n] for n in ('vllm', 'vllm-ascend', 'torch', 'torch-npu')},
                serve_arguments=['serve', config['model'], *public_arguments('joint')],
                reference_capture_profile='submission-v2', formal_repetitions=capture['formal_repetitions'],
                max_tokens=capture['max_tokens'], sampling=dict(max_tokens=capture['max_tokens'], temperature=0,
                    seed=0, ignore_eos=capture['ignore_eos'], top_p=1.0, top_k=-1, presence_penalty=0.0, detokenize=False),
                necessary_work_certified=False, efficiency_percent=None,
                runtime_identity=__import__('official_runtime_identity').runtime_identity(image))
    save(output / 'native-plan.json', plan)
    if native_source is not None:
        plan['purpose'] = 'arithmetic-companion'
        at = plan['serve_arguments'].index('--worker-extension-cls')
        plan['serve_arguments'][at + 1] = 'official_ft_worker.OfficialCompanionWorker'
        save(output / 'native-plan.json', plan)
    weights = json.loads((output.parent / 'model-weights.json').read_bytes())
    guard = load(sdk / 'mods/packed-host/memory_guard.py', 'official_native_memory_guard')
    initial = admit_startup_memory(guard, Path(config['model']), weights, output)
    python, env = environment(image, service_cache(config, output), devices='0,1')
    env.update(TRITON_CACHE_DIR=str(kernel_cache(config, image)), VLLMHUST_OFFICIAL_SDK=str(image))
    inventory = tools / 'third-party-test-kit/proc_capture/m01_model_inventory.py'
    env['VLLMHUST_OFFICIAL_INVENTORY'] = json.dumps(dict(path=str(inventory), sha256=sources[str(inventory)]))
    env.update(PYTHONPATH=str(ROOT) + ':' + str(tools / 'scripts') + ':' + env['PYTHONPATH'],
               VLLM_HUST_TEST_OUTPUT=str(output), VLLM_ENABLE_V1_MULTIPROCESSING='1',
               VLLM_DISABLE_REQUEST_ID_RANDOMIZATION='1')
    if native_source is not None:
        env['PYTHONPATH'] = str(tools / 'third-party-test-kit/proc_capture') + ':' + env['PYTHONPATH']
        env.update(VLLMHUST_M01_PROC_CAPTURE='1', DEEPLINKENGINE_OFFICIAL_FT_COMPANION='1')
    from engine_entry import ipc_directory, temporary_directory
    env['VLLM_RPC_BASE_PATH'] = str(ipc_directory(output))
    env['TMPDIR'] = str(temporary_directory(output))
    logs = output / 'cann-logs'
    logs.mkdir()
    env['ASCEND_PROCESS_LOG_PATH'] = str(logs)
    save(output / 'runtime-environment.json', runtime_environment(env))
    owned = Owned(output)
    receipt = dict(schema='deeplinkengine-official-native-arm/v1', baseline_id=lock['id'], started=time.time(),
                   status='starting', completed=False, model_ready=False, necessary_work_certified=False, efficiency_percent=None)
    save(output / 'receipt.json', receipt)
    process = None
    try:
        with (output / 'native.log').open('xb') as stream:
            entry = 'official_native_entry.py' if native_source is None else 'official_ft_entry.py'
            process = subprocess.Popen([str(python), '-S', '-B', str(ROOT / entry), str(output / 'native-plan.json')],
                                       env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       pass_fds=tuple(v['fd'] for v in leases), start_new_session=True)
        owned.add(process.pid)
        deadline, cache_released, next_sample = time.monotonic() + 14400, False, 0
        while process.poll() is None:
            owned.scan()
            if not cache_released:
                from startup_failure import inspect as inspect_startup_failure
                fatal = inspect_startup_failure(output / 'native.log')
                if fatal:
                    save(output / 'startup-failure.json', fatal)
                    raise RuntimeError('Official native startup failed: ' + (fatal['root_error'] or fatal['fatal']))
            check_memory(guard, initial, output)
            if not cache_released and (output / 'client/model-ready.json').is_file():
                release_loaded_checkpoint_cache(Path(config['model']), weights, output / 'checkpoint-cache-after-load.json', guard.sample)
                cache_released = True
                receipt['status'] = 'native-warmup-and-timing' if native_source is None else 'arithmetic-companion-running'
                receipt['model_ready'] = True
                save(output / 'receipt.json', receipt)
            if time.monotonic() >= next_sample:
                sample = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=15)
                with (output / 'utilization.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(at=time.time(), monotonic=time.monotonic(), raw=sample.stdout, returncode=sample.returncode)) + '\n')
                next_sample = time.monotonic() + 5
            if time.monotonic() > deadline:
                raise TimeoutError('Official native timing phase exceeded 14400 seconds')
            time.sleep(2)
        if process.returncode:
            raise RuntimeError('Official native timing failed; preserve native.log')
        if native_source is None:
            times = json.loads((output / 'client/timings.json').read_bytes())
            if len(times['repetitions']) != capture['formal_repetitions'] or any(v['T_seconds'] <= 0 for v in times['repetitions']):
                raise ValueError('Missing complete original native timing repetitions')
            receipt.update(status='completed', completed=True, T_repetitions=times['repetitions'])
        else:
            companion = json.loads((output / 'client/companion-receipt.json').read_bytes())
            if companion['completed'] is not True or companion['timing_eligible'] is not False:
                raise ValueError('Missing complete independent arithmetic companion')
            receipt.update(status='completed', completed=True, purpose='arithmetic-companion',
                           native_timing_repeated=False, companion=companion)
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        try:
            receipt['cleanup_actions'] = owned.stop()
            receipt['quiescent'] = not owned.scan()
            if process is not None:
                process.wait(timeout=5)
        finally:
            receipt['ended'] = time.time()
            save(output / 'receipt.json', receipt)
            from official_report import seal
            seal(output)
    return receipt
