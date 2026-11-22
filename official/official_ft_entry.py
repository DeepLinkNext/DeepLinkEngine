"""One separate producer cross-check; its clock is never a native T sample."""
import json
from pathlib import Path
import sys
from uuid import uuid4


def run(config, args):
    from dataset_contract import save
    from baseline_native_entry import reset_prefix_cache
    from native_public_client import ranks, save_result, timed_public_requests
    from native_output_geometry import geometry
    from native_state_cache import wait_checkpoint_admission
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.llm_engine import LLMEngine
    output = Path(config['output'])
    workload = json.loads(Path(config['workload']).read_bytes())
    prompts = workload['request_token_ids']
    if len(prompts) != 200 or sum(map(len, prompts)) != 282265:
        raise ValueError('Companion requires the original fixed200 prompt tokens')
    sampling = SamplingParams(**config['sampling'], output_kind=RequestOutputKind.DELTA)
    receipt = dict(schema='deeplinkengine-official-arithmetic-companion/v1', completed=False,
                   timing_eligible=False, native_counts_or_times_replaced=False)
    save(output / 'companion-receipt.json', receipt)
    engine = LLMEngine.from_engine_args(args)
    workers_active = core_active = False
    primary_error = None
    window = uuid4().hex
    raw = output / 'raw-work'
    raw.mkdir()
    try:
        save(output / 'model-ready.json', dict(model_loaded=True, timing_eligible=False))
        wait_checkpoint_admission(Path(config['admission_root']) / 'checkpoint-cache-after-load.json')
        identities = ranks(engine.collective_rpc('hust_ft_identity', timeout=120))
        save(output / 'resolved-worker-identities.json', identities)
        save(output / 'resolved-core.json', engine.engine_core.call_utility('hust_ft_core_identity'))
        inventories = ranks(engine.collective_rpc('hust_native_observer_audit', timeout=1800))
        save(output / 'loaded-inventory-bindings.json', inventories)
        reset_prefix_cache(engine, output / 'companion-reset.json')
        begin = engine.engine_core.call_utility('hust_m01_core_begin', str(raw), window)
        core_active = True
        save(raw / 'core-begin.json', begin)
        workers_active = True
        workers = ranks(engine.collective_rpc('hust_ft_begin', timeout=120, args=(str(raw), window)))
        save(raw / 'workers-begin.json', workers)
        label = 'untimed-producer-' + uuid4().hex
        ids = [f'{label}-{n:03d}' for n in range(200)]
        inputs = [dict(prompt_token_ids=tokens, cache_salt=identity)
                  for tokens, identity in zip(prompts, ids, strict=True)]
        observation = timed_public_requests(engine, inputs, sampling, ids)
        observation.update(timing_eligible=False, purpose='independent-producer-correspondence')
        save_result(output / 'public', observation, list(map(len, prompts)), config['max_tokens'])
        save(output / 'public/public-output-geometry.json',
             geometry(observation, prompts, maximum=config['max_tokens']))
        stopped = ranks(engine.collective_rpc('hust_ft_end', timeout=120, args=(window,)))
        workers_active = False
        save(raw / 'workers-end.json', stopped)
        core = engine.engine_core.call_utility('hust_m01_core_end', window)
        core_active = False
        save(raw / 'core-end.json', core)
        save(output / 'loaded-inventory-bindings-after.json',
             ranks(engine.collective_rpc('hust_native_observer_audit', timeout=1800)))
        if ({r['rank'] for r in stopped} != {0, 1} or len(stopped) != 2
                or any(r['completed'] is not True or r['window'] != window for r in stopped)
                or core['completed'] is not True or core['window'] != window):
            raise ValueError('Independent companion did not close every owned observer')
        receipt.update(completed=True, window=window, requests=200,
                       actual_producer_observation=True, observers_detached=True)
    except BaseException as error:
        primary_error = error
        receipt.update(completed=False, error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        cleanup_errors = []
        actions = (
            ('workers', workers_active,
             lambda: ranks(engine.collective_rpc('hust_ft_end', timeout=120, args=(window,)))),
            ('core', core_active,
             lambda: engine.engine_core.call_utility('hust_m01_core_end', window)),
        )
        for name, active, close in actions:
            if not active:
                continue
            try:
                result = close()
            except BaseException as error:
                result = dict(completed=False, action=name,
                              error=type(error).__name__ + ': ' + str(error))
                cleanup_errors.append(result)
            try:
                save(raw / (name + '-failure-cleanup.json'), result)
            except BaseException as error:
                cleanup_errors.append(dict(action=name + '-receipt',
                    error=type(error).__name__ + ': ' + str(error)))
        if cleanup_errors:
            receipt.update(completed=False, cleanup_errors=cleanup_errors)
        try:
            engine.engine_core.shutdown()
        except BaseException as error:
            receipt.update(completed=False, shutdown_error=type(error).__name__ + ': ' + str(error))
            if primary_error is None:
                raise
        finally:
            try:
                save(output / 'companion-receipt.json', receipt)
            except BaseException as error:
                if primary_error is None:
                    raise
                print('COMPANION_RECEIPT_WRITE_FAILED ' + type(error).__name__ + ': ' + str(error), file=sys.stderr)
        if cleanup_errors and primary_error is None:
            raise ValueError('Independent companion observer cleanup failed')
    return receipt


def main():
    from official_native_entry import prepare
    config = json.loads(Path(sys.argv[1]).read_bytes())
    args = prepare(config)
    run(config, args)


if __name__ == '__main__':
    main()
