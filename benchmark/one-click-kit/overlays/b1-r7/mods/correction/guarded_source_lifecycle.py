"""Rebind only an unused native source guard after a one-time input-batch replacement."""


def state(worker, guard_type, owned_type, refresh, guard_run):
    runner = worker.model_runner
    owned = worker._owned_optimistic_operand
    guard = runner._m01_progress_source_snapshots
    batch = runner.input_batch
    flags = dict(owned_native_type=type(owned) is owned_type,
        owned_runner_is_current=owned.runner is runner,
        owned_refresh_is_declared=owned.refresh.__func__ is refresh,
        guard_native_type=type(guard) is guard_type,
        guard_run_is_native=guard.run.__func__ is guard_run,
        guard_idle=not guard.active, guard_not_failed=not guard.failed,
        owned_not_failed=not owned.failed,
        guard_batch_is_current=guard.batch is batch,
        live_source_is_guard_original=batch.num_computed_tokens_cpu_tensor is guard.original,
        stored_batch_source_unchanged=guard.batch.num_computed_tokens_cpu_tensor is guard.original,
        destination_independent=owned.buffer.data_ptr() != runner.num_computed_tokens.data_ptr())
    return dict(flags=flags, owned_calls=owned.calls,
        preparation_calls=runner._m01_progress_snapshot_calls,
        source_events_used=list(guard.used), preparation_stream=getattr(owned, '_hust_operand_copy_stream', None),
        batch_identity=id(batch), guard_batch_identity=id(guard.batch),
        live_source_pointer=batch.num_computed_tokens_cpu_tensor.data_ptr(),
        guard_original_pointer=guard.original.data_ptr(), guard_identity=id(guard),
        destination_pointer=owned.buffer.data_ptr())


def initialize(worker, torch, guard_type, owned_type, refresh, guard_run):
    before = state(worker, guard_type, owned_type, refresh, guard_run)
    if all(before['flags'].values()):
        return dict(before=before, after=before, rebound=False, native_arithmetic_modified=False)
    allowed = {'guard_batch_is_current', 'live_source_is_guard_original'}
    if any(not value for key, value in before['flags'].items() if key not in allowed):
        raise ValueError('A foreign, active, failed or mutated source owner cannot be rebound')
    if before['flags']['guard_batch_is_current']:
        raise ValueError('Same batch with changed source storage is not this one-time KV batch transition')
    if (before['owned_calls'] != 0 or before['preparation_calls'] != 0
            or before['preparation_stream'] is not None or any(before['source_events_used'])):
        raise ValueError('Rebinding is allowed only before the first native preparation or copy')
    runner = worker.model_runner
    prior = runner._m01_progress_source_snapshots
    # Native constructor rechecks CPU/pinned/integer/NumPy alias before allocating
    # its original two source buffers and events. No arithmetic/copy method changes.
    replacement = guard_type(torch, runner.input_batch)
    runner._m01_progress_source_snapshots = replacement
    worker._hust_retired_unused_source_guard = prior
    runner._hust_retired_unused_source_guard = prior
    after = state(worker, guard_type, owned_type, refresh, guard_run)
    if not all(after['flags'].values()) or after['destination_pointer'] != before['destination_pointer']:
        raise ValueError('The original owned destination or live native source binding was not preserved')
    return dict(before=before, after=after, rebound=True, native_arithmetic_modified=False,
                retained_unused_guard_identity=id(prior), source_buffers=2, source_events=2,
                scope='Common initialization correction in both copy modes; never rebind after traffic starts')
