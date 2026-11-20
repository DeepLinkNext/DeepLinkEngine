"""Recover cache headroom only at the owned healthy idle service boundary."""
import json
from pathlib import Path


def reclaim_idle_headroom(sample, check, required, out):
    from checkpoint_cache import allocate_private_chunk, reclaim_plan
    before = check()
    if not isinstance(before, dict):
        raise ValueError('Validated memory sample is required before idle cache recovery')
    plan = reclaim_plan(before, required)
    plan['phase'] = 'healthy-idle-before-warmup'
    plan['reason'] = 'Controller-owned pages bounded by clean unmapped cache; original model working-set and OOM gates remain active'
    result = dict(schema='healthy-idle-cache-recovery/v1', before=before, plan=plan,
        required_raw_bytes=required, status='declined', allocated_bytes=0,
        healthy_idle_before_warmup=True, requires_healthy_idle_service=True,
        warmup_started=False, formal_client_started=False,
        global_drop_caches=False, memory_limit_changed=False,
        temporary_memory_released=True)
    buffers = []
    failure = None
    try:
        if plan['eligible']:
            result.update(status='completed', temporary_memory_released=False)
            for _ in range(plan['temporary_bytes'] // (64 * 2**20)):
                buffers.append(allocate_private_chunk(64 * 2**20))
                result['allocated_bytes'] += 64 * 2**20
                try:
                    check()
                except RuntimeError as error:
                    # Release ONLY the controller's optional buffers when the
                    # unchanged original working-set gate asks for its margin.
                    # Every OOM event and every different failure remain fatal.
                    if str(error) != '4GiB conservative working-set memory reserve reached':
                        raise
                    observed = sample()
                    if any(observed.get('events',{}).get(n,0) > before.get('events',{}).get(n,0)
                           for n in ('oom','oom_kill','oom_group_kill')):
                        raise RuntimeError('New cgroup OOM event during idle recovery') from error
                    result.update(status='stopped-at-original-working-set-reserve', guard_boundary=observed)
                    break
    except BaseException as error:
        failure = error
        result.update(status='failed', error=type(error).__name__+': '+str(error))
    finally:
        for buffer in buffers:
            buffer.close()
        result.update(temporary_memory_released=True, after=sample())
        Path(out).write_text(json.dumps(result,indent=2)+'\n')
    # This check runs only AFTER every temporary page has been released. A new
    # OOM, genuine model pressure or any failed original guard cannot be hidden.
    check()
    if failure is not None:
        raise failure
    return result
