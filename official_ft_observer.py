"""Official-only producer observation; never admitted to a native T clock."""
from functools import wraps
import json
from pathlib import Path


def start(worker, directory, window):
    from m01_observer_worker import native_rows
    from m01_proc_plugin import restore_fields, snapshot_fields
    from official_ft_draft import DraftObservation
    runner = worker.model_runner
    if hasattr(worker, '_official_ft_capture'):
        raise ValueError('Official companion already active')
    path = Path(directory)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise ValueError('Companion requires a new absolute output directory')
    fields = snapshot_fields(runner, ('execute_model',))
    execute = runner.execute_model
    events = (path / f'rank-{worker.rank}-waves.jsonl').open('x', encoding='utf-8')
    state = dict(window=window, waves=0, last_wave=None, draft=None, fields=fields, events=events)

    @wraps(execute)
    def observed(schedule, *args, **kwargs):
        if not schedule.total_num_scheduled_tokens:
            return execute(schedule, *args, **kwargs)
        wave = state['waves']
        state['last_wave'] = wave
        rows = native_rows(schedule)
        result = execute(schedule, *args, **kwargs)
        events.write(json.dumps(dict(window=window, wave=wave, rows=rows,
            scheduled_draft_width=getattr(schedule, 'num_spec_tokens_to_schedule', None)),
            separators=(',', ':')) + '\n')
        state['waves'] += 1
        return result

    try:
        state['draft'] = DraftObservation(runner, path, int(worker.rank), window, lambda: state['last_wave'])
        runner.execute_model = observed
        worker._official_ft_capture = state
    except BaseException:
        if state['draft'] is not None:
            state['draft'].stop()
        restore_fields(runner, fields)
        events.close()
        raise
    return dict(rank=int(worker.rank), window=window, active=True, timing_eligible=False)


def stop(worker, window):
    from m01_proc_plugin import restore_fields
    state = getattr(worker, '_official_ft_capture', None)
    if state is None or state['window'] != window:
        raise ValueError('Cannot close an absent or foreign official companion')
    result = dict(rank=int(worker.rank), window=window, completed=False, timing_eligible=False)
    try:
        calls = state['draft'].stop()
        result.update(draft_calls=calls, waves=state['waves'], completed=True)
    finally:
        restore_fields(worker.model_runner, state['fields'])
        state['events'].close()
        del worker._official_ft_capture
    return result
