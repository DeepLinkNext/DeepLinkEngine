"""Record scheduled extents and retained output in the actual engine core."""

from functools import wraps
import json

from native_m01_worker import key, native_rows


def attach(scheduler, path, window):
    output = (path / "scheduler-waves.jsonl").open("x", encoding="utf-8", buffering=1024 * 1024)
    update = scheduler.update_from_output
    retain = scheduler._update_request_with_output
    state = {"frame": None, "wave": 0}

    @wraps(update)
    def observed_update(schedule, model_output, *args, **kwargs):
        if not schedule.total_num_scheduled_tokens:
            return update(schedule, model_output, *args, **kwargs)
        frame = dict(window=window, wave=state["wave"], rows=native_rows(schedule), retained=[])
        if state["frame"] is not None:
            raise ValueError("Unexpected nested scheduler update")
        state["frame"] = frame
        try:
            result = update(schedule, model_output, *args, **kwargs)
        finally:
            state["frame"] = None
        # Closing the capture flushes the remaining userspace buffer.
        output.write(json.dumps(frame, separators=(",", ":")) + "\n")
        state["wave"] += 1
        return result

    @wraps(retain)
    def observed_retain(request, token_ids, *args, **kwargs):
        offered = len(token_ids)
        kept, stopped = retain(request, token_ids, *args, **kwargs)
        if state["frame"] is not None:
            state["frame"]["retained"].append(dict(
                request=key(request.request_id), offered=offered, kept=len(kept), stopped=bool(stopped),
                stale=bool(kwargs.get("is_stale", args[0] if args else False)),
            ))
        return kept, stopped

    scheduler.update_from_output = observed_update
    scheduler._update_request_with_output = observed_retain

    def finish():
        scheduler.update_from_output = update
        scheduler._update_request_with_output = retain
        output.close()
        return state["wave"]

    return finish
