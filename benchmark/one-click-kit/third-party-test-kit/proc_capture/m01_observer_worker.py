"""Collect live Ascend runner schedules and device traces through vLLM worker RPC."""

from functools import wraps
import hashlib
import inspect
from m01_metadata_signature import resolve as resolve_metadata_signature
import json
import os
from pathlib import Path


def key(request_id):
    return hashlib.sha256(request_id.encode()).hexdigest()


def native_rows(schedule):
    starts = {row.req_id: row.num_computed_tokens for row in schedule.scheduled_new_reqs}
    cached = schedule.scheduled_cached_reqs
    starts.update(zip(cached.req_ids, cached.num_computed_tokens, strict=True))
    return [
        dict(request=key(req), start=starts[req], count=count,
             draft=len(schedule.scheduled_spec_decode_tokens.get(req, ())),
             draft_token_ids=list(schedule.scheduled_spec_decode_tokens.get(req, ())))
        for req, count in schedule.num_scheduled_tokens.items()
    ]


def gdn_metadata(metadata):
    groups = {}
    entries = metadata if isinstance(metadata, list) else [metadata]
    for microbatch, layers in enumerate(entries):
        for name, item in layers.items():
            core = getattr(item, "owned", None)
            owned = getattr(core, "host_metadata", None)
            if owned is not None:
                identity = (microbatch, id(core))
                if identity in groups:
                    groups[identity]["layers"].append(name)
                    continue
                host = owned.h
                live, verify = int(core.live), int(core.verify_count)
                groups[identity] = dict(layers=[name], microbatch=microbatch, live=live,
                                        decode=bool(core.decode), verify_count=verify)
                if not core.decode:
                    groups[identity].update(cumulative_lengths=host.cu[:live + 1].tolist(),
                                            verify_indices=host.verify_ids[:verify].tolist(),
                                            initial_state=host.initial[:live].tolist())
            elif hasattr(item, "num_spec_decodes") and hasattr(item, "num_prefills"):
                identity = (microbatch, id(item))
                if identity in groups:
                    groups[identity]["layers"].append(name)
                    continue
                groups[identity] = dict(layers=[name], microbatch=microbatch,
                                        prefill=int(item.num_prefills), decode=int(item.num_decodes),
                                        spec_decode=int(item.num_spec_decodes),
                                        prefill_tokens=int(item.num_prefill_tokens))
    return list(groups.values())


def start(worker, directory, profile, window):
    import torch
    import torch_npu

    rank = torch.distributed.get_rank()
    runner = worker.model_runner
    if hasattr(worker, "_hust_m01_capture"):
        raise ValueError("Worker capture already active")
    path = Path(directory)
    from m01_model_inventory import collect as inventory
    from m01_proc_io import write
    from m01_route_capture import describe
    route_active = hasattr(runner, "_hust_m01_route_capture")
    if os.environ.get("VLLMHUST_M01_HTTP_OUTPUT") and not route_active:
        raise ValueError("HTTP route recording was not prepared before graph capture")

    write(path / f"rank-{rank}-model-inventory.json", dict(window=window, rank=rank,
          target=inventory(runner.model), draft=inventory(runner.drafter.model),
          route_observer=describe(runner) if route_active else None))
    events = (path / f"rank-{rank}-waves.jsonl").open("x", encoding="utf-8", buffering=1024 * 1024)
    execute = runner.execute_model
    metadata = runner._build_attention_metadata
    signature, signature_identity = resolve_metadata_signature(runner)
    with (path / f"rank-{rank}-metadata-abi.json").open("x") as stream:
        json.dump(signature_identity, stream, indent=2)
        stream.write("\n")
    profiler = None
    if profile:
        from persistent_npu_profiler import PersistentNPUProfiler
        profiler = PersistentNPUProfiler(torch_npu, path, rank=rank, window=window)
    state = dict(events=events, execute=execute, metadata=metadata, profiler=profiler,
                 waves=0, path=path, window=window, gdn=None, last_wave=None, draft=None)

    @wraps(metadata)
    def observed_metadata(*args, **kwargs):
        result = metadata(*args, **kwargs)
        if state["gdn"] is not None:
            arguments = signature.bind(*args, **kwargs).arguments
            if not arguments.get("for_cudagraph_capture", False):
                ids = runner.input_batch.req_ids[:arguments["num_reqs"]]
                if not all(isinstance(req, str) for req in ids):
                    raise ValueError("Attention metadata contains an unmapped request")
                cu = runner.query_start_loc.np[:arguments["num_reqs"] + 1].tolist()
                state["gdn"].append(dict(requests=[key(req) for req in ids], groups=gdn_metadata(result[0]),
                                         target_query_cumulative_lengths=cu))
        return result

    @wraps(execute)
    def observed(schedule, *args, **kwargs):
        if not schedule.total_num_scheduled_tokens:
            return execute(schedule, *args, **kwargs)
        wave = state["waves"]
        state["last_wave"] = wave
        rows = native_rows(schedule)
        state["gdn"] = []
        try:
            from m01_route_capture import begin as begin_routes, finish as finish_routes
            if route_active:
                begin_routes(runner, "target")
            if profiler:
                with torch.profiler.record_function(f"hust_m01_wave_{wave}"):
                    result = execute(schedule, *args, **kwargs)
            else:
                result = execute(schedule, *args, **kwargs)
            routes = (finish_routes(runner, "target", path / f"rank-{rank}-wave-{wave}-routes.npz")
                      if route_active else None)
            events.write(json.dumps({"window": window, "wave": wave, "rows": rows, "gdn": state["gdn"], "routes": routes},
                                    separators=(",", ":")) + "\n")
        finally:
            state["gdn"] = None
        state["waves"] += 1
        return result

    try:
        if profiler:
            profiler.start()
        from m01_draft_observer import DraftObservation

        state["draft"] = DraftObservation(runner, path, rank, window, lambda: state["last_wave"])
        runner.execute_model = observed
        runner._build_attention_metadata = observed_metadata
        worker._hust_m01_capture = state
    except BaseException:
        if state["draft"] is not None:
            worker._hust_m01_raw_draft_calls = state["draft"].stop()
        events.close()
        raise
    return rank, window


def stop(worker):
    import torch

    rank = torch.distributed.get_rank()
    state = getattr(worker, "_hust_m01_capture", None)
    if state is None:
        return rank, None, 0
    del worker._hust_m01_capture
    worker.model_runner.execute_model = state["execute"]
    worker.model_runner._build_attention_metadata = state["metadata"]
    try:
        if state["draft"] is not None:
            worker._hust_m01_raw_draft_calls = state["draft"].stop()
        profiler = state["profiler"]
        if profiler:
            torch.npu.synchronize(worker.device)
            profiler.stop()
            # Persist raw bytes only. Daemon workers cannot run Ascend analysis.
            # The parent parses sealed copies after every owned model exits.
    finally:
        state["events"].close()
    return rank, state["window"], state["waves"]


def runtime_files(worker):
    import torch
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    wrapper_type = type(worker).__module__ + "." + type(worker).__name__
    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
    runner = worker.model_runner
    sources = {}
    for item in (type(worker), type(runner), type(runner.model), runner.execute_model, runner.sample_tokens):
        path = Path(inspect.getfile(item)).resolve()
        sources[item.__module__ + "." + item.__qualname__] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"rank": torch.distributed.get_rank(), "sources": sources, "rpc_wrapper": wrapper_type}
