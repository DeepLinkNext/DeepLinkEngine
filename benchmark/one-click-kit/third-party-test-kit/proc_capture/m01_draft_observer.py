"""Observe actual merged MTP inputs/length planes in a raw-only service.

Device reads deliberately synchronize. These records cannot be a clean timing
numerator. The observer leaves input/state tensors and every returned value
unchanged; it restores the exact original callable when the session closes.
"""

import json
import hashlib
import inspect
from pathlib import Path

from m01_proc_plugin import restore_fields, snapshot_fields


def integers(tensor, *, label, limit):
    if tensor.ndim != 1 or tensor.numel() > limit:
        raise ValueError("Unsupported draft " + label + " geometry")
    result = tensor.detach().cpu().tolist()
    if any(type(value) is not int for value in result):
        raise ValueError("Draft " + label + " must contain integers")
    return result


def position_planes(proposer, capacity):
    """Read exactly the accessor consumed by the merged draft implementation.

    Text-only Qwen35 still allocates its MRoPE buffer when uses_mrope is true.
    Preserve every real plane and reduce to scalar geometry only after proving
    the three text axes equal. Nonidentical multimodal/xdrope axes are refused.
    """
    accessor = getattr(proposer, "_get_positions", None)
    if not callable(accessor):
        raise ValueError("Draft merged position accessor is absent")
    tensor = accessor(capacity)
    if tensor.ndim not in (1, 2) or tensor.numel() > 3 * 4096:
        raise ValueError("Unsupported draft position plane geometry")
    values = tensor.detach().cpu().tolist()
    planes = [values] if tensor.ndim == 1 else values
    if (len(planes) not in (1, 3) or any(len(row) != capacity for row in planes)
            or any(type(value) is not int for row in planes for value in row)):
        raise ValueError("Draft position planes omit integer captured rows")
    if any(row != planes[0] for row in planes[1:]):
        raise ValueError("Draft text-only geometry requires equal MRoPE position axes")
    return planes[0], planes


def length_planes(metadatas, requests):
    result = []
    for step, layers in enumerate(metadatas):
        groups = {}
        for name, meta in layers.items():
            if id(meta) in groups:
                groups[id(meta)]["layers"].append(name)
                continue
            actual = getattr(meta, "_mtp_device_seq_lens", None)
            if actual is None or actual.device.type == "cpu":
                raise ValueError("The actual device-authoritative MTP length plane is required")
            result_row = dict(step=step, layers=[name],
                query_end_envelope=list(meta.actual_seq_lengths_q),
                kv_length_envelope=list(meta.seq_lens_list),
                actual_kv_lengths=integers(actual[:requests], label="KV lengths", limit=36))
            if (len(result_row["actual_kv_lengths"]) != requests
                    or len(result_row["query_end_envelope"]) < requests):
                raise ValueError("Draft length planes omit a live request")
            groups[id(meta)] = result_row
        result.extend(groups.values())
    return result


class DraftObservation:
    def __init__(self, runner, directory, rank, window, current_wave):
        from m01_observer_worker import key

        self.runner = runner
        self.proposer = runner.drafter
        config = getattr(getattr(self.proposer, "vllm_config", None), "speculative_config", None)
        width = config.num_speculative_tokens if config is not None else getattr(self.proposer, "num_speculative_tokens", None)
        if self.proposer is None or self.proposer.method != "mtp" or width != 2:
            raise ValueError("This raw draft observer requires the frozen MTP2 proposer")
        self.original_fields = snapshot_fields(self.proposer, ("_runnable",))
        self.original = self.proposer._runnable
        if (getattr(self.proposer, "needs_extra_input_slots", None) is not False
                or getattr(self.proposer, "pass_hidden_states_to_model", None) is not True
                or getattr(runner, "dcp_manager", None) is not None):
            raise ValueError("Draft target-hidden origins require the unreindexed first-pass source contract")
        position_planes(self.proposer, 1)
        source_methods = {}
        for name in ("set_inputs_first_pass", "_run_merged_draft", "prepare_inputs_padded", "_get_positions", "_set_positions"):
            method = getattr(self.proposer, name, None)
            if method is not None:
                path = Path(inspect.getfile(method)).resolve()
                source_methods[name] = dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    qualified_name=method.__qualname__)
        runnable = self.original
        seen = set()
        while hasattr(runnable, "runnable") and id(runnable) not in seen:
            seen.add(id(runnable))
            runnable = runnable.runnable
        if inspect.isfunction(runnable) or inspect.ismethod(runnable):
            path = Path(inspect.getfile(runnable)).resolve()
            source_methods["actual_merged_runnable"] = dict(file=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), qualified_name=runnable.__qualname__)
        from m01_proc_io import write
        write(directory / f"rank-{rank}-proposer-source.json", dict(window=window, rank=rank,
            target_hidden_input_contract="unreindexed_same_position_first_pass", methods=source_methods,
            needs_extra_input_slots=False, pass_hidden_states_to_model=True,
            context_parallel_reindexing=False, source_runtime_applicability_certified=False))
        self.events = (directory / f"rank-{rank}-drafts.jsonl").open("x", buffering=1024 * 1024)
        self.window, self.current_wave, self.key = window, current_wave, key
        self.directory, self.rank = directory, rank
        self.routes_enabled = hasattr(runner, "_hust_m01_route_capture")
        self.calls = 0
        self.proposer._runnable = self

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, **kwargs):
        batch, capacity = kwargs["batch_size"], kwargs["num_input_tokens"]
        if type(batch) is not int or not 1 <= batch <= 36 or type(capacity) is not int or not 1 <= capacity <= 4096:
            raise ValueError("MTP raw observation is outside the frozen request/token envelope")
        ids = self.runner.input_batch.req_ids[:batch]
        if len(ids) != batch or any(not isinstance(r, str) for r in ids) or len(set(ids)) != batch:
            raise ValueError("Draft live request ownership is not unique")
        actual = kwargs["num_tokens"]
        if type(actual) is not int or not 1 <= actual <= capacity:
            raise ValueError("Draft actual token count is outside its captured capacity")
        positions, raw_position_planes = position_planes(self.proposer, capacity)
        tokens = integers(self.proposer.input_ids[:capacity], label="input tokens", limit=4096)
        indices = integers(kwargs["token_indices_to_sample"][:batch], label="sample indices", limit=36)
        if len(indices) != batch or any(not 0 <= index < len(tokens) for index in indices):
            raise ValueError("A draft sample root is absent from its first-pass rows")
        before = length_planes(kwargs["multi_steps_attn_metadata"], batch)
        if self.routes_enabled:
            from m01_route_capture import begin as begin_routes
            begin_routes(self.runner, "draft")
        result = self.original(**kwargs)
        # This post-call device read also waits for every merged draft output.
        output = result.detach().cpu().tolist()
        width = self.proposer.num_speculative_tokens
        if type(width) is not int or not 1 <= width <= 2 or len(output) < batch or any(len(row) != width for row in output[:batch]):
            raise ValueError("The merged draft result does not match its live step count")
        after = length_planes(kwargs["multi_steps_attn_metadata"], batch)
        routes = None
        if self.routes_enabled:
            from m01_route_capture import finish as finish_routes
            routes = finish_routes(self.runner, "draft", self.directory / f"rank-{self.rank}-draft-{self.calls}-routes.npz")
        record = dict(window=self.window, call=self.calls, wave=self.current_wave(),
            requests=[self.key(r) for r in ids], capacity=capacity,
            draft_width=width,
            target_hidden_input_contract="unreindexed_same_position_first_pass",
            actual_input_tokens=actual,
            first_pass_positions=positions, first_pass_position_planes=raw_position_planes,
            position_source="_get_positions(num_input_tokens)", text_position_axes_equal=True,
            first_pass_input_ids=tokens,
            sample_indices=indices, length_planes_before=before, length_planes_after=after,
            routes=routes,
            draft_token_ids=output[:batch], full_model_certificate=False,
            timing_eligible_for_performance_comparison=False)
        self.events.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.calls += 1
        return result

    def stop(self):
        try:
            if self.proposer._runnable is not self:
                raise ValueError("The draft observer lost ownership of the merged callable")
            restore_fields(self.proposer, self.original_fields)
        finally:
            self.events.close()
        return self.calls
