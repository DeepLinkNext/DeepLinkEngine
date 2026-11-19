"""Compile serving-only metadata kernels before sealing the worker JIT cache.

Use the real launch helpers and banked tensor addresses (including alignment),
not an independently maintained copy of their signatures. Triton's warmup mode
compiles/cache-loads but never launches: captured State and metadata stay intact.
"""
from contextlib import contextmanager
import copy
from functools import wraps


def row_counts(capacity):
    """Represent every next-power-of-two bucket, including a ragged final one."""
    if type(capacity) is not int or capacity < 1:
        raise ValueError("Serving warmup requires positive integer request capacity")
    rows = 1
    while rows < capacity:
        yield rows
        rows *= 2
    yield capacity


@contextmanager
def compile_only(*kernels):
    """Worker-startup-only bridge from existing [grid] calls to JIT warmup."""
    originals = []
    try:
        for kernel in kernels:
            old = kernel.run
            local = kernel.__dict__.get('run')
            originals.append((kernel, local))

            @wraps(old)
            def run(*args, _original=old, **kwargs):
                kwargs['warmup'] = True
                compiled = _original(*args, **kwargs)
                if hasattr(compiled, 'result'):
                    compiled = compiled.result()
                # CompiledKernel.run is a lazy launcher/module-load property.
                # Resolve it now, but never call it on captured State tensors.
                if compiled is None or not callable(compiled.run):
                    raise RuntimeError("Serving warmup did not materialize a launcher")
                return compiled

            kernel.run = run
        yield
    finally:
        for kernel, local in reversed(originals):
            if local is None:
                del kernel.run
            else:
                kernel.run = local


def warmup(runner):
    import torch
    from . import token_progress, position_reuse, state_commit, state_slots

    capacity = runner.scheduler_config.max_num_seqs
    counts = tuple(row_counts(capacity))
    variants = dict(progress=0, positions=0, commit=0, slots=0, native_slots=0)
    # Private scratch only. No changing of runner fields or captured buffers.
    optimistic = torch.empty(capacity, device=runner.device,
                             dtype=runner.input_batch.num_computed_tokens_cpu_tensor.dtype)
    with compile_only(token_progress.correct, position_reuse.positions,
                      state_commit.commit, state_slots.publish):
        if getattr(runner, '_bs_fused_progress', False) and runner.use_async_spec_decode:
            for dtype in (torch.int32, torch.int64):
                valid = torch.empty(capacity, dtype=dtype, device=runner.device)
                for n in counts:
                    token_progress.update(
                        runner.num_computed_tokens, runner.num_accepted_tokens.gpu[:n],
                        runner.prev_positions.gpu[:n], valid,
                        runner.prev_num_draft_tokens.gpu, optimistic[:n])
                    variants['progress'] += 1
        # N/T are runtime scalars; CAP, rope dimensions and strides come from
        # this runner, not from a hard-coded capture-size list.
        # Compile against the real captured tensor addresses, but never enter
        # M01's live-request snapshot refresh during synthetic startup. Its
        # wrapper delegates private runners to the original position helper.
        # compile_only prevents a launch; the live runner/guard stay untouched.
        private = copy.copy(runner)
        variants['positions'] += int(position_reuse.prepare(private, 1, 1, optimistic))
        if hasattr(runner, '_live_state_root'):
            from .state_address import publish_slots

            private = copy.copy(runner)
            private._live_previous_verify = set()
            frames = runner._owned_frames
            if not frames:
                raise ValueError("Resident warmup requires captured metadata frames")
            for frame in frames.values():
                for original in frame.metas.values():
                    meta = copy.copy(original)
                    meta.resident_runner = private
                    meta.live = min(capacity, meta.requests)
                    # No normalization, host metadata edits or history movement.
                    # NP/NV are runtime scalars, so empty lists keep host work inert.
                    meta.device_slot_source = (None, runner.seq_lens, 0, [], [])
                    for source in (None, runner.num_accepted_tokens.gpu):
                        meta.accepted_source = source
                        publish_slots(meta)
                        variants['slots'] += 1
            if getattr(runner, '_bs_fused_state_commit', False):
                # Verify-role storage belongs to the selected frame at runtime.
                # Compile all actual bank alignments, MTP widths and token dtypes.
                width_max = runner._live_state_root.capacity.speculative_tokens + 1
                for frame in frames.values():
                    for meta in frame.metas.values():
                        private._live_verify_roles = meta.verify_roles
                        for dtype in (torch.int32, torch.int64):
                            for width in range(1, width_max + 1):
                                for stride in sorted({width, width_max}):
                                    tokens = torch.empty((capacity, stride), dtype=dtype,
                                                         device=runner.device)[:, :width]
                                    state_commit.publish(private, tokens)
                                    variants['commit'] += 1
    # School runtime's input-preparation slot mapper is outside forward capture.
    # Invoke its pinned wrapper rather than duplicating launch policy/constants.
    import os
    if os.environ.get("BETTERSCALE_QWEN35_CONTRACT") == "school":
        from vllm_ascend.ops.triton.compute_slot_mapping import _compute_slot_mapping_kernel

        with compile_only(_compute_slot_mapping_kernel):
            if runner.input_batch.block_table._can_fuse_slot_mapping:
                raise ValueError("Serving slot warmup expects the admitted single FA group")
            for table in runner.input_batch.block_table.block_tables:
                if table.is_mamba_group:
                    continue
                if table.dcp_world_size != 1:
                    raise ValueError("Serving slot warmup requires the admitted non-DCP route")
                table.compute_slot_mapping(1, runner.query_start_loc.gpu[:2], runner.positions[:1])
                variants['native_slots'] += 1
    torch.npu.synchronize()
    return variants
