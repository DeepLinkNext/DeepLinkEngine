"""Keep positional drift's optimistic operand independent of device progress.

The source is the active protected CPU snapshot used to construct RoPE. The
owned device buffer is populated asynchronously on the protected preparation stream and
retained for the runner lifetime. Two pinned host sources retain their native
consumer fences. The r7 default absorbs the tested K03 async copy policy.
"""
from functools import wraps

from guarded_owned_copy_policy import make_refresh


class OwnedOptimisticOperand:
    def __init__(self, torch, runner):
        from m01_progress_source_snapshot import ProgressSourceSnapshots
        self._guarded_refresh = make_refresh(True, torch.npu.current_stream, ProgressSourceSnapshots)
        self.runner = runner
        self.buffer = torch.empty_like(runner.num_computed_tokens)
        if self.buffer.data_ptr() == runner.num_computed_tokens.data_ptr():
            raise ValueError('Owned optimistic operand aliases corrected progress')
        self.calls = 0
        self.failed = False

    def refresh(self, n):
        return self._guarded_refresh(self, n)


def install(worker):
    import torch
    from betterscale.models.qwen35 import position_reuse
    runner = worker.model_runner
    original = position_reuse.prepare
    if getattr(original, '_m01_owned_optimistic', False):
        raise ValueError('Optimistic operand policy already installed')
    owned = OwnedOptimisticOperand(torch, runner)
    worker._owned_optimistic_operand = owned

    @wraps(original)
    def prepare(current_runner, n, total, incoming):
        if current_runner is not runner:
            return original(current_runner, n, total, incoming)
        if not getattr(runner, '_bs_fused_positions', False) or not runner.use_async_spec_decode:
            raise ValueError('Owned operand requires qualified fused async position path')
        return original(current_runner, n, total, owned.refresh(n))

    prepare._m01_owned_optimistic = True
    position_reuse.prepare = prepare


def snapshot(worker):
    owned = worker._owned_optimistic_operand
    return dict(enabled=True, policy='owned-asynchronous-protected-snapshot/v2', non_blocking=True,
                calls=owned.calls, failed=owned.failed,
                performance_cost_included=True, full_stream_qualification=False)
