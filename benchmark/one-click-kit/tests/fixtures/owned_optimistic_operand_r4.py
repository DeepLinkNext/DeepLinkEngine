"""Keep positional drift's optimistic operand independent of device progress.

The source is the active protected CPU snapshot used to construct RoPE. The
owned device buffer is populated synchronously after progress correction and
retained for the runner lifetime. This is an explicit serving variant; its
copy/synchronization cost belongs in future uninstrumented measurements.
"""
from functools import wraps


class OwnedOptimisticOperand:
    def __init__(self, torch, runner):
        self.runner = runner
        self.buffer = torch.empty_like(runner.num_computed_tokens)
        if self.buffer.data_ptr() == runner.num_computed_tokens.data_ptr():
            raise ValueError('Owned optimistic operand aliases corrected progress')
        self.calls = 0
        self.failed = False

    def refresh(self, n):
        try:
            runner = self.runner
            guard = getattr(runner, '_m01_progress_source_snapshots', None)
            if guard is None or not guard.active or guard.failed:
                raise ValueError('Active protected host progress snapshot required')
            if not 0 < n <= self.buffer.numel():
                raise ValueError('Invalid live optimistic extent')
            source = runner.input_batch.num_computed_tokens_cpu_tensor[:n]
            if source.device.type != 'cpu' or source.dtype != self.buffer.dtype:
                raise ValueError('Expected matching CPU integer progress source')
            if source.tolist() != runner.input_batch.num_computed_tokens_cpu[:n].tolist():
                raise ValueError('RoPE host progress differs from protected source')
            result = self.buffer[:n]
            result.copy_(source, non_blocking=False)
            self.calls += 1
            return result
        except BaseException:
            self.failed = True
            raise


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
    return dict(enabled=True, policy='owned-synchronous-optimistic-snapshot/v1',
                calls=owned.calls, failed=owned.failed,
                performance_cost_included=True, full_stream_qualification=False)
