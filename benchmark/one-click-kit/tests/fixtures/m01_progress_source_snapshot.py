"""Candidate pinned-source protection; not installed into the serving runtime.

The prepare callback must enqueue source reads on the current NPU stream.
This guard does not protect callbacks that transfer source use to other streams.
"""


class ProgressSnapshotRunnerMixin:
    def _prepare_inputs(self, *args, **kwargs):
        prepare = super()._prepare_inputs
        guard = getattr(self, '_m01_progress_source_snapshots', None)
        if guard is None:
            return prepare(*args, **kwargs)
        result = guard.run(lambda: prepare(*args, **kwargs))
        self._m01_progress_snapshot_calls += 1
        return result


class ProgressSourceSnapshots:
    """Two source buffers, reused only after their consuming stream reaches a fence."""

    def __init__(self, torch, batch):
        source = batch.num_computed_tokens_cpu_tensor
        if source.device.type != 'cpu' or not source.is_pinned() or source.ndim != 1:
            raise ValueError('Pinned one-dimensional CPU progress source required')
        if source.dtype not in (torch.int32, torch.int64):
            raise ValueError('Integer progress source required')
        if batch.num_computed_tokens_cpu.__array_interface__['data'][0] != source.data_ptr():
            raise ValueError('Expected native NumPy/tensor alias')
        self.batch, self.original = batch, source
        self.buffers = [torch.empty_like(source, pin_memory=True) for _ in range(2)]
        self.events = [torch.npu.Event() for _ in range(2)]
        self.used, self.index = [False, False], 0
        self.active, self.failed = False, False

    def run(self, prepare):
        if self.failed or self.active:
            raise RuntimeError('Snapshot guard failed or reentered')
        if self.batch.num_computed_tokens_cpu_tensor is not self.original:
            raise ValueError('Native progress source identity changed')
        index = self.index
        if self.used[index]:
            self.events[index].synchronize()
        self.buffers[index].copy_(self.original)
        self.active = True
        self.batch.num_computed_tokens_cpu_tensor = self.buffers[index]
        try:
            return prepare()
        finally:
            try:
                # Queued H2D reads must finish before this source is overwritten.
                self.events[index].record()
                self.used[index] = True
                self.index = 1 - index
            except BaseException:
                self.failed = True
                raise
            finally:
                self.batch.num_computed_tokens_cpu_tensor = self.original
                self.active = False
