"""Common guarded refresh for sync/async copies; retain native source fences."""


def make_refresh(non_blocking, current_stream, guard_type):
    if type(non_blocking) is not bool:
        raise ValueError('An explicit Boolean copy mode is required')

    def refresh(self, n):
        try:
            runner = self.runner
            guard = getattr(runner, '_m01_progress_source_snapshots', None)
            if type(guard) is not guard_type or not guard.active or guard.failed:
                raise ValueError('Active native protected host progress snapshot required')
            if not 0 < n <= self.buffer.numel():
                raise ValueError('Invalid live optimistic extent')
            protected = runner.input_batch.num_computed_tokens_cpu_tensor
            if not any(protected.data_ptr() == value.data_ptr() for value in guard.buffers):
                raise ValueError('The active host source is outside the native fenced buffers')
            source = protected[:n]
            if source.device.type != 'cpu' or source.dtype != self.buffer.dtype or not source.is_pinned():
                raise ValueError('Expected matching pinned CPU integer progress source')
            if source.tolist() != runner.input_batch.num_computed_tokens_cpu[:n].tolist():
                raise ValueError('RoPE host progress differs from protected source')
            stream = current_stream()
            identity = (str(stream.device), int(stream.npu_stream))
            if identity[0] != str(self.buffer.device):
                raise ValueError('Copy stream and owned device buffer differ')
            previous = getattr(self, '_hust_operand_copy_stream', None)
            if previous is not None and previous != identity:
                raise ValueError('The owned-copy experiment requires one unchanged preparation stream')
            self._hust_operand_copy_stream = identity
            result = self.buffer[:n]
            result.copy_(source, non_blocking=non_blocking)
            self.calls += 1
            return result
        except BaseException:
            self.failed = True
            raise

    return refresh
