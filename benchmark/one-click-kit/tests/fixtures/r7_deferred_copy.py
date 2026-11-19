"""Adversarial copy queue derived from the original K04 source-fence test."""

from dataclasses import dataclass
from types import SimpleNamespace as NS

@dataclass(frozen=True)
class Device:
    type: str
    index: int = 0

    def __str__(self):
        return self.type if self.type == 'cpu' else f'{self.type}:{self.index}'


class Queue:
    def __init__(self):
        self.tasks = []

    def enqueue(self, callback):
        self.tasks.append(callback)

    def step(self):
        assert self.tasks, 'A fence without a queued completion would hang'
        self.tasks.pop(0)()

    def drain(self):
        while self.tasks:
            self.step()


class Event:
    def __init__(self, queue):
        self.queue, self.done = queue, True

    def record(self):
        self.done = False
        self.queue.enqueue(lambda: setattr(self, 'done', True))

    def synchronize(self):
        while not self.done:
            self.queue.step()


class Tensor:
    def __init__(self, count, queue, *, device='cpu', pinned=False, backing=None, offset=0):
        self.count, self.queue, self.device = count, queue, Device(device)
        self.pinned, self.offset = pinned, offset
        self.backing = [0] * count if backing is None else backing
        self.dtype, self.ndim = 'int64', 1

    def __getitem__(self, selected):
        assert isinstance(selected, slice)
        start, stop, step = selected.indices(self.count)
        assert step == 1
        view = Tensor(stop-start, self.queue, device=self.device.type, pinned=self.pinned,
                      backing=self.backing, offset=self.offset+start)
        view.dtype = self.dtype
        return view

    def data_ptr(self):
        return id(self.backing) + self.offset * 8

    def is_pinned(self):
        return self.pinned

    def numel(self):
        return self.count

    def tolist(self):
        return self.backing[self.offset:self.offset+self.count]

    def values(self, values):
        assert len(values) == self.count
        self.backing[self.offset:self.offset+self.count] = values

    def copy_(self, source, non_blocking=False):
        assert source.count == self.count

        def transfer():
            self.values(source.tolist())

        if self.device.type == 'cpu':
            transfer()
        else:
            self.queue.enqueue(transfer)
            if not non_blocking:
                self.queue.drain()
        return self


class Array:
    def __init__(self, tensor):
        self.tensor = tensor

    @property
    def __array_interface__(self):
        return {'data': (self.tensor.data_ptr(), False)}

    def __getitem__(self, selected):
        return Array(self.tensor[selected])

    def tolist(self):
        return self.tensor.tolist()


class Torch:
    int32, int64 = 'int32', 'int64'

    def __init__(self, queue):
        self.queue = queue
        self.npu = NS(Event=lambda: Event(queue))

    def empty_like(self, source, pin_memory=False):
        return Tensor(source.count, self.queue, device=source.device.type, pinned=pin_memory)

