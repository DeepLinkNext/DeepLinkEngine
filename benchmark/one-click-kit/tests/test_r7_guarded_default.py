"""Exercise the integrated default against hostile delayed-copy/source reuse."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

TOOLS = Path(__file__).resolve().parents[1]
CORRECTION = TOOLS / 'overlays/b1-r7/mods/correction'
NATIVE = TOOLS / 'tests/fixtures'
sys.path.insert(0, str(NATIVE))
sys.path.insert(0, str(CORRECTION))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


fixture = module('r7_deferred_copy_fixture', NATIVE / 'r7_deferred_copy.py')
from m01_progress_source_snapshot import ProgressSourceSnapshots as Guard  # noqa: E402
from owned_optimistic_operand import OwnedOptimisticOperand as Owned  # noqa: E402
from guarded_source_lifecycle import initialize  # noqa: E402


def setup():
    queue = fixture.Queue()
    torch = fixture.Torch(queue)
    stream = NS(device=fixture.Device('npu'), npu_stream=99)
    torch.npu.current_stream = lambda: stream
    original = fixture.Tensor(24, queue, pinned=True)
    runner = NS(input_batch=NS(num_computed_tokens_cpu_tensor=original,
        num_computed_tokens_cpu=fixture.Array(original)),
        num_computed_tokens=fixture.Tensor(24, queue, device='npu'),
        _m01_progress_snapshot_calls=0)
    runner._m01_progress_source_snapshots = Guard(torch, runner.input_batch)
    owned = Owned(torch, runner)
    return queue, torch, original, runner, owned, stream


def test_actual_default_is_async_and_retains_guard_fences_over_192_overwrites():
    queue, _, original, runner, owned, _ = setup()
    expected, actual = [], []
    for step in range(192):
        n = (1, 24, 8, 3, 16, 7)[step % 6]
        values = [step * 100 + i + 1 for i in range(24)]
        original.values(values)
        expected.append(values[:n])

        def prepare():
            value = owned.refresh(n)
            if step == 0:
                assert value.tolist() != values[:n], 'A synchronous policy would have drained the queue'
            queue.enqueue(lambda value=value: actual.append(value.tolist()))

        runner._m01_progress_source_snapshots.run(prepare)
        original.values([-999999] * 24)
    queue.drain()
    assert actual == expected and owned.calls == 192 and not owned.failed
    assert owned.buffer.data_ptr() != runner.num_computed_tokens.data_ptr()


def test_integrated_default_rebinds_only_unused_guard_and_preserves_destination():
    queue, torch, _, runner, owned, _ = setup()
    old_guard, pointer = runner._m01_progress_source_snapshots, owned.buffer.data_ptr()
    source = fixture.Tensor(24, queue, pinned=True)
    runner.input_batch = NS(num_computed_tokens_cpu_tensor=source, num_computed_tokens_cpu=fixture.Array(source))
    result = initialize(NS(model_runner=runner, _owned_optimistic_operand=owned), torch,
        Guard, Owned, Owned.refresh, Guard.run)
    assert result['rebound'] and all(result['after']['flags'].values())
    assert old_guard is runner._hust_retired_unused_source_guard
    assert pointer == owned.buffer.data_ptr()
    source.values(list(range(24)))
    runner._m01_progress_source_snapshots.run(lambda: owned.refresh(24))
    source.values([-1] * 24)
    queue.drain()
    assert owned.buffer.tolist() == list(range(24))


@pytest.mark.parametrize('fault', ['used', 'prepared', 'copied', 'active', 'failed'])
def test_used_or_failed_guard_cannot_be_rebound(fault):
    queue, torch, _, runner, owned, _ = setup()
    guard = runner._m01_progress_source_snapshots
    source = fixture.Tensor(24, queue, pinned=True)
    runner.input_batch = NS(num_computed_tokens_cpu_tensor=source, num_computed_tokens_cpu=fixture.Array(source))
    if fault == 'used':
        guard.used[0] = True
    elif fault == 'prepared':
        runner._m01_progress_snapshot_calls = 1
    elif fault == 'copied':
        owned.calls = 1
    else:
        setattr(guard, fault, True)
    with pytest.raises(ValueError):
        initialize(NS(model_runner=runner, _owned_optimistic_operand=owned), torch,
            Guard, Owned, Owned.refresh, Guard.run)


def test_changed_stream_is_rejected_by_integrated_default():
    queue, _, _, runner, owned, stream = setup()
    runner._m01_progress_source_snapshots.run(lambda: owned.refresh(24))
    queue.drain()
    stream.npu_stream += 1
    with pytest.raises(ValueError, match='unchanged preparation stream'):
        runner._m01_progress_source_snapshots.run(lambda: owned.refresh(24))
    assert owned.failed
