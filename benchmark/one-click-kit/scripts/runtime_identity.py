"""Named worker RPC implementations; responses use primitive serializable fields."""

from hashlib import sha256
import inspect
from pathlib import Path


def synchronize(worker):
    import torch

    torch.npu.synchronize(worker.device)
    return dict(rank=int(worker.rank), device=worker.device.index)


def identity(worker):
    runner = worker.model_runner
    if hasattr(worker, "_hust_m01_capture"):
        raise ValueError("Diagnostic worker accounting is active")
    aggregate = getattr(runner, "_m01_host_prefill", None)
    if aggregate is not None and aggregate.active:
        raise ValueError("Host prefill accounting is active")
    methods = {}
    for name in ("execute_model", "_build_attention_metadata"):
        method = getattr(runner, name)
        if name in vars(runner):
            raise ValueError("An instance-level runner observer is installed")
        entry_source = Path(inspect.getfile(method)).resolve()
        implementation = inspect.unwrap(method)
        source = Path(inspect.getfile(implementation)).resolve()
        methods[name] = dict(module=method.__module__, name=method.__qualname__,
                             source=str(source), sha256=sha256(source.read_bytes()).hexdigest(),
                             entry_source=str(entry_source),
                             entry_source_sha256=sha256(entry_source.read_bytes()).hexdigest())
    fence = synchronize(worker)
    return dict(rank=fence["rank"], logical_device=fence["device"],
                worker=type(worker).__module__ + "." + type(worker).__name__,
                rpc_transport="named_worker_method",
                worker_mro=[cls.__module__ + "." + cls.__name__ for cls in type(worker).__mro__],
                methods=methods, extra_host_observer=False, host_prefill_accounting=False,
                identity_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
