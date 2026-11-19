"""Install route output recording before model graph capture, raw service only.

The bounded observer writes private device scratch, never changes routing or
model tensors. Both graph warmup and formal observation use this same service.
It is deliberately ineligible for clean performance measurement.
"""

from functools import wraps
import hashlib
import inspect
from pathlib import Path


def prepare(runner):
    import torch
    from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
    from m01_route_device import record

    if hasattr(runner, "_hust_m01_route_capture"):
        raise ValueError("Route capture is already owned")
    layers = []
    for role, model in (("target", runner.model), ("draft", runner.drafter.model)):
        for name, layer in model.named_modules():
            if isinstance(layer, AscendRoutedExperts):
                if (list(layer.w13_weight.shape) != [256, 2048, 512]
                        or list(layer.w2_weight.shape) != [256, 256, 2048]
                        or layer.top_k != 8 or layer.log2phy is not None
                        or getattr(layer, "mix_placement", False)):
                    raise ValueError("Unsupported current TP2 expert storage/placement")
                layers.append((role, name, layer))
    if [role for role, _, _ in layers] != ["target"] * 40 + ["draft"]:
        raise ValueError("Actual target/draft routed layer population differs")
    device = layers[0][2].w13_weight.device
    # Four replay records per layer cover a target microbatch envelope and the
    # two real MTP steps. Overflow is checked, never silently truncated.
    capacity, max_rows, top_k = 4, 4096, 8
    enabled = torch.zeros(2, dtype=torch.int32, device=device)
    counters = torch.zeros(41, dtype=torch.int32, device=device)
    lengths = torch.zeros((41, capacity), dtype=torch.int32, device=device)
    records = torch.empty((41, capacity, max_rows, top_k), dtype=torch.int32, device=device)
    for index, (role, _, layer) in enumerate(layers):
        layer._hust_m01_route_slot = (enabled[0:1] if role == "target" else enabled[1:2],
                                    counters[index:index + 1], lengths[index], records[index])
    original = AscendRoutedExperts._select_experts
    if not getattr(original, "_hust_m01_route_observer", False):
        @wraps(original)
        def observed(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            slot = getattr(self, "_hust_m01_route_slot", None)
            if slot is not None:
                record(result[1], *slot)
            return result
        observed._hust_m01_route_observer = True
        AscendRoutedExperts._select_experts = observed
    runner._hust_m01_route_capture = dict(layers=layers, enabled=enabled, counters=counters,
        lengths=lengths, records=records, capacity=capacity, max_rows=max_rows, active=None,
        allocated_bytes=sum(t.numel() * t.element_size() for t in (enabled, counters, lengths, records)),
        original_router_source_sha256=hashlib.sha256(Path(inspect.getfile(original)).read_bytes()).hexdigest())


def install_before_load():
    # Follow the SDK's ordinary ops registration order. Importing the runner
    # first enters attention -> DeviceOperator -> ops -> MoE -> DeviceOperator
    # while DeviceOperator is still partial during early plugin discovery.
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    original = NPUModelRunner.load_model
    if getattr(original, "_hust_m01_route_preparation", False):
        return
    @wraps(original)
    def load(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        prepare(self)
        return result
    load._hust_m01_route_preparation = True
    NPUModelRunner.load_model = load


def begin(runner, role):
    import torch

    state = runner._hust_m01_route_capture
    if state["active"] is not None or role not in ("target", "draft"):
        raise ValueError("Nested or unknown route observation role")
    span = slice(0, 40) if role == "target" else slice(40, 41)
    torch.npu.synchronize()
    state["counters"][span].zero_()
    state["lengths"][span].zero_()
    state["enabled"][0 if role == "target" else 1].fill_(1)
    torch.npu.synchronize()
    state["active"] = role


def finish(runner, role, path):
    import numpy as np
    import torch

    state = runner._hust_m01_route_capture
    if state["active"] != role:
        raise ValueError("Foreign route observer boundary")
    # Every route output, including any auxiliary device stream, has retired.
    torch.npu.synchronize()
    state["enabled"][0 if role == "target" else 1].zero_()
    torch.npu.synchronize()
    indices = range(40) if role == "target" else range(40, 41)
    counts = state["counters"].cpu().tolist()
    lengths = state["lengths"].cpu().tolist()
    arrays, layers = {}, []
    for index in indices:
        count = counts[index]
        if not 1 <= count <= state["capacity"]:
            raise ValueError("Route device history missing or overflowed")
        calls = []
        for call in range(count):
            rows = lengths[index][call]
            if not 1 <= rows <= state["max_rows"]:
                raise ValueError("Route device history contains an invalid extent")
            name = f"layer{index}_call{call}"
            values = state["records"][index, call, :rows].cpu().numpy()
            if np.any(values < 0) or np.any(values >= 256) or np.any(np.diff(np.sort(values, axis=1), axis=1) == 0):
                raise ValueError("An actual route is outside the unique top8 expert domain")
            arrays[name] = values
            calls.append(dict(array=name, rows=rows))
        layers.append(dict(index=index, name=state["layers"][index][1], calls=calls))
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    from m01_output_cache import release
    release(path, path.parent)
    state["active"] = None
    return dict(file=path.name, sha256=digest, layers=layers,
        allocated_device_bytes=state["allocated_bytes"], role=role, capacity_per_layer=state["capacity"],
        routing_changed=False, tensor_values_are_expert_ids=True, full_model_certificate=False,
        timing_eligible_for_performance_comparison=False)


def describe(runner):
    state = runner._hust_m01_route_capture
    return dict(prepared_before_graph_capture=True, capacity_per_layer=state["capacity"],
                max_rows=state["max_rows"], allocated_device_bytes=state["allocated_bytes"],
                original_router_source_sha256=state["original_router_source_sha256"],
                layers=[dict(role=role, name=name) for role, name, _ in state["layers"]])
