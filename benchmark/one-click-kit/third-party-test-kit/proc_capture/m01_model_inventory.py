"""Loaded shapes, scalar configuration and dispatch; never copy tensor values.

The allowlist follows the archived model_shape_inventory_v2 approach in
benchmark commit 18257625f75799577b81d8e2e71e1eae7904985b. Each current capture
records its own loaded objects and source hashes; historical values are unused.
"""

import enum
import hashlib
import inspect
import math
import numbers
from pathlib import Path

FIELDS = (
    "top_k", "topk", "num_experts", "num_experts_local", "n_routed_experts",
    "n_shared_experts", "hidden_size", "intermediate_size",
    "intermediate_size_per_partition", "tp_size", "ep_size", "num_heads",
    "num_kv_heads", "head_dim", "key_dim", "value_dim", "head_k_dim",
    "head_v_dim", "num_k_heads", "num_v_heads", "is_sequence_parallel",
    "is_fused_shared_expert_enabled", "replicate_shared_expert",
    "multistream_overlap_shared_expert", "enable_eplb", "n_redundant_experts",
    "num_redundant_experts", "rotary_dim", "rotary_ndims", "dim", "group_size",
    "num_groups", "attn_output_gate", "layer_scale", "norm_before_gate",
    "renormalize", "renorm", "apply_router_weight_on_input", "mrope_interleaved",
    "eps", "variance_epsilon", "rms_norm_eps", "routed_scaling_factor",
    "swiglu_limit", "partial_rotary_factor", "activation", "act_fn",
    "scoring_func", "layer_type", "rope_type", "mrope_section",
)
OBJECTS = ("config", "moe_config", "quant_method", "runner", "token_dispatcher", "router")
METHODS = ("forward", "forward_impl", "no_shared_forward_impl", "forward_native",
           "forward_oot", "forward_cuda", "_forward_method", "_select_experts",
           "apply", "apply_gmm1", "apply_gmm2")


def identity(obj):
    return type(obj).__module__ + "." + type(obj).__qualname__


def scalars(obj):
    result = {}
    for name in FIELDS:
        value = getattr(obj, name, None)
        if isinstance(value, enum.Enum):
            result[name] = dict(enum_type=identity(value), name=value.name)
        elif isinstance(value, bool):
            result[name] = value
        elif isinstance(value, numbers.Integral):
            result[name] = int(value)
        elif isinstance(value, numbers.Real) and math.isfinite(value):
            result[name] = float(value)
        elif isinstance(value, str) and len(value) <= 128:
            result[name] = value
        elif isinstance(value, (tuple, list)) and len(value) <= 16 and all(type(n) is int for n in value):
            result[name] = list(value)
    return result


def collect(model):
    if not callable(getattr(model, "named_modules", None)):
        raise ValueError("Actual loaded target/draft module is required for source binding")
    sources, modules = {}, []

    def source(obj):
        try:
            path = Path(inspect.getfile(obj)).resolve()
        except (TypeError, OSError):
            return None
        if not path.is_file():
            return None
        if str(path) not in sources:
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return dict(file=str(path), qualified_name=getattr(obj, "__qualname__", None))

    def describe(obj):
        methods = {}
        for name in METHODS:
            method = getattr(obj, name, None)
            if callable(method):
                found = source(method)
                if found:
                    methods[name] = dict(found, implementation=source(inspect.unwrap(method)),
                                         instance_override=name in getattr(obj, "__dict__", {}))
        return dict(type=identity(obj), scalars=scalars(obj), class_source=source(type(obj)), methods=methods)

    def tensors(items):
        return {name: dict(shape=[int(n) for n in value.shape], dtype=str(value.dtype),
                           device_type=value.device.type) for name, value in items}

    seen = set()
    for name, module in model.named_modules():
        if name in seen:
            raise ValueError("Duplicate loaded module name")
        seen.add(name)
        row = describe(module)
        row.update(name=name, objects={key: describe(getattr(module, key)) for key in OBJECTS
                                      if getattr(module, key, None) is not None},
                   parameters=tensors(module.named_parameters(recurse=False)),
                   buffers=tensors(module.named_buffers(recurse=False)))
        modules.append(row)
    if not modules:
        raise ValueError("No actual loaded model modules")
    return dict(schema="current-loaded-work-inventory/v1", modules=modules,
                source_sha256=sources, tensor_values_observed=False, arithmetic_certified=False)
