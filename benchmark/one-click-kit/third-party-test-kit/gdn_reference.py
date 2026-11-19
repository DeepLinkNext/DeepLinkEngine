"""Independent scalar and tensor references for one GDN decode transition.

This is an operator reference, not a complete M01 work/reference certificate.
Inputs are after convolution, Q/K normalization and gate preprocessing. State
layout is logical [..., value_heads, key_dim, value_dim], with FP32 state.
"""

import math


def operation_counts(key_dim, value_dim):
    """Basic arithmetic per value head; reductions have n-1 additions."""
    if any(type(n) is not int or n < 1 for n in (key_dim, value_dim)):
        raise ValueError("Positive integer key/value dimensions required")
    k, v = key_dim, value_dim
    basic = dict(
        scale_query=k,
        decay_state=k * v,
        predict_value=(2 * k - 1) * v,
        residual_value=v,
        weight_residual=v,
        update_state=2 * k * v,
        project_output=(2 * k - 1) * v,
    )
    return dict(
        basic=basic,
        total_basic=sum(basic.values()),
        special_functions={"exp": 1},
        formula="7 * key_dim * value_dim + key_dim",
        excludes=["Q/K normalization", "gate preprocessing", "convolution",
                  "GDN output normalization/gate", "draft producer closure"],
        complete_model_work=False,
    )


def scalar_step(q, k, v, log_decay, beta, state, scale):
    """Pure Python oracle for a single head, with no tensor/runtime dependency."""
    if not q or len(q) != len(k) or not v or len(state) != len(k):
        raise ValueError("Inconsistent scalar reference dimensions")
    if any(len(row) != len(v) for row in state):
        raise ValueError("State must have key-by-value layout")
    numbers = [*q, *k, *v, log_decay, beta, scale, *(x for row in state for x in row)]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in numbers):
        raise ValueError("Finite real inputs required")
    if log_decay > 0 or not 0 <= beta <= 1 or scale <= 0:
        raise ValueError("Require nonpositive log decay, beta in [0,1], positive scale")
    scaled_q = [x * scale for x in q]
    decayed = [[x * math.exp(log_decay) for x in row] for row in state]
    prediction = [math.fsum(k[i] * decayed[i][j] for i in range(len(k))) for j in range(len(v))]
    delta = [(value - predicted) * beta for value, predicted in zip(v, prediction, strict=True)]
    updated = [[decayed[i][j] + k[i] * delta[j] for j in range(len(v))] for i in range(len(k))]
    output = [math.fsum(scaled_q[i] * updated[i][j] for i in range(len(k))) for j in range(len(v))]
    return output, updated


def tensor_step(q, k, v, log_decay, beta, state, scale):
    """Independent unfused FP32 tensor algebra, usable on CPU or Ascend.

    q/k: [..., key_heads, key_dim]; v: [..., value_heads, value_dim].
    Grouped keys map each consecutive value-head group to one key head.
    Input tensors and state are never modified. Rounding to output dtype is
    the caller's responsibility, matching the production output store.
    """
    import torch

    if q.shape != k.shape or q.ndim < 2 or v.ndim != q.ndim:
        raise ValueError("Require matching grouped Q/K tensors and value rank")
    hk, dk = q.shape[-2:]
    hv, dv = v.shape[-2:]
    batch = q.shape[:-2]
    if hk < 1 or dk < 1 or hv < 1 or dv < 1 or hv % hk or v.shape[:-2] != batch:
        raise ValueError("Inconsistent grouped head dimensions")
    if state.shape != (*batch, hv, dk, dv) or log_decay.shape != (*batch, hv) or beta.shape != log_decay.shape:
        raise ValueError("State or headwise gate shape differs")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Finite positive scale required")
    tensors = (q, k, v, log_decay, beta, state)
    if any(x.device != q.device or not x.is_floating_point() for x in tensors):
        raise ValueError("Floating tensors on one device required")
    # Value-domain checks belong to the frozen input generator, outside timing.
    # The operator has no .item()/host synchronization in its measured path.
    query = torch.repeat_interleave(q.float(), hv // hk, dim=-2) * scale
    key = torch.repeat_interleave(k.float(), hv // hk, dim=-2)
    decayed = state.float() * torch.exp(log_decay.float())[..., None, None]
    prediction = (decayed * key[..., :, :, None]).sum(dim=-2)
    delta = (v.float() - prediction) * beta.float()[..., :, None]
    updated = decayed + key[..., :, :, None] * delta[..., :, None, :]
    output = (updated * query[..., :, :, None]).sum(dim=-2)
    return output, updated
