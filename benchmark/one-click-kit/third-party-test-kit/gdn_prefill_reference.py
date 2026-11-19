"""Independent segmented GDN prefill and source-defined logical arithmetic.

No production kernel is imported here. Tensor algebra uses the inverse of the
unit lower triangular delta system; a scalar recurrence is a separate oracle.
Matrix products count an FMA as two basic operations, matching the fixed M01
catalog. Structural zero/padding work is omitted, special functions are typed.
"""

import math


def positive_integer(n, name):
    if type(n) is not int or n < 1:
        raise ValueError(name + " must be a positive integer")
    return n


def merge_arithmetic(left, right):
    """Two products: lower(right) @ dense @ lower(left), FMA=2."""
    if not left or not right:
        return 0
    return left * right * (left + right + 2)


def chunk_arithmetic(length, initial_state, key_dim=128, value_dim=128):
    """Necessary arithmetic for one real <=64 chunk and one value head.

    The source inverse first solves 16x16 diagonal blocks, then merges two
    pairs at width16 and a final pair at width32. Zero initial states omit the
    state dot/decay/add ancestors, while all genuine K/V/output work remains.
    """
    n = positive_integer(length, "length")
    k = positive_integer(key_dim, "key_dim")
    v = positive_integer(value_dim, "value_dim")
    if n > 64 or type(initial_state) is not bool:
        raise ValueError("Require a real <=64 chunk and Boolean state presence")
    strict, causal = n * (n - 1) // 2, n * (n + 1) // 2
    blocks = [min(16, max(0, n - i)) for i in range(0, 64, 16)]
    # For row i and column j<i, the 16x16 recurrence has i-j-1
    # multiply/add pairs. Signs are accounted once per strict-lower entry.
    inverse_recurrence = sum(b * (b - 1) * (b - 2) // 3 for b in blocks)
    merge = merge_arithmetic(blocks[0], blocks[1])
    merge += merge_arithmetic(blocks[2], blocks[3])
    merge += merge_arithmetic(sum(blocks[:2]), sum(blocks[2:]))
    parts = dict(
        prefix_scan_add=n - 1,
        kkt_dot=2 * k * strict,
        kkt_gate_difference_and_scaling=3 * strict,
        inverse16_recurrence=inverse_recurrence,
        inverse_merge_triangular_dot=merge,
        inverse_negations=strict,
        wy_value_beta=n * v,
        wy_key_beta_gate=2 * n * k,
        wy_lower_triangular_dot=2 * (k + v) * causal,
        h_wh_dot=2 * n * k * v if initial_state else 0,
        h_value_residual_subtract=n * v if initial_state else 0,
        h_decay_difference=n,
        h_value_decay_multiply=n * v,
        h_kv_dot=2 * n * k * v,
        h_state_decay_and_add=2 * k * v if initial_state else 0,
        o_qk_causal_dot=2 * k * causal,
        o_gate_difference_and_multiply=2 * causal,
        o_qh_dot=2 * n * k * v if initial_state else 0,
        o_av_causal_dot=2 * v * causal,
        o_state_decay_and_add=2 * n * v if initial_state else 0,
        o_output_scale=n * v,
    )
    return dict(basic=parts, total_basic=sum(parts.values()),
                special_functions=dict(exp=strict + n + causal + n +
                                       (n + 1 if initial_state else 0)),
                comparisons=dict(clamp=causal),
                length=n, initial_state=initial_state)


def segment_arithmetic(length, initial_state, key_dim=128, value_dim=128):
    """A published segment starts a fresh 64-token chunk grid.

    Segment boundaries must come from actual metadata. Concatenating two
    segments can change inverse/WY work and is deliberately not done here.
    """
    positive_integer(length, "length")
    if type(initial_state) is not bool:
        raise ValueError("State presence must be Boolean")
    chunks = [chunk_arithmetic(min(64, length - start),
                              initial_state or start > 0, key_dim, value_dim)
              for start in range(0, length, 64)]
    return dict(chunks=chunks,
                basic={name: sum(c["basic"][name] for c in chunks)
                       for name in chunks[0]["basic"]},
                total_basic=sum(c["total_basic"] for c in chunks),
                special_functions=dict(exp=sum(c["special_functions"]["exp"] for c in chunks)),
                comparisons=dict(clamp=sum(c["comparisons"]["clamp"] for c in chunks)))


def _validate(q, k, v, g, beta, state, scale):
    if q.ndim != 3 or q.shape != k.shape or v.ndim != 3:
        raise ValueError("Require [tokens, heads, dimension] Q/K/V")
    n, hk, dk = q.shape
    if n < 1 or n > 64 or hk < 1 or dk < 1:
        raise ValueError("Require 1..64 genuine tokens and positive dimensions")
    hv, dv = v.shape[1:]
    if (v.shape[0] != n or hv < 1 or hv % hk or dv < 1
            or g.shape != (n, hv) or beta.shape != g.shape
            or state.shape != (hv, dk, dv)):
        raise ValueError("Inconsistent grouped head/state/gate geometry")
    if (isinstance(scale, bool) or not isinstance(scale, (float, int))
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError("Require finite positive scale")
    if any(x.device != q.device or not x.is_floating_point()
           for x in (q, k, v, g, beta, state)):
        raise ValueError("Floating inputs on one device required")
    return n, hk, hv


def chunk(q, k, v, log_decay, beta, state, scale, *, initial_state=True, storage_dtype=None):
    """Independent chunk solution, no data-dependent host reads.

    The per-row inverse uses Python loops only over frozen geometry, allowing
    CPU and NPU execution without reading a device scalar. The reference does
    optionally models the declared intermediate storage type. The default
    keeps FP32 throughout for comparison with the independent scalar oracle.
    """
    import torch

    def stored(value):
        return value.to(storage_dtype).float() if storage_dtype is not None else value

    n, hk, hv = _validate(q, k, v, log_decay, beta, state, scale)
    if type(initial_state) is not bool:
        raise ValueError("State presence must be Boolean")
    key = torch.repeat_interleave(k.float(), hv // hk, dim=1).transpose(0, 1)
    query = torch.repeat_interleave(q.float(), hv // hk, dim=1).transpose(0, 1)
    value, weight = v.float().transpose(0, 1), beta.float().transpose(0, 1)
    cumulative = log_decay.float().transpose(0, 1).cumsum(dim=1)
    decay = torch.exp(cumulative)
    relative = torch.exp((cumulative[:, :, None] - cumulative[:, None, :]).clamp(max=0))
    dot = key @ key.transpose(1, 2)
    lower = torch.tril(dot * relative * weight[:, :, None], diagonal=-1)
    # Solve (I + lower) B = I by independent forward substitution. No vendor
    # triangular solve or production inverse implementation is called.
    identity = torch.eye(n, dtype=torch.float32, device=q.device)
    rows = []
    for i in range(n):
        row = identity[i].expand(hv, n)
        if i:
            row = row - (lower[:, i, :i, None] * torch.stack(rows, dim=1)).sum(dim=1)
        rows.append(row)
    inverse = stored(torch.stack(rows, dim=1))
    u = inverse @ (value * weight[:, :, None])
    w = inverse @ (key * (weight * decay)[:, :, None])
    u, w = stored(u), stored(w)
    previous = stored(state.float())
    residual = u - w @ previous if initial_state else u
    scores = torch.tril((query @ key.transpose(1, 2)) * relative)
    output = stored(scores) @ stored(residual)
    if initial_state:
        output = output + (query @ previous) * decay[:, :, None]
    output = output * scale
    tail = torch.exp(cumulative[:, -1, None] - cumulative)
    final = key.transpose(1, 2) @ stored(residual * tail[:, :, None])
    if initial_state:
        # The owned H kernel decays the BF16 H workspace, including for the
        # final FP32 store. It does not retain the incoming FP32 bank value
        # for this update. This matters at every real segment boundary.
        final = final + previous * decay[:, -1, None, None]
    return output.transpose(0, 1), final


def segmented(q, k, v, log_decay, beta, state, scale, segment_lengths, *, initial_state=True, storage_dtype=None):
    """Apply real published segment boundaries, keeping causal state across them."""
    if (not segment_lengths or any(type(n) is not int or n < 1 for n in segment_lengths)
            or sum(segment_lengths) != q.shape[0]):
        raise ValueError("Positive segment lengths must cover all tokens exactly")
    if type(initial_state) is not bool:
        raise ValueError("State presence must be Boolean")
    output, offset, current, present = [], 0, state, initial_state
    for length in segment_lengths:
        end = offset + length
        while offset < end:
            stop = min(end, offset + 64)
            result, current = chunk(q[offset:stop], k[offset:stop], v[offset:stop],
                                    log_decay[offset:stop], beta[offset:stop], current, scale,
                                    initial_state=present, storage_dtype=storage_dtype)
            output.append(result)
            offset = stop
            present = True
    import torch

    return torch.cat(output, dim=0), current
