"""Disjoint source arithmetic, with special functions kept out of FLOPs.

Formulas are adapted from the hash-bound archived sources listed in
semantic-work-origins.json. Counts use current model dimensions and observed
populations. This candidate ledger alone certifies neither runtime applicability
nor the completeness of necessary output dependencies.
"""

from collections import Counter


def extent(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Expected an integer semantic extent")
    return value


def record(basic=None, special=None, comparisons=None):
    parts = dict(basic=dict(basic or {}), special_functions=dict(special or {}), comparisons=dict(comparisons or {}))
    for group in parts.values():
        for count in group.values():
            extent(count)
    parts["total_basic"] = sum(parts["basic"].values())
    return parts


def rms(rows, width, *, residual=False, gated=False):
    extent(rows); extent(width, 1)
    return record(dict(square=rows * width, reduction_add=rows * (width - 1),
        mean_scale=rows, epsilon_add=rows, normalize_multiply=rows * width,
        weight_multiply=rows * width, residual_add=rows * width if residual else 0,
        output_gate_multiply=rows * width if gated else 0),
        dict(inverse_sqrt=rows, silu=rows * width if gated else 0))


def convolution(length, *, initial, width, channels):
    extent(length); extent(width, 1); extent(channels, 1)
    if type(initial) is not bool:
        raise ValueError("Convolution initial state must be explicit")
    prefix = min(length, width - 1)
    taps = length * width if initial else prefix * (prefix + 1) // 2 + (length - prefix) * width
    result = record(dict(multiply=taps * channels, add=(taps - length) * channels),
                    dict(silu=length * channels))
    result["excluded_structural_zero_multiply"] = (length * width - taps) * channels
    return result


def softmax(rows, pairs, heads, layers=1):
    for value in (rows, pairs, heads, layers): extent(value)
    if pairs < rows or not rows and pairs:
        raise ValueError("Every real causal query needs at least one key")
    multiplier = heads * layers
    return record(dict(score_scale=pairs * multiplier, subtract_max=pairs * multiplier,
                       denominator_add=(pairs - rows) * multiplier),
                  dict(exp=pairs * multiplier, divide=pairs * multiplier),
                  dict(maximum=(pairs - rows) * multiplier))


def rope(rows, heads, rotary_dim):
    extent(rows); extent(heads, 1); extent(rotary_dim, 2)
    if rotary_dim % 2:
        raise ValueError("Rotary coordinates must form complete pairs")
    return record(dict(multiply=2 * rows * heads * rotary_dim,
                       add_or_subtract=rows * heads * rotary_dim))


def moe(rows, hidden, intermediate, shared, top_k, experts):
    extent(rows)
    for value in (hidden, intermediate, shared, top_k, experts): extent(value, 1)
    if top_k > experts:
        raise ValueError("Top-k exceeds the actual expert population")
    return record(dict(routed_swiglu_outer=rows * top_k * intermediate,
        shared_swiglu_outer=rows * shared, routed_weight=rows * top_k * hidden,
        routed_weighted_sum_add=rows * (top_k - 1) * hidden,
        shared_gate=rows * hidden, shared_routed_add=rows * hidden,
        router_subtract_max=rows * experts, router_full_sum_add=rows * (experts - 1),
        router_normalize_multiply=rows * experts,
        router_selected_sum_add=rows * (top_k - 1), router_epsilon_add=rows),
        dict(silu=rows * (top_k * intermediate + shared), shared_sigmoid=rows,
             router_exp=rows * experts, router_reciprocal=rows,
             router_selected_divide=rows * top_k),
        dict(router_maximum=rows * (experts - 1)))


def gdn_preprocess(rows, key_heads, value_heads, key_dim):
    extent(rows)
    for value in (key_heads, value_heads, key_dim): extent(value, 1)
    vectors, gates = 2 * rows * key_heads, rows * value_heads
    return record(dict(qk_square=vectors * key_dim, qk_reduce_add=vectors * (key_dim - 1),
        qk_epsilon_add=vectors, qk_normalize=vectors * key_dim,
        gate_bias_add=gates, softplus_expression_add=gates, gate_scale=gates),
        dict(qk_inverse_sqrt=vectors, softplus_exp=gates, softplus_log=gates,
             parameter_exp_expression=gates, sigmoid=gates, unary_negation=gates),
        dict(softplus_threshold=gates))


def combine(records):
    groups = {name: Counter() for name in ("basic", "special_functions", "comparisons")}
    for row in records:
        for name in groups:
            groups[name].update(row[name])
    return record(groups["basic"], groups["special_functions"], groups["comparisons"])


def candidate(model, requests, draft, target=None):
    """Current canonical target population plus backward-closed draft subsets.

    Promoting this to necessary arithmetic requires live final-layer/MTP hidden
    dependency review and the loaded dispatch inventory from this capture.
    """
    if (model.attention_bias or model.attn_output_gate is not True
            or model.mtp_num_hidden_layers != 1 or model.hidden_act != "silu"
            or getattr(model, "layer_scale", False)
            or getattr(model, "norm_topk_prob", True) is not True):
        raise ValueError("Unsupported model scalar semantics")
    layers = model.num_hidden_layers
    linear, full = model.layer_types.count("linear_attention"), model.layer_types.count("full_attention")
    if linear + full != layers:
        raise ValueError("Unmapped model layer family")
    lengths = [extent(row["input_tokens"], 1) + extent(row["output_tokens"], 1) - 1 for row in requests]
    if not lengths:
        raise ValueError("No current request population")
    rows, pairs = sum(lengths), sum(n * (n + 1) // 2 for n in lengths)
    final_rows = rows if target is None else target["necessary_last_full_rows"]
    final_pairs = pairs if target is None else target["necessary_last_full_causal_pairs"]
    if (target is not None and (target["necessary_prefix_rows"] != rows
            or model.layer_types[-1] != "full_attention")):
        raise ValueError("Target closure does not match the final full-attention layer")
    attention_rows = (full - 1) * rows + final_rows
    attention_pairs = (full - 1) * pairs + final_pairs
    moe_rows = (layers - 1) * rows + final_rows
    prefix, outputs, vocab, draft_pairs = (extent(draft[key]) for key in (
        "necessary_kv_prefix_rows", "necessary_full_rows", "necessary_vocabulary_rows", "necessary_full_causal_pairs"))
    if not vocab <= outputs <= prefix or draft_pairs < outputs:
        raise ValueError("Draft source populations do not form a closed subset")
    rotary = model.rope_parameters["partial_rotary_factor"] * model.head_dim
    if not float(rotary).is_integer():
        raise ValueError("Fractional rotary coordinate count")
    rotary = int(rotary)
    h, qh, kh, d = model.hidden_size, model.num_attention_heads, model.num_key_value_heads, model.head_dim
    components = {
        "target.hidden_first": rms(rows, h),
        "target.hidden_residual": rms(2 * (layers - 1) * rows + 2 * final_rows, h, residual=True),
        "target.qk_norm": rms(rows * full * kh + attention_rows * qh, d),
        "target.gdn_output_norm_gate": rms(rows * linear * model.linear_num_value_heads,
                                             model.linear_value_head_dim, gated=True),
        "target.key_rope": rope(rows * full, kh, rotary),
        "target.query_rope": rope(attention_rows, qh, rotary),
        "target.moe_router": moe(moe_rows, h, model.moe_intermediate_size,
            model.shared_expert_intermediate_size, model.num_experts_per_tok, model.num_experts),
        "target.attention_softmax": softmax(attention_rows, attention_pairs, qh),
        "target.attention_gate": record(dict(multiply=attention_rows * qh * d), dict(sigmoid=attention_rows * qh * d)),
        "target.gdn_preprocess": gdn_preprocess(rows * linear, model.linear_num_key_heads,
            model.linear_num_value_heads, model.linear_key_head_dim),
        "target.causal_convolution": combine(convolution(n, initial=False,
            width=model.linear_conv_kernel_dim,
            channels=(2 * model.linear_num_key_heads * model.linear_key_head_dim
                      + model.linear_num_value_heads * model.linear_value_head_dim)) for n in lengths for _ in range(linear)),
        "draft.prefix_input_and_first_norms": rms(3 * prefix, h),
        "draft.full_post_attention_and_final_norms": rms(2 * outputs, h, residual=True),
        "draft.prefix_key_norm": rms(prefix * kh, d),
        "draft.full_query_norm": rms(outputs * qh, d),
        "draft.prefix_key_rope": rope(prefix, kh, rotary),
        "draft.full_query_rope": rope(outputs, qh, rotary),
        "draft.full_moe_router": moe(outputs, h, model.moe_intermediate_size,
            model.shared_expert_intermediate_size, model.num_experts_per_tok, model.num_experts),
        "draft.full_attention_softmax": softmax(outputs, draft_pairs, qh),
        "draft.full_attention_gate": record(dict(multiply=outputs * qh * d), dict(sigmoid=outputs * qh * d)),
    }
    return dict(scope="current_source_nonmatrix_candidate", components=components,
        total_basic=sum(row["total_basic"] for row in components.values()),
        target_rows=rows, target_causal_pairs=pairs, draft_kv_rows=prefix, draft_full_rows=outputs,
        target_all_layer_valid_row_convention=target is None, source_runtime_applicability_certified=False,
        full_model_certificate=False, effective_flops=None,
        remaining=["Live loaded dispatch and scalar configuration applicability",
                   "Target final-layer and draft hidden-state dependency closure",
                   "Shape-matched independent timings and complete resource dependencies"])
