"""Last target layer roots from retained logits and real draft hidden inputs.

For this architecture the last target decoder uses full attention. All earlier
rows produce its necessary KV prefix. Its Q/output/MoE/final norm are needed
only at retained target-logit roots or at first-pass MTP hidden-input roots.
The current proposer copies target hidden rows without reindexing; that source
contract must be separately bound before this geometry becomes a certificate.
"""

from collections import defaultdict


def close(requests, draft):
    lengths, intervals = {}, defaultdict(list)
    for row in requests:
        request, prompt, output = row["request"], row["input_tokens"], row["output_tokens"]
        if (request in lengths or type(prompt) is not int or prompt < 1
                or type(output) is not int or output < 1):
            raise ValueError("Unique positive current target request extents are required")
        lengths[request] = prompt + output - 1
        # A retained token requires its target verification/bonus logit at the
        # preceding input position, including draft tokens accepted by target.
        intervals[request].append((prompt - 1, prompt + output - 2))
    if not lengths:
        raise ValueError("No current target request population")
    for row in draft["kv_intervals"]:
        if row["step"] != 0:
            continue
        request, start, stop = row["request"], row["start"], row["stop"]
        if (request not in lengths or type(start) is not int or type(stop) is not int
                or not 0 <= start <= stop < lengths[request]):
            raise ValueError("An MTP hidden input has no current accepted target producer")
        intervals[request].append((start, stop))
    merged = []
    for request, spans in sorted(intervals.items()):
        joined = []
        for start, stop in sorted(spans):
            if joined and start <= joined[-1][1] + 1:
                joined[-1][1] = max(joined[-1][1], stop)
            else:
                joined.append([start, stop])
        merged.extend(dict(request=request, start=start, stop=stop) for start, stop in joined)
    rows = sum(stop - start + 1 for start, stop in ((r["start"], r["stop"]) for r in merged))
    pairs = sum((r["start"] + r["stop"] + 2) * (r["stop"] - r["start"] + 1) // 2 for r in merged)
    return dict(scope="target_last_full_attention_output_geometry", necessary_prefix_rows=sum(lengths.values()),
        necessary_last_full_rows=rows, necessary_last_full_causal_pairs=pairs,
        full_intervals=merged, target_hidden_input_contract="unreindexed_same_position_first_pass",
        source_runtime_applicability_certified=False, full_model_certificate=False)


def matrix_components(model, closure):
    if (model.layer_types[-1] != "full_attention" or model.attention_bias
            or model.attn_output_gate is not True):
        raise ValueError("This closure requires a final full-attention target decoder")
    rows, final = closure["necessary_prefix_rows"], closure["necessary_last_full_rows"]
    if not 0 < final <= rows:
        raise ValueError("Invalid target last-layer subset")
    h = model.hidden_size
    q, kv = model.num_attention_heads * model.head_dim, model.num_key_value_heads * model.head_dim
    full, linear = model.layer_types.count("full_attention"), model.layer_types.count("linear_attention")
    layers = model.num_hidden_layers
    if full + linear != layers:
        raise ValueError("Unknown target layer family")
    attention_rows = (full - 1) * rows + final
    moe_rows = (layers - 1) * rows + final
    lk = model.linear_num_key_heads * model.linear_key_head_dim
    lv = model.linear_num_value_heads * model.linear_value_head_dim
    return dict(target_qkv_projection=4 * h * (q * attention_rows + kv * full * rows),
        target_attention_output=2 * q * h * attention_rows,
        target_shared_expert=6 * h * model.shared_expert_intermediate_size * moe_rows,
        target_router=2 * h * (model.num_experts + 1) * moe_rows,
        target_gdn_projection=2 * h * (2 * lk + 3 * lv + 2 * model.linear_num_value_heads) * linear * rows,
        target_active_expert=6 * h * model.moe_intermediate_size * model.num_experts_per_tok * moe_rows)
