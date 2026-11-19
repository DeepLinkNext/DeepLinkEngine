"""Disjoint current necessary-arithmetic interval, separate from applicability.

Terminal MTP uncertainty and unknown prompt partitions remain intervals.
An independent source/runtime audit must qualify all listed obligations before
this ledger can serve as a numerator. No duration is an input to any formula.
"""

from collections import Counter

from draft_state_closure import matrix_work
from gdn_partition_bounds import bounds as partition_bounds
from gdn_reference import operation_counts
from public_request_work import bounds as request_bounds
from semantic_work import candidate as scalar_work
from target_state_closure import close as target_close, matrix_components


def ledger(model, geometry, *, allow_recurrent_prompt=False):
    rows = geometry["requests"]
    if (not rows or [r["request_ordinal"] for r in rows] != list(range(len(rows)))
            or model.layer_types[-1] != "full_attention"
            or model.linear_key_head_dim != 128 or model.linear_value_head_dim != 128):
        raise ValueError("Current ordered population and qualified model family required")
    work = [request_bounds(r) for r in rows]
    source_rows = [dict(request=str(r["request_ordinal"]), input_tokens=r["input_tokens"],
                        output_tokens=r["output_tokens"]) for r in rows]
    prompts = [r["input_tokens"] for r in rows]
    prefix_rows = sum(r["input_tokens"] + r["output_tokens"] - 1 for r in rows)
    target_pairs = sum((r["input_tokens"] + r["output_tokens"] - 1)
                       * (r["input_tokens"] + r["output_tokens"]) // 2 for r in rows)
    output_rows = sum(r["output_tokens"] for r in rows)
    partitions = partition_bounds(prompts, key_dim=model.linear_key_head_dim,
                                  value_dim=model.linear_value_head_dim,
                                  allow_recurrent_prompt=allow_recurrent_prompt)
    layers, heads = model.layer_types.count("linear_attention"), model.linear_num_value_heads
    endpoints = {}
    for branch, extremum in (("lower", "minimum"), ("upper", "maximum")):
        combined = Counter()
        intervals = []
        for item in work:
            closure = item[branch]["closure"]
            for key in ("necessary_kv_prefix_rows", "necessary_full_rows",
                        "necessary_vocabulary_rows", "necessary_full_causal_pairs"):
                combined[key] += closure[key]
            intervals.extend(closure["kv_intervals"])
        draft = dict(combined, kv_intervals=intervals)
        target = target_close(source_rows, draft)
        parts = matrix_components(model, target)
        parts["target_causal_qk_av"] = 4 * model.num_attention_heads * model.head_dim * (
            (model.layer_types.count("full_attention") - 1) * target_pairs
            + target["necessary_last_full_causal_pairs"])
        parts["target_vocabulary"] = 2 * model.hidden_size * model.vocab_size * output_rows
        parts.update(matrix_work(model, draft)["components"])
        scalars = scalar_work(model, source_rows, draft, target)
        parts.update({name: count["total_basic"] for name, count in scalars["components"].items()})
        parts["target_gdn_prefill"] = partitions[extremum + "_basic"] * layers * heads
        parts["target_gdn_recurrence"] = (prefix_rows - sum(prompts)) * layers * heads * operation_counts(
            model.linear_key_head_dim, model.linear_value_head_dim)["total_basic"]
        if any(type(value) is not int or value < 0 for value in parts.values()):
            raise ValueError("Arithmetic component is negative or nonintegral")
        endpoints[branch] = dict(total_basic=sum(parts.values()), components=parts,
            draft_population=dict(combined), target_population=target,
            nonmatrix_special_functions_excluded_from_flops={
                name: row["special_functions"] for name, row in scalars["components"].items()})
    if endpoints["lower"]["total_basic"] > endpoints["upper"]["total_basic"]:
        raise ValueError("Necessary arithmetic envelope is not ordered")
    return dict(schema="current-public-necessary-arithmetic-interval/v1",
        geometry_sha256=geometry["geometry_sha256"], requests=len(rows),
        F_interval=[endpoints[b]["total_basic"] for b in ("lower", "upper")],
        endpoints=endpoints, prefill_partition_envelope=partitions,
        formula_families=["projection", "full_attention", "vocabulary", "scalar",
                          "convolution", "gdn_prefill", "gdn_decode", "draft_dependency"],
        elapsed_time_used=False, historical_work_or_timing_reused=False,
        necessary_work_certified=False, source_runtime_applicability_certified=False,
        full_model_certificate=False, efficiency_percent=None,
        required_source_obligations=[
            "Loaded TP2 dimensions, scalar settings, unquantized dispatch and formula sources match.",
            "One public DELTA emission is one uncoalesced core delivery; fixed MTP2 nonterminal c maps to c-1 accepted.",
            "Final trimmed acceptance is contained by both terminal branches.",
            "Request-owned fresh target/draft cache prefixes have no reused external or missing producers.",
            "Capacity and source admission prove no preemption/recomputation of a necessary prefix.",
            "MTP first-pass target hidden inputs use the same logical positions; overwritten draft suffixes cannot feed earlier reads.",
            "Every prompt token belongs to one zero-first/stateful-later chunk chain of lengths 1..64.",
            "Final target full-attention roots and MTP KV/full/vocabulary backward closure are complete."])
