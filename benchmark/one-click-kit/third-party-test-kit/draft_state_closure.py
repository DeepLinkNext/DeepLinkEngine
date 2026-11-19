"""Backward source dependencies for an observed single-block Qwen MTP2 trace.

Persistent prefix versions avoid expanding every causal attention edge into
an O(sequence_length**2) Python graph. A version stores the real contiguous KV
write and its previous prefix owner; overwriting a suffix does not resurrect
or double-count the old speculative suffix. Unknown/missing state is rejected.
This module supplies geometry, never a full-model arithmetic/time certificate.
"""

from collections import deque


def positive(value, label):
    if type(value) is not int or value < 0:
        raise ValueError("Invalid " + label)
    return value


def versions_from_calls(calls):
    versions, latest, roots = [], {}, {}
    for ordinal, call in enumerate(calls):
        if call["call"] != ordinal:
            raise ValueError("MTP call order is incomplete")
        before, after = call["length_planes_before"], call["length_planes_after"]
        if before != after:
            raise ValueError("MTP length plane changed across the merged call; its read lifetime needs qualification")
        planes = {}
        for plane in before:
            if plane["step"] in planes:
                raise ValueError("This closure supports one shared attention metadata group per MTP step")
            planes[plane["step"]] = plane
        width = call.get("draft_width", len(call["draft_token_ids"][0]))
        if width not in (1, 2) or set(planes) != set(range(width)):
            raise ValueError("MTP step metadata is absent or unsupported")
        previous_end = 0
        for row, request in enumerate(call["requests"]):
            sample = call["sample_indices"][row]
            stop = planes[0]["query_end_envelope"][row]
            if not previous_end <= sample < stop <= call["capacity"]:
                raise ValueError("The real MTP sample does not belong to its query envelope")
            positions = call["first_pass_positions"][previous_end:sample + 1]
            if (not positions or any(type(p) is not int or p < 0 for p in positions)
                    or positions != list(range(positions[0], positions[-1] + 1))):
                raise ValueError("A first-pass MTP KV write is not a real contiguous request interval")
            start, end = positions[0], positions[-1]
            contract = call.get("first_pass_kv_extent_contract", "selected_prefix")
            if contract == "full_query_envelope":
                complete_positions = call["first_pass_positions"][previous_end:stop]
                if complete_positions != list(range(start, start + stop - previous_end)):
                    raise ValueError("MTP complete query envelope is not contiguous")
                extent = complete_positions[-1] + 1
            elif contract == "selected_prefix":
                # Public-emission reconstruction describes only the logical
                # selected prefix; it is not a physical device observation.
                extent = end + 1
            else:
                raise ValueError("Unknown first-pass MTP KV extent contract")
            if planes[0]["actual_kv_lengths"][row] != extent:
                raise ValueError("MTP query position and device-authoritative KV extent disagree")
            parent = latest.get(request)
            if start and (parent is None or versions[parent]["stop"] < start - 1):
                raise ValueError("A draft KV prefix has an unobserved producer")
            first = len(versions)
            versions.append(dict(call=ordinal, request=request, row=row, step=0,
                start=start, stop=end, parent=parent, input_hidden=None, input_token=None))
            roots[ordinal, request, 0] = first
            latest[request] = first
            if width == 2:
                if planes[1]["actual_kv_lengths"][row] != end + 2:
                    raise ValueError("The second MTP step does not extend the same actual prefix once")
                second = len(versions)
                versions.append(dict(call=ordinal, request=request, row=row, step=1,
                    start=end + 1, stop=end + 1, parent=first,
                    input_hidden=first, input_token=first))
                roots[ordinal, request, 1] = second
                latest[request] = second
            previous_end = stop
    return versions, roots


def backward(versions, vocabulary_roots):
    """Return the exact union of required prefix/full/logit producer geometries."""
    pending = deque(("vocabulary", root, None) for root in vocabulary_roots)
    needed, full, vocabulary = {}, set(), set()
    while pending:
        kind, index, end = pending.popleft()
        if type(index) is not int or not 0 <= index < len(versions):
            raise ValueError("A necessary MTP dependency is missing")
        version = versions[index]
        start, stop = version["start"], version["stop"]
        if (type(start) is not int or type(stop) is not int or not 0 <= start <= stop
                or version["parent"] is not None and not 0 <= version["parent"] < index):
            raise ValueError("MTP cache version has a gap, foreign owner or future dependency")
        if kind == "vocabulary":
            if index in vocabulary:
                continue
            vocabulary.add(index)
            pending.append(("full", index, None))
        elif kind == "full":
            if index in full:
                continue
            full.add(index)
            pending.append(("prefix", index, stop))
        elif kind == "prefix":
            if type(end) is not int or not 0 <= end <= stop:
                raise ValueError("Attention consumes an absent draft KV position")
            if needed.get(index, -1) >= end:
                continue
            needed[index] = end
            previous = min(start - 1, end)
            if previous >= 0:
                parent = version["parent"]
                if parent is None or versions[parent]["stop"] < previous or versions[parent]["request"] != version["request"]:
                    raise ValueError("A necessary draft KV read has no matching request-owned producer")
                pending.append(("prefix", parent, previous))
            if end >= start:
                # Step1's embedding ID and recycled hidden state come from
                # step0's sample and full output, even if step1 is needed only
                # as a KV producer. No whole-layer cost is assigned to other
                # first-pass KV-only rows.
                if version["input_hidden"] is not None:
                    pending.append(("full", version["input_hidden"], None))
                if version["input_token"] is not None:
                    pending.append(("vocabulary", version["input_token"], None))
        else:
            raise ValueError("Unknown MTP dependency kind")
    intervals = [dict(version=index, call=versions[index]["call"],
        request=versions[index]["request"], step=versions[index]["step"],
        start=versions[index]["start"], stop=end)
        for index, end in sorted(needed.items()) if end >= versions[index]["start"]]
    return dict(scope="backward_observed_mtp_prefix_full_and_vocabulary_geometry",
        necessary_kv_prefix_rows=sum(r["stop"] - r["start"] + 1 for r in intervals),
        necessary_full_rows=len(full), necessary_vocabulary_rows=len(vocabulary),
        necessary_full_causal_pairs=sum(versions[index]["stop"] + 1 for index in full),
        kv_intervals=intervals, full_versions=sorted(full), vocabulary_versions=sorted(vocabulary),
        missing_dependencies_replaced_by_zero=False, source_runtime_applicability_certified=False,
        full_model_certificate=False, effective_flops=None)


def close(calls, producer_bindings):
    versions, root_index = versions_from_calls(calls)
    roots = []
    for binding in producer_bindings["bindings"]:
        for step in range(binding["retained_drafts"]):
            key = binding["producer_call"], binding["request"], step
            if key not in root_index:
                raise ValueError("A retained token lacks its actual full/logit producer")
            roots.append(root_index[key])
    result = backward(versions, roots)
    result["observed_versions"] = len(versions)
    return result


def matrix_work(model, closure):
    """Global TP2 matrices for the verified prefix/full/logit populations.

    Qwen3.5 MTP has two input norms, concat -> FC -> one full-attention/MoE
    decoder. KV-only ancestors require FC and K/V; Q/gate, attention output
    and MoE belong only to full output ancestors. Scalars remain separate.
    """
    if (getattr(model, "mtp_num_hidden_layers", None) != 1
            or model.attention_bias or model.attn_output_gate is not True):
        raise ValueError("This matrix mapping requires the pinned one-layer gated MTP architecture")
    names = ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
             "num_experts", "num_experts_per_tok", "moe_intermediate_size",
             "shared_expert_intermediate_size", "vocab_size")
    for name in names:
        value = getattr(model, name)
        if type(value) is not int or value < 1:
            raise ValueError("Missing positive MTP model dimension: " + name)
    h = model.hidden_size
    q = model.num_attention_heads * model.head_dim
    kv = model.num_key_value_heads * model.head_dim
    prefix = closure["necessary_kv_prefix_rows"]
    full, vocabulary = closure["necessary_full_rows"], closure["necessary_vocabulary_rows"]
    pairs = closure["necessary_full_causal_pairs"]
    if (any(type(n) is not int or n < 0 for n in (prefix, full, vocabulary, pairs))
            or not vocabulary <= full <= prefix):
        raise ValueError("MTP populations do not form a closed source dependency subset")
    parts = dict(
        draft_prefix_concat_fc=4 * h * h * prefix,
        draft_prefix_key_value_projection=4 * h * kv * prefix,
        draft_full_query_gate_projection=4 * h * q * full,
        draft_full_attention_output=2 * q * h * full,
        draft_full_active_experts=6 * h * model.moe_intermediate_size * model.num_experts_per_tok * full,
        draft_full_shared_expert=6 * h * model.shared_expert_intermediate_size * full,
        draft_full_router=2 * h * (model.num_experts + 1) * full,
        draft_full_causal_qk_av=4 * model.num_attention_heads * model.head_dim * pairs,
        draft_retained_or_dependency_vocabulary=2 * h * model.vocab_size * vocabulary,
    )
    return dict(scope="global_mtp_source_matrix_arithmetic", components=parts,
                total_basic=sum(parts.values()), scalar_arithmetic_included=False,
                source_runtime_applicability_certified=False, full_model_certificate=False)
