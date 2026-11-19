"""Time-independent pure-decode domain after every request's first output.

Entire earlier deliveries are omitted from a prospective reference lower
bound. This avoids assigning a prompt partition to a public timing run.
The original output list is neither filtered for speed nor borrowed from a
different run. Publication/producer applicability must be qualified separately.
"""

from public_request_work import emission_contract


def domain(geometry, *, execution_seats):
    if type(execution_seats) is not int or execution_seats < 1:
        raise ValueError("Positive source execution capacity required")
    rows = geometry["requests"]
    if not rows or [r["request_ordinal"] for r in rows] != list(range(len(rows))):
        raise ValueError("Complete ordered own public population required")
    first = {}
    for row in rows:
        emission_contract(row)
        first[row["request_ordinal"]] = next(e["public_step"] for e in row["ordered_public_emissions"]
                                             if e["emitted_tokens"])
    cutoff = max(first.values())
    waves = {}
    for row in rows:
        consumed = 0
        for event in row["ordered_public_emissions"]:
            count, step = event["emitted_tokens"], event["public_step"]
            if step > cutoff and count:
                if not consumed:
                    raise ValueError("A retained tail delivery still contains a first token")
                begin = row["input_tokens"] + consumed - 1
                waves.setdefault(step, []).append(dict(request_ordinal=row["request_ordinal"],
                    canonical_target_interval=[begin, begin + count], necessary_target_rows=count,
                    retained_output_rows=count, terminal=event["finished"]))
            consumed += count
        if consumed != row["output_tokens"]:
            raise ValueError("Tail domain lost its own output population")
    result = []
    for step, entries in sorted(waves.items()):
        if len(entries) > execution_seats:
            raise ValueError("Public tail population exceeds source execution capacity")
        result.append(dict(public_step=step, entries=entries,
            necessary_target_rows=sum(e["necessary_target_rows"] for e in entries)))
    return dict(schema="public-pure-decode-tail-domain/v1",
        geometry_sha256=geometry["geometry_sha256"], cutoff_public_step=cutoff,
        selection_rule="Every delivery strictly after all requests emitted their first token; retain all such deliveries",
        execution_seats=execution_seats, waves=result,
        target_rows=sum(w["necessary_target_rows"] for w in result),
        omitted_entire_earlier_deliveries=True, prompt_partition_assumed=False,
        elapsed_time_used=False, source_runtime_applicability_certified=False,
        full_model_certificate=False, efficiency_percent=None)


def projection_shapes(model, kind):
    if (model.hidden_size != 2048 or model.num_attention_heads != 16
            or model.num_key_value_heads != 2 or model.head_dim != 256
            or model.linear_num_key_heads != 16 or model.linear_num_value_heads != 32
            or model.linear_key_head_dim != 128 or model.linear_value_head_dim != 128):
        raise ValueError("Reviewed unquantized TP2 projection dimensions required")
    if kind == "linear_attention":
        return [(6144, 2048), (32, 2048), (2048, 2048)]
    if kind == "full_attention":
        return [(4608, 2048), (2048, 2048)]
    raise ValueError("Unmapped projection layer")


def dense_shapes(tail, model):
    """Retain complete dense leaves from the first39 target layers only.

    These dimensions are specific to the separately source-bound TP2 Qwen3.5
    architecture. Other operations and the entire final decoder remain omitted
    leaves, never measured zero-duration operations.
    """
    layer_types = model.layer_types
    if (len(layer_types) != 40 or layer_types[-1] != "full_attention"
            or layer_types.count("linear_attention") != 30
            or layer_types.count("full_attention") != 10):
        raise ValueError("Reviewed 40-layer TP2 architecture required")
    cases = set()
    for wave in tail["waves"]:
        m = wave["necessary_target_rows"]
        for kind in set(layer_types[:-1]):
            projections = projection_shapes(model, kind)
            cases.update((m, n, k) for n, k in projections)
    return [dict(case=i, m=m, n=n, k=k) for i, (m, n, k) in enumerate(sorted(cases))]
