"""Necessary GDN prefill envelopes over every ordered 1..64 composition.

This superset contains every source-feasible segmentation once its first
zero state, subsequent state ownership and no-recomputation are qualified.
The endpoints have explicit witnesses; neither endpoint asserts an observed
partition and no midpoint is substituted for missing metadata.
"""

from gdn_prefill_reference import chunk_arithmetic
from gdn_reference import operation_counts


def bounds(lengths, *, key_dim=128, value_dim=128, allow_recurrent_prompt=False):
    if not lengths or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError("Positive current prompt lengths required")
    if type(allow_recurrent_prompt) is not bool:
        raise ValueError("Prompt execution-family uncertainty must be explicit")
    options = {state: [None] + [[(chunk_arithmetic(n, state, key_dim, value_dim)["total_basic"], "wy")]
                              for n in range(1, 65)] for state in (False, True)}
    if allow_recurrent_prompt:
        # original018 may route a one-token non-speculative prompt remainder
        # through recurrence. The first zero state omits decay, prediction,
        # residual subtraction and addition of a structural-zero state.
        cold = 3 * key_dim * value_dim + key_dim
        options[False][1].append((cold, "recurrent"))
        options[True][1].append((operation_counts(key_dim, value_dim)["total_basic"], "recurrent"))
    size = max(lengths)
    tables, choices = {}, {}
    for label, choose in (("minimum", min), ("maximum", max)):
        costs = {state: [None] + [choose(values) for values in options[state][1:]] for state in (False, True)}
        table, decision = [0] * (size + 1), [None] * (size + 1)
        for n in range(1, size + 1):
            cost, first = choose((costs[True][c][0] + table[n - c], c)
                                 for c in range(1, min(64, n) + 1))
            table[n], decision[n] = cost, first
        tables[label], choices[label] = table, (decision, costs)
    rows = []
    for length in lengths:
        endpoints = {}
        for label, choose in (("minimum", min), ("maximum", max)):
            decision, costs = choices[label]
            cost, first = choose((costs[False][c][0] + tables[label][length - c], c)
                                 for c in range(1, min(64, length) + 1))
            chunks, remaining = [first], length - first
            while remaining:
                c = decision[remaining]
                chunks.append(c)
                remaining -= c
            replay = sum(costs[i > 0][c][0] for i, c in enumerate(chunks))
            if sum(chunks) != length or replay != cost:
                raise ValueError("Partition endpoint witness does not conserve arithmetic")
            endpoints[label] = dict(total_basic=cost, witness_chunk_lengths=chunks,
                                   witness_families=[costs[i > 0][c][1] for i, c in enumerate(chunks)])
        rows.append(dict(length=length, **endpoints))
    return dict(per_request_single_head_layer=rows,
        minimum_basic=sum(r["minimum"]["total_basic"] for r in rows),
        maximum_basic=sum(r["maximum"]["total_basic"] for r in rows),
        recurrent_one_token_prompt_allowed=allow_recurrent_prompt,
        interpretation="All-compositions necessary arithmetic envelope; not observed prefill segmentation",
        elapsed_time_used=False, historical_chunk_multiplicities_used=False,
        source_runtime_applicability_certified=False, full_model_certificate=False)
