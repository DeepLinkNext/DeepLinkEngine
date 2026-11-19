"""Request-local MTP2 dependencies from source-qualified public DELTA outputs.

The decoder/output-processor and cache invariants remain separate obligations.
This module computes both possible terminal acceptance branches; it never
identifies a worker wave, imports an elapsed time, or certifies applicability.
Half-open intervals refer to logical positions, not physical cache addresses.
"""

from collections import defaultdict

from draft_state_closure import backward, versions_from_calls


def emission_contract(row):
    prompt, output = row["input_tokens"], row["output_tokens"]
    if any(type(n) is not int or n < 1 for n in (prompt, output)):
        raise ValueError("Positive current prompt and output lengths required")
    events = row["ordered_public_emissions"]
    if not events or any(type(e["finished"]) is not bool for e in events):
        raise ValueError("Ordered public events and boolean terminal flags required")
    if not events[-1]["finished"] or any(e["finished"] for e in events[:-1]):
        raise ValueError("Exactly one final public event required")
    steps = [e["public_step"] for e in events]
    if (any(type(n) is not int or n < 0 for n in steps)
            or any(a >= b for a, b in zip(steps, steps[1:]))):
        raise ValueError("Public event order is incomplete or ambiguous")
    counts = [e["emitted_tokens"] for e in events]
    if any(type(n) is not int or not 0 <= n <= 3 for n in counts):
        raise ValueError("Public DELTA emission exceeds the fixed MTP2 width")
    positive = [n for n in counts if n]
    if not positive or positive[0] != 1 or sum(positive) != output:
        raise ValueError("Initial target token and complete own output are required")
    # An empty terminal frame cannot reveal whether its preceding final output
    # was trimmed. Keep this unsupported case explicit, rather than assigning
    # an exact acceptance sequence to it.
    if not counts[-1]:
        raise ValueError("Empty terminal publication needs a separate source proof")
    if row["finish_reason"] not in ("stop", "length"):
        raise ValueError("Unsupported public stop semantics")
    return prompt, output, positive


def derive(row, *, terminal="lower"):
    """Reconstruct one endpoint without enumerating quadratic cache-read edges."""
    if terminal not in ("lower", "upper"):
        raise ValueError("Choose the lower or upper terminal branch")
    prompt, output, emitted = emission_contract(row)
    request = row["request_ordinal"]
    if type(request) is not int or request < 0:
        raise ValueError("A stable current request ordinal is required")
    identity = str(request)
    calls, root_keys = [], []
    target = [dict(local_stage=0, kind="prefill", position_interval=[0, prompt])]
    consumed, start, selected = 1, 0, prompt - 1
    for stage, count in enumerate(emitted[1:]):
        final = stage == len(emitted) - 2
        retained = count - 1
        if final and terminal == "upper":
            retained = min(2, count)
        stop = prompt if stage == 0 else start + 3
        positions = list(range(start, stop))
        sample = selected - start
        planes = [dict(step=0, query_end_envelope=[len(positions)],
                       actual_kv_lengths=[selected + 1]),
                  dict(step=1, query_end_envelope=[1],
                       actual_kv_lengths=[selected + 2])]
        calls.append(dict(call=stage, requests=[identity], capacity=len(positions),
            sample_indices=[sample], first_pass_positions=positions,
            length_planes_before=planes, length_planes_after=planes,
            draft_width=2, draft_token_ids=[[0, 0]]))
        root_keys.extend((stage, identity, step) for step in range(retained))
        begin = prompt + consumed - 1
        target.append(dict(local_stage=stage + 1, kind="decode",
                           position_interval=[begin, begin + count]))
        consumed += count
        start, selected = selected + 1, selected + count
    versions, indices = versions_from_calls(calls)
    roots = [indices[key] for key in root_keys]
    closure = backward(versions, roots)
    groups = defaultdict(lambda: dict(kv_position_intervals=[], full_queries=[]))
    for interval in closure["kv_intervals"]:
        key = interval["call"], interval["step"]
        groups[key]["kv_position_intervals"].append([interval["start"], interval["stop"] + 1])
    for index in closure["full_versions"]:
        version = versions[index]
        groups[version["call"], version["step"]]["full_queries"].append(dict(
            position=version["stop"], retained_output=index in roots,
            vocabulary_required=index in closure["vocabulary_versions"],
            prefix_version=index, hidden_parent_version=version["input_hidden"]))
    draft = [dict(local_stage=stage, draft_step=step, **group,
                  kv_rows=sum(b - a for a, b in group["kv_position_intervals"]))
             for (stage, step), group in sorted(groups.items())]
    total = sum(b - a for a, b in (t["position_interval"] for t in target))
    if total != prompt + output - 1 or consumed != output:
        raise ValueError("Current useful target population does not close")
    return dict(request_id=request, input_ids_sha256=row["input_ids_sha256"],
        output_ids_sha256=row["output_ids_sha256"], prompt_tokens=prompt,
        output_tokens=output, terminal_branch=terminal, target=target, draft=draft,
        versions=versions, closure=closure, retained_output_roots=len(roots),
        necessary_target_rows=total, vocabulary_output_rows=output,
        source_runtime_applicability_certified=False, full_model_certificate=False)


def bounds(row):
    lower, upper = (derive(row, terminal=branch) for branch in ("lower", "upper"))
    for key in ("necessary_kv_prefix_rows", "necessary_full_rows",
                "necessary_vocabulary_rows", "necessary_full_causal_pairs"):
        if lower["closure"][key] > upper["closure"][key]:
            raise ValueError("Terminal dependency interval is not ordered")
    return dict(lower=lower, upper=upper, elapsed_time_used=False,
        ordered_acceptance_source_binding_certified=False,
        necessary_work_certified=False, efficiency_percent=None)
