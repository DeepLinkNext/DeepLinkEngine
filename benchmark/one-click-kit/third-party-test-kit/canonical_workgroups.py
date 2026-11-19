"""Conserved semantic workgroups with explicit finite admission dependencies.

This is a source-feasible reference scheduling model, not a reconstruction of
the measured native worker waves. Prompt partitions must be supplied by a
separately qualified binding; no historical histogram or duration is used.
"""

from collections import Counter, deque


def build(requests, order, segments, *, budget_limit, max_active):
    if any(type(n) is not int or n < 1 for n in (budget_limit, max_active)):
        raise ValueError("Positive source-bound token and admission capacities required")
    if (not order or len(set(order)) != len(order) or set(order) != set(requests)
            or set(segments) != set(requests)):
        raise ValueError("A complete unique current request order and partition binding required")
    for rid in order:
        r, parts = requests[rid], segments[rid]
        if (r["request_id"] != rid or not parts
                or any(type(n) is not int or not 1 <= n <= budget_limit for n in parts)
                or sum(parts) != r["prompt_tokens"]):
            raise ValueError("Prompt partitions do not conserve current request inputs")
        target = r["target"]
        if (not target or target[0]["position_interval"] != [0, r["prompt_tokens"]]
                or [t["local_stage"] for t in target] != list(range(len(target)))
                or any(a["position_interval"][1] != b["position_interval"][0]
                       for a, b in zip(target, target[1:]))
                or target[-1]["position_interval"][1] != r["prompt_tokens"] + r["output_tokens"] - 1):
            raise ValueError("Necessary target intervals have gaps or foreign population")
    if budget_limit < 3 and any(len(r["target"]) > 1 for r in requests.values()):
        raise ValueError("MTP2 decode requires three scheduled verification rows")
    pending, active = deque(order), []
    free = deque((seat, None) for seat in range(max_active))
    state = {rid: dict(stage=0, segment=0, prefilled=0, last_wave=None) for rid in order}
    draft = {}
    for rid, r in requests.items():
        index = {(d["local_stage"], d["draft_step"]): d for d in r["draft"]}
        if len(index) != len(r["draft"]):
            raise ValueError("Duplicate necessary draft group")
        draft[rid] = index
    waves, totals, admissions = [], Counter(dict(target_rows=0, vocabulary_output_rows=0,
        draft_kv_rows=0, draft_full_queries=0, draft_attention_pairs=0, prefill_segments=0)), []
    while pending or active:
        wave_id, budget, cursor = len(waves), budget_limit, 0
        entries, retired = [], []
        while budget and (cursor < len(active) or (pending and free)):
            if cursor == len(active):
                rid, (seat, predecessor) = pending.popleft(), free.popleft()
                active.append(rid)
                state[rid]["seat"] = seat
                admissions.append(dict(request_id=rid, seat=seat, wave=wave_id,
                    previous_seat_owner=predecessor,
                    feedback="previous owner's terminal CPU output parse to new admission metadata"))
            rid = active[cursor]
            cursor += 1
            r, s = requests[rid], state[rid]
            stage = s["stage"]
            if stage:
                if budget < 3:
                    continue
                begin, end = r["target"][stage]["position_interval"]
                scheduled, complete = 3, True
                if not 1 <= end - begin <= scheduled:
                    raise ValueError("Invalid canonical MTP2 target interval")
            else:
                begin, scheduled = s["prefilled"], segments[rid][s["segment"]]
                if scheduled > budget:
                    continue
                s["segment"] += 1
                end = begin + scheduled
                s["prefilled"] = end
                complete = end == r["prompt_tokens"]
            budget -= scheduled
            groups = []
            for step in (0, 1):
                source = draft[rid].get((stage, step))
                if source is None:
                    continue
                intervals, queries = source["kv_position_intervals"], source["full_queries"]
                if stage == 0:
                    intervals = ([[max(a, begin), min(b, end)] for a, b in intervals
                                  if max(a, begin) < min(b, end)] if step == 0
                                 else intervals if complete else [])
                    if not complete:
                        queries = []
                kv_rows = sum(b - a for a, b in intervals)
                if kv_rows or queries:
                    group = dict(draft_step=step, kv_position_intervals=intervals,
                        kv_rows=kv_rows, full_queries=queries,
                        attention_pairs=sum(q["position"] + 1 for q in queries))
                    groups.append(group)
                    totals.update(draft_kv_rows=kv_rows, draft_full_queries=len(queries),
                                  draft_attention_pairs=group["attention_pairs"])
            outputs = (1 if stage == 0 else end - begin) if complete else 0
            entries.append(dict(request_id=rid, local_stage=stage,
                kind="prefill" if stage == 0 else "decode", stage_completed=complete,
                previous_request_wave=s["last_wave"], seat=s["seat"],
                scheduled_target_rows=scheduled, canonical_target_interval=[begin, end],
                canonical_target_rows=end - begin, retained_output_rows=outputs, draft=groups))
            s["last_wave"] = wave_id
            totals.update(target_rows=end - begin, vocabulary_output_rows=outputs,
                          prefill_segments=int(stage == 0))
            if complete:
                s["stage"] += 1
                if s["stage"] == len(r["target"]):
                    retired.append(rid)
        if not entries:
            raise ValueError("The declared partition/admission model cannot make progress")
        waves.append(dict(wave=wave_id, active_requests=len(active),
            scheduled_target_rows=budget_limit - budget, entries=entries))
        for rid in retired:
            active.remove(rid)
            free.append((state[rid]["seat"], dict(request_id=rid, terminal_wave=wave_id)))
    expected = dict(target_rows=sum(r["necessary_target_rows"] for r in requests.values()),
        vocabulary_output_rows=sum(r["output_tokens"] for r in requests.values()),
        draft_kv_rows=sum(r["closure"]["necessary_kv_prefix_rows"] for r in requests.values()),
        draft_full_queries=sum(r["closure"]["necessary_full_rows"] for r in requests.values()),
        draft_attention_pairs=sum(r["closure"]["necessary_full_causal_pairs"] for r in requests.values()))
    if any(totals[key] != value for key, value in expected.items()):
        raise ValueError("Semantic regrouping lost or duplicated necessary work")
    return dict(waves=waves, admissions=admissions, totals=dict(totals),
        budget_limit=budget_limit, max_active=max_active,
        interpretation="Finite-capacity semantic reference model; not native measured worker-wave identity",
        elapsed_time_used=False, source_runtime_applicability_certified=False,
        full_model_certificate=False)
