"""Bind retained MTP tokens to actual prior per-request merged producers.

This is a producer identity/geometry join. It does not yet certify the backward
KV/hidden-state closure or turn physical candidate work into effective FLOPs.
"""

from collections import defaultdict


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Invalid draft " + label)
    return value


def canonical(record):
    n = integer(record["actual_input_tokens"], 1, 4096, "actual rows")
    if not n <= record["capacity"] <= 4096:
        raise ValueError("Draft actual rows exceed capacity")
    result = {key: (value[:n] if key in ("first_pass_positions", "first_pass_input_ids") else value)
              for key, value in record.items()}
    if isinstance(result.get("routes"), dict):
        # Each rank writes its own filename. Compare the exact route content
        # SHA and all other metadata, while the route loader independently
        # checks each rank's file identity and bytes. A filename is not a
        # producer-geometry disagreement.
        result["routes"] = {k: v for k, v in result["routes"].items() if k != "file"}
    return result


def bind_producers(rank0, rank1, scheduler):
    if len(rank0) != len(rank1) or any(canonical(a) != canonical(b) for a, b in zip(rank0, rank1, strict=True)):
        raise ValueError("The two TP ranks disagree on real draft ownership/geometry")
    history = defaultdict(list)
    all_slots = set()
    for ordinal, record in enumerate(rank0):
        if record["call"] != ordinal:
            raise ValueError("Draft producer call sequence has a gap or duplicate")
        wave = integer(record["wave"], 0, len(scheduler) - 1, "producer wave")
        requests = record["requests"]
        if len(requests) != len(set(requests)) or len(record["draft_token_ids"]) != len(requests):
            raise ValueError("Draft output rows do not bind unique live requests")
        if len(record["sample_indices"]) != len(requests):
            raise ValueError("A live draft sampling root is missing")
        for row, request in enumerate(requests):
            slot = (wave, request)
            if slot in all_slots:
                raise ValueError("Repeated draft producer in one request/wave")
            all_slots.add(slot)
            tokens = record["draft_token_ids"][row]
            if not 1 <= len(tokens) <= 2 or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("MTP2 producer must return its actual one/two live token IDs")
            sample = integer(record["sample_indices"][row], 0, record["actual_input_tokens"] - 1, "sampling row")
            history[request].append(dict(wave=wave, call=ordinal, row=row,
                                         tokens=tokens, sample_index=sample))
    used, bindings = set(), []
    finished, postcompletion_unretained = set(), 0
    for wave in scheduler:
        retained = {}
        for row in wave["retained"]:
            request = row["request"]
            if request in retained:
                raise ValueError("Repeated native retained-output row")
            retained[request] = row
        for row in wave["rows"]:
            count = integer(row["draft"], 0, 2, "scheduled width")
            if not count:
                continue
            request = row["request"]
            scheduled = row["draft_token_ids"]
            if len(scheduled) != count:
                raise ValueError("Scheduled draft width differs from its actual token IDs")
            prior = [p for p in history[request] if p["wave"] < wave["wave"]]
            if not prior:
                raise ValueError("Draft consumer has no prior producer for this request")
            producer = max(prior, key=lambda p: p["wave"])
            identity = (producer["wave"], request)
            if identity in used or scheduled != producer["tokens"][:count]:
                raise ValueError("Draft consumer reused or mismatched its latest actual producer")
            used.add(identity)
            if request not in retained:
                if request not in finished:
                    raise ValueError("Draft verification lacks a native retained-output record")
                postcompletion_unretained += 1
                continue
            kept = retained[request]
            raw = integer(kept["offered"], 0, count + 1, "raw verified outputs")
            actual = integer(kept["kept"], 0, raw, "retained outputs")
            if type(kept["stale"]) is not bool or len(kept["kept_token_ids"]) != actual:
                raise ValueError("Retained draft IDs or stale-output status are incomplete")
            accepted = 0 if kept["stale"] else min(max(0, raw - 1), actual)
            if kept["kept_token_ids"][:accepted] != scheduled[:accepted]:
                raise ValueError("Native retained draft IDs differ from their matched producer")
            bindings.append(dict(request=request, producer_wave=producer["wave"],
                producer_call=producer["call"], producer_row=producer["row"],
                consumer_wave=wave["wave"], scheduled_drafts=count,
                retained_drafts=accepted, sample_index=producer["sample_index"]))
        finished.update(r["request"] for r in wave["retained"] if r.get("stopped") is True)
    return dict(scope="current_actual_draft_producer_to_retained_token_join",
                postcompletion_unretained_rows=postcompletion_unretained,
                producer_request_slots=len(all_slots), consuming_rows=len(bindings),
                retained_first_drafts=sum(b["retained_drafts"] >= 1 for b in bindings),
                retained_second_drafts=sum(b["retained_drafts"] >= 2 for b in bindings),
                producer_slots_without_scheduled_consumer=len(all_slots - used),
                bindings=bindings, backward_state_closure_certified=False,
                full_model_certificate=False, effective_flops=None)
