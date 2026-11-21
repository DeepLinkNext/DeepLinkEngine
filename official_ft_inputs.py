"""Bind official async placeholders to independently observed device producers.

The strict existing producer join remains downstream. Unlike the optimized
engine's special second-step length publication, these source families expose
stable upstream seq_lens/seq_lens_list arguments for both merged MTP steps.
"""
import copy


def stable_lengths(record):
    before, after = record["length_planes_before"], record["length_planes_after"]
    if before != after or [p["step"] for p in before] != [0, 1]:
        raise ValueError("Official MTP kernel length arguments changed or are incomplete")
    result = copy.deepcopy(record)
    result["first_pass_kv_extent_contract"] = "full_query_envelope"
    return result


def resolve_records(scheduler, rank0, rank1, drafts0, drafts1):
    from draft_producer_geometry import canonical

    streams = (scheduler, rank0, rank1, drafts0, drafts1)
    if not scheduler or any(len(s) != len(scheduler) for s in streams):
        raise ValueError("Async device input streams have different wave counts")
    derived = [copy.deepcopy(s) for s in streams]
    latest, finished, changes = {}, set(), []
    for wave, (host, left, right, draft, peer) in enumerate(zip(*streams, strict=True)):
        if (any(r["wave"] != wave or r["window"] != host["window"] for r in (host, left, right, draft, peer))
                or canonical(draft) != canonical(peer)
                or draft["call"] != wave or peer["call"] != wave):
            raise ValueError("Async device producer ranks/window/order disagree")
        derived[3][wave] = stable_lengths(draft)
        derived[4][wave] = stable_lengths(peer)
        rows = [{r["request"]: r for r in stream["rows"]} for stream in (host, left, right)]
        if (any(len(m) != len(s["rows"]) for m, s in zip(rows, (host, left, right), strict=True))
                or rows[0] != rows[1] or rows[0] != rows[2]
                or set(rows[0]) != set(draft["requests"])
                or len(draft["requests"]) != len(set(draft["requests"]))):
            raise ValueError("Async host and device request ownership disagree")
        planes = [p for p in draft["length_planes_before"] if p["step"] == 0]
        if len(planes) != 1:
            raise ValueError("Async device length planes are ambiguous")
        ends = planes[0]["query_end_envelope"]
        retained = {r["request"]: r for r in host["retained"]}
        if len(retained) != len(host["retained"]):
            raise ValueError("Repeated async retained output")
        updates, begin = {}, 0
        for i, request in enumerate(draft["requests"]):
            row = rows[0][request]
            end, sample = ends[i], draft["sample_indices"][i]
            positions = draft["first_pass_positions"][begin:end]
            if (type(end) is not int or type(sample) is not int
                    or end - begin != row["count"] or not begin <= sample < end
                    or not positions or positions != list(range(positions[0], positions[0] + row["count"]))):
                raise ValueError("Device query segment cannot bind the scheduled request")
            start = positions[0]
            prior, kept = latest.get(request), retained.get(request)
            if prior and prior["position"] + 1 != start:
                raise ValueError("Async device positions do not continue the previous live producer")
            count, scheduled = row["draft"], row["draft_token_ids"]
            if type(count) is not int or not 0 <= count <= 2 or len(scheduled) != count:
                raise ValueError("Invalid async speculative width")
            actual = scheduled
            if count:
                if prior is None or prior["wave"] != wave - 1:
                    raise ValueError("Async consumer lacks its immediately previous device producer")
                actual = prior["tokens"][:count]
                if len(actual) != count or any(type(t) is not int or t < 0 for t in actual):
                    raise ValueError("Incomplete previous live device output")
                if scheduled != [-1] * count and scheduled != actual:
                    raise ValueError("Concrete or mixed async draft IDs disagree with the device producer")
                if kept is None:
                    if request not in finished:
                        raise ValueError("Async request lacks retained output before completion")
                else:
                    offered, n = kept["offered"], kept["kept"]
                    if (type(offered) is not int or not 0 <= offered <= count + 1
                            or type(n) is not int or not 0 <= n <= offered
                            or type(kept["stale"]) is not bool
                            or len(kept["kept_token_ids"]) != n
                            or sample - begin + 1 != offered):
                        raise ValueError("Device sampling position and native verified output disagree")
                    accepted = 0 if kept["stale"] else min(max(0, offered - 1), n)
                    if (kept["kept_token_ids"][:accepted] != actual[:accepted]
                            or draft["first_pass_input_ids"][begin:begin+accepted] != actual[:accepted]):
                        raise ValueError("Independent current device and retained draft prefixes disagree")
            if row["start"] != start or actual != scheduled:
                changes.append(dict(wave=wave, request=request, observed_start=row["start"],
                    device_start=start, observed_draft_ids=scheduled, device_producer_ids=actual,
                    producer_wave=None if prior is None else prior["wave"],
                    retained_output_absent_after_completed=kept is None and request in finished))
            updates[request] = dict(start=start, draft_token_ids=actual)
            latest[request] = dict(wave=wave, position=draft["first_pass_positions"][sample],
                                   tokens=draft["draft_token_ids"][i])
            begin = end
        if begin != draft["actual_input_tokens"]:
            raise ValueError("Device request segments do not cover exactly the actual live rows")
        for stream in derived[:3]:
            for row in stream[wave]["rows"]:
                row.update(updates[row["request"]])
        finished.update(r["request"] for r in host["retained"] if r["stopped"])
    return derived, changes

