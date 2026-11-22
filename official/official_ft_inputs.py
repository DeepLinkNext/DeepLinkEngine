"""Bind official async placeholders to independently observed device producers.

The strict existing producer join remains downstream. Unlike the optimized
engine's special second-step length publication, these source families expose
stable upstream seq_lens/seq_lens_list arguments for both merged MTP steps.
"""
import copy
from collections import Counter


ENVELOPE_VERSIONS = frozenset({"0.23.0", "0.30.0"})


class OfficialDraftPrefixMismatch(ValueError):
    """A reproducible source-admission failure, not a rewritten length plane."""
    def __init__(self, counterexample):
        self.counterexample = counterexample
        super().__init__('Observed official second-step KV extent is outside the selected-prefix producer closure'
                         f" (call={counterexample['call']}, row={counterexample['row']},"
                         f" expected={counterexample['expected_selected_prefix_kv_extent']},"
                         f" observed={counterexample['observed_step1_kv_extent']})")


class OfficialDraftEnvelopeUnresolved(OfficialDraftPrefixMismatch):
    """A known source layout with a remaining dependency proof obligation."""

    def __init__(self, audit):
        self.envelope_audit = audit
        super().__init__(audit["first_full_query_advance"])
        self.args = (f"{self.args[0]}. Official MTP full-query advance requires "
                     + audit["remaining_evidence"].replace("_", " "),)


def length_rows(record):
    before, after = record["length_planes_before"], record["length_planes_after"]
    if before != after or [p["step"] for p in before] != [0, 1]:
        raise ValueError("Official MTP kernel length arguments changed or are incomplete")
    batch = len(record["requests"])
    if (len(record["sample_indices"]) != batch
            or any(len(p["actual_kv_lengths"]) != batch or len(p["query_end_envelope"]) < batch
                   for p in before)):
        raise ValueError("Official MTP length arguments omit live request rows")
    begin = 0
    for row, request in enumerate(record['requests']):
        sample = record['sample_indices'][row]
        stop = before[0]['query_end_envelope'][row]
        positions = record['first_pass_positions']
        if (type(sample) is not int or type(stop) is not int
                or not begin <= sample < stop <= record['actual_input_tokens']
                or stop > len(positions)):
            raise ValueError('Official draft sampling/query ownership is invalid')
        selected = positions[sample]
        if type(selected) is not int or selected < 0:
            raise ValueError('Official draft selected position is invalid')
        yield dict(window=record['window'], call=record['call'],
                wave=record['wave'], row=row, request=request, begin=begin,
                sample_index=sample, query_stop=stop, selected_position=selected,
                first_pass_query_positions=positions[begin:stop],
                observed_step0_kv_extent=before[0]['actual_kv_lengths'][row],
                observed_step1_kv_extent=before[1]['actual_kv_lengths'][row],
                expected_selected_prefix_kv_extent=selected + 2,
                observed_step0_query_end=stop,
                observed_step1_query_end=before[1]['query_end_envelope'][row],
                source_derived_second_step_write_position=selected + 1,
                actual_slot_values_or_kernel_access_trace_observed=False,
                raw_length_planes_modified=False)
        begin = stop
    if begin != record["actual_input_tokens"]:
        raise ValueError("Official draft query envelopes omit live tokens")


def stable_lengths(record, *, logical_projection=False):
    result = copy.deepcopy(record)
    for row in length_rows(record):
        observed = row["observed_step1_kv_extent"]
        expected = row["expected_selected_prefix_kv_extent"]
        if observed == expected:
            continue
        if not logical_projection:
            raise OfficialDraftPrefixMismatch(row)
        # Only for reachability analysis after classify_envelopes. Never use
        # a logical projection as physical metadata or in the numerical ledger.
        for boundary in ("length_planes_before", "length_planes_after"):
            plane = result[boundary][1]
            plane["actual_kv_lengths"][row["row"]] = expected
            plane["kv_length_envelope"][row["row"]] = expected
    result["first_pass_kv_extent_contract"] = "full_query_envelope"
    return result


def classify_envelopes(records):
    """Distinguish the official full-query advance from unsupported geometry."""
    counts = Counter()
    widened = {}
    for record in records:
        for row in length_rows(record):
            positions = row["first_pass_query_positions"]
            end = positions[-1]
            selected = row["selected_position"]
            first = row["observed_step0_kv_extent"]
            second = row["observed_step1_kv_extent"]
            query_end = row["observed_step1_query_end"]
            if (any(type(p) is not int or p < 0 for p in positions)
                    or positions != list(range(positions[0], end + 1))
                    or type(first) is not int or first != end + 1
                    or type(second) is not int or type(query_end) is not int or query_end != row["row"] + 1
                    or not 0 <= end - selected <= 2):
                raise ValueError("Unsupported official MTP query/KV geometry")
            if second == selected + 2:
                counts["selected_prefix"] += 1
            elif second == first + 1:
                counts["full_query_advance"] += 1
                key = row["call"], row["request"]
                if key in widened:
                    raise ValueError("Repeated widened KV request in one draft call")
                widened[key] = dict(
                    row, extra_reported_positions=list(range(selected + 2, second)),
                    final_position_without_current_call_writer=end + 1,
                )
            else:
                raise OfficialDraftPrefixMismatch(row)
    return counts, widened


def envelope_dependencies(streams, widened):
    """Find widened attention calls in the retained-output dependency graph."""
    from draft_producer_geometry import bind_producers
    from draft_state_closure import backward, versions_from_calls

    projected, _ = resolve_records(*streams, logical_projection=True)
    scheduler, _, _, rank0, rank1 = projected
    bindings = bind_producers(rank0, rank1, scheduler)
    versions, indices = versions_from_calls(rank0)
    roots = {
        indices[binding["producer_call"], binding["request"], step]
        for binding in bindings["bindings"]
        for step in range(binding["retained_drafts"])
    }
    closure = backward(versions, roots)
    widened_full = []
    for index in closure["full_versions"]:
        version = versions[index]
        key = version["call"], version["request"]
        if version["step"] == 1 and key in widened:
            widened_full.append(widened[key])
    direct = sum(
        versions[index]["step"] == 1
        and (versions[index]["call"], versions[index]["request"]) in widened
        for index in roots
    )
    return dict(
        retained_logits=len(roots),
        direct_widened_roots=direct,
        widened_full_attention_calls=len(widened_full),
        widened_calls_not_needed_for_retained_output=len(widened) - len(widened_full),
        first_widened_full_attention_call=next(iter(widened_full), None),
        scope="reachability_in_logical_projection_only",
    )


def resolve_records(scheduler, rank0, rank1, drafts0, drafts1, *, logical_projection=False):
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
        derived[3][wave] = stable_lengths(draft, logical_projection=logical_projection)
        derived[4][wave] = stable_lengths(peer, logical_projection=logical_projection)
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

