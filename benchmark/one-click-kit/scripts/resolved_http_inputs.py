"""Resolve async host placeholders using source-qualified live device inputs.

This CPU derivation never edits capture streams, runs a model, or creates token
IDs. It records both observed and derived fields and keeps the strict producer
join downstream. Device first-pass segments independently verify retained IDs.
"""

import copy
import json
from pathlib import Path
import sys

from dataset_contract import save, sha

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/async-http-device-input-source-domain-v1.json"
PROTOCOL_SHA256 = "67cae5dfef4c71315be3ad38a1bd48dbd1991768f417c72a391e2c77a70bb1b5"


def qualify(raw, deployment):
    raw, deployment = Path(raw), Path(deployment).resolve()
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Reviewed async device input source domain changed")
    protocol = json.loads(PROTOCOL.read_bytes())
    installed = json.loads((deployment / "provenance/installed-files.json").read_bytes())
    for name, pin in protocol["sources"].items():
        if installed.get(name) != pin or sha(deployment / name) != pin:
            raise ValueError("Async input implementation differs: " + name)
    begin = json.loads((raw / "capture-begin.json").read_bytes())
    source = begin["source"]
    if (source["engine_core_class"] != protocol["engine_core_class"]
            or source["scheduler"] != protocol["scheduler_class"]
            or source["engine_core_mode"] != "proc"
            or source["state_cache_hook_present"] is not True):
        raise ValueError("Unreviewed async scheduler observation")

    def observed(path, pin):
        path = Path(path).resolve()
        return (path.is_relative_to(deployment)
            and protocol["sources"].get(str(path.relative_to(deployment))) == pin)

    if not observed(source["scheduler_source"], source["scheduler_sha256"]):
        raise ValueError("Captured scheduler differs from the qualified source")
    if {w["rank"] for w in begin["worker_sources"]} != {0, 1}:
        raise ValueError("Both captured worker sources are required")
    for worker in begin["worker_sources"]:
        if (worker["helper_sha256"] != protocol["worker_helper_sha256"]
                or worker["sources"].get("corrected_serving_runtime.CorrectedWorker")
                   != protocol["sources"]["mods/correction/corrected_serving_runtime.py"]
                or worker["rpc_wrapper"] != "corrected_serving_runtime.CorrectedWorker"
                or worker["active"] is not False or worker["poisoned"] is not False
                or any(not observed(x["path"], x["sha256"])
                       for x in worker["effective_method_sources"].values())):
            raise ValueError("Captured async worker source differs")
    for rank in (0, 1):
        proposer = json.loads((raw / f"rank-{rank}-proposer-source.json").read_bytes())
        if (proposer["window"] != begin["window"] or proposer["rank"] != rank
                or proposer["target_hidden_input_contract"] != "unreindexed_same_position_first_pass"
                or proposer["needs_extra_input_slots"] is not False
                or proposer["pass_hidden_states_to_model"] is not True
                or proposer["context_parallel_reindexing"] is not False
                or set(proposer["methods"]) != set(protocol["proposer_methods"])):
            raise ValueError("Unreviewed captured MTP first-pass input contract")
        for method, expected in protocol["proposer_methods"].items():
            actual = proposer["methods"][method]
            if (actual["sha256"] != expected["sha256"]
                    or Path(actual["file"]).resolve() != deployment / expected["relative"]
                    or actual["qualified_name"] != expected["qualified_name"]):
                raise ValueError("Captured MTP method differs: " + method)
    return dict(protocol_sha256=PROTOCOL_SHA256, source_sha256=protocol["sources"],
                captured_window=begin["window"], model_inference_executed=False)


def resolve_length_publication(record):
    """Bind the source-ordered second-step publication to both device reads."""
    before, after = record["length_planes_before"], record["length_planes_after"]
    if (len(before) != 2 or len(after) != 2
            or [p["step"] for p in before] != [0, 1]
            or [p["step"] for p in after] != [0, 1]):
        raise ValueError("Causal MTP2 length publication requires both unique steps")
    expected = copy.deepcopy(before)
    expected[1]["actual_kv_lengths"] = after[1]["actual_kv_lengths"]
    if expected != after:
        raise ValueError("Unqualified device length/envelope mutation across the merged call")
    batch = len(record["requests"])
    if any(len(p["actual_kv_lengths"]) != batch for p in before + after):
        raise ValueError("Incomplete device length publication")
    for i, sample in enumerate(record["sample_indices"]):
        query_end = before[0]["query_end_envelope"][i]
        extent = record["first_pass_positions"][query_end - 1] + 1
        selected_extent = record["first_pass_positions"][sample] + 2
        if (before[0]["actual_kv_lengths"][i] != extent
                or before[1]["actual_kv_lengths"][i] != extent + 1
                or after[1]["actual_kv_lengths"][i] != selected_extent
                or not 0 <= extent + 1 - selected_extent <= 2):
            raise ValueError("Observed device lengths do not match the frozen causal publication")
    result = copy.deepcopy(record)
    result["length_planes_before"] = copy.deepcopy(after)
    result["first_pass_kv_extent_contract"] = "full_query_envelope"
    return result


def resolve_records(scheduler, rank0, rank1, drafts0, drafts1):
    sys.path.insert(0, str(ROOT / "third-party-test-kit"))
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
        derived[3][wave] = resolve_length_publication(draft)
        derived[4][wave] = resolve_length_publication(peer)
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


def prepare(raw, deployment, view):
    raw, view = Path(raw).resolve(), Path(view)
    source = qualify(raw, deployment)
    names = ["scheduler-waves.jsonl", "rank-0-waves.jsonl", "rank-1-waves.jsonl",
             "rank-0-drafts.jsonl", "rank-1-drafts.jsonl"]
    def lines(name):
        with (raw / name).open() as stream:
            return [json.loads(line) for line in stream]
    streams, changes = resolve_records(*(lines(n) for n in names))
    view.mkdir(exist_ok=False)
    for name, rows in zip(names, streams, strict=True):
        with (view / name).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    for path in raw.iterdir():
        if path.is_file() and path.name.startswith(("rank-0-", "rank-1-")) and path.name.endswith("-routes.npz"):
            (view / path.name).symlink_to(path)
    proof = dict(schema="source-qualified-async-http-input-derivation/v1", completed=True,
        original_streams_changed=False, source=source,
        raw_input_sha256={n: sha(raw / n) for n in names},
        resolved_wave_sha256={n: sha(view / n) for n in names},
        causal_length_publication=dict(policy="selected-position-second-draft-length/v1",
            original_first_pass_planes_preserved=True, source_order_qualified=True,
            changed_calls=sum(d["length_planes_before"] != d["length_planes_after"] for d in (lines(names[3])))),
        changed_request_rows=len(changes),
        corrected_host_starts=sum(r["observed_start"] != r["device_start"] for r in changes),
        resolved_placeholder_consumers=sum(r["observed_draft_ids"] != r["device_producer_ids"] for r in changes),
        changes=changes, full_model_certificate=False)
    save(view / "async-input-derivation.json", proof)
    return proof


def remove_input_aliases(view):
    for path in Path(view).iterdir():
        if path.is_symlink():
            path.unlink()
