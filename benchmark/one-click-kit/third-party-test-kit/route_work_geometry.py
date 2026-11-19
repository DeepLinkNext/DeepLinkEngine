"""Bind necessary target/draft MoE rows to observed per-token expert choices."""

from collections import defaultdict
import gzip
import hashlib
import json

from draft_state_closure import versions_from_calls


def target_selection(waves, requests, target):
    """Select the latest actual accepted-position producer, never all padding."""
    lengths, seen, final = {}, {}, {}
    for row in requests:
        key = row["request"]
        length = row["input_tokens"] + row["output_tokens"] - 1
        if key in lengths or type(length) is not int or length < 1:
            raise ValueError("Invalid target route population")
        lengths[key], seen[key], final[key] = length, bytearray(length), bytearray(length)
    for span in target["full_intervals"]:
        if (span["request"] not in lengths or not 0 <= span["start"] <= span["stop"] < lengths[span["request"]]):
            raise ValueError("A target full-output root lacks an accepted position")
        final[span["request"]][span["start"]:span["stop"] + 1] = b'\1' * (span["stop"] - span["start"] + 1)
    selected = {}
    for wave in reversed(waves):
        geometries = []
        for capture in wave["gdn"]:
            geometry = (capture["requests"], capture.get("target_query_cumulative_lengths"))
            if geometry not in geometries:
                geometries.append(geometry)
        if len(geometries) != 1 or geometries[0][1] is None:
            raise ValueError("Route rows require one observed actual target query layout")
        order, cu = geometries[0]
        rows = {row["request"]: row for row in wave["rows"]}
        if (set(order) != set(rows) or len(order) != len(rows) or len(cu) != len(order) + 1
                or cu[0] != 0 or any(type(n) is not int for n in cu)):
            raise ValueError("Target route layout ownership is incomplete")
        common, last = [], []
        for index, key in enumerate(order):
            row = rows[key]
            if key not in lengths or cu[index + 1] - cu[index] != row["count"]:
                raise ValueError("Target route query rows differ from actual scheduled rows")
            start, count = row["start"], row["count"]
            if type(start) is not int or start < 0 or type(count) is not int or count < 1:
                raise ValueError("Invalid target route position range")
            for offset, position in enumerate(range(start, min(start + count, lengths[key]))):
                if not seen[key][position]:
                    token_row = cu[index] + offset
                    common.append(token_row)
                    if final[key][position]:
                        last.append(token_row)
                    seen[key][position] = 1
        selected[wave["wave"]] = dict(prefix=sorted(common), final=sorted(last), actual_rows=cu[-1])
    if any(not all(values) for values in seen.values()):
        raise ValueError("A necessary target route producer is absent")
    if (sum(len(row["prefix"]) for row in selected.values()) != target["necessary_prefix_rows"]
            or sum(len(row["final"]) for row in selected.values()) != target["necessary_last_full_rows"]):
        raise ValueError("Target route populations do not match the hidden-state closure")
    return selected


def draft_selection(calls, closure):
    versions, _ = versions_from_calls(calls)
    selected = defaultdict(lambda: defaultdict(list))
    for index in closure["full_versions"]:
        if type(index) is not int or not 0 <= index < len(versions):
            raise ValueError("Missing real draft full-output route producer")
        row = versions[index]
        call = calls[row["call"]]
        token_row = call["sample_indices"][row["row"]] if row["step"] == 0 else row["row"]
        selected[row["call"]][row["step"]].append(token_row)
    if sum(len(rows) for steps in selected.values() for rows in steps.values()) != closure["necessary_full_rows"]:
        raise ValueError("Draft route rows do not match the backward-closed full population")
    for steps in selected.values():
        for step, rows in steps.items():
            if len(rows) != len(set(rows)):
                raise ValueError("A draft route producer is counted twice")
            steps[step] = sorted(rows)
    return dict(selected)


def collect(directory, window, ranks, drafts, requests, target, draft):
    """Write sealed shape catalogs; raw route IDs remain in bounded NPZ files."""
    import numpy as np

    selection = target_selection(ranks[0], requests, target)
    draft_rows = draft_selection(drafts[0], draft)
    summaries, signatures = [], []
    for rank in (0, 1):
        target_rows = draft_count = contexts = 0
        signature = hashlib.sha256()
        destination = directory / f"rank-{rank}-necessary-moe.jsonl.gz"
        with gzip.open(destination, "xt") as stream:
            def emit(capture, role, ordinal, selected, *, steps=None):
                nonlocal target_rows, draft_count, contexts
                if not capture or capture["role"] != role:
                    raise ValueError("Actual per-token route capture is missing")
                name = capture["file"]
                if '/' in name or '\\' in name:
                    raise ValueError("Foreign route artifact path")
                source = directory / name
                if hashlib.sha256(source.read_bytes()).hexdigest() != capture["sha256"]:
                    raise ValueError("Actual route artifact changed")
                layers = capture["layers"]
                expected = list(range(40)) if role == "target" else [40]
                if [row["index"] for row in layers] != expected:
                    raise ValueError("Actual route layer population differs")
                with np.load(source, allow_pickle=False) as arrays:
                    expected_arrays = {call["array"] for layer in layers for call in layer["calls"]}
                    if set(arrays.files) != expected_arrays:
                        raise ValueError("Route archive has foreign or missing arrays")
                    for layer in layers:
                        calls = layer["calls"]
                        if role == "target" and len(calls) != 1:
                            raise ValueError("Multiple target route microbatches need explicit layout binding")
                        if role == "draft" and len(calls) != steps:
                            raise ValueError("Actual route history differs from the merged draft step count")
                        for step, call in enumerate(calls):
                            values = arrays[call["array"]]
                            if (values.dtype != np.int32 or values.shape != (call["rows"], 8)
                                    or np.any(values < 0) or np.any(values >= 256)
                                    or np.any(np.diff(np.sort(values, axis=1), axis=1) == 0)):
                                raise ValueError("Invalid unique top8 source route vectors")
                            if role == "target" and len(values) < selected["actual_rows"]:
                                raise ValueError("Actual route output omits scheduled target rows")
                            ids = selected["final"] if role == "target" and layer["index"] == 39 else (
                                selected["prefix"] if role == "target" else selected.get(step, []))
                            if not ids:
                                continue
                            if max(ids) >= len(values):
                                raise ValueError("A necessary route row is outside its actual device output")
                            useful = np.ascontiguousarray(values[ids], dtype='<i4')
                            counts = np.bincount(useful.reshape(-1), minlength=256).tolist()
                            row = dict(window=window, role=role, ordinal=ordinal, layer=layer["index"], step=step,
                                rows=len(ids), top_k=8, expert_counts=counts,
                                selected_route_sha256=hashlib.sha256(useful.tobytes()).hexdigest())
                            encoded = json.dumps(row, sort_keys=True, separators=(',', ':'))
                            stream.write(encoded + '\n')
                            signature.update(encoded.encode())
                            contexts += 1
                            if role == "target": target_rows += len(ids)
                            else: draft_count += len(ids)
            for wave in ranks[rank]:
                emit(wave["routes"], "target", wave["wave"], selection[wave["wave"]])
            for call in drafts[rank]:
                emit(call["routes"], "draft", call["call"], draft_rows.get(call["call"], {}), steps=call["draft_width"])
        expected_target = 39 * target["necessary_prefix_rows"] + target["necessary_last_full_rows"]
        if target_rows != expected_target or draft_count != draft["necessary_full_rows"]:
            raise ValueError("Necessary expert work did not close across actual source routes")
        signatures.append(signature.hexdigest())
        summaries.append(dict(rank=rank, file=destination.name, sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            contexts=contexts, target_token_layers=target_rows, draft_token_layers=draft_count,
            expert_vector_signature=signature.hexdigest()))
    if signatures[0] != signatures[1]:
        raise ValueError("TP ranks disagree on necessary expert choices")
    return dict(scope="actual_necessary_moe_shape_catalog", ranks=summaries, peer_routes_match=True,
                full_model_certificate=False, effective_flops=None,
                source_runtime_applicability_certified=False)
