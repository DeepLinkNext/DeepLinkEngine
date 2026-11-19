"""Join live scheduler and rank records and derive the covered Qwen3.5 work."""

from collections import Counter, defaultdict
import json

from m01_observer_worker import key  # noqa: F401 -- retain the upstream import
from m01_proc_io import canonical_sha, write
from gdn_work_geometry import target_prefill_work
from draft_producer_geometry import bind_producers
from draft_state_closure import close as close_draft_states, matrix_work as draft_matrix_work
from semantic_work import candidate as nonmatrix_candidate
from target_state_closure import close as close_target_states, matrix_components as target_matrices
from route_work_geometry import collect as collect_routes


def read_waves(path, window, expected):
    with path.open(encoding="utf-8") as stream:
        waves = [json.loads(line) for line in stream]
    if len(waves) != expected or any(row["window"] != window or row["wave"] != i
                                     for i, row in enumerate(waves)):
        raise ValueError(f"Incomplete or foreign engine waves: {path.name}")
    return waves


def projection_coefficients(c):
    h, layers = c.hidden_size, c.num_hidden_layers
    full = c.layer_types.count("full_attention")
    linear = c.layer_types.count("linear_attention")
    if full + linear != layers or c.attention_bias or not c.attn_output_gate:
        raise ValueError("Unsupported Qwen3.5 projection architecture")
    q = c.num_attention_heads * c.head_dim
    kv = c.num_key_value_heads * c.head_dim
    lk = c.linear_num_key_heads * c.linear_key_head_dim
    lv = c.linear_num_value_heads * c.linear_value_head_dim
    shared = c.shared_expert_intermediate_size
    return {
        "target_qkv_projection": 2 * h * (2 * q + 2 * kv) * full,
        "target_attention_output": 2 * q * h * full,
        "target_shared_expert": 6 * h * shared * layers,
        "target_router": 2 * h * (c.num_experts + 1) * layers,
        "target_gdn_projection": (2 * h * (2 * lk + 3 * lv + 2 * c.linear_num_value_heads)) * linear,
        "target_active_expert": 6 * h * c.moe_intermediate_size * c.num_experts_per_tok * layers,
    }


def collect(directory, window, counts, output_ids, outputs, model, *, request_counts=None, draft_counts=None):
    rank0, rank1 = (read_waves(directory / f"rank-{rank}-waves.jsonl", window, counts[rank])
                    for rank in (0, 1))
    scheduler = read_waves(directory / "scheduler-waves.jsonl", window, counts["scheduler"])
    if not scheduler:
        raise ValueError("No scheduled model work recorded")
    worker_join = len(rank0) == len(rank1) == len(scheduler) and all(
        left["rows"] == right["rows"] for left, right in zip(rank0, rank1)
    ) and all(
        sorted((row["request"], row["start"], row["count"], row["draft"], tuple(row["draft_token_ids"])) for row in actual["rows"]) ==
        sorted((row["request"], row["start"], row["count"], row["draft"], tuple(row["draft_token_ids"])) for row in host["rows"])
        for actual, host in zip(rank0, scheduler)
    )
    requests = dict(request_counts or {})
    if request_counts is not None and (outputs or any(
            len(pair) != 2 or any(type(n) is not int or n < 1 for n in pair)
            for pair in requests.values())):
        raise ValueError("Positive counts and an unambiguous HTTP/native source are required")
    for response in outputs:
        identity = output_ids.get(response.request_id)
        if identity is None or identity in requests:
            raise ValueError("Native request ID does not bind uniquely to the returned output")
        requests[identity] = (len(response.prompt_token_ids), len(response.outputs[0].token_ids))
    intervals = defaultdict(list)
    kept = Counter()
    draft_slots = Counter()
    for wave in scheduler:
        for row in wave["rows"]:
            identity = row["request"]
            if identity not in requests:
                raise ValueError("A scheduled request is absent from current output")
            intervals[identity].append((row["start"], row["start"] + row["count"]))
            draft_slots[identity] += row["draft"]
        for row in wave["retained"]:
            if row["request"] not in requests or row["kept"] > row["offered"]:
                raise ValueError("Unbound or inconsistent retained-output record")
            if not row["stale"]:
                kept[row["request"]] += row["kept"]
    coverage = []
    for identity, (prompt, generated) in requests.items():
        length = prompt + generated - 1
        accounted = 0
        for start, end in sorted(intervals[identity]):
            if start > accounted:
                break
            accounted = max(accounted, end)
        coverage.append(dict(request=identity, input_tokens=prompt, output_tokens=generated,
                             retained=kept[identity], scheduled_drafts=draft_slots[identity],
                             covered=accounted >= length))
    if not worker_join or any(row["retained"] != row["output_tokens"] or not row["covered"] for row in coverage):
        write(directory / "necessary-work-coverage.json", dict(worker_schedule_matches=worker_join, requests=coverage))
        raise ValueError("Native rank, scheduler and retained-output accounting did not close")
    coefficients = projection_coefficients(model)
    lengths = [prompt + generated - 1 for prompt, generated in requests.values()]
    rows = sum(lengths)
    pairs = sum(length * (length + 1) // 2 for length in lengths)
    full = model.layer_types.count("full_attention")
    components = {name: value * rows for name, value in coefficients.items()}
    components["target_causal_qk_av"] = 4 * model.num_attention_heads * model.head_dim * full * pairs
    components["target_vocabulary"] = 2 * model.hidden_size * model.vocab_size * sum(kept.values())
    gdn_calls = [sum(len(wave["gdn"]) for wave in rank) for rank in (rank0, rank1)]
    request_edges, last = [], {}
    for wave in scheduler:
        for row in wave["rows"]:
            identity = row["request"]
            if identity in last:
                request_edges.append(dict(request=identity, predecessor=last[identity], successor=wave["wave"]))
            last[identity] = wave["wave"]
    write(directory / "request-wave-order.json", dict(window=window, edges=request_edges,
          meaning="Per-request scheduler order; device state and communication dependencies require separate evidence"))
    result = dict(scope="partial_current_source_work", matrix_components=components,
                  projection_coefficients=coefficients, retained_output_tokens=sum(kept.values()),
                  necessary_target_rows=rows, causal_pairs=pairs, worker_schedule_matches=worker_join,
                  requests=coverage, gdn_metadata_calls_by_rank=gdn_calls, effective_flops=None,
                  remaining=["GDN current source/runtime applicability", "scalar and convolution applicability",
                             "retained target/draft producer dependency certification", "certified reference operator bindings"])
    if any(left["gdn"] != right["gdn"] for left, right in zip(rank0, rank1, strict=True)):
        raise ValueError("The two TP ranks disagree on current GDN host geometry")
    gdn_layers = sorted({name for wave in rank0 for capture in wave["gdn"]
                         for group in capture["groups"] for name in group["layers"]})
    if len(gdn_layers) != model.layer_types.count("linear_attention"):
        raise ValueError("The current model GDN layer population is incomplete")
    prefill = target_prefill_work(rank0, coverage, expected_layers=gdn_layers,
                                 global_heads=model.linear_num_value_heads)
    write(directory / "gdn-prefill-work.json", prefill)
    result["gdn_prefill_work"] = dict(total_basic=prefill["total_basic"],
                                     segments=len(prefill["segments"]),
                                     request_layer_coverage_verified=True)
    result["gdn_recurrence_candidate"] = prefill["recurrence_candidate"]
    if draft_counts is None or set(draft_counts) != {0, 1}:
        raise ValueError("Both worker-acknowledged merged draft call counts are required")
    drafts = []
    for rank in (0, 1):
        with (directory / f"rank-{rank}-drafts.jsonl").open() as stream:
            records = [json.loads(line) for line in stream]
        if (len(records) != draft_counts[rank] or any(r["window"] != window for r in records)):
            raise ValueError("Draft raw records are truncated or belong to another window")
        if any(r.get("target_hidden_input_contract") != "unreindexed_same_position_first_pass" for r in records):
            raise ValueError("Draft target-hidden producer contract was not observed")
        drafts.append(records)
    producers = bind_producers(*drafts, scheduler)
    write(directory / "draft-producer-bindings.json", producers)
    result["draft_producer_bindings"] = {key: value for key, value in producers.items() if key != "bindings"}
    states = close_draft_states(drafts[0], producers)
    states["matrix_work"] = draft_matrix_work(model, states)
    write(directory / "draft-state-closure.json", states)
    target = close_target_states(coverage, states)
    write(directory / "target-state-closure.json", target)
    components.update(target_matrices(model, target))
    components["target_causal_qk_av"] = 4 * model.num_attention_heads * model.head_dim * (
        (full - 1) * pairs + target["necessary_last_full_causal_pairs"])
    result["target_state_geometry"] = {key: value for key, value in target.items() if key != "full_intervals"}
    scalar = nonmatrix_candidate(model, coverage, states, target)
    write(directory / "nonmatrix-candidate.json", scalar)
    result["nonmatrix_candidate"] = dict(total_basic=scalar["total_basic"],
                                         source_runtime_applicability_certified=False,
                                         full_model_certificate=False)
    if all(row.get("routes") is not None for rank in (rank0, rank1) for row in rank):
        result["moe_routes"] = collect_routes(directory, window, (rank0, rank1), drafts,
                                              coverage, target, states)
    result["draft_state_geometry"] = {key: value for key, value in states.items()
                                     if key not in ("kv_intervals", "full_versions", "vocabulary_versions")}
    request_ordinals = {identity: index for index, identity in enumerate(requests)}
    result["schedule_signature"] = canonical_sha([
        sorted((request_ordinals[row["request"]], row["start"], row["count"], row["draft"])
               for row in wave["rows"]) for wave in scheduler
    ])
    write(directory / "necessary-work-coverage.json", result)
    return result
