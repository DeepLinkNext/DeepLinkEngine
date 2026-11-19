"""Join live scheduler and rank records and derive the covered Qwen3.5 work."""

from collections import Counter, defaultdict
import json

from native_m01_worker import key
from reference_capture_worker import canonical_sha, write


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


def collect(directory, window, counts, output_ids, outputs, model):
    rank0, rank1 = (read_waves(directory / f"rank-{rank}-waves.jsonl", window, counts[rank])
                    for rank in (0, 1))
    scheduler = read_waves(directory / "scheduler-waves.jsonl", window, counts["scheduler"])
    if not scheduler:
        raise ValueError("No scheduled model work recorded")
    worker_join = len(rank0) == len(rank1) == len(scheduler) and all(
        left["rows"] == right["rows"] for left, right in zip(rank0, rank1)
    ) and all(
        sorted((row["request"], row["start"], row["count"], row["draft"]) for row in actual["rows"]) ==
        sorted((row["request"], row["start"], row["count"], row["draft"]) for row in host["rows"])
        for actual, host in zip(rank0, scheduler)
    )
    requests = {}
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
    result = dict(scope="partial_target_matrix_semantic_demand", matrix_components=components,
                  projection_coefficients=coefficients, retained_output_tokens=sum(kept.values()),
                  necessary_target_rows=rows, causal_pairs=pairs, worker_schedule_matches=worker_join,
                  requests=coverage, gdn_metadata_calls_by_rank=gdn_calls, effective_flops=None,
                  remaining=["GDN prefill and recurrence", "scalar and convolution arithmetic",
                             "retained draft producer dependencies", "certified reference operator bindings"])
    request_ordinals = {identity: index for index, identity in enumerate(requests)}
    result["schedule_signature"] = canonical_sha([
        sorted((request_ordinals[row["request"]], row["start"], row["count"], row["draft"])
               for row in wave["rows"]) for wave in scheduler
    ])
    write(directory / "necessary-work-coverage.json", result)
    return result
