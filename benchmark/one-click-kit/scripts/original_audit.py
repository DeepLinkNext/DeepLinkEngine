"""Unchanged pure audit functions from the pinned public benchmark."""

import gzip
import hashlib
import json
import math


def audit_activation(files, prefix="retrieval-b1-r1/", expected_requests=7):

    def rows(name):
        return [
            json.loads(line)
            for line in files[prefix + name].splitlines()
            if line.strip()
        ]

    policy = rows("kvmat-policy.jsonl")
    engine = rows("kvmat-engine.jsonl")
    if len(policy) != expected_requests or len(engine) != expected_requests:
        raise ValueError("Unexpected number of request admissions")
    salts = {}
    for row in policy:
        if (
            row.get("runtime_effective_decision") != "recompute"
            or row.get("runtime_control_path") != "request_scoped_prefix_cache_bypass"
            or row.get("reused_tokens") != 0
            or row.get("runtime_decision_supported") is not True
        ):
            raise ValueError("Unexpected KV action")
        digest = hashlib.sha256(row["runtime_cache_salt"].encode()).hexdigest()
        if digest in salts:
            raise ValueError("Duplicate request salt")
        salts[digest] = row["prompt_tokens"]
    admitted = {
        r["cache_salt_sha256"]: r["prompt_tokens"]
        for r in engine
        if r["event"] == "school028_request_salt_admitted"
    }
    if len(admitted) != expected_requests or admitted != salts:
        raise ValueError("Policy/engine linkage mismatch")
    ranks = {}
    for path in files:
        if not path.startswith(prefix + "microflow-") or not path.endswith(".jsonl"):
            continue
        events = [json.loads(line) for line in files[path].splitlines() if line.strip()]
        loaded = [r for r in events if r.get("event") == "model_loaded"]
        if len(loaded) != 1 or loaded[0].get("active") is not True:
            raise ValueError("Inactive Microflow rank")
        rank = loaded[0]["rank"]
        if rank not in (0, 1) or rank in ranks:
            raise ValueError("Duplicate or unexpected rank")
        equivalence = [r for r in events if r.get("event") == "partition_equivalence"]
        if {r["tokens"] for r in equivalence} != {1024, 2048, 4096} or any(
            r.get("exact") != [True] * 5 or r.get("rank") != rank for r in equivalence
        ):
            raise ValueError("Microflow partition qualification failed")
        split = sum(
            r.get("event") == "prefix" and r.get("split") is True for r in events
        )
        replay = sum(
            r.get("event") == "graph_replay"
            and r.get("active") is True
            and r.get("target") is True
            and r.get("speculative_enabled") is True
            for r in events
        )
        if not split or not replay:
            raise ValueError("Microflow not exercised")
        ranks[str(rank)] = dict(split_events=split, target_mtp_graph_replays=replay)
    if set(ranks) != {"0", "1"}:
        raise ValueError("Both ranks required")
    identity = json.loads(files[prefix + "runtime-identity.json"])
    if identity.get("worker") != "corrected_serving_runtime.CorrectedWorker":
        raise ValueError("Serving corrections missing")
    return dict(
        request_scoped_recompute_admissions=expected_requests,
        reused_tokens=0,
        microflow=ranks,
        corrected_worker=identity["worker"],
        scope="Actual activation and request admission, not general state-isolation proof",
    )


def reconstruct(raw):
    first = last = usage = finish = None
    done = False
    prior = -math.inf
    for line in gzip.decompress(raw).splitlines():
        saved = json.loads(line)
        stamp = saved["clock"]["monotonic"]
        if stamp < prior:
            raise ValueError("Non-monotonic stream")
        prior = stamp
        wire = saved["raw"]
        if not wire.startswith("data:"):
            continue
        payload = wire[5:].strip()
        if payload == "[DONE]":
            done = True
            continue
        event = json.loads(payload)
        if event.get("error"):
            raise ValueError("Recorded stream error")
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if any(delta.get(k) for k in ("content", "reasoning", "reasoning_content")):
                first = stamp if first is None else first
                last = stamp
            finish = choice.get("finish_reason") or finish
    if not done or first is None or not usage or not finish:
        raise ValueError("Incomplete recorded stream")
    return first, last, usage, finish
