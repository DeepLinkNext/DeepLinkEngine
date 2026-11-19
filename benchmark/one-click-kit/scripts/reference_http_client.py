"""Versioned fixed200 source-domain capture; ordinary full-answer budgets stay frozen."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path

from dataset_contract import save, sha
from reference_capture_profile import profile


def requests_for_capture(original, capture):
    if (original["max_completion_tokens"] != 8192 or original["temperature"] != 0
            or original["seed"] != 0 or len(original["requests"]) != 200):
        raise ValueError("Original full-answer fixed200 source differs")
    requests = copy.deepcopy(original["requests"])
    for request in requests:
        if request["max_completion_tokens"] != 8192 or request["temperature"] != 0 or request["seed"] != 0:
            raise ValueError("Original full-answer request controls differ")
        request["max_completion_tokens"] = capture["max_tokens"]
    return requests


async def run(client, out, name):
    capture = profile(name)
    if name == "full-answer-v1":
        return await client.main(out)
    import aiohttp

    plan = json.loads((out / "plan.json").read_bytes())
    original = json.loads(Path(plan["quality_plan"]).read_bytes())
    verified = json.loads((client.ROOT / "verified-mmlu-quality-inputs.json").read_bytes())
    expected = json.loads((client.ROOT / "mmlu-quality018-r2/quality-tokenize-preflight.json").read_bytes())
    requests = requests_for_capture(original, capture)
    golds = verified["cases"]
    if len(golds) != 200 or len(expected) != 200:
        raise ValueError("Frozen fixed200 token/gold population differs")
    extract = client.scorer(Path(plan["scorer"]), original["scorer_sha256"])
    save(out / "request-plan.json", dict(source=plan["quality_plan"], source_sha256=sha(Path(plan["quality_plan"])),
        capture_profile=capture, observation_repetitions=1, request_count=200, request_rate="inf", warmups=0,
        requests=requests, cache_salts="unique output-directory/case-id", scorer_sha256=original["scorer_sha256"],
        source_domain_observation_only=True, native_counts_or_times_taken_from_HTTP=False,
        full_answer_quality_or_performance_eligible=False))
    timeout = aiohttp.ClientTimeout(total=6600, sock_read=6600)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=timeout, trust_env=False) as session:
        checked = []
        for index, request in enumerate(requests):
            if golds[index]["case_id"] != expected[index]["case_id"]:
                raise ValueError("Fixed200 input order differs")
            async with session.post(client.URL + "/tokenize", json=dict(model=request["model"], messages=request["messages"], add_generation_prompt=True)) as response:
                data = await response.json()
                if response.status != 200:
                    raise ValueError("Reference HTTP tokenizer rejected original input")
            digest = hashlib.sha256(json.dumps(data["tokens"]).encode()).hexdigest()
            if (data["count"], digest) != (expected[index]["count"], expected[index]["token_ids_sha256"]):
                raise ValueError("Original fixed200 prompt tokens differ")
            checked.append(dict(case_id=golds[index]["case_id"], count=data["count"], token_ids_sha256=digest))
        save(out / "tokenize-preflight.json", checked)
        start = client.clock()
        save(out / "phase-start.json", start)
        rows = await asyncio.gather(*(client.one(session, out, index, requests[index], golds[index], expected[index], extract) for index in range(200)))
        end = client.clock()
    if any(row["success"] and not 1 <= row["usage"]["completion_tokens"] <= capture["max_tokens"] for row in rows):
        raise ValueError("Observed output exceeds the declared reference capture budget")
    summary = dict(schema="fixed200-reference-source-domain-http/v2", role=plan["role"], capture_profile=capture,
        total=200, completed=sum(row["success"] for row in rows), failed=sum(not row["success"] for row in rows),
        start=start, end=end, duration_seconds=end["monotonic"] - start["monotonic"],
        successful_output_tokens=sum(row.get("usage", {}).get("completion_tokens", 0) for row in rows if row["success"]),
        source_domain_observation_only=True, timing_eligible_for_performance_comparison=False,
        full_answer_quality_eligible=False, efficiency_percent=None)
    save(out / "joint.json", summary)
    return int(summary["failed"] != 0)
