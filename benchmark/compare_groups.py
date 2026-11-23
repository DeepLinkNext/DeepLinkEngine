"""Read completed group receipts and score the original fixed 200 MMLU prompts."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re

from benchmark.unified_run import canonical_mmlu, load


def scoring(deployment):
    inputs = deployment / "inputs"
    raw = canonical_mmlu(inputs)
    protocol = load(Path(__file__).resolve().parents[1] / "datasets/mmlu-fixed200.json")
    for name in ("inputs/verified-mmlu-quality-inputs.json", "scorers/mmlu_pro_pyramidkv.py",
                 "inputs/mmlu-quality018-r2/quality-request-plan.json"):
        if hashlib.sha256((deployment / name).read_bytes()).hexdigest() != protocol["frozen_files_sha256"][name]:
            raise ValueError("Frozen scoring input differs: " + name)
    client = deployment / "scorers/mmlu_pro_pyramidkv.py"
    nodes = [node for node in ast.parse(client.read_bytes()).body
             if isinstance(node, ast.FunctionDef) and node.name == "extract_answer"]
    if len(nodes) != 1:
        raise ValueError("Ambiguous answer extractor")
    namespace = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(client), "exec"), namespace)
    gold = {case["case_id"]:case["answer"] for case in load(inputs/"verified-mmlu-quality-inputs.json")["cases"]}
    expected = {case["case_id"]:case["body"] for case in map(json.loads, raw.splitlines())}
    return expected, gold, namespace["extract_answer"], hashlib.sha256(raw).hexdigest()


def percentile(values, quantile):
    if values:
        return sorted(values)[math.ceil(len(values)*quantile)-1]
    return None


def summarize(folder, expected, gold, score, digest):
    plan = load(folder / "plan.json")
    workload = plan["workload"]
    if workload["request_sha256"] != digest or workload["requests"] != len(expected):
        raise ValueError("Run uses different inputs: " + str(folder))
    seen, complete, correct, released = set(), 0, 0, 0
    output_tokens, length_stops, unparsed = 0, 0, 0
    first_token, final_text_token, latency = [], [], []
    final_answers = reasoning_only = cached_tokens = 0
    for path in folder.glob("*/result.json"):
        row = load(path)
        case = row["case_id"]
        if case not in expected or case in seen:
            raise ValueError("Unexpected or repeated case: " + case)
        seen.add(case)
        body = dict(row["request"])
        salt = body.pop("cache_salt")
        if plan["route"] == "request_throttling":
            if body.pop("program_id") != salt:
                raise ValueError("Unexpected program identity: " + case)
            release = path.with_name("release.json")
            if release.exists() and load(release) == {"program_id":salt,"released":True}:
                released += 1
        original = dict(expected[case])
        original.pop("max_completion_tokens")
        original["max_tokens"] = workload["max_tokens"]
        if body != original:
            raise ValueError("Executed request differs from frozen case: " + case)
        if row["status"] != "ok":
            continue
        capture = path.with_name(Path(row["stream_capture"]).name)
        if hashlib.sha256(capture.read_bytes()).hexdigest() != row["stream_sha256"]:
            raise ValueError("Retained stream digest mismatch: " + case)
        complete += 1
        answer = score(row["content"])
        correct += answer == gold[case]
        unparsed += answer is None
        length_stops += row["finish_reason"] == "length"
        output_tokens += row["usage"]["completion_tokens"]
        final_answers += bool(row["content"].strip())
        reasoning_only += bool(row.get("reasoning", "").strip()) and not row["content"].strip()
        cached_tokens += (row["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        generated = row.get("first_output_token") or row["first_token"]
        first_token.append(generated["monotonic"] - row["start"]["monotonic"])
        if row["first_token"] is not None:
            final_text_token.append(row["first_token"]["monotonic"] - row["start"]["monotonic"])
        latency.append(row["elapsed_s"])
    measured = folder / "measurement.json"
    seconds = load(measured)["elapsed_seconds"] if measured.is_file() else None
    status = load(folder/"status.json") if (folder/"status.json").is_file() else {"status":"incomplete"}
    return dict(group=folder.name, engine=plan["engine"], release=plan["release"], route=plan["route"],
                status=status, requested=workload["requests"], recorded=len(seen), completed=complete,
                correct=correct, unparsed=unparsed, length_stops=length_stops, released=released,
                final_answers=final_answers, reasoning_only=reasoning_only, cached_tokens=cached_tokens,
                first_final_text_p50_seconds=percentile(final_text_token,.5), output_tokens=output_tokens, seconds=seconds, output_tokens_per_second=output_tokens/seconds if seconds else None,
                ttft_p50_seconds=percentile(first_token, .5), ttft_p95_seconds=percentile(first_token, .95),
                latency_p50_seconds=percentile(latency, .5), latency_p95_seconds=percentile(latency, .95))


def compare(folders, deployment):
    inputs = scoring(deployment)
    context = None
    report = []
    for folder in folders:
        plan = load(folder / "plan.json")
        frozen = [plan["workload"], plan["tp"], plan["replicas"]]
        if context is not None and context != frozen:
            raise ValueError("Compared groups differ in workload, TP or replica count")
        context = frozen
        report.append(summarize(folder, *inputs))
    return report
