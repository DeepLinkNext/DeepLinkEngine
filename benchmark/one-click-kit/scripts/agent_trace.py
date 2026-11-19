"""Frozen samples of published agent requests; real inference, no task execution."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/public-agent-trace-fixed32-v1.json"
# Updated only when publishing a new prospectively frozen protocol.
PROTOCOL_SHA256 = "eacf3628cda8d7accc2e39b20fce687b1deb1cb0b0f470e8b11c8dd4517f71f8"
TRACE_DATASETS = {"swe", "terminal"}
SWE_TOOLS_SHA256 = "95c0546bbc92c4a6a9e6780ae825d78e05e2c436211603790502303b7f74fcf8"


class InvalidPublishedRequest(ValueError):
    pass


def template_messages(body):
    """Mirror API JSON argument decoding for counting; preserve wire bodies."""
    messages = copy.deepcopy(body["messages"])
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call.get("function", call)
            if isinstance(function.get("arguments"), str):
                try:
                    function["arguments"] = json.loads(function["arguments"])
                except json.JSONDecodeError as error:
                    raise InvalidPublishedRequest(
                        "invalid_published_tool_arguments"
                    ) from error
            if not isinstance(function.get("arguments"), dict):
                raise InvalidPublishedRequest("non_object_published_tool_arguments")
    return messages


def select_tasks(dataset, available):
    """Prospective SHA256 ranking, SWE repository-proportional largest remainder."""
    available = sorted(set(available))

    def rank(identity):
        return hashlib.sha256(
            ("public-agent-trace-fixed32-v1\0" + dataset + "\0" + identity).encode()
        ).hexdigest(), identity

    if len(available) < 32:
        raise ValueError("Published source has fewer than32 eligible tasks")
    if dataset == "terminal":
        return sorted(sorted(available, key=rank)[:32]), None
    cohorts = {}
    for identity in available:
        repo = re.fullmatch(r"instance_(.+)-[0-9a-f]{40}(?:-v.*)?", identity).group(1)
        cohorts.setdefault(repo, []).append(identity)
    quota = {repo: 32 * len(ids) // len(available) for repo, ids in cohorts.items()}
    remaining = 32 - sum(quota.values())
    for repo in sorted(
        cohorts, key=lambda k: (-(32 * len(cohorts[k]) % len(available)), k)
    )[:remaining]:
        quota[repo] += 1
    chosen = [
        identity
        for repo, ids in cohorts.items()
        for identity in sorted(ids, key=rank)[: quota[repo]]
    ]
    return sorted(chosen), quota


def select_requests(candidates, count_tokens):
    """First and last distinct eligible published request; no truncation or padding."""
    eligible, excluded, seen = [], [], set()
    for index, candidate in enumerate(candidates):
        if candidate.get("invalid_reason"):
            excluded.append(
                {
                    "source_index": candidate["source_index"],
                    "reason": candidate["invalid_reason"],
                }
            )
            continue
        body = candidate["body"]
        image = any(
            isinstance(m.get("content"), list)
            and any(p.get("type") not in ("text", "input_text") for p in m["content"])
            for m in body["messages"]
        )
        if image:
            excluded.append(
                {
                    "source_index": candidate["source_index"],
                    "reason": "non_text_content",
                }
            )
            continue
        try:
            tokens = count_tokens(body)
        except InvalidPublishedRequest as error:
            excluded.append(
                {"source_index": candidate["source_index"], "reason": str(error)}
            )
            continue
        if tokens > 16384:
            excluded.append(
                {
                    "source_index": candidate["source_index"],
                    "reason": "prompt_over16384",
                    "prompt_tokens": tokens,
                }
            )
            continue
        body_hash = digest(body)
        if body_hash in seen:
            excluded.append(
                {
                    "source_index": candidate["source_index"],
                    "reason": "duplicate_request_body",
                }
            )
            continue
        seen.add(body_hash)
        eligible.append(
            dict(
                candidate,
                extracted_index=index,
                prompt_tokens=tokens,
                source_body_sha256=body_hash,
            )
        )
    selected = eligible[:1] + eligible[-1:] if len(eligible) > 1 else eligible
    return selected, {
        "published_requests": len(candidates),
        "eligible_requests": len(eligible),
        "selected_requests": len(selected),
        "excluded": excluded,
    }


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def protocol():
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Public agent trace protocol differs from external pin")
    data = json.loads(PROTOCOL.read_bytes())
    for name, selected in data["datasets"].items():
        identities, quota = select_tasks(name, selected["available_published_task_ids"])
        if (
            identities != selected["scope"]["selected_task_ids"]
            or quota != selected["scope"]["repository_quotas"]
        ):
            raise ValueError("Published task selection differs from prospective rank")
        if not 0 < len(selected["requests"]) <= 64:
            raise ValueError("Public trace request budget exceeds fixed scope")
    return data


def fetch(source, cache):
    """No credentials; a mirror may supply only the already pinned source bytes."""
    target = Path(cache) / source["cache_key"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != source["sha256"]:
            raise ValueError("Cached public trajectory changed: " + source["cache_key"])
        return target
    failures = []
    for url in [source["url"]] + source.get("alternate_urls", []):
        partial = target.with_name(target.name + ".partial-" + uuid.uuid4().hex)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with (
                urllib.request.urlopen(request, timeout=180) as response,
                partial.open("xb") as stream,
            ):
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > source["size"]:
                        raise ValueError("Published trajectory exceeds frozen size")
                    stream.write(block)
            if size != source["size"] or sha(partial) != source["sha256"]:
                raise ValueError("Published trajectory differs from frozen SHA256")
            partial.replace(target)
            return target
        except (OSError, ValueError) as error:
            failures.append(type(error).__name__ + ": " + str(error))
            partial.unlink(missing_ok=True)
    raise RuntimeError(
        "Cannot restore published trajectory "
        + source["cache_key"]
        + ": "
        + "; ".join(failures)
    )


def extract(dataset, path):
    """Keep the published messages; strip only SWE-agent non-wire metadata."""
    if dataset == "swe":
        data = json.loads(Path(path).read_bytes())
        tools = None
        if data.get("replay_config"):
            config = data["replay_config"]
            config = json.loads(config) if isinstance(config, str) else config
            manifest = ROOT / "protocols/swe-agent-v110-tools.json"
            if sha(manifest) != SWE_TOOLS_SHA256:
                raise ValueError("SWE official tool schema manifest changed")
            original = json.loads(manifest.read_bytes())
            tool_config = config["agent"]["tools"]
            if (
                data["info"]["swe_agent_version"] != original["agent_version"]
                or hashlib.sha256(tool_config["command_docs"].encode()).hexdigest()
                != original["command_docs_sha256"]
                or [b["path"] for b in tool_config["bundles"]]
                != original["tool_bundle_paths"]
                or not tool_config["enable_bash_tool"]
                or tool_config["parse_function"]["type"] != "function_calling"
                or config["agent"]["model"]["convert_system_to_user"]
            ):
                raise ValueError(
                    "SWE trajectory tool/version configuration differs from pinned official bundles"
                )
            tools = original["tools"]
        result = []
        allowed = {
            "role",
            "content",
            "name",
            "tool_calls",
            "tool_call_id",
            "function_call",
            "reasoning_content",
        }
        for index, step in enumerate(data["trajectory"]):
            query = step.get("query")
            if not isinstance(query, list) or not query:
                result.append(
                    {"source_index": index, "invalid_reason": "missing_published_query"}
                )
                continue
            if any(
                not isinstance(message, dict)
                or "role" not in message
                or "content" not in message
                for message in query
            ):
                result.append(
                    {
                        "source_index": index,
                        "invalid_reason": "incomplete_published_message",
                    }
                )
                continue
            body = {
                "messages": [
                    {k: copy.deepcopy(v) for k, v in m.items() if k in allowed}
                    for m in query
                ]
            }
            # Match SWE-Agent v1.1.0 LiteLLMModel._history_to_messages;
            # tool_call_ids is an internal field, not a wire field to discard.
            for source, message in zip(query, body["messages"], strict=True):
                if source["role"] == "tool" and source.get("tool_call_ids"):
                    message["tool_call_id"] = source["tool_call_ids"][0]
            if tools is not None:
                body["tools"] = copy.deepcopy(tools)
            result.append({"source_index": index, "body": body})
        return result
    rows = [
        json.loads(line)
        for line in Path(path).read_bytes().splitlines()
        if line.strip()
    ]
    return [
        {
            "source_index": index,
            "original_timestamp_ms": row["t"],
            "body": {
                k: copy.deepcopy(row["payload"][k])
                for k in ("messages", "tools", "tool_choice", "parallel_tool_calls")
                if k in row["payload"]
            },
        }
        for index, row in enumerate(rows)
    ]


def load_plan(plan):
    plan = Path(plan)
    contract = json.loads((plan / "contract.json").read_bytes())
    pin = protocol()
    name = contract["dataset"]
    if name not in TRACE_DATASETS or contract["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("Unknown trace protocol or dataset")
    tasks = [
        json.loads(line)
        for line in (plan / "tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    expected = pin["datasets"][name]["requests"]
    metadata = [{k: v for k, v in task.items() if k != "source_body"} for task in tasks]
    if metadata != expected or contract["request_ids"] != [
        t["case_id"] for t in expected
    ]:
        raise ValueError("Frozen trace request identity, order or body changed")
    if any(digest(t["source_body"]) != t["source_body_sha256"] for t in tasks):
        raise ValueError("Trace message or tool content changed")
    if (
        contract["generation"] != pin["generation"]
        or contract["concurrency"] != pin["concurrency"]
    ):
        raise ValueError("Trace budget or arrival profile changed")
    if (
        contract["model"] != "qwen3.5-35b"
        or contract["base_url"] != "http://127.0.0.1:18390"
        or contract["timeout"] != pin["timeout_seconds"]
    ):
        raise ValueError("Trace endpoint, model or timeout differs")
    if (
        contract["scope"] != pin["datasets"][name]["scope"]
        or contract["tokenizer_files"] != pin["tokenizer_files"]
    ):
        raise ValueError("Trace scope or tokenizer binding changed")
    return contract, tasks


def prepare(name, cache, out, model):
    pin = protocol()
    data = pin["datasets"][name]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = data["sources"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        local = dict(
            zip(
                (s["cache_key"] for s in sources),
                pool.map(lambda s: fetch(s, cache), sources),
                strict=True,
            )
        )
    tasks = []
    for task in data["requests"]:
        original = extract(name, local[task["source_key"]])[task["extracted_index"]]
        if original["source_index"] != task["source_index"]:
            raise ValueError("Published request no longer matches frozen selection")
        if digest(original["body"]) != task["source_body_sha256"]:
            raise ValueError("Original message/tool hash differs")
        tasks.append(dict(task, source_body=original["body"]))
    metadata = {
        p.name: sha(p)
        for p in Path(model).iterdir()
        if p.name in pin["tokenizer_files"]
    }
    if metadata != pin["tokenizer_files"]:
        raise ValueError("Trace sample belongs to another tokenizer or chat template")
    contract = {
        "schema": "vllm-hust-public-agent-trace/v1",
        "dataset": name,
        "protocol_sha256": PROTOCOL_SHA256,
        "request_ids": [t["case_id"] for t in data["requests"]],
        "generation": pin["generation"],
        "concurrency": pin["concurrency"],
        "timeout": pin["timeout_seconds"],
        "base_url": "http://127.0.0.1:18390",
        "model": "qwen3.5-35b",
        "tokenizer_files": metadata,
        "scope": data["scope"],
        "quality_measured": False,
        "source_files": {k: sha(v) for k, v in local.items()},
        "engine_transport": {
            "adapter_sha256": sha(Path(__file__).with_name("streaming_transport.py"))
        },
    }
    save(out / "contract.json", contract)
    (out / "tasks.jsonl").write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
    )
    save(out / "sample-scope.json", data["scope"])
    load_plan(out)
    print(
        "PUBLIC_TRACE_SCOPE",
        name,
        json.dumps(data["scope"], ensure_ascii=False),
        flush=True,
    )
    return {
        "name": name,
        "plan": str(out.resolve()),
        "tasks": len(data["requests"]),
        "contract_sha256": sha(out / "contract.json"),
        "execution_scope": data["scope"],
        "quality_measured": False,
    }


def checked(spec):
    data = json.loads(Path(spec).read_bytes())
    plan = Path(data["plan"])
    if sha(plan / "contract.json") != data["contract_sha256"]:
        raise ValueError("Frozen trace contract changed")
    contract, tasks = load_plan(plan)
    if data["dataset"] != contract["dataset"] or data["case_ids"] != [
        t["case_id"] for t in tasks
    ]:
        raise ValueError("Trace request selection changed after freeze")
    if (
        data["stage"] != "trace"
        or data["diagnostic"]
        or data["service_role"] not in ("B0", "B1")
    ):
        raise ValueError("Public trace replay cannot become a task solver or judge")
    if contract["engine_transport"]["adapter_sha256"] != sha(
        Path(__file__).with_name("streaming_transport.py")
    ):
        raise ValueError("Trace streaming adapter changed")
    return data


def request_body(contract, task):
    return dict(
        copy.deepcopy(task["source_body"]),
        **contract["generation"],
        model=contract["model"],
        stream=True,
        stream_options={"include_usage": True},
        cache_salt=uuid.uuid4().hex,
    )


def tokenization_body(contract, task):
    return {
        "model": contract["model"],
        "messages": copy.deepcopy(task["source_body"]["messages"]),
        "tools": copy.deepcopy(task["source_body"].get("tools")),
        "add_generation_prompt": True,
        "continue_final_message": False,
        "add_special_tokens": False,
        "chat_template_kwargs": contract["generation"]["chat_template_kwargs"],
    }


def admit_tokenization(data, contract, tasks, out):
    """Check the actual service's entire input scope before generating tokens."""
    baseline = None
    if data["service_role"] == "B1":
        from qualification import sealed_files

        prior = Path(data["baseline_trace_run"])
        sealed_files(prior)
        if not json.loads((prior / "receipt.json").read_bytes()).get("completed"):
            raise ValueError("B1 trace requires its sealed successful B0 arm")
        prior_spec = checked(prior / "dataset-spec.json")
        if (
            prior_spec["service_role"] != "B0"
            or prior_spec["contract_sha256"] != data["contract_sha256"]
        ):
            raise ValueError("B1 trace baseline belongs to another frozen input scope")
        baseline = json.loads(
            (prior / "service/trace-tokenizer-admission.json").read_bytes()
        )
        if not baseline["complete"]:
            raise ValueError("B0 input tokenization was not qualified")
    out = Path(out)
    (out / "tokenize").mkdir(parents=True, exist_ok=False)
    receipt = {
        "complete": False,
        "generation_requests": 0,
        "inputs": {},
        "failure": None,
        "started": time.time(),
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for task in tasks:
            body = tokenization_body(contract, task)
            request = urllib.request.Request(
                contract["base_url"] + "/tokenize",
                json.dumps(body).encode(),
                {"Content-Type": "application/json"},
            )
            with opener.open(request, timeout=60) as response:
                result = json.load(response)
            capture = out / "tokenize" / (task["case_id"] + ".json.gz")
            with gzip.open(capture, "wt") as stream:
                json.dump({"request": body, "response": result}, stream)
            tokens = result.get("tokens")
            if (
                not isinstance(tokens, list)
                or result.get("count") != len(tokens)
                or len(tokens) != task["prompt_tokens"]
            ):
                raise ValueError(
                    "Actual service tokenization differs from frozen public input: "
                    + task["case_id"]
                )
            entry = {
                "count": len(tokens),
                "token_ids_sha256": digest(tokens),
                "capture_sha256": sha(capture),
            }
            if (
                baseline
                and entry["token_ids_sha256"]
                != baseline["inputs"][task["case_id"]]["token_ids_sha256"]
            ):
                raise ValueError(
                    "B0/B1 actual input token IDs differ: " + task["case_id"]
                )
            receipt["inputs"][task["case_id"]] = entry
        receipt["complete"] = True
    except urllib.error.HTTPError as error:
        receipt["failure"] = (
            "Actual service /tokenize HTTP"
            + str(error.code)
            + ": "
            + error.read(8192).decode(errors="replace")
        )
        raise ValueError(receipt["failure"]) from error
    except BaseException as error:
        receipt["failure"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        receipt["ended"] = time.time()
        save(out / "trace-tokenizer-admission.json", receipt)
    return receipt


def run(spec, out):
    from streaming_transport import call

    data = checked(spec)
    contract, tasks = load_plan(data["plan"])
    directory = Path(out) / "evaluation"
    requests = directory / "requests"
    requests.mkdir(parents=True, exist_ok=False)
    admit_tokenization(data, contract, tasks, out)
    start = time.monotonic()

    def send(task):
        body = request_body(contract, task)
        record = call(contract, body, Path(out) / "transport")
        if (
            record["status"] == "ok"
            and record["usage"]["prompt_tokens"] != task["prompt_tokens"]
        ):
            record.update(
                status="tokenization_error",
                error="Actual prompt tokenization differs from prospectively frozen template",
            )
        record.update(
            case_id=task["case_id"],
            source_body_sha256=task["source_body_sha256"],
            request_sha256=digest(body),
        )
        record["record_sha256"] = digest(record)
        save(requests / (task["case_id"] + ".json"), record)
        print("PUBLIC_TRACE_RESPONSE", task["case_id"], record["status"], flush=True)
        return record

    with ThreadPoolExecutor(max_workers=contract["concurrency"]) as pool:
        rows = list(pool.map(send, tasks))
    save(
        directory / "phase.json",
        {
            "dataset": data["dataset"],
            "planned": len(tasks),
            "returned": len(rows),
            "successful": sum(r["status"] == "ok" for r in rows),
            "start_monotonic": start,
            "end_monotonic": time.monotonic(),
            "quality_measured": False,
        },
    )
    return 0 if all(r["status"] == "ok" for r in rows) else 2


def inspect(root):
    """Independently reconstruct frozen inputs and metrics from sealed raw SSE."""
    from qualification import sealed_files
    from report_dataset import replay
    import math
    import statistics

    root = Path(root).resolve()
    sealed_files(root)
    receipt = json.loads((root / "receipt.json").read_bytes())
    if receipt.get("quiescent") is not True:
        raise ValueError("Trace run is not quiescent")
    data = checked(root / "dataset-spec.json")
    contract, tasks = load_plan(data["plan"])
    originals = {t["case_id"]: t for t in tasks}
    admission_path = root / "service/trace-tokenizer-admission.json"
    admission = (
        json.loads(admission_path.read_bytes()) if admission_path.exists() else None
    )
    if receipt.get("completed") and (not admission or not admission["complete"]):
        raise ValueError("Completed trace has no qualified actual input tokenization")
    if admission and admission["complete"]:
        if set(admission["inputs"]) != set(originals):
            raise ValueError("Trace tokenization omitted planned inputs")
        for identity, entry in admission["inputs"].items():
            capture = root / "service/tokenize" / (identity + ".json.gz")
            if sha(capture) != entry["capture_sha256"]:
                raise ValueError("Actual trace tokenization capture changed")
            with gzip.open(capture, "rt") as stream:
                raw = json.load(stream)
            if (
                raw["request"] != tokenization_body(contract, originals[identity])
                or raw["response"]["count"] != originals[identity]["prompt_tokens"]
                or digest(raw["response"]["tokens"]) != entry["token_ids_sha256"]
            ):
                raise ValueError(
                    "Trace input tokens differ from their original service response"
                )
    rows = {}
    for path in sorted((root / "service/evaluation/requests").glob("*.json")):
        record = json.loads(path.read_bytes())
        identity = record["case_id"]
        if (
            digest({k: v for k, v in record.items() if k != "record_sha256"})
            != record["record_sha256"]
            or identity not in originals
            or identity in rows
        ):
            raise ValueError("Unknown, duplicate or altered trace response")
        task = originals[identity]
        body = record["request"]
        expected = request_body(contract, task)
        expected["cache_salt"] = body.get("cache_salt")
        if (
            not expected["cache_salt"]
            or body != expected
            or digest(body) != record["request_sha256"]
            or record["source_body_sha256"] != task["source_body_sha256"]
        ):
            raise ValueError(
                "Actual trace request differs from published input or frozen budget"
            )
        rows[identity] = record
        if record["status"] != "ok":
            continue
        wire = Path(record["stream_capture"])
        if (
            not wire.resolve().is_relative_to(root)
            or sha(wire) != record["stream_sha256"]
        ):
            raise ValueError("Trace response stream binding changed")
        content, usage, finish, first, last = replay(wire)
        if (content, usage, finish, first, last) != (
            record["content"],
            record["usage"],
            record["finish_reason"],
            record["first_token"],
            record["last_token"],
        ):
            raise ValueError("Trace cached content or timing differs from raw SSE")
        times = [
            record["start"]["monotonic"],
            first["monotonic"],
            last["monotonic"],
            record["end"]["monotonic"],
        ]
        if not all(math.isfinite(t) for t in times) or times != sorted(times):
            raise ValueError("Trace timestamps are invalid")
        if (
            usage["prompt_tokens"] != task["prompt_tokens"]
            or not 0
            < usage["completion_tokens"]
            <= contract["generation"]["max_tokens"]
        ):
            raise ValueError("Actual trace input tokenization or output budget differs")
        if usage.get("prompt_tokens_details", {}).get("cached_tokens", 0):
            raise ValueError("Trace unexpectedly reused KV")
    good = [r for r in rows.values() if r["status"] == "ok"]
    duration = max((r["end"]["monotonic"] for r in rows.values()), default=0) - min(
        (r["start"]["monotonic"] for r in rows.values()), default=0
    )
    tokens = sum(r["usage"]["completion_tokens"] for r in good)
    ttft = [
        (r["first_token"]["monotonic"] - r["start"]["monotonic"]) * 1000 for r in good
    ]
    tpot = [
        (r["end"]["monotonic"] - r["first_token"]["monotonic"])
        * 1000
        / (r["usage"]["completion_tokens"] - 1)
        for r in good
        if r["usage"]["completion_tokens"] > 1
    ]

    def percentile(values, q):
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "dataset": data["dataset"],
        "stage": "trace",
        "service_role": data["service_role"],
        "solver_role": None,
        "planned": len(tasks),
        "recorded": len(rows),
        "successful": len(good),
        "failed": len(rows) - len(good),
        "unreturned": len(tasks) - len(rows),
        "length_stops": sum(r["finish_reason"] == "length" for r in good),
        "duration_seconds": duration if duration > 0 else None,
        "successful_output_tokens": tokens,
        "output_tokens_per_second": tokens / duration if duration > 0 else None,
        "mean_ttft_ms": statistics.mean(ttft) if ttft else None,
        "mean_tpot_ms": statistics.mean(tpot) if tpot else None,
        "ttft_p95_ms": percentile(ttft, 0.95),
        "ttft_p99_ms": percentile(ttft, 0.99),
        "tpot_p95_ms": percentile(tpot, 0.95),
        "tpot_p99_ms": percentile(tpot, 0.99),
        "percentile_definition": "Linear interpolation at(n-1)*q over this finite sample; not a continuous-window stability claim",
        "actual_context_16k_cases": sum(
            r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] >= 16384
            for r in good
        ),
        "tpot_definition": "V1.5: (request completion - first token)/(completion tokens - 1); per-request mean",
        "execution_scope": contract["scope"],
        "diagnostic": False,
        "quality_measured": False,
        "formal_complete": False,
        "receipt": receipt,
        "failure": receipt.get("failure"),
        "source_seal_sha256": sha(root / "checksums.json"),
        "actual_tokenizer_admission": admission,
        "scope": "Published agent request sample with real generation; no tool execution or task solve-rate grading. New fixed concurrency/budget; original arrival timestamps are provenance only.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(TRACE_DATASETS), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset, args.cache, args.out, args.model)


if __name__ == "__main__":
    main()
