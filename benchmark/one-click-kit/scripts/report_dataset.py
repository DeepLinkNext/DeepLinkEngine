"""Audit sealed dataset SSE and original grade parsing before rendering a report."""

import argparse
import gzip
import html
import json
import math
from pathlib import Path
import statistics

from dataset_contract import client, sha, save, DATASETS
from qualification import sealed_files


def replay(path):
    content, usage, finish, first, last, done = "", None, None, None, None, False
    with gzip.open(path, "rt") as stream:
        for line in stream:
            item = json.loads(line)
            raw = item["raw"]
            if not raw.startswith("data:"):
                continue
            value = raw[5:].strip()
            if value == "[DONE]":
                done = True
                continue
            event = json.loads(value)
            if event.get("error"):
                raise ValueError("Retained SSE has an in-stream error")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                text = choice.get("delta", {}).get("content") or ""
                content += text
                if text:
                    first = first or item["clock"]
                    last = item["clock"]
                finish = choice.get("finish_reason") or finish
    if not done or not first or usage is None or finish not in ("stop", "length"):
        raise ValueError("Retained SSE is incomplete")
    return content, usage, finish, first, last


def inspect(root):
    sealed_files(root)
    from serving_warmup import audit as audit_warmup

    audit_warmup(root)
    receipt = json.loads((root / "receipt.json").read_bytes())
    if receipt.get("quiescent") is not True:
        raise ValueError("Dataset run is not quiescent")
    specfile = root / "dataset-spec.json"
    if not specfile.exists():
        return {
            "receipt": receipt,
            "recorded": 0,
            "successful": 0,
            "failed": 0,
            "failure": receipt.get("failure"),
            "formal_complete": False,
            "scope": "Preparation failed before a dataset contract was admitted",
        }
    data = json.loads(specfile.read_bytes())
    if data["dataset"] in ("swe", "terminal"):
        from agent_trace import inspect as inspect_trace

        return inspect_trace(root)
    module = client(data["dataset"])
    plan = Path(data["plan"])
    if sha(plan / "contract.json") != data["contract_sha256"]:
        raise ValueError("Dataset contract changed")
    contract, tasks = module.load_plan(plan)
    if (
        data["dataset"] == "hle"
        and contract.get("execution_scope", {}).get("profile") == "fixed200-v1"
    ):
        from hle_scope import verify_selected

        verify_selected(plan, contract, tasks)
    originals = {t["case_id"]: t for t in tasks}
    solver_records = None
    if data["stage"] == "judge":
        _, solver_records = module.verify_run(
            data["responses"], data["contract_sha256"], "solver"
        )
    records = {}
    for path in sorted((root / "service/evaluation/requests").glob("*.json")):
        record = json.loads(path.read_bytes())
        payload_hash = record.pop("record_sha256")
        if (
            module.digest(record) != payload_hash
            or record["case_id"] in records
            or record["case_id"] not in data["case_ids"]
        ):
            raise ValueError("Duplicate, unknown or altered task record")
        records[record["case_id"]] = record
        if record["status"] in ("ok", "invalid_judge"):
            wire = Path(record["stream_capture"])
            if (
                not wire.resolve().is_relative_to(root.resolve())
                or sha(wire) != record["stream_sha256"]
            ):
                raise ValueError("Dataset stream binding differs")
            content, usage, finish, first, last = replay(wire)
            if (content, usage, finish, first, last) != (
                record["content"],
                record["usage"],
                record["finish_reason"],
                record["first_token"],
                record["last_token"],
            ):
                raise ValueError("Cached answer or timing differs from original stream")
            times = [
                record["start"]["monotonic"],
                first["monotonic"],
                last["monotonic"],
                record["end"]["monotonic"],
            ]
            if not all(math.isfinite(t) for t in times) or times != sorted(times):
                raise ValueError("Dataset stream timing is nonfinite or out of order")
            if module.digest(record["request"]) != record["request_sha256"]:
                raise ValueError("Actual dataset request binding differs")
            if data["stage"] == "solver":
                task = originals[record["case_id"]]
                # Reconstruct frozen task semantics independently of the cached
                # request. References never become solver prompt material.
                if data["dataset"] == "hle":
                    parts = [{"type": "text", "text": task["problem"]}]
                    if task["image"]:
                        parts.append(
                            {"type": "image_url", "image_url": {"url": task["image"]}}
                        )
                    expected_messages = [
                        {"role": "system", "content": contract["solver_prompt"]},
                        {"role": "user", "content": parts},
                    ]
                else:
                    expected_messages = [{"role": "user", "content": task["problem"]}]
                if record["request"]["messages"] != expected_messages:
                    raise ValueError(
                        "Solver input differs from frozen task or drops an image"
                    )
            else:
                source = solver_records[record["case_id"]]
                if record["solver_record_sha256"] != module.digest(source):
                    raise ValueError("Judge record belongs to another solver answer")
                task = originals[record["case_id"]]
                if data["dataset"] == "hle":
                    prompt = contract["judge_prompt"].format(
                        question=task["problem"],
                        response=source.get("content") or "",
                        correct_answer=task["answer"],
                    )
                else:
                    prompt = contract["judge_prompts"][task["track"]].format_map(
                        {
                            "problem": task["problem"],
                            "answer": source.get("content") or "",
                            "reference answer": task["answer"],
                            "rubric": task["answer"],
                        }
                    )
                expected_messages = [{"role": "user", "content": prompt}]
                try:
                    grade = module.parse_grade(
                        record["track"], content, finish, contract["scoring_policy"]
                    )
                except (TypeError, ValueError):
                    if record["status"] != "invalid_judge":
                        raise ValueError("Invalid grade was silently accepted")
                else:
                    if record["status"] != "ok" or record.get("grade") != grade:
                        raise ValueError("Cached grade differs from original parser")
            expected_body = dict(
                contract[data["stage"]]["request"],
                model=contract[data["stage"]]["model"],
                messages=expected_messages,
                stream=True,
                stream_options={"include_usage": True},
                cache_salt=record["request"].get("cache_salt"),
            )
            if data["stage"] == "judge" and data["dataset"] == "hle":
                from judge_grammar import effective_format

                expected_body["response_format"] = effective_format(
                    data, contract["judge_response_format"]
                )
            if not expected_body["cache_salt"] or record["request"] != expected_body:
                raise ValueError(
                    "Actual generation parameters, schema or request semantics differ from the frozen contract"
                )
    good = [r for r in records.values() if r["status"] == "ok"]
    completed = [r for r in records.values() if r["status"] in ("ok", "invalid_judge")]
    output = sum(r["usage"]["completion_tokens"] for r in completed)
    duration = max(
        (r["end"]["monotonic"] for r in records.values() if "end" in r), default=0
    ) - min(
        (r["start"]["monotonic"] for r in records.values() if "start" in r), default=0
    )
    ttft = [
        1000 * (r["first_token"]["monotonic"] - r["start"]["monotonic"])
        for r in completed
    ]
    tpot = [
        1000
        * (r["end"]["monotonic"] - r["first_token"]["monotonic"])
        / (r["usage"]["completion_tokens"] - 1)
        for r in completed
        if r["usage"]["completion_tokens"] > 1
    ]
    content_span = [
        1000
        * (r["last_token"]["monotonic"] - r["first_token"]["monotonic"])
        / (r["usage"]["completion_tokens"] - 1)
        for r in completed
        if r["usage"]["completion_tokens"] > 1
    ]
    judge_policy = None
    if data["dataset"] == "hle" and data["stage"] == "judge" and records:
        from judge_grammar import audit

        judge_policy = audit(
            data,
            json.loads((root / "service/plan.json").read_bytes()),
            json.loads((root / "service/launch-command.json").read_bytes()),
        )
    return {
        "run": str(root),
        "source_seal_sha256": sha(root / "checksums.json"),
        "receipt": receipt,
        "dataset": data["dataset"],
        "stage": data["stage"],
        "service_role": data["service_role"],
        "solver_role": data["solver_role"],
        "diagnostic": data["diagnostic"],
        "judge_grammar": judge_policy,
        "planned": len(data["case_ids"]),
        "full_dataset_tasks": DATASETS[data["dataset"]]["tasks"],
        "execution_scope": contract.get("execution_scope"),
        "recorded": len(records),
        "successful": len(good),
        "transport_completed": len(completed),
        "failed": len(records) - len(completed),
        "unreturned": len(data["case_ids"]) - len(records),
        "unresolved_grades": sum(
            r["status"] == "invalid_judge" for r in records.values()
        ),
        "length_stops": sum(
            r.get("finish_reason") == "length" for r in records.values()
        ),
        "successful_output_tokens": output,
        "duration_seconds": duration if duration > 0 else None,
        "output_tokens_per_second": output / duration if duration > 0 else None,
        "mean_ttft_ms": statistics.mean(ttft) if ttft else None,
        "mean_tpot_ms": statistics.mean(tpot) if tpot else None,
        "tpot_definition": "V1.5 3.4.2: (request completion - first token)/(completion tokens - 1); per-request equal-weight mean",
        "mean_content_span_per_token_ms": statistics.mean(content_span)
        if content_span
        else None,
        "content_span_scope": "Auxiliary first-to-last content interval only; not the outline TPOT or its acceptance metric",
        "actual_context_16k_cases": sum(
            sum(
                r.get("usage", {}).get(k, 0)
                for k in ("prompt_tokens", "completion_tokens")
            )
            >= 16384
            for r in records.values()
        ),
        "failure": receipt.get("failure"),
        "formal_complete": False,
        "scope": "Current frozen dataset phase; judge measurements are excluded from solver performance/cost. All planned cases, errors, length stops and unresolved grades remain visible.",
    }


def render(root, out):
    data = inspect(root.resolve())
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib

    if matplotlib.__version__ != "3.10.8":
        raise RuntimeError("Report requires matplotlib3.10.8")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 3.2))
    keys = ("successful", "failed", "unreturned", "unresolved_grades")
    axis.bar(
        keys,
        [data.get(k, 0) for k in keys],
        color=["#166534", "#b91c1c", "#64748b", "#b45309"],
    )
    label = (
        "HLE fixed stratified200; source1811"
        if (data.get("execution_scope") or {}).get("profile") == "fixed200-v1"
        else "Current dataset phase; planned denominator retained"
    )
    axis.set_title(label)
    fig.tight_layout()
    fig.savefig(out / "dataset-status.png", dpi=180)
    plt.close(fig)
    save(out / "summary.json", data)
    body = html.escape(json.dumps(data, ensure_ascii=False, indent=2))
    (out / "report.html").write_text(
        "<!doctype html><meta charset=utf-8><title>数据集实际执行</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:32px auto}pre{white-space:pre-wrap}img{max-width:100%}</style><h1>数据集实际执行</h1><p>本轮独立请求与原始流复核；诊断子集不冒充全量，裁判流量不计入solver性能。</p><img src='dataset-status.png'><pre>"
        + body
        + "</pre>"
    )
    save(out / "checksums.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print("DATASET_REPORT", out / "report.html")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.run, args.out.resolve())


if __name__ == "__main__":
    main()
