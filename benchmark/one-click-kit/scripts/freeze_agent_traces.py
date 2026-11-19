"""Rebuild a public trace scope before measurements; never resample during a run."""

import argparse
import importlib.metadata
import hashlib
import json
from pathlib import Path
import tarfile

from agent_trace import (
    PROTOCOL,
    extract,
    save,
    select_requests,
    select_tasks,
    sha,
    template_messages,
)


def freeze(plan, cache, tokenizer_root, out, terminal_archive):
    from transformers import AutoTokenizer

    if importlib.metadata.version("transformers") != "5.14.1":
        raise ValueError("Scope construction requires transformers5.14.1")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    if sha(terminal_archive) != plan["terminal_task_archive_sha256"]:
        raise ValueError("Frozen Terminal2.1 original source archive changed")
    instructions = {}
    with tarfile.open(terminal_archive) as archive:
        for member in archive.getmembers():
            if "/tasks/" in member.name and member.name.endswith("/instruction.md"):
                identity = member.name.split("/tasks/", 1)[1].split("/", 1)[0]
                instructions[identity] = archive.extractfile(member).read().decode()
    receipt = cache / "download-receipt.json"
    if receipt.exists():
        source_files = json.loads(receipt.read_bytes())
    else:
        pinned = json.loads(PROTOCOL.read_bytes())
        source_files = [
            dict(
                source,
                dataset=name,
                lfs_sha256=source.get("git_lfs_sha256"),
                mirror_url=source.get("alternate_urls", [None])[0]
                if source.get("alternate_urls")
                else None,
            )
            for name, data in pinned["datasets"].items()
            for source in data["sources"]
        ]
    files = {s["cache_key"]: s for s in source_files}
    result = {
        "schema": "vllm-hust-public-agent-trace-fixed32/v1",
        "sampling": {
            "body_digest": "SHA256(UTF8(JSON(sort_keys=true,ensure_ascii=false,separators=(',',':')))); original source file SHA256 is over raw bytes",
            "tasks_per_dataset": 32,
            "task_rank": "SHA256(UTF8('public-agent-trace-fixed32-v1\\0'+dataset+'\\0'+task_id)), lexical tie-break",
            "swe_strata": "Canonical v1 repository, proportional largest-remainder quotas over available official published task trajectories",
            "terminal_strata": "Unstratified task hash rank among exact2.1 task IDs; lexicographically first published corpus per task",
            "requests_per_task": "First and last distinct eligible original requests, maximum2; original source order; never duplicate a lone eligible request or replace a missing task",
            "eligibility": "Complete published text history with valid tool-argument JSON, target prompt_tokens<=16384; missing/redacted source entries are excluded with reasons; no repair/truncation/padding or outcome/speed filter",
        },
        "generation": {
            "max_tokens": 2048,
            "temperature": 0,
            "seed": 17,
            "chat_template_kwargs": {"enable_thinking": False},
            "tool_choice": "none",
        },
        "concurrency": 8,
        "timeout_seconds": 600,
        "arrival_profile": "8 worker closed-loop finite request batch, original request order; historical timestamps are provenance only, not serving load replay",
        "tokenizer_files": {
            p.name: sha(p)
            for p in tokenizer_root.iterdir()
            if p.name
            in (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "config.json",
            )
        },
        "tokenizer_implementation": {
            "transformers": "5.14.1",
            "tokenizers": importlib.metadata.version("tokenizers"),
            "class": type(tokenizer).__name__,
        },
        "datasets": {},
    }

    def count(body):
        args = {"tools": body["tools"]} if "tools" in body else {}
        # vLLM decodes OpenAI JSON-string function arguments before applying the
        # target Qwen template. Only this tokenizer view uses mappings; actual
        # requests preserve the published JSON-string arguments unchanged.
        messages = template_messages(body)
        return len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
                **args,
            )
        )

    for name in ("swe", "terminal"):
        universe = plan["task_universes"][name]
        available = plan["available_source_tasks"][name]
        selected, quota = select_tasks(name, available)
        if set(selected) != set(plan["selected_tasks"][name]):
            raise ValueError(
                "Source download selection differs from prospective task rank"
            )
        sources, requests, selection = [], [], []
        for task_index, identity in enumerate(selected):
            candidates = []
            origin = []
            task_sources = sorted(
                (
                    s
                    for s in files.values()
                    if s["dataset"] == name and s["task_id"] == identity
                ),
                key=lambda s: s["source_path"],
            )
            for source in task_sources:
                path = cache / source["cache_key"]
                if sha(path) != source["sha256"]:
                    raise ValueError("Source changed before scope freeze")
                entry = {
                    k: source[k]
                    for k in ("cache_key", "url", "size", "sha256", "source_path")
                }
                entry.update(
                    task_id=identity,
                    alternate_urls=[source["mirror_url"]]
                    if source.get("mirror_url")
                    else [],
                )
                if "git_blob" in source:
                    entry["git_blob"] = source["git_blob"]
                if source.get("lfs_sha256"):
                    entry["git_lfs_sha256"] = source["lfs_sha256"]
                sources.append(entry)
                extracted = extract(name, path)
                if name == "terminal":
                    question = "\n".join(
                        m["content"]
                        if isinstance(m["content"], str)
                        else "\n".join(p.get("text", "") for p in m["content"])
                        for m in extracted[0]["body"]["messages"]
                        if m["role"] == "user"
                    )
                    if instructions[identity].strip() not in question:
                        raise ValueError(
                            "Published request instruction differs from frozen Terminal2.1: "
                            + identity
                        )
                candidates += extracted
                origin += [(source["cache_key"], i) for i in range(len(extracted))]
            chosen, stats = select_requests(candidates, count)
            for index, item in enumerate(chosen):
                source_key, extracted_index = origin[item["extracted_index"]]
                requests.append(
                    {
                        "case_id": name
                        + "-trace-"
                        + str(task_index).zfill(2)
                        + "-"
                        + str(index),
                        "task_id": identity,
                        "source_key": source_key,
                        "extracted_index": extracted_index,
                        "source_index": item["source_index"],
                        "source_body_sha256": item["source_body_sha256"],
                        "prompt_tokens": item["prompt_tokens"],
                        **(
                            {"original_timestamp_ms": item["original_timestamp_ms"]}
                            if "original_timestamp_ms" in item
                            else {}
                        ),
                    }
                )
            selection.append(
                dict(
                    task_id=identity,
                    **stats,
                    **(
                        {
                            "terminal_instruction_sha256": hashlib.sha256(
                                instructions[identity].encode()
                            ).hexdigest(),
                            "instruction_exact_substring_verified": True,
                        }
                        if name == "terminal"
                        else {}
                    ),
                )
            )
            print(
                "FROZEN_TRACE_TASK",
                name,
                task_index + 1,
                "requests",
                len(chosen),
                flush=True,
            )
        scope = {
            "profile": "public-agent-trace-fixed32-v1",
            "measurement": "real inference on published request snapshots; quality_measured=false",
            "benchmark_source_tasks": len(universe),
            "available_published_tasks": len(available),
            "selected_tasks": len(selected),
            "selected_task_ids": selected,
            "selected_requests": len(requests),
            "source_unavailable_task_ids": sorted(set(universe) - set(available)),
            "outside_sample_available_tasks": len(available) - len(selected),
            "selected_tasks_without_eligible_requests": [
                s["task_id"] for s in selection if not s["selected_requests"]
            ],
            "repository_quotas": quota,
            "prompt_tokens_min": min(t["prompt_tokens"] for t in requests),
            "prompt_tokens_max": max(t["prompt_tokens"] for t in requests),
            "prompt_tokens_sum": sum(t["prompt_tokens"] for t in requests),
            "prompt_at_least16k": sum(t["prompt_tokens"] >= 16384 for t in requests),
            "upper_bound_output_tokens_per_arm": len(requests) * 2048,
            "quality_measured": False,
        }
        result["datasets"][name] = {
            "sources": sources,
            "requests": requests,
            "scope": scope,
            "task_selection_audit": selection,
            "task_universe": universe,
            "available_published_task_ids": sorted(available),
        }
    result["provenance"] = {
        "swe": {
            "publisher": "ScaleAI official SWE-bench Pro",
            "run": "claude-sonnet-4-paper",
            "index": "https://github.com/scaleapi/SWE-bench_Pro-os/blob/66f92766bba642462d4bbe5479e83f91f9211862/traj/README.md",
            "source": "s3://scaleapi-results/swe-bench-pro/claude-sonnet-4-paper/traj/",
            "task_version": plan["source_commits"]["swe_dataset"],
        },
        "terminal": {
            "publisher": "shadowpa0327, third-party real provider request capture",
            "repository": "https://huggingface.co/datasets/shadowpa0327/pi_agents_terminal_bench",
            "revision": plan["source_commits"]["terminal_traces"],
            "task_repository": "harbor-framework/terminal-bench-2-1",
            "task_commit": plan["source_commits"]["terminal_tasks"],
            "license": "research-only (author dataset card); original bodies downloaded by user, not redistributed in this protocol or tools archive",
        },
    }
    save(out, result)
    print("PUBLIC_TRACE_PROTOCOL_SHA256", sha(out), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--terminal-source-archive", type=Path, required=True)
    args = parser.parse_args()
    freeze(
        json.loads(args.source_plan.read_bytes()),
        args.cache,
        args.tokenizer,
        args.out,
        args.terminal_source_archive,
    )


if __name__ == "__main__":
    main()
