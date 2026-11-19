"""Freeze real published request histories for context and tenant supplements.

This adds a separate workload; it never changes the five fixed dataset samples.
Selection depends only on source identity/order and token length, never outcomes.
"""

import argparse
import copy
import json
from pathlib import Path

from agent_trace import (
    InvalidPublishedRequest,
    digest,
    extract,
    fetch,
    protocol,
    save,
    sha,
    template_messages,
)

RANGES = ((4096, 8192), (16384, 24576), (32768, 49152))
LANES = 8
TASKS = 5


def select(candidates, encode):
    """First complete original request in each declared length band, unchanged."""
    chosen, observations = {}, []
    for item in candidates:
        entry = dict(source_index=item["source_index"])
        if item.get("invalid_reason"):
            observations.append(dict(entry, excluded=item["invalid_reason"]))
            continue
        body = item["body"]
        if any(
            isinstance(m.get("content"), list)
            and any(
                part.get("type") not in ("text", "input_text") for part in m["content"]
            )
            for m in body["messages"]
        ):
            observations.append(dict(entry, excluded="non_text_original"))
            continue
        try:
            ids = encode(body)
        except InvalidPublishedRequest as error:
            observations.append(dict(entry, excluded=str(error)))
            continue
        if not isinstance(ids, list) or not ids or any(type(n) is not int for n in ids):
            raise ValueError("Tokenizer must return the actual integer token sequence")
        observations.append(dict(entry, tokens=len(ids)))
        for low, high in RANGES:
            if low <= len(ids) < high and str(low) not in chosen:
                chosen[str(low)] = dict(
                    source_index=item["source_index"],
                    source_body_sha256=digest(body),
                    input_tokens=len(ids),
                    token_ids_sha256=digest(ids),
                    source_body=copy.deepcopy(body),
                )
        if len(chosen) == len(RANGES):
            break
    return chosen, observations


def prepare(model, cache, out):
    from transformers import AutoTokenizer

    pin = protocol()
    metadata = {name: sha(Path(model) / name) for name in pin["tokenizer_files"]}
    if metadata != pin["tokenizer_files"]:
        raise ValueError("The frozen task tokenizer/template/model metadata changed")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    def encode(body):
        encoded = tokenizer.apply_chat_template(
            template_messages(body),
            tools=body.get("tools"),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Transformers5 may return BatchEncoding even with one conversation.
        ids = encoded if isinstance(encoded, list) else encoded["input_ids"]
        if len(ids) == 1 and isinstance(ids[0], list):
            ids = ids[0]
        return ids

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    selected, inspected = [], []
    sources = [
        dict(source, dataset=name)
        for name in ("swe", "terminal")
        for source in pin["datasets"][name]["sources"]
    ]
    # Rank the frozen SWE32+Terminal32 source tasks before examining request lengths.
    ordered = sorted(
        sources,
        key=lambda s: (
            digest(["context-tenant-supplement-v2", s["dataset"], s["task_id"]]),
            s["task_id"],
        ),
    )
    for source in ordered:
        path = fetch(source, cache)
        cases, checks = select(extract(source["dataset"], path), encode)
        inspected.append(
            dict(
                dataset=source["dataset"],
                task_id=source["task_id"],
                source_sha256=source["sha256"],
                observations=checks,
                eligible_all_three_bands=len(cases) == len(RANGES),
            )
        )
        if len(cases) == len(RANGES):
            selected.append(
                dict(
                    task_id=source["dataset"] + ":" + source["task_id"],
                    source=source,
                    cases=cases,
                )
            )
        if len(selected) == TASKS:
            break
    save(out / "selection-audit.json", inspected)
    if len(selected) != TASKS:
        raise ValueError(
            "Original frozen SWE32+Terminal32 source population lacks five common 4K/16K/32K tasks; retained selection audit, no padding or replacement corpus"
        )
    cohorts = {}
    for low, _ in RANGES:
        cohorts[str(low)] = []
        for task in selected:
            case = task["cases"][str(low)]
            request = dict(
                copy.deepcopy(case["source_body"]),
                model="qwen3.5-35b",
                stream=True,
                stream_options=dict(include_usage=True),
                max_tokens=256,
                temperature=0,
                seed=17,
                ignore_eos=True,
                tool_choice="none",
                chat_template_kwargs=dict(enable_thinking=False),
            )
            cohorts[str(low)].append(
                dict(
                    task_id=task["task_id"],
                    source_index=case["source_index"],
                    source_body_sha256=case["source_body_sha256"],
                    source_sha256=task["source"]["sha256"],
                    input_tokens=case["input_tokens"],
                    token_ids_sha256=case["token_ids_sha256"],
                    request=request,
                )
            )
    save(out / "acceptance-inputs.json", cohorts)
    contract = dict(
        schema="vllm-hust-context-tenant-supplement/v1",
        input_file="acceptance-inputs.json",
        input_sha256=sha(out / "acceptance-inputs.json"),
        tokenizer_files=metadata,
        task_count=TASKS,
        cohorts=[
            dict(min_tokens=low, max_tokens_exclusive=high, concurrency=LANES)
            for low, high in RANGES
        ],
        phases=[
            dict(
                id=str(low),
                cohort=str(low),
                concurrency=LANES,
                tenants=1,
                warmup_seconds=60,
                formal_seconds=300,
            )
            for low, _ in RANGES
        ]
        + [
            dict(
                id="multi-tenant",
                cohort="4096",
                concurrency=24,
                tenants=4,
                warmup_seconds=60,
                formal_seconds=300,
            )
        ],
        output_tokens=256,
        temperature=0,
        seed=17,
        ignore_eos=True,
        request_timeout_seconds=600,
        automatic_retries=0,
        measurement_window_seconds=60,
        max_model_len_required=49408,
        task_selection="First five source-hash-ranked tasks in the frozen SWE32+Terminal32 population with one complete unchanged published request in each declared band; first eligible source-order request per band. No score, speed, answer, cropping or padding filter.",
        tenant_identity="Four independent closed-loop client queues; OpenAI user field tenant-0..3 and unique per-request cache salts. Shared engine/resources; no security-isolation or dedicated-VM claim.",
        acceptance="Preserve the original qualitative throughput/tail stability and high utilization criteria; no invented pass threshold. Complete execution produces measurements, not an automatic numerical pass.",
        source_tasks_are_not_concurrency="Five distinct real histories; eight closed-loop lanes cycle the fixed cases, repetitions are not independent task samples.",
        five_fixed_datasets_unchanged=True,
        quality_measured=False,
        source_protocol_sha256=sha(
            Path(__file__).resolve().parents[1]
            / "protocols/public-agent-trace-fixed32-v1.json"
        ),
    )
    save(out / "contract.json", contract)
    save(out / "checksums.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(
        json.dumps(
            dict(
                contract=str(out / "contract.json"),
                sha256=sha(out / "contract.json"),
                tasks=TASKS,
                cohort_token_lengths={
                    key: [r["input_tokens"] for r in rows]
                    for key, rows in cohorts.items()
                },
                model_requests=0,
            )
        )
    )
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model, args.cache, args.out)
