"""Recompute live context curves and per-tenant observations from sealed SSE."""

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

from original_audit import reconstruct
from qualification import sealed_files
from telemetry_audit import utilization


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metrics(rows, start, end):
    successful = [row for row in rows if row["success"]]
    ttft = [
        1000 * (r["first_token"]["monotonic"] - r["start"]["monotonic"])
        for r in successful
    ]
    tpot = [
        1000
        * (r["end"]["monotonic"] - r["first_token"]["monotonic"])
        / (r["usage"]["completion_tokens"] - 1)
        for r in successful
    ]

    def describe(values):
        return (
            dict(
                mean=statistics.fmean(values),
                p95=percentile(values, 0.95),
                p99=percentile(values, 0.99),
            )
            if values
            else dict(mean=None, p95=None, p99=None)
        )

    tokens = sum(row["usage"]["completion_tokens"] for row in successful)
    return dict(
        requests=len(rows),
        successful=len(successful),
        failed=len(rows) - len(successful),
        duration_seconds=end - start,
        output_tokens=tokens,
        output_tokens_per_second=tokens / (end - start) if end > start else None,
        ttft_ms=describe(ttft),
        tpot_ms=describe(tpot),
    )


def inspect(root):
    root = Path(root).resolve()
    sealed_files(root)
    receipt = json.loads((root / "receipt.json").read_bytes())
    if receipt.get("quiescent") is not True:
        raise ValueError("An owned released acceptance supplement is required")
    from completion_warmup import audit

    folder = root / "service/acceptance"
    if not (folder / "execution.json").exists():
        if receipt.get("completed"):
            raise ValueError("Completed supplement is missing its execution ledger")
        return dict(
            schema="vllm-hust-context-tenant-report/v1",
            role=receipt["role"],
            execution_complete=False,
            failure=receipt.get("failure", "No formal acceptance execution recorded"),
            model_service_started=receipt.get("model_service_started", False),
            contract=None,
            phases=[],
            serving_warmup=None,
            limits=[
                "Unexecuted phases have no throughput, latency or utilization value."
            ],
        )
    if not receipt.get("acceptance_suite"):
        raise ValueError("Acceptance workload admission is missing")
    warmup = audit(root)
    execution = json.loads((folder / "execution.json").read_bytes())
    contract = json.loads((folder / "contract.json").read_bytes())
    from agent_trace import sha

    if (
        sha(folder / "contract.json") != receipt["acceptance_suite"]["contract_sha256"]
        or sha(folder / "request-plan.json") != contract["input_sha256"]
    ):
        raise ValueError("Actual acceptance plan differs from frozen admission")
    inputs = json.loads((folder / "request-plan.json").read_bytes())
    expected = [
        (spec["id"] + "-" + suffix, spec, seconds)
        for spec in contract["phases"]
        for suffix, seconds in (
            ("warmup", spec["warmup_seconds"]),
            ("formal", spec["formal_seconds"]),
        )
    ]
    actual_names = [phase["name"] for phase in execution["phases"]]
    if actual_names != [name for name, _, _ in expected][: len(actual_names)] or len(
        actual_names
    ) > len(expected):
        raise ValueError("Missing, duplicate or reordered phase ledger")
    if receipt["completed"] and (
        execution["status"] != "completed" or len(actual_names) != len(expected)
    ):
        raise ValueError("Successful run does not prove all eight original windows")
    listed = [name for phase in execution["phases"] for name in phase["request_files"]]
    observed = {str(p.relative_to(folder)) for p in folder.rglob("request.json")}
    if len(listed) != len(set(listed)) or set(listed) != observed:
        raise ValueError("Request inventory is incomplete or duplicated")
    samples = [
        json.loads(line)
        for line in (root / "device-samples.jsonl").read_text().splitlines()
        if line
    ]
    devices = json.loads((root / "resource-admission.json").read_bytes())[
        "physical_cards"
    ]
    phases, seen, salts = [], set(), set()
    for phase, (_, spec, seconds) in zip(execution["phases"], expected, strict=False):
        if (
            phase["cohort"],
            phase["concurrency"],
            phase["tenants"],
            phase["seconds"],
        ) != (spec["cohort"], spec["concurrency"], spec["tenants"], seconds):
            raise ValueError("Phase scope or budget differs from the frozen contract")
        rows = []
        for name in phase["request_files"]:
            path = folder / name
            if not path.resolve().is_relative_to(folder) or path.is_symlink():
                raise ValueError("Request evidence escapes its own run")
            row = json.loads(path.read_bytes())
            if row["id"] in seen or row["phase"] != phase["name"]:
                raise ValueError("Duplicated or foreign phase request")
            if name != f"{phase['name']}/{row['id']}/request.json":
                raise ValueError("Request file is not owned by its declared phase")
            lane = row["lane"]
            if type(lane) is not int or not 0 <= lane < spec["concurrency"]:
                raise ValueError("Request lane is outside the frozen concurrency")
            index = (lane // spec["tenants"]) % len(inputs[spec["cohort"]])
            case = inputs[spec["cohort"]][index]
            body = dict(case["request"], user=f"tenant-{lane % spec['tenants']}")
            salt = row["request"].get("cache_salt")
            if (
                not salt
                or salt in salts
                or row["request"] != dict(body, cache_salt=salt)
                or row["tenant"] != body["user"]
                or row["case_index"] != index
                or row["input_tokens"] != case["input_tokens"]
            ):
                raise ValueError(
                    "Actual request, tenant mapping or fresh KV salt differs"
                )
            salts.add(salt)
            seen.add(row["id"])
            if row["success"]:
                wire = Path(row["stream_capture"])
                if (
                    not wire.resolve().is_relative_to(folder)
                    or hashlib.sha256(wire.read_bytes()).hexdigest()
                    != row["stream_sha256"]
                ):
                    raise ValueError("Raw SSE identity differs")
                first, last, usage, finish = reconstruct(wire.read_bytes())
                if (
                    first != row["first_token"]["monotonic"]
                    or last != row["last_token"]["monotonic"]
                    or usage != row["usage"]
                    or finish != row["finish_reason"]
                    or finish != "length"
                    or usage["completion_tokens"] != 256
                    or usage["prompt_tokens"] != row["input_tokens"]
                    or usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                ):
                    raise ValueError("Original stream does not prove the scalar result")
            rows.append(row)
        start, end = phase["start"]["monotonic"], phase["end"]["monotonic"]
        if not all(
            start <= row["start"]["monotonic"] < phase["submission_deadline_monotonic"]
            and row["end"]["monotonic"] <= end
            for row in rows
        ):
            raise ValueError("Request outside the actual submitted/drained phase")
        if phase["submission_deadline_monotonic"] != start + seconds:
            raise ValueError("Submission clock differs from the complete frozen window")
        if phase["status"] == "completed" and (
            end < start + seconds
            or any(not r["success"] for r in rows)
            or {r["lane"] for r in rows} != set(range(spec["concurrency"]))
        ):
            raise ValueError(
                "Completed phase omitted lanes, failures or part of the window"
            )
        value = dict(
            phase=phase["name"],
            cohort=phase["cohort"],
            tenants=phase["tenants"],
            concurrency=phase["concurrency"],
            status=phase["status"],
            metrics=metrics(rows, start, end),
            utilization=utilization(samples, start, end, tuple(map(str, devices))),
            windows=[],
            per_tenant={},
        )
        for tenant in sorted({r["tenant"] for r in rows}):
            group = [r for r in rows if r["tenant"] == tenant]
            value["per_tenant"][tenant] = metrics(group, start, end)
        for index in range(math.ceil(phase["seconds"] / 60)):
            lo, hi = (
                start + 60 * index,
                min(start + 60 * (index + 1), phase["submission_deadline_monotonic"]),
            )
            group = [r for r in rows if lo <= r["start"]["monotonic"] < hi]
            drained = max([hi] + [r["end"]["monotonic"] for r in group])
            value["windows"].append(
                dict(
                    submission_window=index,
                    start_monotonic=lo,
                    submission_end_monotonic=hi,
                    drain_end_monotonic=drained,
                    metrics=metrics(group, lo, drained),
                )
            )
        phases.append(value)
    return dict(
        schema="vllm-hust-context-tenant-report/v1",
        role=receipt["role"],
        execution_complete=execution["status"] == "completed" and receipt["completed"],
        contract=contract,
        phases=phases,
        failure=execution.get("error") or receipt.get("failure"),
        serving_warmup=warmup,
        limits=[
            "Window throughput uses a60s submission cohort and its full drain; it is not a count of SSE chunks or an instantaneous device rate.",
            "Four user-labelled client queues share one service; no VM/security isolation claim.",
            "Source task histories are unchanged; output256 is an explicitly separate stress budget, not fixed-five task quality.",
            "No automatic numerical stability/utilization cutoff is invented. The original qualitative requirements are judged from all curves, per-tenant counts and raw evidence.",
        ],
    )


def report(root, output):
    data = inspect(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for phase in data["phases"]:
        if not phase["phase"].endswith("-formal"):
            continue
        for axis, kind in zip(axes, ("throughput", "ttft", "tpot"), strict=True):
            values = [
                w["metrics"]["output_tokens_per_second"]
                if kind == "throughput"
                else w["metrics"][kind + "_ms"]["p99"]
                for w in phase["windows"]
            ]
            axis.plot(
                range(1, len(values) + 1), values, marker="o", label=phase["phase"]
            )
            axis.set_ylabel(
                "output tok/s (cohort+drain)"
                if kind == "throughput"
                else kind.upper() + " P99 ms"
            )
            axis.set_xlabel("60s submission cohort")
            axis.grid(alpha=0.2)
    if axes[0].lines:
        axes[0].legend(fontsize=8)
    fig.suptitle("Actual context and tenant load; all formal cohorts retained")
    fig.tight_layout()
    fig.savefig(output / "context-tenant-curves.svg")
    fig.savefig(output / "context-tenant-curves.png", dpi=150)
    plt.close(fig)
    (output / "report.html").write_text(
        '<meta charset="utf-8"><h1>真实上下文与多租户实测</h1><img style="max-width:100%" src="context-tenant-curves.svg"><pre>'
        + html.escape(json.dumps(data, ensure_ascii=False, indent=2))
        + "</pre>"
    )
    print("ACCEPTANCE_REPORT", output / "report.html", flush=True)


def report_pair(source, output):
    from acceptance_suite import paired

    data = json.loads(Path(source).read_bytes())
    checked = paired({role: Path(path) for role, path in data["arms"].items()})
    if data != checked:
        raise ValueError("Paired summary differs from recomputed sealed arm evidence")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), squeeze=False)
    rows = []
    for column, comparison in enumerate(data["comparison"]):
        for role in ("B0", "B1"):
            phase = comparison[role]
            for row, field in enumerate(("throughput", "ttft", "tpot")):
                values = [
                    w["metrics"]["output_tokens_per_second"]
                    if field == "throughput"
                    else w["metrics"][field + "_ms"]["p99"]
                    for w in phase["windows"]
                ]
                axes[row, column].plot(
                    range(1, len(values) + 1), values, marker="o", label=role
                )
                axes[row, column].set_xlabel("60s submission cohort (full drain)")
                axes[row, column].set_ylabel(
                    "output tok/s"
                    if field == "throughput"
                    else field.upper() + " P99 ms"
                )
                axes[row, column].grid(alpha=0.2)
                axes[row, column].legend()
        axes[0, column].set_title(comparison["phase"])
        rows.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(value) + "</td>"
                for value in [
                    comparison["phase"],
                    f"{comparison['throughput_ratio']:.3f}",
                    f"{comparison['mean_ttft_reduction_percent']:.2f}%",
                    f"{comparison['mean_tpot_reduction_percent']:.2f}%",
                    f"{comparison['equal_rate_same_card_count_unit_cost_reduction_percent']:.2f}%",
                ]
            )
            + "</tr>"
        )
    fig.suptitle("Sealed paired context/tenant supplement; quality not measured")
    fig.tight_layout()
    fig.savefig(output / "paired-curves.svg")
    fig.savefig(output / "paired-curves.png", dpi=150)
    plt.close(fig)
    (output / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    (output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Context / tenant B0-B1 supplement</title><style>body{font-family:system-ui;max-width:1200px;margin:auto;padding:24px}img{max-width:100%}table{border-collapse:collapse}th,td{padding:12px;border:1px solid #888}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>真实上下文 / 多租户 B0-B1 专项</h1><p>固定五个真实历史任务，4K–8K / 16K–24K / 32K–48K 输入区间。每项60秒预热、300秒正式窗口；输出256、强制预算终止，属于独立压力工况。八条并发通道重复固定任务不等于新增独立样本；四个逻辑租户共享一个服务，不声称安全隔离。费用降幅仅为同卡数同费率推导；不测任务质量，不包含有效算力参考。原定性稳定性及利用率要求结合全部曲线判断，不另造数值门槛。</p><table><thead><tr><th>正式阶段</th><th>吞吐倍数</th><th>平均TTFT降低</th><th>平均TPOT降低</th><th>同费率单位token成本降低</th></tr></thead><tbody>'
        + "".join(rows)
        + '</tbody></table><img src="paired-curves.svg" alt="All B0 and B1 formal submission windows"><details><summary>封存来源与完整数值</summary><pre>'
        + html.escape(json.dumps(data, ensure_ascii=False, indent=2))
        + "</pre></details>"
    )
    print("ACCEPTANCE_PAIRED_REPORT", output / "report.html", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run", type=Path)
    choice.add_argument("--paired-summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.paired_summary:
        report_pair(args.paired_summary, args.out)
    else:
        report(args.run, args.out)
