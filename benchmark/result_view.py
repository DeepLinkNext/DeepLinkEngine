"""Read an existing full acceptance result for a terminal or a JSON consumer."""

import hashlib
import json
from pathlib import Path
import sys

from benchmark.full_report import DATASETS, NAMES, number
from benchmark.run_progress import _console_line, configured_output, progress_path
from benchmark.run_progress import snapshot as progress_snapshot

SUMMARY = Path("full-report/summary.json")
SCHEMA = "deeplinkengine-full-acceptance-report/v1"
ARMS = ("B0", "B1")
TARGETS = {
    "4.1": "Effective compute",
    "4.2": "Throughput and latency",
    "4.3": "Context stability",
    "4.4": "Capacity and unit cost",
}


def load_summary(path, expected_digest=None):
    raw = path.read_bytes()
    if expected_digest is not None and hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("The saved summary differs from the result receipt: " + str(path))
    summary = json.loads(raw)
    if (not isinstance(summary, dict) or summary.get("schema") != SCHEMA
            or type(summary.get("execution_complete")) is not bool
            or not isinstance(summary.get("targets"), dict)
            or not isinstance(summary.get("workloads"), dict)):
        raise ValueError("Unsupported full acceptance summary: " + str(path))
    return summary


def with_summary(result, path, digest=None):
    summary = load_summary(path, digest)
    if result["state"] == "completed" and not summary["execution_complete"]:
        raise ValueError("The run receipt and report disagree on execution: " + str(path))
    report = path.with_name("report.html")
    return dict(
        result, summary=summary, summary_path=str(path),
        summary_digest_verified=digest is not None,
        report_path=str(report) if report.is_file() else None,
    )


def read_results(source):
    source = Path(source).expanduser().resolve()
    if source.is_file() or (source.suffix == ".json" and not source.exists()):
        if source.name == "results.json":
            return read_results(source.parent)
        if source.parent.name == SUMMARY.parent.name and source.name == SUMMARY.name:
            run = source.parent.parent
            if (run / "results.json").is_file() or progress_path(run).is_file():
                return read_results(run)
        return with_summary(dict(state="summary_only", run=None), source)

    progress = progress_snapshot(source)
    result = {key: progress[key] for key in ("state", "run_id", "stage", "failure", "returncode", "results_path")}
    result["run"] = str(source)
    report = source / SUMMARY
    if result["state"] == "completed":
        receipt = json.loads((source / "results.json").read_bytes())
        if receipt.get("status") != "completed" or receipt.get("progress_id") != result["run_id"]:
            raise ValueError("The run changed while reading its results: " + str(source))
        result["automated_acceptance_complete"] = receipt.get("automated_acceptance_complete")
        digest = receipt.get("evidence_sha256", {}).get(SUMMARY.as_posix())
        return with_summary(result, report, digest)
    if result["state"] == "not_started" and report.is_file():
        result["state"] = "summary_only"
        return with_summary(result, report)
    return result


def decision(goal):
    evidence = goal.get("evidence_status")
    if evidence == "missing":
        return "MISSING COST EVIDENCE"
    if evidence == "missing_measurement":
        return "MISSING MEASUREMENT"
    met = goal.get("met")
    return "PASS" if met is True else "FAIL" if met is False else "--"


def metric(value):
    parsed = number(value)
    return f"{parsed:.2f}" if parsed is not None else "--"


def pair_value(arms, key, percentile=None):
    values = [arms.get(role, {}).get(key) for role in ARMS]
    if percentile is not None:
        values = [value.get(percentile) if isinstance(value, dict) else None for value in values]
    return " -> ".join(metric(value) for value in values)


def table(headers, rows):
    lines = [headers, *rows]
    widths = [max(map(len, column)) for column in zip(*lines)]
    return "\n".join("  ".join(value.ljust(width) for value, width in zip(line, widths)).rstrip()
                     for line in lines)


def render(result):
    lines = [f"Run state  {result['state']}"]
    if result.get("run"):
        lines.append(f"Run        {result['run']}")
    if result.get("stage"):
        lines.append(f"Stage      {result['stage']}")
    if result.get("failure"):
        lines.append(f"Failure    {result['failure']}")
    if result.get("returncode") is not None and result["state"] in {"failed", "interrupted"}:
        lines.append(f"Exit code  {result['returncode']}")

    summary = result.get("summary")
    if summary is None:
        lines.append("Full report unavailable.")
        if result.get("results_path"):
            lines.append(f"Receipt    {result['results_path']}")
        return "\n".join(lines)

    lines.append(f"Engine     {summary.get('engine_release') or '--'}")
    lines.append(f"Execution  {'complete' if summary['execution_complete'] else 'incomplete'}")
    lines.append(f"Summary    {result['summary_path']}")
    summary_identity = "verified against the result receipt" if result["summary_digest_verified"] else "no receipt digest"
    lines.append(f"SHA-256    {summary_identity}")

    targets = summary["targets"]
    lines.extend(("", table(
        ("Target", "Milestone", "Final"),
        [(f"{code} {name}", decision(targets.get(code, {}).get("milestone", {})),
          decision(targets.get(code, {}).get("final", {}))) for code, name in TARGETS.items()],
    )))
    efficiency = targets.get("4.1", {}).get("efficiency_lower_percent")
    if efficiency is not None:
        lines.append(f"B1 compute efficiency lower bound  {metric(efficiency)}%")
    cost = targets.get("4.4", {})
    if cost.get("cost_basis"):
        lines.append(f"Cost basis  {cost['cost_basis']}")
        for label, key in (("MMLU cost reduction", "unit_cost_reduction_percent"),
                           ("Four-tenant cost reduction", "four_tenant_cost_reduction_percent")):
            if cost.get(key) is not None:
                lines.append(f"{label}  {metric(cost[key])}%")

    workloads = summary["workloads"]
    if workloads:
        rows = []
        for name in DATASETS:
            pair = workloads.get(name, {})
            arms = pair.get("arms", {})
            complete = pair.get("complete")
            execution = "complete" if complete is True else "incomplete" if complete is False else "--"
            quality = "trace only" if name in ("swe", "terminal") else decision(pair.get("quality", {}))
            rows.append((NAMES[name], pair_value(arms, "output_tokens_per_second"),
                         pair_value(arms, "mean_ttft_ms"), pair_value(arms, "mean_tpot_ms"),
                         execution, quality))
        lines.extend(("", "Fixed five workloads (B0 -> B1)", table(
            ("Dataset", "Output tok/s", "Mean TTFT ms", "Mean TPOT ms", "Execution", "Quality"),
            rows,
        )))

    if contexts := summary.get("context_results"):
        rows = []
        for row in contexts:
            arms = {role: row.get(role, {}).get("metrics", {}) for role in ARMS}
            rows.append((row["phase"], pair_value(arms, "output_tokens_per_second"),
                         pair_value(arms, "ttft_ms", "p95"), pair_value(arms, "tpot_ms", "p95")))
        lines.extend(("", "Context and tenant workloads (B0 -> B1)", table(
            ("Phase", "Output tok/s", "P95 TTFT ms", "P95 TPOT ms"), rows,
        )))

    if result["report_path"]:
        lines.append(f"\nHTML report  {result['report_path']}")
    return "\n".join(lines)


def show_results(source, *, config=None, as_json=False):
    try:
        out = source if source is not None else configured_output(config or Path("evaluation.json"))
        result = read_results(out)
        if as_json:
            _console_line(json.dumps(result, ensure_ascii=False), fallback=lambda: json.dumps(result))
        else:
            _console_line(render(result))
        return 0
    except (OSError, ValueError) as error:
        _console_line("RESULTS_ERROR " + str(error), stream=sys.stderr)
        return 2
