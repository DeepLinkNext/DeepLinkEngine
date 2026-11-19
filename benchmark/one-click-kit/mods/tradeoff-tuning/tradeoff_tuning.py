"""Offline, source-bound operating-point explorer; no inference or deployment."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

SCHEMAS = {
    "vllm-hust-capacity-analysis/v2",
    "vllm-hust-padding-analysis/v5",
    "vllm-hust-state-policy-pilot-analysis/v2",
    "vllm-hust-native-overlap-pilot-analysis/v1",
    "vllm-hust-decode-capacity-analysis/v1",
    "vllm-hust-native-weight-layout-analysis/v1",
    "vllm-hust-nz-scalar-compat-analysis/v1",
    "vllm-hust-nz-layout-cache-four-cell-analysis/v1",
    "vllm-hust-expert-nz-linear-nd-analysis/v1",
    "vllm-hust-joint-policy-analysis/v1",
    "vllm-hust-native-off-ablation-analysis/v1",
    "vllm-hust-native-capacity28-analysis/v1",
}
LEGACY = {
    "vllm-hust-capacity-analysis/v2",
    "vllm-hust-padding-analysis/v5",
    "vllm-hust-state-policy-pilot-analysis/v2",
    "vllm-hust-native-overlap-pilot-analysis/v1",
}
TIMINGS = (
    "output_tokens_per_second",
    "mean_tpot_ms",
    "mean_ttft_ms",
    "tpot_p95_ms",
    "ttft_p95_ms",
    "duration_seconds",
)
FIELDS = (
    "id",
    "execution_seats",
    "planned",
    "recorded",
    "successful",
    "unreturned",
    "correct",
    "truncated",
    "unparsed",
    "output_tokens",
    "seal_sha256",
) + TIMINGS


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(row, schema):
    result = {k: row.get(k) for k in FIELDS}
    admitted = row.get("timing_eligible") is True
    if schema in LEGACY and "timing_eligible" not in row:
        admitted = row.get("completed") is True
        for flag in ("native_inference_complete", "report_pipeline_complete"):
            if flag in row:
                admitted = admitted and row[flag] is True
    if admitted:
        if (
            tuple(
                row.get(k) for k in ("planned", "recorded", "successful", "unreturned")
            )
            != (200, 200, 200, 0)
            or row.get("completed") is not True
            or row.get("quality_measured") is not True
            or not isinstance(row.get("seal_sha256"), str)
            or len(row["seal_sha256"]) != 64
        ):
            raise ValueError("Incomplete/unbound admitted outcome: " + row["id"])
        for key in TIMINGS + ("output_tokens",):
            value = row.get(key)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Invalid timing/count: " + key)
        for key in ("correct", "truncated", "unparsed"):
            if type(row.get(key)) is not int or not 0 <= row[key] <= 200:
                raise ValueError("Invalid full200 task count: " + key)
    else:
        # Failed/pending outcomes remain ledger entries, never zero-speed or zero-score points.
        for key in TIMINGS + ("output_tokens", "correct", "truncated", "unparsed"):
            result[key] = None
    result.update(
        timing_eligible=admitted,
        state=row.get("state", "sealed-complete" if admitted else "not-admitted"),
        failure=row.get("failure") or row.get("validation_failure"),
    )
    return result


def load_catalog(path, source_root=None):
    path = path.resolve()
    source_root = (
        source_root or path.parents[2] / "experiments/optimization-20261115"
    ).resolve()
    raw = path.read_bytes()
    catalog = json.loads(raw)
    if catalog.get("schema") != "vllm-hust-tradeoff-catalog/v1":
        raise ValueError("Versioned explicit catalog required")
    sources, rows, comparisons = [], {}, []
    for spec in catalog["sources"]:
        source = (source_root / spec["path"]).resolve()
        if not source.is_relative_to(source_root):
            raise ValueError("Evidence path must stay inside its source root")
        source_raw = source.read_bytes()
        if digest(source_raw) != spec["sha256"]:
            raise ValueError("Analysis source changed: " + spec["path"])
        data = json.loads(source_raw)
        schema = data.get("schema")
        if schema not in SCHEMAS:
            raise ValueError("Unsupported native analysis schema")
        ids = [r["id"] for r in data["rows"]]
        if len(set(ids)) != len(ids) or sorted(ids) != sorted(spec["trial_ids"]):
            raise ValueError("Keep every declared outcome/repetition in each source")
        if not (
            data.get("all_planned_outcomes_included") is True
            or data.get("all_repetitions_included") is True
        ):
            raise ValueError("Native analysis must retain its complete planned ledger")
        sources.append(dict(spec, schema=schema, protocol=data["protocol"]))
        for original in data["rows"]:
            row = normalize(original, schema)
            trial = row["id"]
            if trial in rows:
                previous = {
                    k: v for k, v in rows[trial].items() if k != "source_indices"
                }
                if previous != row:
                    raise ValueError("Conflicting reuse of physical trial: " + trial)
                rows[trial]["source_indices"].append(len(sources) - 1)
            else:
                rows[trial] = dict(row, source_indices=[len(sources) - 1])
        declared = data.get(
            "pairs", data.get("comparisons", data.get("comparison", {}))
        )
        comparisons.append(
            dict(source_index=len(sources) - 1, native_comparisons=declared)
        )
    configurations = catalog["configurations"]
    assigned = [t for c in configurations for t in c["trial_ids"]]
    if (
        len(set(c["id"] for c in configurations)) != len(configurations)
        or len(set(assigned)) != len(assigned)
        or set(assigned) != set(rows)
    ):
        raise ValueError("Assign every physical trial once; keep all repetitions")
    for c in configurations:
        for trial in c["trial_ids"]:
            rows[trial].update(configuration_id=c["id"], family=c["family"])
    ordered = list(rows.values())
    valid = [r for r in ordered if r["timing_eligible"]]

    def dominates(a, b):
        av = (a["output_tokens_per_second"], -a["mean_tpot_ms"], -a["mean_ttft_ms"])
        bv = (b["output_tokens_per_second"], -b["mean_tpot_ms"], -b["mean_ttft_ms"])
        return all(x >= y for x, y in zip(av, bv)) and any(
            x > y for x, y in zip(av, bv)
        )

    frontier = [r["id"] for r in valid if not any(dominates(s, r) for s in valid)]
    return dict(
        schema="vllm-hust-tradeoff-evidence/v1",
        catalog_sha256=digest(raw),
        scope=catalog["scope"],
        sources=sources,
        configurations=configurations,
        rows=ordered,
        comparisons=comparisons,
        physical_trials=len(ordered),
        admitted_trials=len(valid),
        source_row_references=sum(len(s["trial_ids"]) for s in sources),
        observed_performance_nondominated=frontier,
        quality_qualified=False,
        stability_qualified=False,
        candidate_promoted=False,
        limits=[
            "Hash-bound native analyses, not a fresh verification of their raw archives; restore raw/seals and use each native analyzer for audit.",
            "Historical protocols share a declared workload but include runtime/source differences. Scatter is descriptive; causal claims use the retained native comparisons only.",
            "Every physical repetition retained; reused references deduplicated by exact trial/metric/seal identity, not new measurements.",
            "Observed nondominance uses throughput/mean TPOT/mean TTFT only, not task quality; no connected/fitted global frontier or production recommendation.",
            "Identical questions are reused; runs are not independent tasks. Generated lengths and scores vary. Burst TTFT includes queueing; startup/warmup excluded from inference clocks.",
            "No equal-budget autotuning contribution, quality equivalence, chip MFU, universal gain or five-dataset acceptance claim.",
        ],
    )


def shortlist(evidence, constraints):
    if evidence.get("schema") != "vllm-hust-tradeoff-evidence/v1":
        raise ValueError("Tradeoff evidence required")
    if set(constraints) != {
        "max_tpot_ms",
        "max_ttft_ms",
        "min_correct",
        "max_truncated",
    }:
        raise ValueError("Explicit latency and observed-task constraints required")
    for key in ("max_tpot_ms", "max_ttft_ms"):
        if not math.isfinite(constraints[key]) or constraints[key] <= 0:
            raise ValueError("Finite positive latency budget required")
    for key in ("min_correct", "max_truncated"):
        if type(constraints[key]) is not int or not 0 <= constraints[key] <= 200:
            raise ValueError("Task count budget must be an integer in [0,200]")
    indexed = {r["id"]: r for r in evidence["rows"]}
    results = []
    for config in evidence["configurations"]:
        reasons = []
        if not config["trial_ids"]:
            raise ValueError("Empty declared configuration")
        for trial in config["trial_ids"]:
            row = indexed[trial]
            if not row["timing_eligible"]:
                reasons.append(
                    dict(trial=trial, reason="not-admitted", state=row["state"])
                )
                continue
            for metric, constraint, maximum in (
                ("mean_tpot_ms", "max_tpot_ms", True),
                ("mean_ttft_ms", "max_ttft_ms", True),
                ("correct", "min_correct", False),
                ("truncated", "max_truncated", True),
            ):
                value, bound = row[metric], constraints[constraint]
                if (maximum and value > bound) or (not maximum and value < bound):
                    reasons.append(
                        dict(
                            trial=trial,
                            metric=metric,
                            observed=value,
                            constraint=constraint,
                            bound=bound,
                        )
                    )
        results.append(
            dict(
                configuration_id=config["id"],
                trial_ids=config["trial_ids"],
                all_observed_repetitions_meet_constraints=not reasons,
                reasons=reasons,
                deployable=False,
            )
        )
    return dict(
        schema="vllm-hust-tradeoff-shortlist/v1",
        constraints=constraints,
        results=results,
        candidate_promoted=False,
        note="All declared repetitions must pass; baseline/all-off choices remain eligible. Counts are observed-task filters, not independent quality equivalence. No best-run ranking, launch or deployment.",
    )


def write_json(path, data):
    with path.open("x") as stream:
        stream.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def render(evidence, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "tradeoff.json", evidence)
    rows = evidence["rows"]
    with (out / "all-trials.csv").open("x", newline="") as stream:
        columns = (
            ["point", "configuration_id", "family", "state", "timing_eligible"]
            + list(FIELDS)
            + ["failure"]
        )
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dict(row, point=i) for i, row in enumerate(rows, 1))
    families = list(dict.fromkeys(r["family"] for r in rows))
    palette = plt.get_cmap("tab10")
    colors = {name: palette(i) for i, name in enumerate(families)}
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    views = [
        ("mean_tpot_ms", 1, "Mean TPOT (ms/token; lower is better)"),
        ("mean_ttft_ms", 1000, "Mean TTFT (s; includes burst queue; lower is better)"),
        ("correct", 2, "Observed correct answers /200 (%)"),
        ("truncated", 1, "Truncated answers /200 (lower is better)"),
    ]
    for ax, (metric, scale, label) in zip(axes.flat, views):
        for i, row in enumerate(rows, 1):
            if not row["timing_eligible"]:
                continue
            ax.scatter(
                row[metric] / scale,
                row["output_tokens_per_second"],
                s=70,
                color=colors[row["family"]],
                alpha=0.85,
                marker="D"
                if row["id"] in evidence["observed_performance_nondominated"]
                else "o",
            )
            ax.annotate(
                str(i),
                (row[metric] / scale, row["output_tokens_per_second"]),
                xytext=(4, (5, -11, 13)[i % 3]),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Output throughput (tokens/s)")
        ax.grid(alpha=0.2)
        ax.margins(0.16)
    handles = [
        Line2D([], [], marker="o", ls="", color=colors[f], label=f) for f in families
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            color="#333333",
            label="Observed performance-nondominated",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Inference tradeoffs | Qwen3.5-35B-A3B, dual Ascend 910B2, MMLU200\n"
        + f"{evidence['admitted_trials']}/{evidence['physical_trials']} physical trials admitted; all repetitions retained, references deduplicated",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.014,
        "Point numbers map to all-trials.csv and the companion table. Failed/pending trials have no plotted timing or score.\nHistorical observations; output lengths vary. Diamonds ignore quality. No deployment or quality-equivalence claim.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.88, bottom=0.18, hspace=0.35, wspace=0.24)
    for suffix in ("png", "svg"):
        fig.savefig(out / ("tradeoff." + suffix), dpi=170)
    plt.close(fig)
    # Keep the full ledger legible separately from the scatter, including failed trials.
    fig, ax = plt.subplots(figsize=(17, max(5, 0.35 * len(rows) + 2)))
    ax.axis("off")
    cells = []
    for i, r in enumerate(rows, 1):
        admitted = r["timing_eligible"]

        def cell(k, scale=1, precision=2):
            return f"{r[k] / scale:.{precision}f}" if admitted else "—"

        cells.append(
            [
                i,
                r["id"],
                r["state"],
                cell("output_tokens_per_second"),
                cell("mean_tpot_ms", precision=3),
                cell("mean_ttft_ms", 1000),
                f"{r['correct']}/200" if admitted else "unmeasured",
                r["truncated"] if admitted else "unmeasured",
            ]
        )
    table = ax.table(
        cellText=cells,
        colLabels=[
            "Point",
            "Physical trial",
            "State",
            "tok/s",
            "TPOT ms",
            "TTFT s",
            "Correct",
            "Trunc",
        ],
        colWidths=[0.05, 0.34, 0.14, 0.10, 0.10, 0.09, 0.10, 0.07],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    fig.suptitle("Complete physical-trial ledger | no best-run selection", fontsize=14)
    for suffix in ("png", "svg"):
        fig.savefig(out / ("all-trials." + suffix), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    plot = subs.add_parser("plot")
    plot.add_argument("--catalog", type=Path, required=True)
    plot.add_argument(
        "--source-root",
        type=Path,
        help="Experiment root; defaults to repository experiments/optimization-20261115",
    )
    plot.add_argument("--out", type=Path, required=True)
    choose = subs.add_parser("shortlist")
    choose.add_argument("--evidence", type=Path, required=True)
    choose.add_argument("--max-tpot-ms", type=float, required=True)
    choose.add_argument("--max-ttft-ms", type=float, required=True)
    choose.add_argument("--min-correct", type=int, required=True)
    choose.add_argument("--max-truncated", type=int, required=True)
    choose.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plot":
        render(load_catalog(args.catalog, args.source_root), args.out)
    else:
        raw = args.evidence.read_bytes()
        evidence = json.loads(raw)
        result = shortlist(
            evidence,
            dict(
                max_tpot_ms=args.max_tpot_ms,
                max_ttft_ms=args.max_ttft_ms,
                min_correct=args.min_correct,
                max_truncated=args.max_truncated,
            ),
        )
        result["evidence_sha256"] = digest(raw)
        write_json(args.out, result)


if __name__ == "__main__":
    main()
