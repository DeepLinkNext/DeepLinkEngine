"""Explicit, offline r6 capacity intervention and evidence-bound tradeoff plots."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

SEATS = (24, 26, 32)
BASE = (3, 6, 12, 16, 24, 32, 40, 48, 64, 128, 256, 512, 1024, 1536, 2048, 4096)
BINS = (3, 6, 12, 24, 40, 48, 80, 96, 108, 144, 180)


def captures(seats):
    if type(seats) is not int or seats not in SEATS:
        raise ValueError("Experimental operating points: 24, 26, 32 only")
    maximum = next(n for n in BINS if n >= seats * 3)
    return sorted(set(BASE) | {n for n in BINS if n <= maximum})


def configure(baseline, seats):
    """Return an explicit command proposal; never launch or silently promote it."""
    captures(seats)
    proposal = copy.deepcopy(baseline)
    command, env = proposal["command"], proposal["env"]

    def value(flag):
        if command.count(flag) != 1:
            raise ValueError("Expected one " + flag)
        return command[command.index(flag) + 1]

    if baseline["engine_release"] != "full-answer-b1-20261113-r6":
        raise ValueError("Requires separately verified r6 composition")
    if (
        "--enforce-eager" in command
        or env.get("MTP_TOKENS") != "2"
        or env.get("BETTERSCALE_CONTEXT_PARALLEL") != "1"
    ):
        raise ValueError("Requires FULL / MTP2 / CP1")
    if value("--max-num-seqs") != "24" or value("--max-num-batched-tokens") != "4096":
        raise ValueError("Baseline must be E24 / prefill4096")
    if json.loads(value("--speculative-config")) != dict(
        method="mtp", num_speculative_tokens=2
    ):
        raise ValueError("Composed M01 correction supports MTP2 only")
    state = json.loads(value("--additional-config"))
    expected = dict(
        using_live_runtime=True,
        state_resident_seats=36,
        state_cache_host_bytes=4 * 1024**3,
        state_cache_incremental=True,
        state_cache_streaming=True,
        state_cache_policy=True,
        state_cache_partial_reclaim=True,
    )
    if any(
        type(state.get(k)) is not type(v) or state[k] != v for k, v in expected.items()
    ):
        raise ValueError("Baseline State policy/budget drift")
    graph = json.loads(value("--compilation-config"))
    if graph.get("cudagraph_mode") != "FULL" or sorted(
        graph.get("cudagraph_capture_sizes", [])
    ) != captures(24):
        raise ValueError("Baseline capture policy drift")
    command[command.index("--max-num-seqs") + 1] = str(seats)
    graph["cudagraph_capture_sizes"] = captures(seats)
    command[command.index("--compilation-config") + 1] = json.dumps(graph)
    proposal["capacity_tuning"] = dict(
        schema="capacity-tuning/v1",
        execution_seats=seats,
        diagnostic_only=True,
        production_default_changed=False,
        quality_qualified=False,
        common_warmup_concurrencies=[1, 2, 4, 8, 16, 24, 32],
        required_before_launch=[
            "Verify complete runtime/source identity and M01 composition",
            "Common live32 synthetic warmup followed by both-rank Torch/Triton seal",
            "Freeze identical inputs, EOS, budget, scorer and paired repeated order",
            "Use native owned cleanup, failure records and sealed raw artifacts",
        ],
        note="Proposal only; no new quality/stability certificate. CANN compilation is outside the existing guard.",
    )
    return proposal


def apply(command, env, plan, path):
    """Native experimental controller hook; enforce the common warmup contract."""
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    if (
        set(config) != {"schema", "id", "execution_seats", "diagnostic_only"}
        or config["schema"] != "vllm-hust-capacity-tuning/v1"
        or config["diagnostic_only"] is not True
    ):
        raise ValueError("Explicit diagnostic capacity configuration required")
    warmup = plan.get("serving_warmup", {})
    if (
        warmup.get("policy") != "observed-torch-triton-sealed-task-warmup-v4-capacity32"
        or warmup.get("dense_decode_concurrencies") != [1, 2, 4, 8, 16, 24, 32]
        or warmup.get("formal_clock_excludes_warmup") is not True
    ):
        raise ValueError(
            "Common live32 pre-clock warmup required for every operating point"
        )
    before = list(command)
    proposal = configure(
        dict(command=command, env=env, engine_release=plan["engine_release"]),
        config["execution_seats"],
    )
    command[:] = proposal["command"]
    plan["optimization"] = dict(
        config=config,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        original_command=before,
        effective_command=list(command),
        formal_acceptance=False,
        candidate_stability_qualified=False,
        capacity_tuning=proposal["capacity_tuning"],
    )
    return config


def load_evidence(paths):
    rows, sources, pairs, seen = [], [], [], set()
    for path in paths:
        raw = path.read_bytes()
        obj = json.loads(raw)
        if obj.get("schema") not in (
            "vllm-hust-capacity-analysis/v2",
            "vllm-hust-padding-analysis/v5",
        ):
            raise ValueError("Unrecognized sealed-analysis schema")
        source = dict(
            path=path.name,
            sha256=hashlib.sha256(raw).hexdigest(),
            protocol=obj["protocol"],
            planned_trials=obj["planned_trials"],
            sealed_trials=obj["sealed_trials"],
        )
        sources.append(source)
        for row in obj["rows"]:
            if row["id"] in seen:
                raise ValueError("Duplicate trial: " + row["id"])
            seen.add(row["id"])
            item = dict(row, protocol=obj["protocol"])
            if item["completed"]:
                if (
                    (
                        item["planned"],
                        item["recorded"],
                        item["successful"],
                        item["unreturned"],
                    )
                    != (200, 200, 200, 0)
                    or not item.get("seal_sha256")
                    or not item.get("quality_measured")
                ):
                    raise ValueError(
                        "Incomplete or unbound successful trial: " + item["id"]
                    )
                for key in ("mean_tpot_ms", "mean_ttft_ms", "output_tokens_per_second"):
                    if not math.isfinite(item[key]) or item[key] <= 0:
                        raise ValueError("Invalid timing: " + item["id"])
            rows.append(item)
        pairs.extend(obj.get("pairs", []))
    return dict(
        schema="capacity-tuning-tradeoff/v1",
        sources=sources,
        rows=rows,
        pairs=pairs,
        all_repetitions_included=True,
        quality_qualified=False,
        candidate_promoted=False,
        note="Sealed-analysis inputs are hash-bound here; their raw seals must be verified by the native analyzer. Repeated runs reuse the same200 questions. Lines connect prospective pairs, not a fitted curve or quality-qualified Pareto frontier.",
    )


def render(evidence, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=False)
    (out / "tradeoff.json").write_text(json.dumps(evidence, indent=2) + "\n")
    colors = {24: "#3367a5", 26: "#c17805", 32: "#138575"}
    valid = {r["id"]: r for r in evidence["rows"] if r["completed"]}
    fig = plt.figure(figsize=(13.5, 8.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1.35])
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    table_ax = fig.add_subplot(grid[1, :])
    table_ax.axis("off")
    fig.subplots_adjust(bottom=0.12, top=0.85, wspace=0.26, hspace=0.4)
    for ax, key, scale, xlabel in [
        (axes[0], "mean_tpot_ms", 1, "Mean TPOT (ms/token; lower is better)"),
        (
            axes[1],
            "mean_ttft_ms",
            1000,
            "Mean TTFT (s; includes burst queue; lower is better)",
        ),
    ]:
        for pair in evidence["pairs"]:
            a, b = valid.get(pair["baseline"]), valid.get(pair["candidate"])
            if a and b:
                ax.plot(
                    [a[key] / scale, b[key] / scale],
                    [a["output_tokens_per_second"], b["output_tokens_per_second"]],
                    color="#bdc4cc",
                    lw=1,
                    zorder=1,
                )
        for i, row in enumerate(valid.values()):
            seats = row["execution_seats"]
            ax.scatter(
                row[key] / scale,
                row["output_tokens_per_second"],
                color=colors[seats],
                marker="o" if row["id"].startswith("E02") else "s",
                s=65,
                zorder=3,
            )
            label = str(i + 1)
            # Alternating offsets preserve the nearby control repetitions.
            offset = (5, 9 if i % 2 == 0 else -17)
            ax.annotate(
                label,
                (row[key] / scale, row["output_tokens_per_second"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=7.5,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Output throughput (tokens/s; higher is better)")
        ax.grid(alpha=0.2)
        ax.margins(x=0.25, y=0.22)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", ls="", color=color, label=f"E{seats}")
        for seats, color in colors.items()
    ]
    handles += [
        Line2D([], [], marker="o", color="gray", ls="", label="E02: E24 vs E32"),
        Line2D([], [], marker="s", color="gray", ls="", label="E04: E24 vs E26"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=5,
        frameon=False,
    )
    cells = [
        [
            str(i + 1),
            r["id"],
            str(r["execution_seats"]),
            f"{r['output_tokens_per_second']:.2f}",
            f"{r['mean_tpot_ms']:.3f}",
            f"{r['mean_ttft_ms'] / 1000:.2f}",
            f"{r['correct']}/200",
            str(r["truncated"]),
        ]
        for i, r in enumerate(valid.values())
    ]
    table = table_ax.table(
        cellText=cells,
        colLabels=[
            "Point",
            "Trial",
            "Seats",
            "tok/s",
            "TPOT ms",
            "TTFT s",
            "Correct",
            "Truncated",
        ],
        loc="center",
        cellLoc="center",
        colWidths=[0.06, 0.19, 0.08, 0.12, 0.12, 0.12, 0.13, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    count = sum(s["sealed_trials"] for s in evidence["sources"])
    planned = sum(s["planned_trials"] for s in evidence["sources"])
    fig.suptitle(
        f"Capacity tuning tradeoff | MMLU200, r6 MTP2, TP2, live32 warmup\n{count}/{planned} trials sealed; each label retains the full200 correctness denominator",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.015,
        "All completed repetitions shown; connecting lines denote paired trials. No best-of selection.\nQuality equivalence and capacity stability remain unqualified; timing gains alone do not promote a configuration.",
        ha="center",
        fontsize=9,
    )
    for suffix in ("png", "svg"):
        fig.savefig(out / ("tradeoff." + suffix), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    cfg = subs.add_parser(
        "configure", help="Write an explicit launch proposal; does not start inference"
    )
    cfg.add_argument("--baseline", type=Path, required=True)
    cfg.add_argument("--seats", type=int, required=True)
    cfg.add_argument("--out", type=Path, required=True)
    plot = subs.add_parser(
        "plot", help="Plot every available sealed trial, retaining planned denominator"
    )
    plot.add_argument("--analysis", type=Path, action="append", required=True)
    plot.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "configure":
        with args.out.open("x") as stream:
            json.dump(
                configure(json.loads(args.baseline.read_bytes()), args.seats),
                stream,
                indent=2,
            )
            stream.write("\n")
    else:
        render(load_evidence(args.analysis), args.out)


if __name__ == "__main__":
    main()
