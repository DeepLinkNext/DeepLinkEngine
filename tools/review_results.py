"""Check and display the arithmetic supported by the public benchmark summaries."""

import json
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "benchmark" / "reports"


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def checked(value, expected, label):
    require(isclose(value, expected, rel_tol=1e-8, abs_tol=1e-8), f"Mismatch: {label}")
    return value


def rate(arm, label):
    return checked(
        arm["output_tokens_per_second"],
        arm["output_tokens"] / arm["duration_seconds"],
        label + " output token/s",
    )


def decrease(b0, b1):
    return 100 * (1 - b1 / b0)


def five_rows():
    standard = read(ROOT / "datasets/standard.json")
    summary = read(REPORTS / "r8-fixed-five-20261118/summary-public-v2.json")
    measurements = {row["id"]: row for row in summary["measurements"]}
    require(
        list(measurements) == ["mmlu", "hle", "frontier", "swe", "terminal"]
        and len(summary["measurements"]) == len(measurements),
        "Unexpected workload identity or order",
    )
    window = read(ROOT / "evidence/mmlu-r8-window.json")
    rows = []
    for dataset, (name, pair) in zip(standard["datasets"], measurements.items(), strict=True):
        require(dataset["id"] == name, f"Dataset contract differs: {name}")
        b0, b1 = pair["arms"]
        require(
            pair["baseline_id"] == "original018-cann8.5.1"
            and (b0["role"], b1["role"]) == ("B0", "r8"),
            f"Unexpected pair identity: {name}",
        )
        count = dataset["requests_per_arm"]
        for arm in (b0, b1):
            require(
                arm["requests"] == count and 0 <= arm["failed"] <= count,
                f"Request counts differ: {name}",
            )
        r0, r1 = (rate(arm, f"{name} {arm['role']}") for arm in (b0, b1))
        quality = "—"
        if name in ("mmlu", "hle", "frontier"):
            quality = f"{b0['correct']}/{count} → {b1['correct']}/{count}"
            require(
                all(0 <= arm["correct"] <= count for arm in (b0, b1)),
                f"Invalid correct counts: {name}",
            )
        if name == "mmlu":
            for key, arm in zip(("B0", "B1"), (b0, b1), strict=True):
                same = window["arms"][key]
                require(
                    same["output_tokens"] == arm["output_tokens"]
                    and same["stage_seal_sha256"] == arm["source_seal_sha256"],
                    f"MMLU-Pro window and arm differ: {key}",
                )
                checked(same["formal_window_seconds"], arm["duration_seconds"], key + " window")
        ttft = decrease(b0["mean_ttft_ms"], b1["mean_ttft_ms"])
        tpot = decrease(b0["mean_tpot_ms"], b1["mean_tpot_ms"])
        rows.append(
            f"| {dataset['name']} | {count} | {r0:.2f} → {r1:.2f} | "
            f"{r1 / r0:.2f}× | {ttft:.2f}% | {tpot:.2f}% | {quality} |"
        )
    return rows


def context_rows():
    summary = read(REPORTS / "r8-context-tenant-20261119/summary-public-v1.json")
    rows = []
    for pair in summary["comparisons"]:
        phase = pair["phase"]
        b0, b1 = (pair[arm]["metrics"] for arm in ("B0", "B1"))
        r0, r1 = rate(b0, phase + " B0"), rate(b1, phase + " B1")
        ratio = checked(pair["throughput_ratio"], r1 / r0, phase + " throughput ratio")
        cost = checked(
            pair["equal_rate_same_card_count_unit_cost_reduction_percent"],
            decrease(r1, r0), phase + " same-rate cost",
        )
        for key, metric in (("ttft", "ttft_ms"), ("tpot", "tpot_ms")):
            checked(
                pair[f"mean_{key}_reduction_percent"],
                decrease(b0[metric]["mean"], b1[metric]["mean"]),
                f"{phase} {key}",
            )
        u0, u1 = (pair[arm]["utilization"]["mean_percent"] for arm in ("B0", "B1"))
        rows.append(f"| {phase} | {r0:.2f} → {r1:.2f} | {ratio:.2f}× | {cost:.2f}% | {u0:.2f}% → {u1:.2f}% |")
    require(len(rows) == 4, "Expected three context phases and one four-tenant phase")
    return rows


def main():
    five = five_rows()
    context = context_rows()
    print("固定五项，B0 → B1\n")
    print("| 数据集 | 每臂请求 | 输出 token/s | 吞吐比 | 平均 TTFT 降幅 | 平均 TPOT 降幅 | 正确数 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(*five, sep="\n")
    print("\n长上下文与四租户，单位 token 成本按同费率估算\n")
    print("| 工况 | 输出 token/s | 吞吐比 | 成本降幅 | 采样利用率 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    print(*context, sep="\n")


if __name__ == "__main__":
    main()
