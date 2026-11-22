"""Recompute candidate report arithmetic from the published, sealed summaries."""

import hashlib
import html
import json
import math
from pathlib import Path
import re

from full_report import COUNTS, NAMES
from release_selection import archived_candidate, archived_candidates


ROOT = Path(__file__).resolve().parent
BASELINE = "original018-cann8.5.1"
DATASET_NAMES = {**NAMES, "swe": "SWE-bench Pro 公开轨迹", "terminal": "Terminal-Bench 2.1 公开轨迹"}
PROFILES = {
    **{key: (DATASET_NAMES[key], count) for key, count in COUNTS.items()},
    "4096-formal": ("合成闭环 4K / C8", None),
    "16384-formal": ("合成闭环 16K / C8", None),
}
ARM_NAMES = {"B0": "固定 B0 0.18", "r10_original": "原始 r10",
             "native_rms": "原生 RMS（封存对照）", "candidate": "所选候选"}
OUT_OF_SCOPE = {
    "4.1": "缺该候选的 V2 原生计时及有效算力认证。",
    "4.2": "已复算固定五项的汇总指标，尚未完成 V2 原件与质量门槛核验。",
    "4.3": "已有两档合成 C8。V2 需要真实数据长上下文和四租户证据。",
    "4.4": "已有吞吐和耗时。V2 还需要相应工况、占卡计量和同窗利用率。",
}


def load_archive(version):
    binding = archived_candidate(version)
    if binding is None:
        choices = ", ".join(row["candidate"]["version"] for row in archived_candidates())
        raise ValueError(f"Unknown archived candidate {version!r}. Available versions: {choices}")
    path = ROOT / binding["path"]
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT / "benchmark/reports"):
        raise ValueError("Invalid archived candidate path")
    raw = path.read_bytes()
    if len(raw) != binding["bytes"] or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("Archived candidate size or SHA-256 mismatch")
    archive = json.loads(raw)
    identity = binding["candidate"]
    if (
        archive.get("schema") != "vllm-hust-r10-full-fixed-five-evidence/v1"
        or archive.get("release") != identity["release"]
        or archive.get("baseline_id") != BASELINE
        or archive.get("full_request_count_per_solver_arm") != sum(COUNTS.values())
        or set(archive.get("measurements", {})) != set(PROFILES)
    ):
        raise ValueError("Archived candidate identity or workload scope differs")
    return binding, archive


def recompute_arm(arm, count):
    if (
        type(arm["requests"]) is not int or arm["requests"] <= 0
        or (count is not None and arm["requests"] != count)
        or type(arm["failed"]) is not int or not 0 <= arm["failed"] <= arm["requests"]
        or not re.fullmatch(r"[0-9a-f]{64}", arm["source_seal_sha256"])
        or not all(math.isfinite(arm[key]) and arm[key] > 0
                   for key in ("output_tokens", "duration_seconds", "output_tokens_per_second"))
    ):
        raise ValueError("Invalid archived candidate arm")
    throughput = arm["output_tokens"] / arm["duration_seconds"]
    if not math.isclose(throughput, arm["output_tokens_per_second"], rel_tol=1e-9):
        raise ValueError("Archived candidate throughput differs from tokens / seconds")
    correct = arm.get("correct")
    if correct is not None and (type(correct) is not int or not 0 <= correct <= arm["requests"]):
        raise ValueError("Invalid archived candidate quality denominator")
    stats = {"output_tokens_per_second": throughput}
    if correct is not None:
        stats["correct_percent"] = 100 * correct / arm["requests"]
    return stats


def collect(version):
    binding, archive = load_archive(version)
    roles = tuple(archive["measurements"]["mmlu"])
    if not {"B0", "r10_original", "candidate"} <= set(roles):
        raise ValueError("Missing fixed B0, original r10, or candidate arm")
    workloads = {}
    for profile, (label, count) in PROFILES.items():
        observed = archive["measurements"][profile]
        if set(observed) != set(roles):
            raise ValueError(f"Archived arm coverage differs for {profile}")
        arms = {role: recompute_arm(observed[role], count) for role in roles}
        candidate = arms["candidate"]
        changes = {}
        for role, arm in arms.items():
            if role == "candidate":
                continue
            change = {"throughput_percent": 100 * (candidate["output_tokens_per_second"] /
                                                   arm["output_tokens_per_second"] - 1)}
            if "correct_percent" in candidate and "correct_percent" in arm:
                change["correct_percentage_points"] = candidate["correct_percent"] - arm["correct_percent"]
            changes[role] = change
        workloads[profile] = {
            "dataset": label, "kind": "fixed-profile" if count else "synthetic-c8",
            "arms": arms, "candidate_change_vs": changes,
        }
    return {
        "schema": "deeplinkengine-archived-candidate-report/v1",
        "selection": {**binding["candidate"], "status": "candidate", "baseline_id": BASELINE},
        "source": {key: binding[key] for key in ("path", "source", "bytes", "sha256")},
        "workloads": workloads,
        "archived": archive,
        "submission_v2": {key: {"status": "not_assessed", "reason": value}
                          for key, value in OUT_OF_SCOPE.items()},
    }


def table(headers, rows):
    head = "".join(f"<th scope='col'>{html.escape(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in row) + "</tr>"
                   for row in rows)
    return f"<div class='table'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def report_html(data):
    selection = data["selection"]
    archive = data["archived"]
    rows = []
    for key, workload in data["workloads"].items():
        for role, computed in workload["arms"].items():
            arm = archive["measurements"][key][role]
            rows.append((workload["dataset"], ARM_NAMES.get(role, role), arm["requests"], arm["failed"],
                         f'{arm["correct"]}/{arm["requests"]}' if "correct_percent" in computed else "—",
                         arm.get("unresolved", "—"), f'{computed["output_tokens_per_second"]:,.2f}',
                         f'{arm["mean_ttft_ms"]:,.2f} / {arm["mean_tpot_ms"]:,.2f}',
                         f'{arm["ttft_p95_ms"]:,.2f} / {arm["ttft_p99_ms"]:,.2f}',
                         f'{arm["tpot_p95_ms"]:,.2f} / {arm["tpot_p99_ms"]:,.2f}',
                         f'{arm["end_to_end_p95_ms"]:,.2f} / {arm["end_to_end_p99_ms"]:,.2f}',
                         arm["source_seal_sha256"]))
    comparisons = [
        (workload["dataset"], ARM_NAMES.get(role, role), f'{change["throughput_percent"]:+.2f}%',
         f'{change["correct_percentage_points"]:+.2f}' if "correct_percentage_points" in change else "—")
        for workload in data["workloads"].values()
        for role, change in workload["candidate_change_vs"].items()
    ]
    judges = [(role, judge["planned"], judge["recorded"], judge["unresolved_grades"])
              for role, judge in archive["judges"].items()]
    targets = [(key, "未判定", item["reason"]) for key, item in data["submission_v2"].items()]
    source = data["source"]
    esc = html.escape
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeepLinkEngine 候选封存 CPU 复算</title>
<style>
body{{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17212b;background:#f8fafc}}
.table{{overflow:auto;margin:1rem 0}} table{{border-collapse:collapse;width:100%;background:white}}
th,td{{padding:.65rem;text-align:left;border-bottom:1px solid #cbd5e1}} td:last-child{{overflow-wrap:anywhere}}
</style></head><body>
<h1>DeepLinkEngine 候选封存 CPU 复算</h1>
<p><strong>{esc(selection["version"])}</strong> · {esc(selection["release"])} · 候选</p>
<p>{esc(selection["model"])} · {esc(selection["precision"])} · TP{selection["tensor_parallel_size"]} ·
{esc(selection["accelerator"])} × {selection["accelerator_count"]}。固定五项每臂
{archive["full_request_count_per_solver_arm"]} 条请求，两档合成 C8 独立计时。吞吐按双卡合计输出 token 数除以耗时。</p>
<p><a href="{esc(source["source"], quote=True)}">封存汇总原件</a> ·
<a href="summary.json">复算 JSON</a> · <a href="receipt.json">复算回执</a><br>
封存汇总 SHA-256 <code>{esc(source["sha256"])}</code></p>
<h2>逐工况与运行臂</h2>
{table(("工况", "运行臂", "请求", "失败", "正确 / 计划", "未决", "双卡 tok/s", "平均 TTFT / TPOT（ms）",
        "TTFT P95 / P99（ms）", "TPOT P95 / P99（ms）", "端到端 P95 / P99（ms）", "运行封存 SHA-256"), rows)}
<h2>候选变化</h2>
{table(("工况", "比较对象", "吞吐变化", "正确率变化（百分点）"), comparisons)}
<h2>裁判记录</h2><p>裁判请求独立于求解器吞吐。各运行臂保留自己的判分和分母。</p>
{table(("裁判阶段", "计划", "已记录", "未决"), judges)}
<h2>送测 V2 判定范围</h2>
{table(("指标", "判定", "证据范围"), targets)}
</body></html>
"""


def render(version, out):
    data = collect(version)
    summary = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    document = report_html(data).encode()
    files = {"summary.json": summary, "report.html": document}
    receipt = {
        "schema": "deeplinkengine-archived-candidate-receipt/v1",
        "status": "completed",
        "mode": "cpu-replay",
        "selection": data["selection"],
        "source": data["source"],
        "workloads": list(data["workloads"]),
        "model_requests": 0,
        "npu_tasks_started": 0,
        "outputs": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                    for name, raw in files.items()},
    }
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        (out / name).write_bytes(raw)
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8", newline="\n")
    return data
