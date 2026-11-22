"""Recompute candidate report arithmetic from the published, sealed summaries."""

import hashlib
import html
import json
import math
from pathlib import Path
import re

from benchmark.candidate_v2 import assess
from benchmark.full_report import COUNTS, LABELS, NAMES
from engine.release_selection import archived_candidate, archived_candidates


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "original018-cann8.5.1"
DATASET_NAMES = {**NAMES, "swe": "SWE-bench Pro 公开轨迹", "terminal": "Terminal-Bench 2.1 公开轨迹"}
PROFILES = {
    **{key: (DATASET_NAMES[key], count) for key, count in COUNTS.items()},
    "4096-formal": ("合成闭环 4K / C8", None),
    "16384-formal": ("合成闭环 16K / C8", None),
}
ARM_NAMES = {"B0": "固定 B0 0.18", "r10_original": "原始 r10",
             "native_rms": "原生 RMS（封存对照）", "candidate": "所选候选"}
FIXED_B0_V2 = "evidence/r10-fixed-b0-v2.json"
ORIGINAL_AUDIT = "evidence/r10-original-evidence-audit-v1.json"
CONTEXT_AUDIT = "evidence/r10-v2-context-supplement.json"
COMPUTE_AUDIT = "evidence/r10-v2-compute-supplement.json"
VERDICTS = {"met": "达成", "not_met": "未达成", "missing_evidence": "证据不足"}


def read_bound_json(binding, scope):
    path = ROOT / binding["path"]
    if path.is_symlink() or not path.resolve().is_relative_to((ROOT / scope).resolve()):
        raise ValueError("Invalid archived candidate path")
    raw = path.read_bytes()
    if len(raw) != binding["bytes"] or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("Archived candidate size or SHA-256 mismatch")
    return json.loads(raw)


def source_identity(binding):
    return {key: binding[key] for key in ("path", "source", "bytes", "sha256")}


def load_archive(version):
    binding = archived_candidate(version)
    if binding is None:
        choices = ", ".join(row["candidate"]["version"] for row in archived_candidates())
        raise ValueError(f"Unknown archived candidate {version!r}. Available versions: {choices}")
    archive = read_bound_json(binding, "benchmark/reports")
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
    pins = {record["path"]: record for record in json.loads((ROOT / "benchmark/provenance.json").read_bytes())["files"]}
    baseline_binding = pins[FIXED_B0_V2]
    baseline = read_bound_json(baseline_binding, "evidence")
    quality_binding = pins[binding["quality_source"]] if "quality_source" in binding else None
    priority = read_bound_json(quality_binding, "benchmark/reports") if quality_binding else None
    audit_binding = pins[ORIGINAL_AUDIT]
    audit = read_bound_json(audit_binding, "evidence")
    if audit.get("schema") != "deeplinkengine-r10-original-evidence-audit/v1":
        raise ValueError("Unknown r10 original evidence audit schema")
    original = audit["candidates"][binding["candidate"]["version"]]
    context_binding = pins.get(CONTEXT_AUDIT)
    context_source = read_bound_json(context_binding, "evidence") if context_binding else {}
    context = context_source.get(binding["candidate"]["version"])
    compute_binding = pins.get(COMPUTE_AUDIT)
    compute_source = read_bound_json(compute_binding, "evidence") if compute_binding else {}
    compute = compute_source.get(binding["candidate"]["version"])
    targets = assess(archive, binding["candidate"], baseline, priority, original, context, compute)
    assessment_sources = {"fixed_b0": {**source_identity(baseline_binding), "projection_inputs": baseline["sources"]}}
    assessment_sources["original_evidence_audit"] = source_identity(audit_binding)
    if context is not None:
        assessment_sources["context_audit"] = source_identity(context_binding)
    if compute is not None:
        assessment_sources["compute_audit"] = source_identity(compute_binding)
    if quality_binding:
        assessment_sources["priority_grades"] = source_identity(quality_binding)
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
        "source": source_identity(binding),
        "assessment_sources": assessment_sources,
        "workloads": workloads,
        "archived": archive,
        "submission_v2": targets,
    }


def table(headers, rows):
    head = "".join(f"<th scope='col'>{html.escape(item)}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in row) + "</tr>"
                   for row in rows)
    return f"<div class='table'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def status_text(result):
    return VERDICTS[result["status"]]


def v2_html(data):
    assessments = data["submission_v2"]
    targets = []
    for indicator, assessment in assessments.items():
        for level, label in (("milestone", "里程碑"), ("final", "结题")):
            result = assessment[level]
            status = status_text(result)
            targets.append((f"{indicator} {LABELS[indicator]}", label, status,
                            assessment.get("available", "两臂固定五项的封存汇总及对应判分。"), result["reason"]))

    speed = assessments["4.2"]
    speed_rows = []
    for name, row in speed["datasets"].items():
        ratios = [f"{float(value):.3f}×" if value is not None else "—" for value in row["ratios"].values()]
        grade = row["quality"]
        drop = grade["drop_pp"]
        if isinstance(drop, dict):
            drop = "、".join(f"{key} {value} pp" for key, value in drop.items())
        elif drop is not None:
            drop += " pp"
        quality = f"{status_text(grade)}（{drop}）" if drop is not None else grade["reason"]
        speed_rows.append((data["workloads"][name]["dataset"], *ratios, quality,
                           status_text(row["milestone"]), row["milestone"]["reason"],
                           status_text(row["final"]), row["final"]["reason"]))

    audit_note = f'<p>{html.escape(speed["raw_reaudit"]["reason"])}</p>'
    if speed["raw_missing"]:
        audit_note += "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in speed["raw_missing"]) + "</ul>"
    costs = assessments["4.4"]
    rent = costs["published_rental"]
    cost_rows = []
    for workload, scenario in (("MMLU-Pro 200 完整回答", costs["mmlu_full_answers"]),
                               ("真实 4K 四租户 C24 · " + "、".join(assessments["4.3"]["baseline_context"]["workload"]["datasets"]),
                                costs["four_tenant"])):
        for role, arm in scenario["arms"].items():
            util = scenario["utilization"][role]
            cost_rows.append((
                workload, "固定 B0 0.18" if role == "B0" else "所选候选",
                f'{float(arm["cost_per_million_output_tokens_cny"]):.4f}' if arm else "缺运行原件",
                f'{util["mean_percent"]:.2f}%' if util else "缺同窗采样",
            ))
    savings = float(costs["mmlu_full_answers"]["unit_cost_reduction_percent"])
    cost_gate = status_text(costs["mmlu_full_answers"]["cost_gate"])
    tenant_block = ""
    if costs["four_tenant"]["unit_cost_reduction_percent"] is not None:
        tenant_savings = float(costs["four_tenant"]["unit_cost_reduction_percent"])
        tenant = next(row for row in assessments["4.3"]["paired"]["comparison"] if row["phase"] == "multi-tenant-formal")
        tenants = []
        for role in ("B0", "B1"):
            for name, metrics in tenant[role]["per_tenant"].items():
                tenants.append((ARM_NAMES["B0" if role == "B0" else "candidate"], name,
                                metrics["requests"], metrics["successful"], metrics["failed"]))
        tenant_block = (f"<p>4K 四租户 C24 单位成本下降 {tenant_savings:.2f}%。</p>"
                        + table(("运行臂", "租户", "请求", "成功", "失败"), tenants))
    labels = {"fixed_b0": "固定 B0 V2 封存", "matched_mmlu_utilization": "同封存 MMLU-Pro 双卡采样",
              "context_workloads": "真实上下文数据集来源", "priority_grades": "原生 RMS 同轮裁判分层对照",
              "original_evidence_audit": "固定五项与同窗 AI Core 原件复算",
              "context_audit": "真实上下文及四租户的封存复算",
              "compute_audit": "MMLU-Pro 200 / out256 原生计时及独立参考复算"}
    links = []
    for kind, source in data["assessment_sources"].items():
        origins = source.get("projection_inputs") or {kind: {"url": source["source"], "sha256": source["sha256"]}}
        for name, origin in origins.items():
            links.append(f'<li><a href="{html.escape(origin["url"], quote=True)}">{html.escape(labels.get(name, name))}</a> '
                         f'<code>{html.escape(origin["sha256"])}</code></li>')
    context = assessments["4.3"]
    context_block = ""
    if "paired" in context:
        rows = []
        phases = {"4096-formal": "4K / C8", "16384-formal": "16K / C8",
                  "32768-formal": "32K / C8", "multi-tenant-formal": "4K 四租户 / C24"}
        workload = context["baseline_context"]["workload"]

        def cell(value):
            return f"{value:.2f}" if value is not None else "—"

        for phase in context["paired"]["comparison"]:
            for role in ("B0", "B1"):
                arm = phase[role]
                metrics = arm["metrics"]
                cards = ", ".join(map(str, context["physical_cards"][role]))
                rows.append((
                    "、".join(workload["datasets"]),
                    phases[phase["phase"]], ARM_NAMES["B0" if role == "B0" else "candidate"], cards,
                    metrics["successful"], metrics["failed"], cell(metrics["output_tokens_per_second"]),
                    " / ".join(cell(metrics["ttft_ms"][q]) for q in ("p95", "p99")),
                    " / ".join(cell(metrics["tpot_ms"][q]) for q in ("p95", "p99")),
                    context["paired"]["arm_seals"][role],
                ))
        headers = ("数据集", "工况", "运行臂", "物理卡", "成功", "失败", "双卡 tok/s",
                   "TTFT P95/P99（ms）", "TPOT P95/P99（ms）", "封存 SHA-256")
        task_ids = "、".join(workload["task_ids"])
        input_sha256 = context["baseline_context"]["input_sha256"]
        context_block = ("<h3>4.3 真实长上下文</h3>"
                         + f'<p>冻结任务 {html.escape(task_ids)}。输入 SHA-256 <code>{input_sha256}</code>。</p>'
                         + table(headers, rows))
    compute = assessments["4.1"]
    compute_block = ""
    if "native" in compute:
        rows = []
        native = compute["native"]
        for role in ("B0", "B1"):
            arm = native["arms"][role]
            card_ids = "、".join(map(str, native["physical_cards"][role]))
            clocks = " / ".join(f'{float(row["T_seconds"]):.3f}' for row in arm["repetitions"])
            rows.append((ARM_NAMES["B0" if role == "B0" else "candidate"], card_ids, clocks,
                         f'{float(compute["achieved_tflops_per_card_lower"][role]):.4f}',
                         f'{float(arm["reference_tflops_per_card_upper"]):.4f}',
                         f'{float(arm["efficiency_lower_percent"]):.4f}%', native["native_seals"][role]))
        compute_block = "<h3>4.1 MMLU-Pro 固定 200 题 / out256</h3>" + table(
            ("运行臂", "物理卡", "三轮原生计时（秒）", "有效算力下界（TFLOP/s/卡）",
             "参考算力上界（TFLOP/s/卡）", "效率下界", "原生封存 SHA-256"), rows)
        if terminal := native.get("native_terminal_audit"):
            proof_sha = html.escape(terminal["proof_sha256"])
            observe_sha = html.escape(native["http_observation_seal_sha256"])
            compute_block += (f'<p>退出时设备采样的独立审计 SHA-256 <code>{proof_sha}</code>。'
                              f'必要算量观测封存 SHA-256 <code>{observe_sha}</code>。</p>')
    return f"""<h2>送测 V2 按证据判定</h2>
{table(("指标", "阶段", "状态", "已有证据", "判定依据或缺项"), targets)}
{compute_block}
<h3>4.2 逐数据集门槛核验</h3>
<p>固定 0.18 B0 与所选修复候选直接比较。里程碑要求吞吐≥1.10×、平均 TTFT/TPOT 各≤0.95×，
结题要求吞吐≥2×、平均 TTFT/TPOT 各≤0.80×。分数按固定计划分母，未决计非成功。</p>
{table(("数据集", "吞吐 B1/B0", "平均 TTFT B1/B0", "平均 TPOT B1/B0", "质量门槛及降幅",
        "里程碑", "里程碑依据", "结题", "结题依据"), speed_rows)}
{audit_note}
{context_block}
<h3>4.4 按公开租金折算</h3>
<p><a href="{html.escape(rent["source_url"], quote=True)}">{html.escape(rent["publisher"])}公开报价</a>
 · {rent["server_monthly_price"]:,} 元/{rent["cards_per_server"]} 卡月，折 {float(rent["monthly_cost_per_card_cny"]):,.0f}
 元/卡月，按 {rent["days_per_month"]} 天。MMLU-Pro 200 单位成本下降 {savings:.2f}%，成本子项{cost_gate}。</p>
{table(("工况", "运行臂", "元/百万输出词元", "同窗双卡 AI Core 均值"), cost_rows)}
{tenant_block}
<h3>判定附件来源</h3><ul>{"".join(links)}</ul>
"""


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
    source = data["source"]
    esc = html.escape
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DeepLinkEngine 候选封存 CPU 复算</title>
<style>
body{{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17212b;background:#f8fafc}}
.table{{overflow:auto;margin:1rem 0}} table{{border-collapse:collapse;width:100%;background:white}}
th,td{{padding:.65rem;text-align:left;border-bottom:1px solid #cbd5e1}} td:last-child,code{{overflow-wrap:anywhere}}
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
{v2_html(data)}
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
        "assessment_sources": data["assessment_sources"],
        "published_rental_sha256": data["submission_v2"]["4.4"]["published_rental"]["evidence_sha256"],
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
