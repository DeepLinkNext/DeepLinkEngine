"""Render the actual campaign checklist and distinguish current from historical evidence."""

import argparse
import html
import json
from pathlib import Path
import shutil

from dataset_contract import save, sha
from judge_disclosure import NOTICE_ZH, policy as judge_policy


def current_reference_requested(reference):
    return bool(reference) and reference.get("status") != "not_requested"


def reference_card(current, historical):
    if current_reference_requested(current):
        from compute_reference import target

        qualified = target(current)
        measured = qualified["status"] == "current_workload_measured"
        return ("3.4.1 Effective FLOPs", qualified["display_value"] if measured else "Incomplete",
            "Current MMLU200 reference lower bound" if measured and qualified.get("value_kind") == "certified_lower_bound"
            else "Current workload independent reference")
    return ("3.4.1 Effective FLOPs",
        historical["formal_targets"]["3.4.1"]["display_value"] if historical else "Unavailable",
        "Historical qualified out256 reference")


def render_reference_bundle(root, out, result=None):
    """Render only sealed current numbers; incomplete runs never get a numeric plot."""
    from compute_reference import bound_display, reference_report, target

    root, out = Path(root), Path(out)
    source = root / "compute-reference"
    if result is None:
        result = json.loads((source / "status.json").read_bytes())
    qualified = target(result)
    measured_bound = (qualified["status"] == "current_workload_measured"
        and qualified.get("value_kind") == "certified_lower_bound")
    files = ["matched-reference-pair.json", "necessary-work-audit.json",
        "reference-graph-audits.json", "qualified-full-reference-statistics.json"] if measured_bound else []
    admitted = {}
    if measured_bound:
        seal = json.loads((source / "checksums.json").read_bytes())
        if sha(source / "status.json") != seal.get("status.json") or json.loads((source / "status.json").read_bytes()) != result:
            raise ValueError("Current reference report status differs from its source seal")
        for name in files:
            path = source / name
            if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()) or sha(path) != seal.get(name):
                raise ValueError("Current reference report evidence changed: " + name)
            admitted[name] = seal[name]
    out.mkdir(parents=True, exist_ok=False)
    for name in files:
        (out / "compute-reference").mkdir(exist_ok=True)
        shutil.copyfile(source / name, out / "compute-reference" / name)
        if sha(out / "compute-reference" / name) != admitted[name]:
            raise ValueError("Current reference report copy differs from its sealed source: " + name)
    image = None
    if measured_bound:
        import matplotlib

        if matplotlib.__version__ != "3.10.8":
            raise RuntimeError("Expected matplotlib3.10.8")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [bound_display(result["arms"][role]["efficiency_lower_percent"]) for role in ("B0", "B1")]
        values = list(map(float, labels))
        fig, axis = plt.subplots(figsize=(7.2, 4.6))
        axis.bar(["B0", "B1"], values, color=["#475569", "#1d4ed8"], hatch="//")
        ceiling = max(100, max(values) * 1.18)
        axis.set_ylim(0, ceiling)
        for index, (value, label) in enumerate(zip(values, labels, strict=True)):
            axis.text(index, value + ceiling * .025, "\u2265" + label + "%", ha="center")
        for threshold, label in [(50, "50% milestone"), (90, "90% target")]:
            axis.axhline(threshold, color="#64748b", linestyle="--", linewidth=1, label=label)
        axis.set_ylabel("Certified reference efficiency lower bound (%)")
        capture = result.get("capture_profile") or dict(max_tokens=8192, formal_repetitions=1)
        axis.set_title("Current fixed MMLU200 / max" + str(capture["max_tokens"]) + " / native TP2")
        axis.legend(loc="upper left", fontsize=8)
        fig.text(.5, .015, str(capture["formal_repetitions"]) + " predeclared native repetition(s) per arm; sums before ratios; exact efficiency unknown. Not hardware peak utilization.",
            ha="center", fontsize=7)
        fig.tight_layout(rect=(0, .045, 1, 1))
        image = "current-reference-bounds.svg"
        fig.savefig(out / image, metadata={"Date": None})
        fig.savefig(out / "current-reference-bounds.png", dpi=180)
        plt.close(fig)
    document = reference_report(result)
    if image:
        document += '<figure><img style="max-width:100%" src="' + image + '" alt="B0/B1本轮完整参考效率下界；精确效率未知，非硬件峰值利用率"></figure>'
    (out / "report.html").write_text(document)
    save(out / "summary.json", dict(compute_reference=result, target_3_4_1=qualified))
    save(out / "source-binding.json", dict(source=str(source.resolve()),
        checksums_sha256=sha(source / "checksums.json") if measured_bound else None,
        copied_source_files=admitted, numeric_plot_created=bool(image)))
    save(out / "checksums.json", {str(p.relative_to(out)): sha(p) for p in out.rglob("*")
        if p.is_file() and p != out / "checksums.json"})
    return dict(plot=image, target=qualified)


def verify_campaign_bindings(root):
    """Validate contents behind the bound child seals, not just seal JSON bytes."""
    root = root.resolve()
    bindings = json.loads((root / "campaign-checksums.json").read_bytes())
    validated = dict(bindings)
    for name, expected in bindings.items():
        path = root / name
        if (
            not path.resolve().is_relative_to(root)
            or path.is_symlink()
            or sha(path) != expected
        ):
            raise ValueError("Campaign source binding changed: " + name)
    for name in bindings:
        seal = root / name
        if seal.name not in ("checksums.json", "pair-checksums.json"):
            continue
        nested = json.loads(seal.read_bytes())
        if not nested:
            raise ValueError("Empty child seal: " + name)
        for relative, expected in nested.items():
            path = seal.parent / relative
            if (
                not path.resolve().is_relative_to(seal.parent)
                or path.is_symlink()
                or sha(path) != expected
            ):
                raise ValueError(
                    "Campaign child content changed: " + name + ": " + relative
                )
            validated[str(path.relative_to(root))] = expected
    return validated


def read_bound(path, root, validated):
    if str(path.relative_to(root)) not in validated:
        raise ValueError(
            "Report input has no campaign binding: " + str(path.relative_to(root))
        )
    return json.loads(path.read_bytes())


def targets(reference, pair, plan, receipt):
    result = {
        "3.4.1": {
            "name": "有效算力效率≥50%",
            "accepted": False,
            "display_value": "Pending",
            "status": "本次未新增F/T测量",
            "reason": "已有独立合格参考按原out256工况单独复算；不能用当前完整回答耗时替换原F/T。",
        },
        "3.4.2": {
            "name": "吞吐及TTFT/TPOT改善",
            "accepted": False,
            "display_value": "Pending",
            "status": "本次配对尚未完成",
            "reason": "本次须完成原200题两臂及原始流复核；历史正式配对已存在并另表复算，不据本次未测称项目缺少配对。",
        },
        "3.4.3": {
            "name": "五数据集与16K稳定性",
            "accepted": False,
            "display_value": "Incomplete",
            "status": "五项范围未完成",
            "reason": "SWE与Terminal任务调度和评分尚未接入本执行清单；CPU连接已另有就绪回执，不能写成工作区未创建。其余三项以冻结的全量或抽样任务清单为分母；诊断不能冒充正式抽样，C8不能替代指定数据集任务。",
        },
        "3.4.4": {
            "name": "单位token成本及利用率",
            "accepted": False,
            "display_value": "Pending",
            "status": "本次成本配对尚未完成",
            "reason": "本次从同一响应集合与同区间双卡采样核算；已有历史费用依据与同费率估算另表复算，不冒充本机账单。",
        },
    }
    if reference:
        result["3.4.1"] = dict(reference["formal_targets"]["3.4.1"])
        result["3.4.1"]["status"] = "历史原工况范围内达标；参考已重新核验"
    if pair:
        for key in ("3.4.2", "3.4.4"):
            result[key] = dict(pair["formal_targets"][key])
            result[key]["status"] = (
                "本次数值达标" if result[key]["accepted"] else "本次数值未达标"
            ) + (
                "（披露的同费率估算范围）" if key == "3.4.4" else "（原固定200题范围）"
            )
    if receipt.get("selected_scope_execution_complete") or receipt.get(
        "selected_full_dataset_execution_complete"
    ):
        result["3.4.3"]["reason"] = (
            "本次冻结的所选范围已执行结束；HLE范围以计划中的分层子集/全集标识为准，不冒充全集成绩。SWE/Terminal仍未接入本执行清单，五项整体验收未完成。任务异常、实际16K覆盖、尾时延和未决评分须按原记录分别核验。"
        )
    if plan and any(name in plan.get("datasets", []) for name in ("swe", "terminal")):
        result["3.4.3"].update(
            status="公开轨迹推理性能已接入；按实际运行核验",
            reason="SWE/Terminal已接入固定公开真实请求的抽样回放，不执行工具、不测任务解决率；不再以CPU容器未接入作为推理性能阻塞。按本次冻结请求数、失败/未返回、实际16K覆盖和尾时延核验性能与稳定性；回放成功不能推导任务质量或五项端到端质量验收通过。",
        )
    if receipt.get("reused_inference_stages"):
        for key in ("3.4.2", "3.4.4"):
            result[key]["status"] = "复用已封存配对重新核验：" + result[key]["status"]
    return result


def inspect(root):
    root = root.resolve()
    validated = verify_campaign_bindings(root)
    receipt = json.loads((root / "campaign.json").read_bytes())
    # Early fresh-run controllers omitted these display fields. Derive them
    # from the sealed checklist without changing any stored receipts.
    receipt.setdefault(
        "reused_inference_stages",
        sum(s["status"] == "reused_verified" for s in receipt["stages"]),
    )
    receipt.setdefault(
        "new_inference_stages",
        len(receipt["stages"]) - receipt["reused_inference_stages"],
    )
    plan = (
        json.loads((root / "frozen-execution-plan.json").read_bytes())
        if (root / "frozen-execution-plan.json").exists()
        else None
    )
    if plan and plan.get("execution_source_snapshots"):
        captured = {
            str(path.relative_to(root / "execution-source")): sha(path)
            for path in (root / "execution-source").rglob("*")
            if path.is_file()
        }
        if captured != plan["sources"]:
            raise ValueError(
                "Archived execution source differs from the prospective plan"
            )
    reference, reference_error = None, None
    try:
        from project_evidence import assess, ensure

        reference = assess(ensure())
    except (OSError, ValueError, RuntimeError) as error:
        reference_error = type(error).__name__ + ": " + str(error)
    from reuse_campaign import load_report

    reused_roots, reused_audits = load_report(root, plan)
    pair = reused_audits.get("mmlu-pair")
    if (root / "mmlu-pair/receipt.json").exists():
        paired_receipt = read_bound(root / "mmlu-pair/receipt.json", root, validated)
        if paired_receipt.get("completed"):
            from paired_execution_report import inspect_pair

            pair = inspect_pair(root / "mmlu-pair")
    observed = []
    traces = {}
    dataset_performance = {}
    for row in receipt["stages"]:
        item = dict(row)
        child = root / row["name"] / "receipt.json"
        if row["status"] == "reused_verified":
            if row["name"] not in reused_audits:
                raise ValueError("Reused receipt has no audited source")
            item["source"] = row["source"]
        if child.exists():
            native = read_bound(child, root, validated)
            item["failure_detail"] = native.get("failure")
        summary = root / (row["name"] + "-report") / "summary.json"
        if row["name"] in reused_audits or summary.exists():
            if row["name"] in reused_audits:
                data = reused_audits[row["name"]]
            else:
                item["report_sha256"] = sha(summary)
                data = read_bound(summary, root, validated)
            item["counts"] = {
                key: data[key]
                for key in (
                    "planned",
                    "recorded",
                    "successful",
                    "failed",
                    "unreturned",
                    "unresolved_grades",
                    "diagnostic",
                )
                if key in data
            }
            if data.get("stage") == "solver":
                dataset_performance[row["name"]] = data
            if data.get("stage") == "trace":
                traces.setdefault(data["dataset"], {})[data["service_role"]] = data
        observed.append(item)
    quality = {}
    for name in ("hle", "frontier"):
        for role in ("B0", "B1"):
            path = root / (name + "-quality-" + role) / "summary.json"
            if path.exists():
                quality[name + "-" + role] = read_bound(path, root, validated)
    trace_pairs = {}
    for name, arms in traces.items():
        complete = all(
            role in arms
            and arms[role]["successful"] == arms[role]["planned"]
            and arms[role]["receipt"].get("completed")
            for role in ("B0", "B1")
        )
        gains = None
        if complete:
            if arms["B0"]["execution_scope"] != arms["B1"]["execution_scope"]:
                raise ValueError("Trace arms belong to different frozen samples")
            left = arms["B0"]["actual_tokenizer_admission"]["inputs"]
            right = arms["B1"]["actual_tokenizer_admission"]["inputs"]
            if {k: v["token_ids_sha256"] for k, v in left.items()} != {
                k: v["token_ids_sha256"] for k, v in right.items()
            }:
                raise ValueError("Trace pair used different actual input token IDs")
            b0, b1 = arms["B0"], arms["B1"]
            gains = {
                "throughput_gain_pct": 100
                * (b1["output_tokens_per_second"] / b0["output_tokens_per_second"] - 1),
                "ttft_reduction_pct": 100
                * (1 - b1["mean_ttft_ms"] / b0["mean_ttft_ms"]),
                "tpot_reduction_pct": 100
                * (1 - b1["mean_tpot_ms"] / b0["mean_tpot_ms"])
                if b0["mean_tpot_ms"] and b1["mean_tpot_ms"] is not None
                else None,
            }
        trace_pairs[name] = {
            "arms": arms,
            "paired_execution_complete": complete,
            "gains": gains,
            "quality_measured": False,
        }
    current_reference = receipt.get("compute_reference")
    formal_targets = targets(reference, pair, plan, receipt)
    if current_reference_requested(current_reference):
        from compute_reference import target as reference_target

        formal_targets["3.4.1"] = reference_target(current_reference)
    return {
        "schema": "vllm-hust-third-party-report/v1",
        "run": str(root),
        "receipt": receipt,
        "plan": plan,
        "current_stages": observed,
        "reused_stage_sources": {
            name: str(path) for name, path in reused_roots.items()
        },
        "evidence_mode": receipt.get("evidence_mode", "new_native_execution"),
        "current_mmlu_pair": pair,
        "current_dataset_quality": quality,
        "quality_scoring_policy": judge_policy(),
        "current_dataset_performance": dataset_performance,
        "current_trace_performance": trace_pairs,
        "trace_execution_scopes": {
            name: data["execution_scope"]
            for name, data in (plan or {}).get("prepared_datasets", {}).items()
            if name in ("swe", "terminal")
        },
        "existing_project_evidence": reference,
        "existing_evidence_read_error": reference_error,
        "current_compute_reference": current_reference,
        "formal_targets": formal_targets,
        "formal_complete": False,
        "scope": "Selected quality subsets and published SWE/Terminal trace inference performance, with separately identified historical evidence. Trace replay does not measure task solve rates; no automatic full five-dataset quality acceptance.",
    }


def render(root, out):
    data = inspect(root)
    out.mkdir(parents=True, exist_ok=False)
    reference, pair = data["existing_project_evidence"], data["current_mmlu_pair"]
    selected = (
        ", ".join((data["plan"] or {}).get("datasets", [])) or "准备阶段未完成范围冻结"
    )
    hle_scope = (
        (data["plan"] or {}).get("prepared_datasets", {}).get("hle") or {}
    ).get("execution_scope")
    hle_description = (
        "HLE固定分层200题（Gold文本64/图像10，Revision文本113/图像13；源1811题）"
        if hle_scope and hle_scope.get("profile") == "fixed200-v1"
        else "HLE Gold668/Revision1143（211图像题；如本次选择HLE）"
    )
    deferred = ", ".join(
        (data["plan"] or {}).get(
            "outside_execution_profile",
            (data["plan"] or {}).get(
                "deferred_by_requested_scope", ["SWE-bench-Pro", "TerminalBench2.1"]
            ),
        )
    )
    cards = [
        reference_card(data["current_compute_reference"], reference),
        (
            "3.4.2 Paired performance",
            f"{pair['gains']['throughput_gain_pct']:+.2f}%" if pair else "Pending",
            "Current original200 pair",
        ),
        (
            "Fixed-five execution",
            "Complete" if data['receipt'].get('selected_execution_complete') else "Incomplete",
            "Frozen quality samples + public agent inference traces",
        ),
        (
            "3.4.4 Cost + utilization",
            f"{-pair['gains']['equal_rate_unit_cost_reduction_pct']:+.2f}%"
            if pair
            else "Pending",
            "Current pair, disclosed equal-rate estimate",
        ),
    ]
    import matplotlib

    if matplotlib.__version__ != "3.10.8":
        raise RuntimeError("Expected matplotlib3.10.8")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for axis, (title, value, scope), target in zip(
        axes, cards, data["formal_targets"].values(), strict=True
    ):
        axis.set_title(title, fontsize=10)
        axis.text(
            0.5,
            0.58,
            value,
            ha="center",
            fontsize=25,
            color="#166534" if target["accepted"] else "#b45309",
        )
        axis.text(0.5, 0.22, scope, ha="center", fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Third-party campaign: current execution and historical evidence have separate scopes",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "four-targets.png", dpi=180)
    fig.savefig(out / "four-targets.svg", metadata={"Date": None})
    plt.close(fig)
    reference_section = ""
    if current_reference_requested(data["current_compute_reference"]):
        bundle = render_reference_bundle(root, out / "current-reference", data["current_compute_reference"])
        reference_section = '<h2>本轮有效算力参考</h2><p><a href="current-reference/report.html">本轮F/T、效率下界、独立审计与来源</a></p>'
        if bundle["plot"]:
            reference_section += '<img src="current-reference/' + bundle["plot"] + '" alt="本轮B0/B1参考效率下界，不表示精确效率或硬件峰值利用率">'
    trace_plots = ""
    if data["current_trace_performance"]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for axis, key, title in zip(
            axes,
            ("output_tokens_per_second", "mean_ttft_ms", "mean_tpot_ms"),
            ("Output tokens/s", "Mean TTFT (ms)", "Mean TPOT (ms)"),
            strict=True,
        ):
            labels, values = [], []
            for name, pair_data in data["current_trace_performance"].items():
                for role, arm in pair_data["arms"].items():
                    if arm.get(key) is not None:
                        labels.append(name + " " + role)
                        values.append(arm[key])
            axis.bar(labels, values)
            axis.set_title(title)
            axis.tick_params(axis="x", labelrotation=25)
        fig.suptitle(
            "Published agent trace sample: real inference; no task solve-rate score"
        )
        fig.tight_layout()
        fig.savefig(out / "trace-performance.png", dpi=180)
        fig.savefig(out / "trace-performance.svg", metadata={"Date": None})
        plt.close(fig)
        trace_plots = (
            "<img src='trace-performance.png' alt='公开轨迹抽样的真实推理性能'>"
        )
    reference_plot = ""
    current_reference = data.get("current_compute_reference")
    from compute_reference import target as compute_reference_target

    if current_reference and compute_reference_target(current_reference)["status"] == "current_workload_measured" and current_reference.get("full_model_certificate") is True:
        from plot_compute_reference import draw

        draw(current_reference, out)
        reference_plot = "<h2>本轮有效算力与参考效率</h2><img src='compute-reference.svg' alt='本轮两臂有效算力区间及参考效率下界'>"
    esc = html.escape
    rows = "".join(
        f"<tr><td>{esc(row['name'])}</td><td>{esc(row['status'])}</td><td>{esc(str(row.get('failure_detail') or row.get('counts') or '见对应执行回执'))}</td></tr>"
        for row in data["current_stages"]
    )
    current_targets = "".join(
        f"<tr><td>{key} {esc(value['name'])}</td><td>{'达成目标' if value['accepted'] else '未达成目标'}</td><td>{esc(value['reason'])}</td></tr>"
        for key, value in data["formal_targets"].items()
    )
    original = (
        "".join(
            f"<tr><td>{key} {esc(value['name'])}</td><td>{esc(value['display_value'])}</td><td>{esc(value['reason'])}<br><a href='{esc(value['source_url'], quote=True)}'>固定来源</a></td></tr>"
            for key, value in reference["formal_targets"].items()
        )
        if reference
        else "<tr><td>既有参考与配对证据读取失败；不据此声称项目没有这些证据。</td><td>"
        + esc(str(data["existing_evidence_read_error"]))
        + "</td></tr>"
    )
    scope_complete = (set((data["plan"] or {}).get("datasets", [])) == {"mmlu", "hle", "frontier", "swe", "terminal"}
        and data["receipt"]["selected_execution_complete"] is True)
    scope_note = "五个固定配置的推理与既定评分阶段均已执行；任务质量和四项目标分别判定。" if scope_complete else "所选范围的完成情况见执行清单；未完成范围不计作通过。"
    body = f"""<!doctype html><meta charset=utf-8><title>第三方测试实际执行报告</title><style>body{{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto}}td,th{{border:1px solid #cbd5e1;padding:10px}}table{{border-collapse:collapse;width:100%}}pre{{white-space:pre-wrap}}img{{max-width:100%}}</style>
<h1>第三方测试实际执行报告</h1><p>本次明确选择：{esc(selected)}。未执行范围：{esc(deferred)}。范围标识：MMLU-Pro固定200题子集（源test12,032题）、{esc(hle_description)}、Frontier Olympiad100/Research60；仅本次选择的数据集执行，诊断模式见冻结计划。{scope_note}</p>
<p>执行状态：{esc(data["receipt"]["status"])}；所选范围执行完成：{data["receipt"]["selected_execution_complete"]}。退出0只表示相应执行与归档成功。失败：{esc(str(data["receipt"].get("failure")))}</p>
<p class="judge-disclosure">{esc(NOTICE_ZH)}</p>
<img src='four-targets.png' alt='本次测试与历史参考的范围分别标注'>
{reference_plot}
<h2>四项目标逐项判定</h2><table><tr><th>目标</th><th>是否达标及适用范围</th><th>证据与具体卡点</th></tr>{current_targets}</table>
{reference_section}
<p>证据方式：{esc(data["evidence_mode"])}；复用项沿用各自原始运行日期、控制器与条件，本次只核验归并，不冒充新一轮推理。复用 {data["receipt"].get("reused_inference_stages", 0)} 阶段，新增 {data["receipt"].get("new_inference_stages", 0)} 阶段。</p><h2>本次执行清单</h2><table><tr><th>阶段</th><th>状态</th><th>分母、失败与具体卡点</th></tr>{rows}</table>
<h2>本次固定200题子集配对</h2><pre>{esc(json.dumps(pair["gains"], ensure_ascii=False, indent=2)) if pair else "本次尚无完成的原始200题配对；历史正式配对见下表，不能写成项目缺少配对。"}</pre>
<h2>HLE/Frontier 实际推理性能</h2><p>两侧原始 solver 墙钟吞吐、TTFT/TPOT独立展示，裁判流量不混入。负收益不删除。</p><pre>{esc(json.dumps(data["current_dataset_performance"], ensure_ascii=False, indent=2))}</pre><h2>本次数据集评分</h2><p>两侧solver使用同一固定B1替代judge；评分流量不计入solver性能。原评分器、长度终止和未决项保留；显式诊断子集不冒充全量。</p><pre>{esc(json.dumps(data["current_dataset_quality"], ensure_ascii=False, indent=2))}</pre>
<h2>SWE/Terminal公开轨迹推理性能</h2><p>固定公开来源与抽样请求，B0/B1同输入同预算，实际生成；不执行工具或任务评分。历史时间戳仅作来源记录，采用本次明确冻结的并发8闭环负载。失败、未返回与长度终止保留；缺测不画成零。</p><pre>{esc(json.dumps(data["trace_execution_scopes"], ensure_ascii=False, indent=2))}</pre>{trace_plots}<pre>{esc(json.dumps(data["current_trace_performance"], ensure_ascii=False, indent=2))}</pre>
<h2>已有独立参考与历史配对（单独复算）</h2><p>本次未测不等于项目没有证据；旧数据也不冒充本次新增运行。</p><table>{original}</table>
<p><a href='summary.json'>完整结果及来源</a> · 源封存SHA256：{sha(root / "campaign-checksums.json")}</p>"""
    (out / "report.html").write_text(body)
    save(out / "summary.json", data)
    save(
        out / "checksums.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*")
         if path.is_file() and path != out / "checksums.json"},
    )
    print("THIRD_PARTY_REPORT", out / "report.html", flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--reference-run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.reference_run:
        render_reference_bundle(args.reference_run.resolve(), args.out.resolve())
    else:
        render(args.run.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
