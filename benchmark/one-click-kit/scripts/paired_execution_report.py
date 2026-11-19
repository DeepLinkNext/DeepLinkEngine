"""Rebuild a new paired report from both sealed original200 streams."""

import argparse
import hashlib
import html
import json
from pathlib import Path

from baseline_admission import PINNED_SOURCE_MAP_SHA256
from project_evidence import assess, ensure
from report_execution import inspect_run
from mmlu_scope import description as mmlu_description


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utilization(root):
    from telemetry_audit import utilization as calculate

    phase = json.loads((root / "service/joint.json").read_bytes())
    rows = [
        json.loads(line)
        for line in (root / "device-samples.jsonl").read_text().splitlines()
        if line
    ]
    devices = json.loads((root / "resource-admission.json").read_bytes())[
        "physical_cards"
    ]
    return calculate(
        rows,
        phase["start"]["monotonic"],
        phase["end"]["monotonic"],
        tuple(map(str, devices)),
    )


def inspect_pair(root, project_root=None):
    root = root.resolve()
    seal = json.loads((root / "pair-checksums.json").read_bytes())
    for name, digest in seal.items():
        p = root / name
        if not p.resolve().is_relative_to(root) or sha(p) != digest:
            raise ValueError("Pair binding changed: " + name)
    receipt = json.loads((root / "receipt.json").read_bytes())
    plan = json.loads((root / "frozen-pair-plan.json").read_bytes())
    reused = plan.get("reused_arms", {})
    expected_attempts = [dict(role=role, reused_verified=True,
        source_seal_sha256=reused[role]["seal_sha256"]) if role in reused
        else dict(role=role, returncode=0) for role in ("B0", "B1")]
    if not (
        receipt["completed"]
        and receipt["status"] == "completed"
        and receipt["attempts"]
        == expected_attempts
    ):
        raise ValueError("A completed original B0/B1 pair is required")
    if not (
        plan["requests"] == 200
        and plan["max_completion_tokens"] == 8192
        and plan["roles"] == ["B0", "B1"]
        and plan["temperature"] == plan["request_seed"] == 0
        and plan["eos_enabled"] is True
        and plan["additional_task_warmups"] == 0
        and plan["baseline"]["source_map_sha256"] == PINNED_SOURCE_MAP_SHA256
    ):
        raise ValueError("Original prospective pair protocol differs")
    arms = {}
    arm_roots = {}
    for role in plan["roles"]:
        from arm_reuse import resolve

        arm = resolve(root, role, plan)
        arm_roots[role] = arm
        data = inspect_run(arm)
        controller = json.loads((arm / "controller-source.json").read_bytes())
        if not (
            data["role"] == role
            and data["receipt"]["completed"]
            and data["recorded_requests"] == data["successful_requests"] == 200
            and data["failed_requests"] == 0
            and data["quality"]["unreturned"] == 0
            and (reused[role]["verified_at"] <= plan["frozen_before_first_arm"]
                 if role in reused else plan["frozen_before_first_arm"] <= data["receipt"]["started"])
        ):
            raise ValueError("Paired role or completed original200 evidence differs")
        for name, digest in ([] if role in reused else plan["native_controller_sources"].items()):
            if controller["scripts/" + name] != digest:
                raise ValueError(
                    "Native controller differs from premeasurement freeze: " + name
                )
        if data["quality"]["scorer_sha256"] != plan["scorer_sha256"]:
            raise ValueError("Paired scorer differs")
        arms[role] = data
    if reused:
        from arm_reuse import weights

        if weights(arm_roots["B0"]) != weights(arm_roots["B1"]):
            raise ValueError("Composed paired model weight bytes differ")
        for name, key in (("deployment.json", "optimized_deployment_sha256"),
                          ("installed-files.json", "optimized_installed_files_sha256")):
            if sha(arm_roots["B1"] / name) != plan[key]:
                raise ValueError("Composed B1 source is not the current frozen optimized deployment")
    else:
        for name in ("model-weights.json", "installed-files.json", "deployment.json"):
            if json.loads((arm_roots["B0"] / name).read_bytes()) != json.loads(
                (arm_roots["B1"] / name).read_bytes()
            ):
                raise ValueError("Paired shared source or weights differ: " + name)
    baseline = json.loads((arm_roots["B0"] / "baseline-source-admission.json").read_bytes())
    source_digest = hashlib.sha256(
        json.dumps(baseline["files"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if source_digest != PINNED_SOURCE_MAP_SHA256:
        raise ValueError("Paired baseline source freeze differs")
    requests = [
        json.loads((arm_roots[role] / "service/request-plan.json").read_bytes())
        for role in plan["roles"]
    ]
    if requests[0]["requests"] != requests[1]["requests"]:
        raise ValueError("Pair did not submit identical frozen tasks")
    a, b = [arms[x]["groups"]["joint200"] for x in plan["roles"]]
    gains = {
        "throughput_gain_pct": 100
        * (b["output_tokens_per_second"] / a["output_tokens_per_second"] - 1),
        "mean_ttft_improvement_pct": 100 * (1 - b["mean_ttft_ms"] / a["mean_ttft_ms"]),
        "mean_tpot_improvement_pct": 100 * (1 - b["mean_tpot_ms"] / a["mean_tpot_ms"]),
        "equal_rate_unit_cost_reduction_pct": 100
        * (1 - a["output_tokens_per_second"] / b["output_tokens_per_second"]),
    }
    observed_util = {role: utilization(arm_roots[role]) for role in plan["roles"]}
    gains["ai_core_utilization_gain_pp"] = (
        observed_util["B1"]["mean_percent"] - observed_util["B0"]["mean_percent"]
    )
    reference = assess(ensure(project_root))
    targets = json.loads(json.dumps(reference["formal_targets"]))
    performance_met = (
        gains["throughput_gain_pct"] >= 10
        and gains["mean_ttft_improvement_pct"] >= 5
        and gains["mean_tpot_improvement_pct"] >= 5
    )
    quality_difference = (
        arms["B1"]["quality"]["correct"] - arms["B0"]["quality"]["correct"]
    )
    targets["3.4.2"].update(
        status=("composed_pair_numeric_pass" if reused else "fresh_pair_numeric_pass")
        if performance_met
        else ("composed_pair_numeric_fail" if reused else "fresh_pair_numeric_fail"),
        accepted=performance_met,
        measured_value=gains["throughput_gain_pct"],
        display_value=f"{gains['throughput_gain_pct']:+.2f}%",
        source_url="summary.json",
        accepted_scope="Explicit sealed-arm composition with original dates" if reused else "This new prospectively frozen original200 pair; observed quality reported with original scorer",
        reason=f"本次对比吞吐变化{gains['throughput_gain_pct']:.2f}%，平均TTFT/TPOT改善{gains['mean_ttft_improvement_pct']:.2f}%/{gains['mean_tpot_improvement_pct']:.2f}%；两侧各200完整请求、零失败，正确数差{quality_difference:+d}/200。保留单次固定子集、长度终止与原服务协议，不推断统计质量等价。",
    )
    targets["3.4.4"].update(
        status="composed_pair_equal_rate_estimate" if reused else "fresh_pair_equal_rate_estimate",
        measured_value=gains["equal_rate_unit_cost_reduction_pct"],
        display_value=f"{(-gains['equal_rate_unit_cost_reduction_pct']):+.2f}%",
        source_url="summary.json",
        accepted=gains["equal_rate_unit_cost_reduction_pct"] >= 10
        and gains["ai_core_utilization_gain_pp"] > 0,
        numeric_threshold_met=gains["equal_rate_unit_cost_reduction_pct"] >= 10,
        accepted_scope="Explicit original sealed arms, same reserved two cards and fixed price basis" if reused else "This new pair, same reserved two cards and same fixed published price basis",
        reason=f"本次两侧同费率估算单位输出token成本变化{-gains['equal_rate_unit_cost_reduction_pct']:+.2f}%；实际采样AI Core均值差{gains['ai_core_utilization_gain_pp']:+.2f}个百分点。保留原费用范围、报价日期与利用率统计规则，不冒充实际账单。",
    )
    return {
        "schema": "vllm-hust-current-paired-report/v1",
        "root": str(root),
        "frozen_plan_sha256": sha(root / "frozen-pair-plan.json"),
        "source_pair_seal_sha256": sha(root / "pair-checksums.json"),
        "arms": arms,
        "arm_roots": {role:str(path) for role,path in arm_roots.items()},
        "reused_roles": sorted(reused),
        "evidence_mode": "composition_of_sealed_arms" if reused else "new_native_execution",
        "original_arm_dates": {role:arms[role]["receipt"]["started"] for role in plan["roles"]},
        "gains": gains,
        "utilization": observed_util,
        "quality_difference_of200": quality_difference,
        "formal_targets": targets,
        "existing_project_evidence": reference,
        "formal_complete": False,
        "scope": "Explicit original200 sealed-arm comparison; original dates and runtime identities retained" if reused else "New original MMLU200 pair plus separately identified existing qualified reference and five-dataset coverage; no all-suite acceptance claim",
        "mmlu_task_scope": plan.get("mmlu_task_scope", mmlu_description()),
    }


def render(root, out, project_root=None):
    data = inspect_pair(root, project_root)
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib

    if matplotlib.__version__ != "3.10.8":
        raise RuntimeError("Expected matplotlib3.10.8")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    measures = [
        ("output_tokens_per_second", "Output tokens/s"),
        ("mean_ttft_ms", "Mean TTFT (ms)"),
        ("mean_tpot_ms", "Mean TPOT (ms)"),
    ]
    for axis, (key, title) in zip(axes, measures):
        axis.bar(
            ["B0", "B1"],
            [data["arms"][role]["groups"]["joint200"][key] for role in ("B0", "B1")],
            color=["#64748b", "#2563eb"],
        )
        axis.set_title(title)
    axes[3].bar(
        ["B0", "B1"],
        [data["arms"][role]["quality"]["correct"] for role in ("B0", "B1")],
        color=["#64748b", "#2563eb"],
    )
    axes[3].set_ylim(0, 200)
    axes[3].set_title("Correct / planned200")
    fig.suptitle("MMLU-Pro fixed200 subset (source test12032); original8192 cap/EOS")
    fig.tight_layout()
    fig.savefig(out / "paired-performance.png", dpi=180)
    fig.savefig(out / "paired-performance.svg", metadata={"Date": None})
    plt.close(fig)
    esc = html.escape
    rows = "".join(
        f"<tr><td>{k} {esc(v['name'])}</td><td>{esc(v['status'])}</td><td>{esc(v['reason'])}</td></tr>"
        for k, v in data["formal_targets"].items()
    )
    provenance = ""
    if data["reused_roles"]:
        from datetime import datetime, timezone

        dates = "; ".join(role + ": " + datetime.fromtimestamp(
            data["original_arm_dates"][role], timezone.utc).isoformat()
            for role in ("B0", "B1"))
        provenance = "<p>本报告组合已封存结果，原始执行日期（UTC）：" + esc(dates) + (
            "。复用侧：" + esc(", ".join(data["reused_roles"])) +
            "；各侧的版本、请求和原始封存位置见完整来源。</p>")
    body = f"""<!doctype html><meta charset=utf-8><title>本次B0/B1配对与四项目标</title><style>body{{font:16px/1.7 system-ui;max-width:1150px;margin:30px auto}}td,th{{border:1px solid #cbd5e1;padding:10px}}table{{border-collapse:collapse}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>本次MMLU-Pro固定200题子集B0/B1配对</h1>{provenance}<p>原固定任务、8192输出上限、EOS、temperature0/seed0、一次突发200请求；两侧各200完成，失败和长度终止保留。</p><img src='paired-performance.png' alt='本次实际配对吞吐时延和正确数'>
<p>MMLU-Pro源test共12,032题，本工况复用14学科的固定200题子集及请求顺序；seed0是生成种子，不是抽样种子。结果仅适用于该子集，不能称为官方全集成绩或五数据集整体验收。逐题ID、学科和输入哈希见冻结清单。</p><pre>{esc(json.dumps(data["mmlu_task_scope"], ensure_ascii=False, indent=2))}</pre>
<h2>本次改善率</h2><pre>{esc(json.dumps(data["gains"], ensure_ascii=False, indent=2))}</pre>
<h2>四项目标及来源范围</h2><p>算力参考仍属于原out256合格三轮；本次完整回答配对不借用该F/T。五数据集总体尚未完成。费用按固定公开类似机型同费率估算，非实际账单。</p><table><tr><th>目标</th><th>状态</th><th>证据与边界</th></tr>{rows}</table>
<h2>五数据集覆盖</h2><pre>{esc(json.dumps(data["existing_project_evidence"]["dataset_coverage"], ensure_ascii=False, indent=2))}</pre>
<p>冻结计划SHA256：{data["frozen_plan_sha256"]}<br>配对源清单SHA256：{data["source_pair_seal_sha256"]}<br><a href='summary.json'>完整复算与逐项来源</a></p>"""
    (out / "summary.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "report.html").write_text(body)
    (out / "checksums.json").write_text(
        json.dumps({p.name: sha(p) for p in out.iterdir() if p.is_file()}, indent=2)
        + "\n"
    )
    print("PAIRED_REPORT", out / "report.html", flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project-evidence", type=Path)
    args = parser.parse_args()
    render(args.run, args.out.resolve(), args.project_evidence)


if __name__ == "__main__":
    main()
