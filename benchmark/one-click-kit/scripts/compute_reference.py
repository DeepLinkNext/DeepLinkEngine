"""Run automatic current-reference certification or an explicit frozen live plan."""

import json
import html
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from pathlib import Path
import shutil
import subprocess

from dataset_contract import save, sha
from child_runner import run_stage
from reference_evidence import evidence_manifest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "third-party-test-kit"


def binding(frozen):
    return {
        "profile": "mmlu-fixed200-v1",
        "quality_plan_sha256": frozen["mmlu_inputs"]["quality_plan_sha256"],
        "requests": 200,
        "max_completion_tokens": 8192,
        "temperature": 0,
        "seed": 0,
        "eos": "enabled",
        "model": frozen["model"],
        "physical_cards": frozen["physical_cards"],
        "engine_release": frozen["engine_release"],
    }


def prepare(args, out, frozen, *, force_live=False):
    folder = out / "plans/compute-reference"
    folder.mkdir(parents=True)
    result = {
        "schema": "one-click-current-compute-reference/v1",
        "status": "blocked",
        "binding": binding(frozen),
        "historical_substitution_allowed": False,
        "blockers": [],
    }
    save(
        folder / "execution-checklist.json",
        {
            "binding": result["binding"],
            "stages": [
                {"name": name, "status": "not_executed"}
                for name in (
                    "independent_calibration",
                    "independent_reference_dependency_audit",
                    "target_path_observation",
                    "synchronized_dual_card_timing",
                    "sealed_record_recomputation",
                    "efficiency_report_and_plot",
                )
            ],
            "kit_files": {p.name: sha(p) for p in KIT.glob("*.py")},
        },
    )
    if getattr(args, "performance_only", False):
        result.update(
            status="not_requested",
            blockers=["显式performance-only范围；未申请有效算力验收"],
        )
        save(folder / "admission.json", result)
        return result
    plan = getattr(args, "compute_reference_plan", None)
    digest = getattr(args, "compute_reference_sha256", None)
    if plan is None:
        from automatic_reference import prepare as prepare_automatic

        return prepare_automatic(args, out, frozen, result)
    else:
        try:
            plan = Path(plan).resolve()
            if not digest or sha(plan) != digest:
                raise ValueError("现场标定计划SHA256缺失或不匹配")
            document = json.loads(plan.read_bytes())
            if document.get("compute_reference_binding") != result["binding"]:
                raise ValueError("参考工况与本轮固定MMLU200/模型/双卡/引擎身份不一致")
            cases = [x for x in document["cases"] if x["kind"] == "compute"]
            if len(cases) != 1 or set(cases[0]["arms"]) != {"B0", "B1"}:
                raise ValueError("只接受一个MMLU200现场标定case及完整B0/B1")
            for name, expected in document["files"].items():
                source = plan.parent / name
                if (
                    not source.resolve().is_relative_to(plan.parent)
                    or source.is_symlink()
                    or sha(source) != expected
                ):
                    raise ValueError("参考输入缺失、越界或哈希改变: " + name)
                target = folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            for arm in cases[0]["arms"].values():
                if "live_capture" not in arm:
                    raise ValueError("只接受本轮现场采集，禁止复用旧参考回执")
                profile = json.loads((folder / arm["live_capture"]).read_bytes())
                workload = json.loads((folder / profile["workload"]).read_bytes())
                sampling = workload.get("sampling", {})
                if (
                    len(workload.get("request_token_ids", [])) != 200
                    or sampling.get("max_tokens") != 8192
                    or sampling.get("temperature") != 0
                    or sampling.get("seed") != 0
                    or sampling.get("ignore_eos", False)
                ):
                    raise ValueError(
                        "现场参考必须保持MMLU200/max8192/seed0/temperature0/EOS工况"
                    )
                if (
                    workload.get("mmlu_quality_plan_sha256")
                    != result["binding"]["quality_plan_sha256"]
                ):
                    raise ValueError("参考token清单未绑定原固定MMLU请求计划")
            # The shipped live collector is the authority, not arbitrary plan argv.
            for name in (p.name for p in KIT.glob("*.py")):
                if document["files"].get(name) != sha(KIT / name):
                    raise ValueError("必须冻结当前实际现场采集器: " + name)
            target_plan = folder / "execution-plan.json"
            shutil.copyfile(plan, target_plan)
            from report_environment import report_python

            interpreter, environment = report_python(
                getattr(args, "report_python", None)
            )
            python = str(interpreter)
            command = [
                python,
                str(folder / "run_vllm_hust_tests.py"),
                "preflight",
                "--plan",
                str(target_plan),
                "--sha256",
                digest,
                "--case",
                cases[0]["id"],
            ]
            checked = subprocess.run(
                command, text=True, capture_output=True, timeout=90, env=environment
            )
            (folder / "preflight.log").write_text(checked.stdout + checked.stderr)
            if checked.returncode:
                raise ValueError(
                    "现场参考预检失败: " + (checked.stderr or checked.stdout).strip()
                )
            result.update(
                status="ready",
                plan="plans/compute-reference/execution-plan.json",
                plan_sha256=digest,
                case=cases[0]["id"],
                python=python,
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as error:
            result["blockers"] = [str(error)]
    save(folder / "admission.json", result)
    print(
        "COMPUTE_REFERENCE_PREFLIGHT",
        result["status"],
        json.dumps(result["blockers"], ensure_ascii=False),
        flush=True,
    )
    return result


def execute(out, admission):
    folder = out / "compute-reference"
    folder.mkdir()
    result = dict(admission)
    try:
        if result["status"] == "automatic_ready":
            from automatic_reference import execute as execute_automatic

            result = execute_automatic(out, result)
            return result
        if result["status"] == "reference_cached":
            from project_evidence import assess, ensure

            evidence = assess(ensure(result["source_root"]))
            archived = out / result["evidence_path"]
            if (
                sha(archived) != result["evidence_sha256"]
                or json.loads(archived.read_bytes()) != evidence
                or evidence["formal_targets"]["3.4.1"] != result["historical_target"]
            ):
                raise ValueError("复用参考在预检后改变")
            from reference_statistics import statistics

            reference = statistics(evidence)
            cached = out / "plans/compute-reference/denominator-statistics.json"
            if (
                sha(cached) != result["reference_statistics_sha256"]
                or json.loads(cached.read_bytes()) != reference
                or result["reference_statistics"] != reference
            ):
                raise ValueError("参考分母统计在预检后改变")
            save(folder / "qualified-evidence.json", evidence)
            save(folder / "denominator-statistics.json", reference)
            return result
        if result["status"] != "ready":
            return result
        plan = out / result["plan"]
        if sha(plan) != result["plan_sha256"]:
            raise ValueError("参考计划在预检后改变")
        command = [
            result["python"],
            str(plan.parent / "run_vllm_hust_tests.py"),
            "run-live-reference",
            "--plan",
            str(plan),
            "--sha256",
            result["plan_sha256"],
            "--case",
            result["case"],
            "--out",
            str(folder / "native"),
        ]
        result["command"] = command
        result["status"] = "running"
        save(folder / "status.json", result)
        print(
            "THIRD_PARTY_REFERENCE mmlu200 independent-calibration/validation/target/timing",
            flush=True,
        )
        # No enclosing resource lease: the kit acquires and releases the same two cards itself.
        with (folder / "collector.log").open("wb") as log:
            run = run_stage(command, stdout=log, stderr=subprocess.STDOUT)
        result["returncode"] = run.returncode
        summary = json.loads((folder / "native/summary.json").read_bytes())
        rows = summary["results"]
        if not rows or any(
            x["status"] not in ("criteria_met", "criteria_not_met") for x in rows
        ):
            raise ValueError("参考结果仍有未决项，不能出效率数值")
        values = [x["arms"]["B1"]["efficiency_pct"] for x in rows]
        if any(
            not isinstance(v, (int, float)) or not 0 < v < float("inf") for v in values
        ):
            raise ValueError("无有效B1参考效率")
        # Recompute from the just-captured sealed records, never trust a prefilled percentage.
        review = command[:2] + ["review"] + command[3:]
        with (folder / "review.log").open("wb") as log:
            checked = run_stage(review, stdout=log, stderr=subprocess.STDOUT)
        if checked.returncode not in (0, 1) or run.returncode not in (0, 1):
            raise ValueError("现场标定记录复算失败")
        result.update(
            status="measured",
            efficiency_lower_observed_percent=min(values),
            milestone_met=all(v >= 50 for v in values),
            final_numeric_met=all(v >= 90 for v in values),
            summary_sha256=sha(folder / "native/summary.json"),
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        result.update(status="blocked", blockers=[str(error)])
    finally:
        save(folder / "status.json", result)
        save(
            folder / "checksums.json",
            evidence_manifest(folder),
        )
        print("COMPUTE_REFERENCE_RESULT", result["status"], flush=True)
    return result


def bound_display(value, *, upper=False):
    """Round a displayed inequality outwards, never across an unproved target."""
    value = Decimal(str(value))
    with localcontext() as context:
        context.prec = max(40, value.adjusted() + 6)
        rounded = value.quantize(Decimal("0.0001"), rounding=ROUND_CEILING if upper else ROUND_FLOOR)
    return format(rounded, ".4f").rstrip("0").rstrip(".")


def target(result):
    # Retire the dense-tail provider's old certificate identity. Its output
    # files remain immutable history; reading them with repaired tools must
    # not revive the withdrawn full-reference claim.
    if result.get("certificate_kind") in (
        "matched_complete_necessary_work_and_full_reference_lower_bound",
        "partial_dense_projection_diagnostic",
    ):
        return {
            "name": "有效算力效率 ≥50%",
            "accepted": False,
            "status": "current_workload_reference_not_measured",
            "display_value": "完整参考未完成",
            "measured_value": None,
            "measured_value_decimal": None,
            "value_kind": None,
            "accepted_scope": None,
            "reason": "部分 dense 子图只能作为诊断，不能替代已确认的完整参考方法；原完整认证标记与据此生成的50%/90%判定已撤回。",
        }
    if result.get("status") == "reference_cached":
        reference = result["reference_statistics"]["arms"]["B1"]
        return {
            "name": "有效算力效率 ≥50%",
            "accepted": False,
            "status": "reference_available_current_numerator_not_measured",
            "display_value": "参考分母已缓存；本轮分子待测",
            "measured_value": None,
            "accepted_scope": None,
            "reference_tflops_per_card_upper": reference[
                "reference_tflops_per_card_upper"
            ],
            "reference_statistics": result["reference_statistics"],
            "historical_evidence": result["historical_target"],
            "reason": "复用原独立参考分母统计；效率由本轮同工况必要算量和同步计时重新计算，未沿用旧效率百分比。",
        }
    full_reuse = result.get("certificate_kind") == "qualified_full_reference_denominator_reuse/v1"
    lower_bound = full_reuse
    required = {"F_interval", "T_seconds", "achieved_tflops_per_card_interval",
        "reference_tflops_per_card_upper", "efficiency_lower_percent"}
    complete_arms = all(isinstance(result.get("arms", {}).get(role), dict)
        and required <= result["arms"][role].keys() for role in ("B0", "B1"))
    reuse_qualified = (result.get("full_model_certificate") is True and complete_arms
        and result.get("full_reference_denominator_qualified") is True
        and result.get("original_reference_work_and_time_retained") is True
        and result.get("original_efficiency_used") is False
        and result.get("shared_work_cancellation_used") is False)
    measured = result.get("status") == "measured" and (not full_reuse or reuse_qualified)
    displayed_lower = bound_display(result["arms"]["B1"]["efficiency_lower_percent"]) if measured and lower_bound else None
    return {
        "name": "有效算力效率 ≥50%",
        "accepted": measured and result["milestone_met"],
        "status": "current_workload_measured"
        if measured
        else "current_workload_reference_not_measured",
        "display_value": (
            ("≥" + displayed_lower if lower_bound else str(round(result["efficiency_lower_observed_percent"], 4))) + "%"
        )
        if measured
        else "现场参考未完成",
        "measured_value": (float(displayed_lower) if lower_bound else result.get("efficiency_lower_observed_percent")) if measured else None,
        "measured_value_decimal": result["arms"]["B1"]["efficiency_lower_percent"] if measured and lower_bound else None,
        "value_kind": "certified_lower_bound" if measured and lower_bound else None,
        "accepted_scope": ("Current fixed MMLU200 native F/T with qualified original full-reference denominator"
            if full_reuse else "Current fixed MMLU200 live independent reference") if measured else None,
        "reason": ("原完整参考的算量、参考时间与每卡分母已独立复算，并通过同工况输入、权重、精度、双卡及预算核验；本轮必要F和原生同步T重新计算分子，再除以原参考每卡上界。两份F不相消，未沿用旧效率。"
                   if lower_bound else "本轮MMLU200独立标定、整链验证、双卡同步计时及原件复算；各重复逐项判定。") if measured
        else "；".join(result.get("blockers") or (["完整B0/B1统计或资格证据缺失，当前工况参考未完成"]
            if lower_bound and result.get("status") == "measured" else ["参考阶段尚未执行；仅完成预检"])),
    }


def reference_report(result, *, plot=False):
    """Display a qualified bound explicitly, preserving exact source JSON links."""
    target_result = target(result)
    text = ['<meta charset="utf-8"><title>MMLU200有效算力参考</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px;color:#17212b;background:#fff}.table-wrap{overflow-x:auto}.table-wrap:focus-visible{outline:3px solid #004b9b}table{border-collapse:collapse;width:100%;min-width:800px}th,td{padding:12px;border:1px solid #667085;text-align:left}a{color:#004b9b}details{margin-top:24px}</style>',
        '<h1>MMLU200有效算力参考</h1><p>' + html.escape(result["status"]) + '</p>',
        '<p>' + html.escape(target_result["reason"]) + '</p>']
    if (target_result["status"] == "current_workload_measured"
            and result.get("certificate_kind") == "qualified_full_reference_denominator_reuse/v1"):
        if plot:
            text.append('<img style="width:100%;height:auto" src="compute-reference.svg" alt="B0/B1 effective compute intervals and reference efficiency lower bounds">')
        capture = result.get("capture_profile") or {"max_tokens": 8192, "formal_repetitions": 1}
        text += ['<p>BF16 · TP2 · 固定200题 · 每题上限' + str(capture["max_tokens"]) + ' token · 每臂' + str(capture["formal_repetitions"]) + '轮 · EOS保留。以下时间来自纯原生同步区间，HTTP与分词在区间外。</p>',
            '<div class="table-wrap" tabindex="0" aria-label="左右滚动查看本轮参考数据"><table><caption>本轮每卡算力及完整参考效率下界</caption><thead><tr><th>运行臂</th><th>原生时间 (s)</th><th>本轮必要算量区间 (FLOPs)</th><th>原参考算量区间 (FLOPs)</th><th>原参考时间下界 (s)</th><th>每卡有效算力区间 (TFLOP/s)</th><th>每卡参考算力上界 (TFLOP/s)</th><th>效率下界</th></tr></thead><tbody>']
        for role in ("B0", "B1"):
            arm = result["arms"][role]
            text.append('<tr><th>' + role + '</th><td>' + html.escape(str(arm["T_seconds"])) + '</td><td>['
                + ', '.join(html.escape(str(v)) for v in arm["F_interval"]) + ']</td><td>['
                + ', '.join(html.escape(str(v)) for v in arm["reference_F_interval"]) + ']</td><td>≥'
                + html.escape(arm["reference_seconds_lower"]) + '</td><td>['
                + ', '.join(bound_display(v, upper=bool(i)) for i, v in enumerate(arm["achieved_tflops_per_card_interval"])) + ']</td><td>≤'
                + bound_display(arm["reference_tflops_per_card_upper"], upper=True) + '</td><td>≥'
                + bound_display(arm["efficiency_lower_percent"]) + '%</td></tr>')
        text += ['</tbody></table></div><p>B1里程碑50%：' + html.escape(result["milestone_decision"])
            + '；后续90%目标：' + html.escape(result["final_decision"]) + '。</p>',
            '<p>下界低于门槛表示当前证据未证明达标，不能据此判定未知的精确效率低于门槛。每臂保留全部预先声明的完整正式测量，算量和时间先求和再相除；不取最快轮，也不平均各轮效率。算量界与参考模型界不是统计置信区间。</p>',
            '<details><summary>统计口径与可复算证据</summary><p>C本轮下界=ΣF本轮下界/(2×ΣT本轮×10¹²)；C参考上界=ΣF原参考上界/(2×ΣT原参考下界×10¹²)；η下界=100×C本轮下界/C参考上界。保留两份原始算量及其各自时间，不消去不同F，不沿用旧效率，不表示硬件额定峰值利用率。</p>',
            '<p><a href="compute-reference/matched-reference-pair.json">配对复算JSON</a> · <a href="compute-reference/necessary-work-audit.json">本轮必要工作审计</a> · <a href="compute-reference/reference-graph-audits.json">原参考图与同工况复用审计</a> · <a href="compute-reference/qualified-full-reference-statistics.json">原完整参考分母统计</a></p></details>']
    text.append('<p>本目录追加参考认证，原性能批次的原件与统计口径保留。</p>')
    return ''.join(text)


def reference_only(args):
    """Append an independent reference run without modifying or rerunning its parent."""
    root = args.reference_campaign
    if root is None:
        raise ValueError("--compute-reference-only需要--reference-campaign")
    root = root.resolve()
    seal = json.loads((root / "campaign-checksums.json").read_bytes())
    for name in ("frozen-execution-plan.json", "campaign.json"):
        if sha(root / name) != seal.get(name):
            raise ValueError("原性能批次绑定改变: " + name)
    frozen = json.loads((root / "frozen-execution-plan.json").read_bytes())
    campaign = json.loads((root / "campaign.json").read_bytes())
    if "mmlu" not in frozen["datasets"] or campaign["status"] != "completed":
        raise ValueError("参考补测只接续已完成且含MMLU200的封存批次")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(
        out / "source-campaign.json",
        {
            "campaign": str(root),
            "campaign_seal_sha256": sha(root / "campaign-checksums.json"),
            "prepared_plan_sha256": sha(root / "frozen-execution-plan.json"),
            "performance_inference_rerun": False,
            "reference_collector_sources": {
                str(p.relative_to(ROOT)): sha(p) for p in KIT.glob("*.py")
            },
        },
    )
    result = execute(out, prepare(args, out, frozen, force_live=True))
    save(
        out / "summary.json",
        {"compute_reference": result, "target_3_4_1": target(result)},
    )
    document = reference_report(result)
    plot_failed = False
    if (hasattr(args, "report_python") and target(result)["status"] == "current_workload_measured"
            and result.get("certificate_kind") == "qualified_full_reference_denominator_reuse/v1"):
        try:
            from report_environment import report_python

            python, environment = report_python(args.report_python)
            subprocess.run([str(python), "-B", str(ROOT / "scripts/third_party_report.py"),
                "--reference-run", str(out), "--out", str(out / "reference-report")],
                env=environment, check=True)
            document += '<figure><img style="max-width:100%" src="reference-report/current-reference-bounds.svg" alt="本轮B0/B1参考效率下界；精确效率未知"></figure><p><a href="reference-report/report.html">图表、复算与来源</a></p>'
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            plot_failed = True
            save(out / "report-plot-failure.json", dict(failure=type(error).__name__ + ": " + str(error),
                certificate_preserved=True, model_inference_rerun=False))
            document += '<p>本轮认证原件已保留；图表生成失败，详见report-plot-failure.json。</p>'
    (out / "report.html").write_text(document)
    save(
        out / "checksums.json",
        evidence_manifest(out),
    )
    print("COMPUTE_REFERENCE_REPORT", out / "report.html", flush=True)
    return 0 if target(result)["status"] == "current_workload_measured" and not plot_failed else 2
