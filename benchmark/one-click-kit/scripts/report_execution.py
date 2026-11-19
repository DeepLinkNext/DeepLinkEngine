"""Rebuild a B1 run report from sealed streams, retaining failed denominators."""

import argparse
import ast
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics

from original_audit import audit_activation, reconstruct
from qualification import qualify, sealed_files
from restore_b1_r4 import verify_installed_sources


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_run(root, existing_evidence=None):
    root = root.resolve()
    manifest = sealed_files(root)
    from serving_warmup import audit as audit_warmup

    warmup_evidence = audit_warmup(root)
    receipt = json.loads((root / "receipt.json").read_bytes())
    if receipt.get("quiescent") is not True:
        raise ValueError("Run is not quiescent")
    if receipt.get("kind") not in ("continuous", "joint"):
        raise ValueError("Not an original Engine protocol run")
    role = receipt.get("role", "B1")
    identity = root / "service/runtime-identity.json"
    if receipt.get("client_started"):
        if not identity.is_file():
            raise ValueError("Actual serving identity missing")
        actual = json.loads(identity.read_bytes())
        if role == "B0":
            from baseline_admission import validate_identity

            validate_identity(
                actual,
                json.loads((root / "baseline-source-admission.json").read_bytes()),
            )
        elif actual.get("worker") != "corrected_serving_runtime.CorrectedWorker":
            raise ValueError("Actual B1 worker identity missing or different")
    records = []
    for path in sorted((root / "service").glob("*.json")):
        row = json.loads(path.read_bytes())
        if not (
            isinstance(row, dict)
            and "success" in row
            and "start" in row
            and "end" in row
            and ("phase" in row or "case_id" in row)
        ):
            continue
        wire = path.with_name(path.stem + ".sse.jsonl.gz")
        if row["success"]:
            first, last, usage, finish = reconstruct(wire.read_bytes())
            if not (
                row["http_status"] == 200
                and first == row["first_token"]["monotonic"]
                and last == row["last_token"]["monotonic"]
                and usage == row["usage"]
                and finish == row["finish_reason"]
                and row["start"]["monotonic"]
                <= first
                <= last
                <= row["end"]["monotonic"]
                and usage.get("prompt_tokens_details", {}).get("cached_tokens")
                in (None, 0)
            ):
                raise ValueError("Successful stream/scalar mismatch: " + path.name)
            if "phase" in row and not (
                finish == "length"
                and usage["completion_tokens"] == 256
                and usage["prompt_tokens"] == row["input_tokens"]
            ):
                raise ValueError("Original continuous token budget differs")
        elif wire.exists():
            # Preserve partial/error streams; an HTTP 200 never overrides failure.
            for line in gzip.decompress(wire.read_bytes()).splitlines():
                json.loads(line)
        row["raw_stream_present"] = wire.exists()
        row["record_file"] = str(path.relative_to(root))
        records.append(row)
    identifiers = [row.get("id", row.get("index")) for row in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate request records")
    groups = {}
    for name in sorted({r.get("phase", "joint200") for r in records}):
        cohort = [r for r in records if r.get("phase", "joint200") == name]
        successes = [r for r in cohort if r["success"]]

        def mean(values):
            return statistics.fmean(values) if values else None

        groups[name] = {
            "recorded": len(cohort),
            "successful": len(successes),
            "failed": len(cohort) - len(successes),
            "successful_output_tokens": sum(
                r["usage"]["completion_tokens"] for r in successes
            ),
            "mean_ttft_ms": mean(
                [
                    (r["first_token"]["monotonic"] - r["start"]["monotonic"]) * 1000
                    for r in successes
                ]
            ),
            "mean_tpot_ms": mean(
                [
                    (r["end"]["monotonic"] - r["first_token"]["monotonic"])
                    * 1000
                    / (r["usage"]["completion_tokens"] - 1)
                    for r in successes
                    if r["usage"]["completion_tokens"] > 1
                ]
            ),
        }
    # Use original submission-to-drain phase clocks, not summed overlapping latencies.
    phase_file = (
        root
        / "service"
        / ("continuous.json" if receipt["kind"] == "continuous" else "joint.json")
    )
    if phase_file.is_file():
        phases = json.loads(phase_file.read_bytes())
        phases = (
            phases["phases"]
            if receipt["kind"] == "continuous"
            else [{**phases, "name": "joint200"}]
        )
        for phase in phases:
            name = phase["name"]
            if name not in groups:
                continue
            cohort = [r for r in records if r.get("phase", "joint200") == name]
            start, end = phase["start"]["monotonic"], phase["end"]["monotonic"]
            if not (
                math.isfinite(start)
                and math.isfinite(end)
                and end > start
                and all(
                    start <= r["start"]["monotonic"] <= r["end"]["monotonic"] <= end
                    for r in cohort
                )
            ):
                raise ValueError("Phase clock does not contain its actual requests")
            groups[name]["cohort_wall_seconds"] = end - start
            groups[name]["output_tokens_per_second"] = groups[name][
                "successful_output_tokens"
            ] / (end - start)
            if receipt["kind"] == "joint" and not (
                phase["total"] == 200 == len(cohort)
                and phase["completed"] == groups[name]["successful"]
                and phase["failed"] == groups[name]["failed"]
                and phase["successful_output_tokens"]
                == groups[name]["successful_output_tokens"]
                and math.isclose(phase["duration_seconds"], end - start)
                and math.isclose(
                    phase["output_tokens_per_second"],
                    groups[name]["output_tokens_per_second"],
                )
            ):
                raise ValueError(
                    "Joint aggregate does not replay against actual requests"
                )
    if (
        receipt["kind"] == "joint"
        and receipt.get("completed")
        and not (
            len(records) == 200
            and all(r["success"] for r in records)
            and phase_file.is_file()
        )
    ):
        raise ValueError(
            "Completed joint run must have all200 successful raw request records"
        )
    qualification = None
    qualification_error = None
    if receipt.get("completed") and receipt["kind"] == "continuous" and role == "B1":
        deployment = Path(json.loads((root / "deployment.json").read_bytes())["root"])
        try:
            qualification = qualify(root, deployment)
        except (OSError, KeyError, ValueError) as error:
            qualification_error = type(error).__name__ + ": " + str(error)
    if (
        receipt["kind"] == "joint"
        and (root / "continuous-qualification.json").is_file()
    ):
        # The admission receipt binds the exact sealed continuous prerequisite.
        # Re-audit it instead of treating the joint run as another C8 run.
        saved = json.loads((root / "continuous-qualification.json").read_bytes())
        deployment = Path(json.loads((root / "deployment.json").read_bytes())["root"])
        qualification = qualify(Path(saved["run"]), deployment)
        if qualification != saved:
            raise ValueError("Joint continuous qualification receipt does not replay")
    quality = None
    if receipt["kind"] == "joint" and records:
        deployment = Path(json.loads((root / "deployment.json").read_bytes())["root"])
        verify_installed_sources(deployment)
        gold = json.loads(
            (deployment / "inputs/verified-mmlu-quality-inputs.json").read_bytes()
        )["cases"]
        original_requests = json.loads(
            (
                deployment / "inputs/mmlu-quality018-r2/quality-request-plan.json"
            ).read_bytes()
        )
        actual_requests = json.loads((root / "service/request-plan.json").read_bytes())
        expected_tokens = json.loads(
            (
                deployment / "inputs/mmlu-quality018-r2/quality-tokenize-preflight.json"
            ).read_bytes()
        )
        expected_tokens = [
            {k: r[k] for k in ("case_id", "count", "token_ids_sha256")}
            for r in expected_tokens
        ]
        if not (
            len(gold) == len(expected_tokens) == 200
            and actual_requests["requests"] == original_requests["requests"]
            and actual_requests["request_count"] == 200
            and actual_requests["warmups"] == 0
            and actual_requests["request_rate"] == "inf"
            and actual_requests["scorer_sha256"] == original_requests["scorer_sha256"]
            and json.loads((root / "service/tokenize-preflight.json").read_bytes())
            == expected_tokens
        ):
            raise ValueError("Joint request protocol or server token binding differs")
        scorer = deployment / "scorers/mmlu_pro_pyramidkv.py"
        nodes = [
            n
            for n in ast.parse(scorer.read_bytes()).body
            if isinstance(n, ast.FunctionDef) and n.name == "extract_answer"
        ]
        if len(nodes) != 1:
            raise ValueError("Ambiguous frozen scorer")
        namespace = {"re": re}
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(scorer), "exec"),
            namespace,
        )
        correct = unparsed = 0
        for row in records:
            index = row["index"]
            if not isinstance(index, int) or not 0 <= index < 200:
                raise ValueError("Joint request index is outside original200")
            if row["success"] and not (
                row["usage"]["prompt_tokens"] == expected_tokens[index]["count"]
                and 0 < row["usage"]["completion_tokens"] <= 8192
                and row["output"]
                and not row["reasoning"]
            ):
                raise ValueError(
                    "Joint successful response violates original wire/budget"
                )
            if (
                row["case_id"] != gold[index]["case_id"]
                or row["gold"] != gold[index]["answer"]
            ):
                raise ValueError("Frozen MMLU gold binding differs")
            content = reasoning = ""
            wire = root / Path(row["record_file"]).with_suffix(".sse.jsonl.gz")
            if wire.exists():
                for line in gzip.decompress(wire.read_bytes()).splitlines():
                    saved = json.loads(line)["raw"]
                    if not saved.startswith("data:") or saved[5:].strip() == "[DONE]":
                        continue
                    event = json.loads(saved[5:].strip())
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        content += delta.get("content") or ""
                        reasoning += (
                            delta.get("reasoning_content")
                            or delta.get("reasoning")
                            or ""
                        )
            if content != row["output"] or reasoning != row["reasoning"]:
                raise ValueError("Joint raw content/scalar mismatch")
            answer = namespace["extract_answer"](content) if row["success"] else None
            matched = bool(row["success"] and answer == gold[index]["answer"])
            if answer != row["answer"] or matched != row["correct"]:
                raise ValueError("Joint score does not replay")
            correct += matched
            unparsed += answer is None
        quality = {
            "planned_denominator": 200,
            "recorded": len(records),
            "correct": correct,
            "accuracy_pct_of_planned200": 100 * correct / 200,
            "truncated_recorded": sum(r["finish_reason"] == "length" for r in records),
            "unparsed_recorded": unparsed,
            "unreturned": 200 - len(records),
            "scorer_sha256": sha(scorer),
            "scope": "Descriptive MMLU200 only; not the five-dataset acceptance suite",
        }
    joint_activation = None
    if receipt["kind"] == "joint" and receipt.get("completed") and role == "B1":
        activation_files = {
            name.removeprefix("service/"): (root / name).read_bytes()
            for name in manifest
            if name.startswith("service/")
            and (name.endswith(".jsonl") or name.endswith("runtime-identity.json"))
        }
        joint_activation = audit_activation(
            activation_files,
            prefix="",
            expected_requests=200
            + (warmup_evidence["requests"] if warmup_evidence else 0),
        )
        plan = json.loads((root / "service/plan.json").read_bytes())
        runtime_env = json.loads(
            (root / "service/runtime-environment.json").read_bytes()
        )
        expected_packed = json.loads((root / "installed-files.json").read_bytes())[
            "mods/packed-host/packed_host_state.py"
        ]
        rank_pids = [
            name.removeprefix("microflow-").removesuffix(".jsonl")
            for name in activation_files
            if name.startswith("microflow-") and name.endswith(".jsonl")
        ]
        if not (
            plan["host_budget_per_rank_bytes"] == 4 * 2**30
            and runtime_env["BETTERSCALE_STATE_CACHE_HOST_BYTES"] == str(4 * 2**30)
            and plan["physical_cards"]
            == json.loads((root / "resource-admission.json").read_bytes())[
                "physical_cards"
            ]
            and len(rank_pids) == 2
            and all(
                json.loads(
                    (
                        root / ("service/packed-host-install-" + pid + ".json")
                    ).read_bytes()
                )["adapter_sha256"]
                == expected_packed
                for pid in rank_pids
            )
        ):
            raise ValueError("Joint packed-host activation or original budget differs")
        joint_activation.update(
            host_budget_per_rank_bytes=4 * 2**30, packed_host_sha256=expected_packed
        )
    targets = {
        "3.4.1": {
            "name": "有效算力效率 ≥50%",
            "reason": "本单臂未执行原out256同步F/T测量；项目已有合格参考由正式证据入口复算，不能据本记录推断项目缺少参考。",
        },
        "3.4.2": {
            "name": "吞吐 ≥10%，TTFT/TPOT 各改善 ≥5%",
            "reason": "本记录为单臂；项目已有原固定200题配对，本次新配对由冻结pair清单及两臂原始记录单独核验，不事后增设SLO或质量否决项。",
        },
        "3.4.3": {
            "name": "五数据集与 16K 稳定性",
            "reason": "本单臂不覆盖完整五数据集；项目既有覆盖由正式证据入口逐项核对。"
            + (
                "连续测试资格已复核；正式任务与阈值仍须单独验证。"
                if qualification
                else "4K/16K 连续测试资格未通过或未完成，保留已失败请求。"
            ),
        },
        "3.4.4": {
            "name": "单位 token 成本降低 ≥10%及高负载利用率改善",
            "reason": "本单臂不能计算配对成本降幅；已有公开报价与原同费率配对另行复算，保留类似机型估算范围，不新增必须发票的要求。",
        },
    }
    for value in targets.values():
        value.update(status="unassessable", accepted=False, measured_value=None)
    current_run_targets = targets
    if existing_evidence is not None:
        targets = existing_evidence["formal_targets"]
    failure_detail = receipt.get("failure")
    log = root / "service.log"
    if log.exists():
        text = log.read_text(errors="replace")
        if "aclrtMallocHostWithCfg, error code is 207001" in text:
            failure_detail = "主机 State 换出时，锁页分配 aclrtMallocHostWithCfg 返回207001；详见封存的 service.log。"
        elif "AF_UNIX path too long" in text:
            failure_detail = "服务初始化时 Unix socket 路径超过内核长度限制；详见封存的 service.log。"
    return {
        "serving_warmup": warmup_evidence,
        "schema": "vllm-hust-engine-execution-report/v1",
        "run": str(root),
        "source_seal_sha256": sha(root / "checksums.json"),
        "report_sources": {
            p.name: sha(p) for p in Path(__file__).parent.glob("*.py") if p.is_file()
        },
        "sealed_files": len(manifest),
        "receipt": receipt,
        "recorded_requests": len(records),
        "successful_requests": sum(r["success"] for r in records),
        "failed_requests": sum(not r["success"] for r in records),
        "groups": groups,
        "quality": quality,
        "joint_activation": joint_activation,
        "continuous_qualification": qualification,
        "continuous_qualification_error": qualification_error,
        "failure_detail": failure_detail,
        "formal_targets": targets,
        "current_run_targets": current_run_targets,
        "existing_project_evidence": existing_evidence,
        "formal_complete": False,
        "scope": f"真实 {role} 原协议补充记录；失败及未执行项保留。",
        "role": role,
    }


def render(root, out, project_root=None):
    from project_evidence import assess, ensure

    data = inspect_run(root, assess(ensure(project_root)))
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib

    if matplotlib.__version__ != "3.10.8":
        raise RuntimeError("Report environment requires matplotlib==3.10.8")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.1))
    titles = [
        "3.4.1 Effective FLOPs",
        "3.4.2 Paired performance",
        "3.4.3 Five datasets + 16K",
        "3.4.4 Cost + utilization",
    ]
    subtitles = [
        "Existing qualified M01, out256",
        "Existing matched200, max8192",
        "SWE26/731; Terminal4/89",
        "Existing equal-rate estimate",
    ]
    for axis, title, value, subtitle in zip(
        axes, titles, data["formal_targets"].values(), subtitles, strict=True
    ):
        axis.set_title(title, fontsize=10)
        axis.text(
            0.5,
            0.58,
            value.get("display_value", "N/A"),
            ha="center",
            va="center",
            fontsize=25,
            color="#166534" if value.get("accepted") else "#b45309",
        )
        axis.text(0.5, 0.24, subtitle, ha="center", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Existing project evidence, separate frozen workloads; current run shown separately",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "four-targets.png", dpi=180)
    fig.savefig(out / "four-targets.svg", metadata={"Date": None})
    plt.close(fig)
    if data["groups"]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for axis, metric, title in zip(
            axes,
            ("output_tokens_per_second", "mean_ttft_ms", "mean_tpot_ms"),
            ("Successful output tokens/s", "Mean TTFT (ms)", "Mean TPOT (ms)"),
            strict=True,
        ):
            observed = [
                (name, row[metric])
                for name, row in data["groups"].items()
                if row.get(metric) is not None
            ]
            if observed:
                axis.bar(
                    [
                        x[0].replace("16384", "16K").replace("4096", "4K")
                        for x in observed
                    ],
                    [x[1] for x in observed],
                    color="#2563eb",
                )
                axis.tick_params(axis="x", labelrotation=25)
            else:
                axis.text(0.5, 0.5, "N/A", transform=axis.transAxes, ha="center")
            axis.set_title(title, fontsize=10)
        fig.suptitle(
            f"Observed {data['role']} protocol only; paired improvement not assessed",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out / "observed-performance.png", dpi=180)
        fig.savefig(out / "observed-performance.svg", metadata={"Date": None})
        plt.close(fig)
    (out / "summary.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    esc = html.escape
    statuses = {
        "verified_existing_pass": "已有证据适用范围内达标",
        "verified_existing_pair_numeric_pass": "已有原配对数值达标",
        "verified_existing_pair_numeric_fail": "已有原配对数值未达标",
        "incomplete_dataset_coverage": "任务范围未完成",
        "verified_existing_equal_rate_estimate": "达标（同费率估算范围）",
    }
    rows = "".join(
        f"<tr><td>{key} {esc(value['name'])}</td><td>{esc(statuses.get(value['status'], value['status']))}</td><td>{esc(value['reason'])}<br><a href='{esc(value['source_url'], quote=True)}'>固定来源</a></td></tr>"
        for key, value in data["formal_targets"].items()
    )
    observed_image = (
        "<img src='observed-performance.png' alt='本轮实际时延与吞吐，不计算对照改善率'>"
        if data["groups"]
        else ""
    )
    body = f"""<!doctype html><meta charset=utf-8><title>vLLM-HUST Engine 原始运行报告</title>
<style>body{{font:16px/1.6 system-ui;max-width:1100px;margin:32px auto}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cbd5e1;padding:10px;text-align:left}}img{{max-width:100%}}pre{{white-space:pre-wrap;background:#f1f5f9;padding:16px}}</style>
<h1>vLLM-HUST Engine 原始运行报告</h1><p>{esc(data["scope"])}</p>
<p>本轮状态：{esc(data["receipt"]["status"])}。记录 {data["recorded_requests"]} 请求，成功 {data["successful_requests"]}，失败 {data["failed_requests"]}。联合任务计划 200 请求，是否执行见原始回执；未返回项不当作成功。</p>
<p>阻塞：{esc(str(data["failure_detail"] or data["continuous_qualification_error"] or "本轮执行结束；正式验收证据仍不完整"))}</p>
<p><strong>四目标表采用已校验复算的既有项目证据，各项保留原工况与来源。本次新增单臂记录单独列示，不跨工况借用F/T，不把旧配对冒充新配对。</strong></p>
<img src='four-targets.png' alt='既有正式证据与覆盖缺口，适用工况分别注明'>
<table><tr><th>正式测试目标</th><th>是否达标</th><th>具体卡点</th></tr>{rows}</table>
<h2>五数据集既有覆盖</h2><pre>{esc(json.dumps(data["existing_project_evidence"]["dataset_coverage"], ensure_ascii=False, indent=2))}</pre>
<h2>本轮实际任务质量</h2><pre>{esc(json.dumps(data["quality"], ensure_ascii=False, indent=2)) if data["quality"] else "未执行联合200题"}</pre>
{observed_image}
<h2>原始分母与描述性时延</h2><p>C8 时延只描述本轮闭环补充测试，不与旧诊断负载计算改善率。</p><pre>{esc(json.dumps(data["groups"], ensure_ascii=False, indent=2))}</pre>
<p>源运行：{esc(data["run"])}<br>封存清单 SHA256：{data["source_seal_sha256"]}<br>完整计算结果：<a href='summary.json'>summary.json</a></p>"""
    (out / "report.html").write_text(body)
    hashes = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / "checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("REPORT", out / "report.html")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--project-evidence",
        type=Path,
        help="Verified pinned benchmark evidence cache; otherwise fetch automatically",
    )
    args = parser.parse_args()
    render(args.run, args.out.resolve(), args.project_evidence)


if __name__ == "__main__":
    main()
