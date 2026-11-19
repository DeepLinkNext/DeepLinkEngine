"""Verify and recompute existing formal evidence without borrowing its scope for new runs."""

import argparse
from decimal import localcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tarfile
import tempfile
import urllib.request

from fetch_b1_r4 import SafeRedirect, resolve_token
from telemetry_audit import utilization as calculate_utilization

BENCHMARK_COMMIT = "53541118b3461705f1e61532710ff0a759d59af0"
MANIFEST_SHA256 = "5c7194b483d96e7a682183fa62c569737312d8139fc0b20b51e91a5479772ad6"
ASSET_ID = 737688443
ASSET_BYTES = 17200471
ASSET_SHA256 = "e887a90707989281d3421d16c3cc73fc3827a2a440c636e1ad134a8ad1c13cb9"
BASE = "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/" + BENCHMARK_COMMIT + "/"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = root / "source-readback.json"
    if digest(manifest) != MANIFEST_SHA256:
        raise ValueError(
            "Formal evidence manifest differs from the pinned benchmark snapshot"
        )
    data = json.loads(manifest.read_bytes())
    if data["commit"] != BENCHMARK_COMMIT:
        raise ValueError("Formal benchmark revision differs")
    for name, binding in data["files"].items():
        source = root / name
        if not (
            source.resolve().is_relative_to(root)
            and not source.is_symlink()
            and source.stat().st_size == binding["bytes"]
            and digest(source) == binding["sha256"]
        ):
            raise ValueError("Formal evidence changed: " + name)
    return data


def ensure(explicit=None, env_file=None):
    selected = explicit or os.environ.get("VLLM_HUST_PROJECT_EVIDENCE")
    if selected:
        root = Path(selected).expanduser().resolve()
        verify(root)
        return root
    cache = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "vllm-hust-engine"
        / "project-evidence-53541118"
    )
    root = cache / "benchmark-evidence"
    if root.exists():
        verify(root)
        return root
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "source.tar.gz"
    if not (
        archive.is_file()
        and archive.stat().st_size == ASSET_BYTES
        and digest(archive) == ASSET_SHA256
    ):
        token, _ = resolve_token(env_file)
        request = urllib.request.Request(
            "https://api.github.com/repos/vLLM-HUST/vllm-hust-engine/releases/assets/"
            + str(ASSET_ID),
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/octet-stream",
                "User-Agent": "vllm-hust-qualified-project-evidence",
            },
        )
        partial = cache / "source.partial"
        with (
            urllib.request.build_opener(SafeRedirect()).open(
                request, timeout=120
            ) as response,
            partial.open("wb") as output,
        ):
            while block := response.read(1024 * 1024):
                output.write(block)
                if output.tell() > ASSET_BYTES:
                    raise ValueError("Formal evidence archive exceeds its pinned size")
        if partial.stat().st_size != ASSET_BYTES or digest(partial) != ASSET_SHA256:
            raise ValueError("Formal evidence archive SHA256 differs")
        partial.replace(archive)
    with tempfile.TemporaryDirectory(prefix="unpack-", dir=cache) as temp:
        with tarfile.open(archive) as source:
            names = set()
            for member in source.getmembers():
                path = Path(member.name)
                if (
                    not member.isfile()
                    or path.is_absolute()
                    or ".." in path.parts
                    or path.parts[0] != "benchmark-evidence"
                    or member.name in names
                ):
                    raise ValueError("Unsafe formal evidence archive member")
                names.add(member.name)
            source.extractall(temp, filter="data")
        candidate = Path(temp) / "benchmark-evidence"
        verify(candidate)
        candidate.rename(root)
    return root


def load_module(root, name):
    path = root / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("qualified_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_original_utilization(root, joint):
    report = root / "reports/full-answer-joint-20261030"
    manifest = json.loads((report / "telemetry-manifest.json").read_bytes())["files"]
    with tarfile.open(report / "original-telemetry.tar.gz") as archive:
        members = {}
        for item in archive.getmembers():
            if not item.isfile() or item.name in members or item.name not in manifest:
                raise ValueError("Unexpected original telemetry member")
            raw = archive.extractfile(item).read()
            if hashlib.sha256(raw).hexdigest() != manifest[item.name]:
                raise ValueError("Original telemetry member changed: " + item.name)
            members[item.name] = raw
        if set(members) != set(manifest):
            raise ValueError("Original telemetry archive is incomplete")
    result = {}
    for arm in joint["arms"]:
        prefix = {"B0": "joint-b0-r2", "B1": "joint-b1-r4"}[arm["role"]]
        phase = json.loads(members[prefix + "/joint.json"])
        rows = [
            json.loads(line)
            for line in members[prefix + "/device-samples.jsonl"].splitlines()
            if line
        ]
        actual = calculate_utilization(
            rows, phase["start"]["monotonic"], phase["end"]["monotonic"]
        )
        for key in ("covered_seconds", "missing_seconds", "mean_percent"):
            if not math.isclose(actual[key], arm["utilization"][key], abs_tol=1e-8):
                raise ValueError(
                    "Original utilization does not recompute: " + arm["role"]
                )
        for card, expected in arm["utilization"]["per_card_percent"].items():
            if not math.isclose(
                actual["per_card_percent"][card], expected, abs_tol=1e-8
            ):
                raise ValueError("Original per-card utilization differs")
        result[arm["role"]] = actual
    return result


def assess(root):
    root = Path(root).resolve()
    sources = verify(root)
    with localcontext() as context:
        context.prec = 40
        m01 = load_module(root, "recompute_m01_qualified_pair").calculate()
    expected = json.loads(
        (root / "reports/m01-qualified-pair-20261103/summary.json").read_bytes()
    )
    if m01 != expected:
        raise ValueError("Existing qualified M01 evidence does not recompute")
    joint = json.loads(
        (root / "reports/full-answer-joint-20261030/joint-summary.json").read_bytes()
    )
    paired = load_module(root, "recompute_full_answer_joint").paired_metrics(joint)
    if (
        paired["paired_quality"] != joint["paired_quality"]
        or not math.isclose(
            paired["output_throughput_change_percent"],
            joint["output_throughput_change_percent"],
        )
        or not math.isclose(
            paired["rental_cost_reduction_percent"],
            joint["rental_cost_reduction_percent"],
        )
    ):
        raise ValueError("Existing complete-answer paired metrics do not recompute")
    a, b = paired["arms"]
    performance = {
        "throughput_gain_pct": paired["output_throughput_change_percent"],
        "mean_ttft_improvement_pct": 100
        * (1 - b["timing"]["ttft_ms"]["mean"] / a["timing"]["ttft_ms"]["mean"]),
        "mean_tpot_improvement_pct": 100
        * (1 - b["timing"]["tpot_ms"]["mean"] / a["timing"]["tpot_ms"]["mean"]),
    }
    numeric_pass = (
        performance["throughput_gain_pct"] >= 10
        and min(
            performance["mean_ttft_improvement_pct"],
            performance["mean_tpot_improvement_pct"],
        )
        >= 5
    )
    eta = next(
        float(x["aggregate"]["eta_percent_lower"])
        for x in m01["arms"]
        if x["role"] == "B1"
    )
    utilization = audit_original_utilization(root, joint)
    util_gain = utilization["B1"]["mean_percent"] - utilization["B0"]["mean_percent"]
    targets = {
        "3.4.1": {
            "name": "有效算力效率 ≥50%",
            "status": "verified_existing_pass",
            "accepted": True,
            "accepted_scope": "Existing qualified M01 fixed200/out256, three synchronized uninstrumented repetitions per arm",
            "measured_value": eta,
            "display_value": f"≥{eta:.2f}%",
            "source_url": BASE + "reports/m01-qualified-pair-20261103/README.md",
            "reason": "已有独立合格参考，原脚本已重新核验算量、三轮计时及关键路径。B1保守效率下界达到50%；适用于原out256工况，不套用到新out8192运行。",
        },
        "3.4.2": {
            "name": "吞吐 ≥10%，TTFT/TPOT 各改善 ≥5%",
            "status": "verified_existing_pair_numeric_pass"
            if numeric_pass
            else "verified_existing_pair_numeric_fail",
            "accepted": numeric_pass,
            "accepted_scope": "Existing original matched200/max8192 pair; single fixed subset with original quality and service protocol",
            "measured_value": performance["throughput_gain_pct"],
            "display_value": f"+{performance['throughput_gain_pct']:.2f}%",
            "source_url": BASE + "reports/full-answer-joint-20261030/README.md",
            "reason": f"已有固定0.18 B0/B1原200题配对，原标量复算通过：吞吐+{performance['throughput_gain_pct']:.2f}%，平均TTFT/TPOT改善{performance['mean_ttft_improvement_pct']:.2f}%/{performance['mean_tpot_improvement_pct']:.2f}%。保留143/153正确、长度终止与单次子集边界；不是全集或统计非劣证明，不事后新增SLO否决项。",
        },
        "3.4.3": {
            "name": "五数据集与16K稳定性",
            "status": "incomplete_dataset_coverage",
            "accepted": False,
            "accepted_scope": None,
            "measured_value": None,
            "display_value": "Incomplete",
            "source_url": BASE + "reports/vllm-hust-midterm-20261019/README.md",
            "reason": "已有MMLU、HLE和FrontierScience记录；SWE为26/731次尝试、705未执行，Terminal为4/89、85未尝试，部分退出/判分仍未决。合成C8与单臂MMLU补跑不能填齐这些任务。",
        },
        "3.4.4": {
            "name": "单位token成本降低 ≥10%及高负载利用率改善",
            "status": "verified_existing_equal_rate_estimate",
            "accepted": paired["rental_cost_reduction_percent"] >= 10 and util_gain > 0,
            "numeric_threshold_met": paired["rental_cost_reduction_percent"] >= 10,
            "accepted_scope": "Conditional public analogous-machine equal-rate estimate; actual invoice is not asserted or newly required",
            "measured_value": paired["rental_cost_reduction_percent"],
            "display_value": f"−{paired['rental_cost_reduction_percent']:.2f}%",
            "source_url": BASE + "reports/full-answer-joint-20261030/README.md",
            "reason": f"已有同负载配对和公开报价费用依据，同费率估算降低{paired['rental_cost_reduction_percent']:.2f}%；原始双卡采样按同请求区间重新复算，AI Core均值上升{util_gain:.2f}个百分点。该配对在披露的类似机型同费率估算范围内达到成本与利用率要求，不冒充本机账单或全场景结论。",
        },
    }
    coverage = {
        "MMLU-Pro": {
            "existing_scope": "Original matched200; separate out256 M01",
            "fresh_scope": "Counted separately in current run/pair",
        },
        "HLE-Verified": {
            "existing_solver_cases": 1811,
            "gold": 668,
            "revision": 1143,
            "scope": "Frozen substitute judge; unresolved grades retained; no new private regrading claimed",
        },
        "FrontierScience": {
            "existing_solver_cases_per_arm": 160,
            "olympiad": 100,
            "research": 60,
            "scope": "Designated018 and frozen optimized candidate; original substitute judge and unresolved cases retained",
        },
        "SWE-bench-Pro": {
            "attempted": 26,
            "planned": 731,
            "unexecuted": 705,
            "original_successes": 3,
            "same_answer_dependency_recovery_successes": 4,
        },
        "TerminalBench2.1": {
            "attempted": 4,
            "planned": 89,
            "unattempted": 85,
            "scope": "Includes context termination and null verifier; attempts are not successful completions",
        },
    }
    return {
        "schema": "vllm-hust-existing-project-evidence/v1",
        "benchmark_commit": BENCHMARK_COMMIT,
        "manifest_sha256": MANIFEST_SHA256,
        "verified_source_files": len(sources["files"]),
        "m01": m01,
        "paired_full_answer": paired,
        "paired_performance": performance,
        "original_utilization": utilization,
        "utilization_gain_pp": util_gain,
        "price_basis": joint["arms"][0]["cost"]["price_basis"],
        "dataset_coverage": coverage,
        "formal_targets": targets,
        "formal_complete": False,
        "scope": "Recomputed existing project evidence and raw physical-card utilization, not new hardware runs or private-answer regrading",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = assess(ensure(args.root))
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(
        "PROJECT_EVIDENCE",
        args.out,
        "existing M01 and paired metrics verified; overall coverage incomplete",
    )


if __name__ == "__main__":
    main()
