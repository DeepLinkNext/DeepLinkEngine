"""Resume the archived 9B submission through the public one-click entry."""

from argparse import ArgumentParser
from html import escape
import json
from pathlib import Path

from dataset_contract import save, sha
from reference_9b_archive import CARDS, MODEL, RELEASE, REVISION, native_clocks, retained_sections


def run(args):
    archive = args.archive.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sections, retained = retained_sections(archive)
    reference_dir = out / "compute-reference"
    reference_dir.mkdir()
    reference = dict(reference_time_seconds=None, efficiency_percent=None,
        full_model_certificate=False, new_model_inference=False)
    phase = "native-clocks"
    try:
        clocks = native_clocks(archive)
        reference["native_clocks"] = clocks
        if args.original018_target_kv_proof is not None or args.r11_target_kv_proof is not None:
            phase = "original-target-kv-logical-evidence"
            from reference_9b_logical_target import admit as admit_target_kv

            evidence = {}
            for role, path in (("B0", args.original018_target_kv_proof), ("B1", args.r11_target_kv_proof)):
                if path is not None:
                    evidence[role] = admit_target_kv(path, role, clocks[role])
            reference["native_target_kv_logical_evidence"] = evidence
            save(reference_dir / "original-target-kv-logical-evidence.json", evidence)
        ffn_branches = b0_order = retained_work = r11_prefill = None
        if args.ffn_branch_proof is not None:
            phase = "native-ffn-branch-admission"
            from reference_9b_ffn_branch import BASE_PROJECTION, admit

            if args.component_projection is None or sha(args.component_projection) != BASE_PROJECTION:
                raise ValueError("FFN branch clocks require the original 9B component graph")
            ffn_branches = admit(args.ffn_branch_proof, clocks)
        if args.original018_request_order is not None:
            phase = "original018-required-request-order"
            from reference_9b_ffn_branch import admit_b0_order

            if ffn_branches is None:
                raise ValueError("Original018 request order must be paired with its native FFN branch proof")
            b0_order = admit_b0_order(args.original018_request_order, ffn_branches, clocks)
        if args.retained_query_work_proof is not None:
            phase = "native-retained-query-arithmetic"
            from reference_9b_projection import read_retained_work

            if args.component_projection is None:
                raise ValueError("Retained-query arithmetic requires a component projection")
            retained_work = read_retained_work(args.retained_query_work_proof, clocks)
        gdn_work = None
        if args.gdn_arithmetic_proof is not None:
            phase = "native-target-gdn-arithmetic"
            from reference_9b_projection import read_gdn_work

            if retained_work is None:
                raise ValueError("GDN arithmetic needs the original retained-query arithmetic terms")
            gdn_work = read_gdn_work(args.gdn_arithmetic_proof, clocks, retained_work)
        if args.r11_first_prefill_proof is not None:
            phase = "r11-first-prefill-admission"
            from reference_9b_ffn_branch import admit_r11_prefill

            if ffn_branches is None or args.original018_prefill_proof is None:
                raise ValueError("The r11 first prefill requires the native FFN domain and signed 1369-row timings")
            r11_prefill = admit_r11_prefill(args.r11_first_prefill_proof, clocks)
        r11_prefill_hccl = None
        if args.r11_prefill_tp2_cache is not None or args.r11_prefill_tp2_cache_sha256 is not None:
            phase = "r11-first-prefill-collective-clock"
            from reference_9b_r11_prefill_hccl import resolve as resolve_prefill_tp2

            if r11_prefill is None or args.r11_prefill_tp2_cache is None or args.r11_prefill_tp2_cache_sha256 is None:
                raise ValueError("The r11 first prefill TP2 clock needs a proved prefill and a sealed cache index")
            r11_prefill_hccl, reference["r11_first_prefill_collective_clock"] = resolve_prefill_tp2(
                archive, args.baseline_sdk.resolve(), reference_dir, clocks, r11_prefill,
                args.r11_prefill_tp2_cache.resolve(), args.r11_prefill_tp2_cache_sha256)
        cold_kv = None
        if args.r11_prefill_kv_cache is not None or args.r11_prefill_kv_cache_sha256 is not None:
            phase = "r11-cold-target-kv-clock"
            from reference_9b_r11_prefill_kv import resolve as resolve_prefill_kv

            if (r11_prefill_hccl is None or args.target_fia_cache is None or args.r11_prefill_kv_cache is None
                    or args.r11_prefill_kv_cache_sha256 is None):
                raise ValueError("Cold target KV clocks require the exact prefill graph and both sealed clock caches")
            cold_kv, reference["r11_first_prefill_kv_calibration"] = resolve_prefill_kv(
                archive, reference_dir, args.r11_prefill_kv_cache.resolve(), args.r11_prefill_kv_cache_sha256,
                args.target_fia_cache.resolve(), clocks)
        cold_gdn = None
        if args.r11_cold_gdn_cache is not None or args.r11_cold_gdn_cache_sha256 is not None:
            phase = "r11-cold-gdn-convolution-clock"
            from reference_9b_r11_gdn_conv import resolve as resolve_gdn_conv

            if (cold_kv is None or args.r11_cold_gdn_cache is None or args.r11_cold_gdn_cache_sha256 is None):
                raise ValueError("Cold GDN requires the proved first-packet graph and a sealed convolution cache")
            cold_gdn, reference["r11_first_prefill_gdn_calibration"] = resolve_gdn_conv(
                archive, reference_dir, args.r11_cold_gdn_cache.resolve(), args.r11_cold_gdn_cache_sha256,
                args.target_fia_cache.resolve(), clocks)
        phase = "canonical-device-task-clocks"
        from reference_9b_task_cache import resolve

        cache = args.cache.resolve()
        if not (cache / "task-clock-cache.json").is_file():
            raise ValueError("9B device task clocks are required. The legacy external graph replay clock "
                             "also measured host submission gaps and cannot be used here.")
        reference["dense_calibration"] = resolve(archive, args.calibration_archive.resolve(),
            cache, args.cache_sha256, reference_dir, clocks, args.baseline_sdk.resolve(),
            archive / "installation/profiles/qwen9b/sdk", args.original018_prefill_proof, ffn_branches)
        fia = b0_fia = r11_draft = b0_cold_kv = None
        phase = "independent-reference-graph"
        if (args.b0_cold_kv_cache is not None or args.b0_cold_kv_cache_sha256 is not None) and args.component_projection is None:
            raise ValueError("Original018 cold KV requires an original 9B component projection")
        if (args.first_draft_cache is not None or args.first_draft_cache_sha256 is not None) and args.component_projection is None:
            raise ValueError("The first-draft cache requires an original 9B graph")
        if args.component_projection is not None:
            from reference_9b_b0_fia import PROJECTION as DUAL_FIA_PROJECTION, resolve as resolve_b0_fia
            from reference_9b_projection import DRAFT_PROJECTION_SHA256, compose
            from reference_9b_target_fia import PROJECTION, resolve as resolve_fia

            projection_pin = sha(args.component_projection) if args.component_projection.is_file() else None
            if projection_pin in {PROJECTION, DUAL_FIA_PROJECTION, DRAFT_PROJECTION_SHA256}:
                phase = "target-attention-fia-component-clocks"
                fia, reference["target_fia_calibration"] = resolve_fia(
                    archive, reference_dir, args.target_fia_cache, args.target_fia_cache_sha256, clocks)
            if projection_pin in {DUAL_FIA_PROJECTION, DRAFT_PROJECTION_SHA256}:
                phase = "original018-direct-fia-component-clocks"
                b0_fia, reference["b0_direct_fia_calibration"] = resolve_b0_fia(
                    archive, reference_dir, args.target_fia_cache, args.target_fia_cache_sha256, clocks)
            if args.first_draft_cache is not None or args.first_draft_cache_sha256 is not None:
                phase = "r11-first-retained-draft-component-clocks"
                from reference_9b_draft_clock import resolve as resolve_draft

                if projection_pin != DRAFT_PROJECTION_SHA256:
                    raise ValueError("The first-draft cache requires the matching original 9B graph")
                r11_draft, reference["first_draft_calibration"] = resolve_draft(
                    archive, reference_dir, args.first_draft_cache, args.first_draft_cache_sha256, clocks)
            if args.b0_cold_kv_cache is not None or args.b0_cold_kv_cache_sha256 is not None:
                phase = "original018-cold-kv-write-clock"
                from reference_9b_b0_cold_kv import resolve as resolve_b0_cold_kv

                if b0_fia is None or args.b0_cold_kv_cache is None or args.b0_cold_kv_cache_sha256 is None:
                    raise ValueError("Original018 cold KV requires the admitted direct prefill and a sealed KV cache")
                b0_cold_kv, reference["b0_cold_kv_calibration"] = resolve_b0_cold_kv(
                    archive, reference_dir, args.b0_cold_kv_cache.resolve(), args.b0_cold_kv_cache_sha256,
                    args.target_fia_cache.resolve(), clocks)
            phase = "independent-reference-graph"
            reference["component_projection"] = compose(args.component_projection.resolve(),
                out, reference["dense_calibration"], clocks, fia, b0_fia, r11_draft, ffn_branches, b0_order, retained_work, r11_prefill, r11_prefill_hccl, cold_kv, gdn_work, cold_gdn=cold_gdn, b0_cold_kv=b0_cold_kv)
        reference.update(status="incomplete", failed_stage=phase,
            reason="The formal 4.1 denominator remains open. The projected lower bounds use the six "
                   "original clocks and the qualified subset of the dependency graph.",
            unresolved=[
                dict(role="B0", repetitions=[3], component="Physical target KV and scatter clocks",
                    source="original018 native output packets 806 and 875",
                    detail="Logical writer edges are recoverable. The original window does not determine "
                           "the physical page, slot and write-version history for these reads."),
                dict(role="B1", repetitions=[1, 2, 3], component="Target and MTP KV after cold prefill",
                    source="r11 original native clocks and sealed observer companion",
                    detail="Some initial operators are source-bound. Later rank-local page and writer "
                           "histories in the companion are not identities for the original windows."),
                dict(role="B0/B1", repetitions=[1, 2, 3], component="Remaining reference operations",
                    detail="Unqualified operation clocks and ordering constraints are left outside the "
                           "projected graph. See its coverage notes for the measured scope."),
            ])
        if args.conservative_reference_evidence is not None:
            phase = "formal-conservative-reference"
            from reference_9b_conservative import qualify

            if "component_projection" not in reference:
                raise ValueError("Formal conservative 4.1 requires an admitted original component graph")
            reference["conservative_reference"] = qualify(out, args.conservative_reference_evidence.resolve(), reference)
            reference.update(status="qualified-lower-bound",
                             reason="The original six windows have a qualified conservative reference. "
                                    "The complete independent reference time and exact efficiency remain unobserved.")
            del reference["failed_stage"]
    except Exception as error:
        reference.update(status="failed", failed_stage=phase,
            reason=f"{type(error).__name__}: {error}")
    report = dict(schema="vllm-hust-9b-original018-r11-one-click/v1", model=MODEL,
        checkpoint_revision=REVISION, baseline="original018", engine=RELEASE,
        physical_npus=CARDS, input_requests=200, max_output_tokens=256, tensor_parallel_size=2,
        retained_evidence=retained, **{"4.1": reference, **sections})
    save(out / "summary.json", report)
    document = [
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
        "<title>Qwen3.5-9B 一键验收</title>",
        "<h1>Qwen3.5-9B 一键验收</h1><p>original018 B0 与 r11 RC4 B1</p>",
        "<p>200 题，256 输出 token，BF16 TP2，910B2 物理卡 4、5。"
        " <a href='summary.json'>完整机器回执</a></p>",
    ]
    for number in ("4.1", "4.2", "4.3", "4.4"):
        contents = report[number]
        document.append(f"<section><h2>{number}</h2>")
        if number == "4.1":
            location = contents.get("failed_stage", contents["status"])
            document.append(f"<p>{escape(location)} · {escape(contents['reason'])}</p>")
            if "conservative_reference" in contents:
                receipt = contents["conservative_reference"]
                href = escape(receipt["path"], quote=True)
                document.append(f"<p><a href='{href}'>正式保守下界与原件清单</a></p>")
            projection = contents.get("component_projection", {})
            if "aggregate_lower_bounds" in projection:
                document.extend(render_lower_bounds(projection))
        document.append(f"<pre>{escape(json.dumps(contents, indent=2, ensure_ascii=False))}</pre></section>")
    document.append("</html>")
    (out / "report.html").write_text("\n".join(document) + "\n", encoding="utf-8")
    files = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    save(out / "checksums.json", files)
    print("REFERENCE_9B_REPORT", out / "report.html", flush=True)
    return 0 if reference["status"] == "qualified-lower-bound" else 2


def render_lower_bounds(projection):
    rows = [
        "<h3>按原墙钟复算的保守下界</h3>",
        "<p>表中参考时长来自已验收子图。<a href='" + escape(projection["replay_file"], quote=True)
        + "'>查看逐轮关键路径、节点前驱和独立计时</a>。</p>",
        "<table><thead><tr><th>工况</th><th>轮次</th><th>原墙钟（秒）</th>"
        "<th>必要基本运算下界</th><th>TFLOPS/卡下界</th><th>参考时长下界（秒）</th>"
        "<th>效率下界（%）</th><th>关键路径节点数</th></tr></thead><tbody>",
    ]
    for role, rounds in projection["arms"].items():
        for row in rounds:
            values = [role, row["repetition"], row["original_wall_seconds"], row["arithmetic_basic_lower"],
                      row["measured_tflops_per_card_lower"], row["projected_seconds_lower"],
                      row["efficiency_percent_lower"], row["critical_path_nodes"]]
            rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in values) + "</tr>")
    rows.append("</tbody></table>")
    return rows

def main(argv):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--calibration-archive", type=Path, required=True)
    parser.add_argument("--baseline-sdk", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--component-projection", type=Path)
    parser.add_argument("--original018-prefill-proof", type=Path)
    parser.add_argument("--target-fia-cache", type=Path)
    parser.add_argument("--target-fia-cache-sha256")
    parser.add_argument("--first-draft-cache", type=Path)
    parser.add_argument("--first-draft-cache-sha256")
    parser.add_argument("--ffn-branch-proof", type=Path)
    parser.add_argument("--original018-request-order", type=Path)
    parser.add_argument("--retained-query-work-proof", type=Path)
    parser.add_argument("--r11-first-prefill-proof", type=Path)
    parser.add_argument("--original018-target-kv-proof", type=Path)
    parser.add_argument("--r11-target-kv-proof", type=Path)
    parser.add_argument("--r11-prefill-tp2-cache", type=Path)
    parser.add_argument("--r11-prefill-tp2-cache-sha256")
    parser.add_argument("--r11-prefill-kv-cache", type=Path)
    parser.add_argument("--r11-prefill-kv-cache-sha256")
    parser.add_argument("--gdn-arithmetic-proof", type=Path)
    parser.add_argument("--r11-cold-gdn-cache", type=Path)
    parser.add_argument("--r11-cold-gdn-cache-sha256")
    parser.add_argument("--b0-cold-kv-cache", type=Path)
    parser.add_argument("--b0-cold-kv-cache-sha256")
    parser.add_argument("--conservative-reference-evidence", type=Path,
                        help="Verify the historical clock panels and qualify the original six-round reference lower bound")
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args(argv))
