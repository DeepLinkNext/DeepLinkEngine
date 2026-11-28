"""Source-bound arithmetic, canonical calibration, independent graph and pair.

All model services must have exited before these providers run. A source
continuation may supply previously sealed unchanged model/calibration stages;
the caller admits that continuation before passing its directories here.
"""

import json
import argparse
import contextlib
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from child_runner import run_stage
from dataset_contract import save, sha

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "third-party-test-kit"
sys.path.insert(0, str(KIT))

from audit_dense_lower_graph import audit as audit_graph
from canonical_protocol import protocol, locations
from canonical_results import audit as audit_calibration
from dense_lower_graph import compile_graph
from qualify_dense_source import qualify as qualify_dense
from qualify_necessary_work import qualify as qualify_work, sealed_read
from reference_pair_statistics import arm as arm_statistics, pair as pair_statistics, aggregate
from reference_capture_profile import repetitions, profile as capture_profile, validate_native_plan


def release_owned_model_artifact_cache(folder, model_source=None, http_source=None):
    """Discard only verified, immutable completed-run data pages before NPU calibration."""
    source = Path(model_source or folder).resolve()
    rows = []
    roots = [*native_sources(source).values(), Path(http_source) if http_source else source / "http-current-work"]
    before = Path("/sys/fs/cgroup/memory.current")
    initial = int(before.read_text()) if before.exists() else None
    for root in roots:
        receipt = sealed_read(root, "receipt.json")
        closed_recovery = False
        if root == roots[-1] and receipt.get("completed") is False and receipt.get("quiescent") is True:
            independent = Path(folder) / "http-independent-audit"
            recovered = sealed_read(independent, "receipt.json")
            provenance = sealed_read(independent, "source-inputs.json")
            closed_recovery = (recovered.get("completed") is True
                and recovered.get("original_completed") is False
                and recovered.get("original_failure") == receipt.get("failure")
                and recovered.get("all_request_closures_contained") is True
                and Path(recovered["source"]).resolve() == root.resolve()
                and Path(provenance["source"]).resolve() == root.resolve()
                and provenance["source_checksums_sha256"] == sha(root / "checksums.json")
                and provenance["original_receipt_sha256"] == sha(root / "receipt.json"))
        if (receipt["completed"] is not True and not closed_recovery) or receipt["quiescent"] is not True:
            raise ValueError("Artifact cache release requires completed quiescent owned model stages")
        seal = json.loads((root / "checksums.json").read_bytes())
        for name, expected in seal.items():
            path = root / name
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Owned artifact cache path escaped its sealed run")
            if path.stat().st_size < 2**20:
                continue
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                identity = os.fstat(stream.fileno())
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("Owned artifact changed before cache release: " + name)
                current = os.fstat(stream.fileno())
                if (identity.st_dev, identity.st_ino, identity.st_size, identity.st_mtime_ns) != (
                        current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
                    raise ValueError("Owned artifact identity changed during cache admission")
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            rows.append(dict(path=str(path), sha256=expected, bytes=identity.st_size,
                device=identity.st_dev, inode=identity.st_ino, mtime_ns=identity.st_mtime_ns))
    result = dict(operation="Read-only POSIX_FADV_DONTNEED on hash-verified completed model-run artifacts",
        files=rows, file_content_written=False, global_drop_caches=False,
        model_services_quiescent=True, before_current=initial,
        after_current=int(before.read_text()) if before.exists() else None)
    save(Path(folder) / "owned-artifact-cache-release.json", result)
    return result


def native_sources(folder):
    return dict(B0=Path(folder) / "native-baseline-timing", B1=Path(folder) / "native-current-timing")


def repetition_prefix(role, repetition):
    return role if repetition == 1 else f"{role}-r{repetition:03d}"


def necessary(plan, folder, model_source=None, http_source=None):
    from audit_http_reference import run as audit_http

    folder = Path(folder)
    origin = Path(model_source or folder)
    audit_directory = folder / "http-independent-audit"
    audit_http(Path(http_source) if http_source else origin / "http-current-work", plan["deployment"], audit_directory)
    result = {}
    for role, source in native_sources(origin).items():
        keywords = dict(baseline_deployment=plan["baseline_deployment"]) if role == "B0" else {}
        receipt = sealed_read(source, "receipt.json")
        if receipt["formal_repetitions"] != len(repetitions(plan)):
            raise ValueError("All predeclared native repetitions are required")
        native_plan = sealed_read(source, "service/native-plan.json")
        capture = validate_native_plan(native_plan)
        if capture != (plan.get("capture_profile") or capture_profile("full-answer-v1")):
            raise ValueError("Native budget and automatic reference capture profile differ")
        rows = []
        for repetition in repetitions(plan):
            work = qualify_work(plan["deployment"], source, audit_directory, repetition=repetition, **keywords)
            dense = qualify_dense(plan["deployment"], source, repetition=repetition, **keywords)
            prefix = repetition_prefix(role, repetition)
            save(folder / (prefix + "-necessary-work.json"), work)
            save(folder / (prefix + "-dense-source.json"), dense)
            rows.append(dict(repetition=repetition, prefix=prefix,
                necessary_work_sha256=sha(folder / (prefix + "-necessary-work.json")),
                dense_source_sha256=sha(folder / (prefix + "-dense-source.json")),
                geometry_sha256=work["geometry_sha256"], F_interval=work["F_interval"]))
        result[role] = dict(rows[0], repetitions=rows)
    save(folder / "necessary-work-audit.json", dict(completed=True, arms=result,
        model_source=str(origin.resolve()), model_inference_executed=False))
    return result


def dense_request(deployment):
    deployment = Path(deployment).resolve()
    names = ["runtime/engine-core/vllm/model_executor/models/qwen3_next.py",
        "runtime/engine-core/vllm/model_executor/models/qwen3_5.py",
        "runtime/engine-core/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "runtime/engine-core/vllm/model_executor/layers/linear.py",
        "runtime/ascend/vllm_ascend/ops/linear.py"]
    cases = [dict(case=index, m=m, n=n, k=2048) for index, (n, m) in enumerate(
        (n, m) for n in (32, 2048, 4608, 6144) for m in range(1, 109))]
    return dict(schema="current-canonical-sampler-request/v1", family="dense", cases=cases,
        source_bindings={str(deployment / name): sha(deployment / name) for name in names},
        selection="Prospective full possible TP2 pure-decode canonical row domain1..108 from E36 and fixedMTP2; B0E24 is a subset. Independent of observed T or efficiency. No old timing reused.")


def calibration(plan, folder, calibration_source=None):
    folder = Path(folder)
    request = dense_request(plan["deployment"])
    save(folder / "canonical-dense-request.json", request)
    source = Path(calibration_source) if calibration_source else folder / "canonical-dense"
    if calibration_source is None:
        command = [sys.executable, "-B", str(ROOT / "scripts/run_canonical_reference.py"),
            "--deployment", plan["deployment"], "--request", str(folder / "canonical-dense-request.json"),
            "--out", str(source), "--physical-devices", ",".join(map(str, plan["physical_cards"])),
            "--devices", plan["logical_devices"], "--lease-timeout", "86400"]
        with (folder / "canonical-dense.log").open("x") as log:
            process = run_stage(command, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise ValueError("Exact canonical dense calibration failed; raw artifacts retained")
    receipt = sealed_read(source, "receipt.json")
    if receipt["complete"] is not True or receipt["quiescent"] is not True or receipt.get("failure"):
        raise ValueError("Only a complete quiescent canonical calibration can be used")
    mapping = sealed_read(source, "mapping.json")
    if (mapping["physical_cards"] != plan["physical_cards"]
            or mapping["logical_cards"] != list(map(int, plan["logical_devices"].split(",")))):
        raise ValueError("Canonical calibration belongs to another physical/logical device pair")
    if sealed_read(source, "request.json") != request:
        raise ValueError("Canonical calibration source/shape request differs")
    preparation = sealed_read(source, "sampler/preparation.json")
    # Re-audit all original independent samples and numerical checks.
    row, _ = protocol("dense")
    raw_plan = sealed_read(source, "sampler/" + row["original_plan"])
    validate_canonical_program(source, preparation, row, raw_plan)
    dependencies = sealed_read(source, "dependencies.json")
    deployment = Path(plan["deployment"])
    profile = json.loads((deployment / "deployment.json").read_bytes())
    contract = json.loads((deployment / "mods/microflow/hust_prefill_microflow/school028/host-contract.json").read_bytes())
    if (dependencies["python"] != profile["python"]
            or dependencies["metadata"] != contract["versions"]
            or dependencies["native_packages_imported"] is not False):
        raise ValueError("Canonical reference dependency domain differs from the frozen SDK")
    if raw_plan["cases"] != request["cases"] or raw_plan["current_source_sha256"] != request["source_bindings"]:
        raise ValueError("Canonical sampler plan differs from current source shapes")
    for key, value in row["controls"].items():
        if raw_plan[key] != value:
            raise ValueError("Original canonical numerical/statistical control changed: " + key)
    # Audit raw result identities against the containing stage seal first.
    for rank in (0, 1):
        sealed_read(source, "sampler/" + preparation["output"] + f"/rank-{rank}.json")
    result = audit_calibration(source / "sampler" / preparation["output"],
        source / "sampler" / row["original_plan"], plan["physical_cards"])
    if result != sealed_read(source, "component-timings.json"):
        raise ValueError("Canonical component medians do not independently reproduce")
    save(folder / "qualified-dense-calibration.json", dict(source=str(source.resolve()),
        checksums_sha256=sha(source / "checksums.json"), reused=calibration_source is not None,
        request_sha256=sha(folder / "canonical-dense-request.json"), result=result))
    return result


def validate_canonical_program(source, preparation, row, raw_plan):
    """A sealed result cannot substitute a changed program or relaxed protocol."""
    source = Path(source).resolve()
    name = "sampler/" + row["sampler"]
    path = source / name
    seal = json.loads((source / "checksums.json").read_bytes())
    if (path.is_symlink() or not path.resolve().is_relative_to(source)
            or seal.get(name) != row["sampler_sha256"]
            or sha(path) != row["sampler_sha256"]
            or preparation["sampler"] != row["sampler"]
            or preparation["sampler_sha256"] != row["sampler_sha256"]
            or preparation["plan"] != row["original_plan"]
            or preparation["plan_sha256"] != sha(source / "sampler" / row["original_plan"])
            or raw_plan["original_sampler_sha256"] != row["sampler_sha256"]
            or raw_plan["original_protocol_plan_sha256"] != row["original_plan_sha256"]
            or any(raw_plan[k] is not False for k in (
                "numerical_or_statistical_thresholds_relaxed", "historical_counts_or_timings_reused"))
            or preparation["output"] != locations(path.read_bytes())["OUT"]):
        raise ValueError("Original canonical sampler bytes or protocol identity changed")


def graphs(plan, folder):
    folder = Path(folder)
    if (folder / "qualified-full-reference-statistics.json").exists():
        from full_reference_reuse import audit

        necessary = json.loads((folder / "necessary-work-audit.json").read_bytes())
        return audit(plan, folder, native_sources(necessary["model_source"]))
    calibrated = json.loads((folder / "qualified-dense-calibration.json").read_bytes())
    request = json.loads((folder / "canonical-dense-request.json").read_bytes())
    necessary_audit = json.loads((folder / "necessary-work-audit.json").read_bytes())
    result = {}
    for role in ("B0", "B1"):
        rows = []
        proofs = necessary_audit["arms"][role].get("repetitions", [necessary_audit["arms"][role]])
        if len(proofs) != len(repetitions(plan)):
            raise ValueError("Reference graphs require every native repetition")
        for repetition, proof in zip(repetitions(plan), proofs, strict=True):
            prefix = repetition_prefix(role, repetition)
            if sha(folder / (prefix + "-dense-source.json")) != proof["dense_source_sha256"]:
                raise ValueError("Retained source-domain proof changed after necessary-work audit")
            source = json.loads((folder / (prefix + "-dense-source.json")).read_bytes())
            if source["source_runtime_applicability_certified"] is not True:
                raise ValueError("Complete retained graph source-domain proof required")
            model = SimpleNamespace(**source["model_text_config"])
            graph = compile_graph(source["tail"], model, request["cases"], calibrated["result"])
            audit = audit_graph(graph, source["tail"], model, request["cases"], calibrated["result"])
            if audit["graph_integrity_passed"] is not True:
                raise ValueError("Independent retained reference graph audit failed")
            graph.update(source_runtime_applicability_certified=True,
                source_proof_sha256=sha(folder / (prefix + "-dense-source.json")),
                reference_seconds_lower=audit["reference_model_lower_seconds"])
            save(folder / (prefix + "-reference-lower-graph.json"), graph)
            save(folder / (prefix + "-reference-graph-audit.json"), audit)
            rows.append(dict(repetition=repetition, prefix=prefix,
                reference_seconds_lower=audit["reference_model_lower_seconds"],
                graph_sha256=sha(folder / (prefix + "-reference-lower-graph.json")),
                independent_audit_sha256=sha(folder / (prefix + "-reference-graph-audit.json"))))
        result[role] = dict(rows[0], repetitions=rows)
    save(folder / "reference-graph-audits.json", dict(completed=True, arms=result,
        qualified_calibration_sha256=sha(folder / "qualified-dense-calibration.json"),
        source_embedding_and_all_retained_leaf_timings_qualified=True))
    return result


def validate_native_timing(raw, timing, audit, work, *, repetition=1):
    if (work["necessary_work_certified"] is not True
            or work["source_runtime_applicability_certified"] is not True
            or any(raw.get(key) is not False for key in ("http_inside_timing", "tokenizer_inside_timing",
                "per_step_engine_internals_read", "per_step_disk_writes"))
            or raw["public_step_output_only"] is not True):
        raise ValueError("Complete own-work/clean native timing correspondence required")
    for key in ("start_fences", "end_fences"):
        fences = raw[key]
        if (len(fences) != 2 or {r["rank"] for r in fences} != {0, 1}
                or any(r["device"] != r["rank"] for r in fences)):
            raise ValueError("Both source-matched device fences required")
    if (len(audit) != 2 or {r["rank"] for r in audit} != {0, 1}
            or any(r["native_per_step_observer"] is not False for r in audit)):
        raise ValueError("Both clean native worker audits required")
    if (any(type(raw[k]) is not int for k in ("start_ns", "end_ns"))
            or raw["end_ns"] <= raw["start_ns"]
            or type(timing["repetition"]) is not int or timing["repetition"] != repetition
            or len(timing["boundary"]) != 2 or {r["rank"] for r in timing["boundary"]} != {0, 1}
            or timing["ordinary_public_outputs_retained"] is not True
            or any(timing[k] != raw[k] for k in ("start_ns", "end_ns", "T_seconds"))
            or abs(raw["T_seconds"] - (raw["end_ns"] - raw["start_ns"]) / 1e9) > 1e-9
            or len(raw["request_ids"]) != work["requests"]
            or any(r["sealed"] is not True or r["failure"] is not None for r in timing["boundary"])):
        raise ValueError("Native repetition, clean boundary, raw clock or population differs")
    return raw["T_seconds"]


def matched_pair(plan, folder, model_source=None):
    folder = Path(folder)
    if (folder / "qualified-full-reference-statistics.json").exists():
        from full_reference_reuse import matched

        return matched(plan, folder, native_sources(model_source or folder), validate_native_timing)
    summaries, pair_identity = {}, {}
    graphs_audited = json.loads((folder / "reference-graph-audits.json").read_bytes())
    necessary_audit = json.loads((folder / "necessary-work-audit.json").read_bytes())
    if (graphs_audited["completed"] is not True or necessary_audit["completed"] is not True
            or sha(folder / "qualified-dense-calibration.json") != graphs_audited["qualified_calibration_sha256"]):
        raise ValueError("Independent source/shape/dependency audit is incomplete")
    capture = plan.get("capture_profile") or capture_profile("full-answer-v1")
    freeze = None
    if capture["id"] == "submission-v2":
        from reference_statistics import canonical_sha

        freeze = json.loads((folder / "reference-calibration-freeze.json").read_bytes())
        calibrated = json.loads((folder / "qualified-dense-calibration.json").read_bytes())
        if (freeze["calibration_result_sha256"] != canonical_sha(calibrated["result"])
                or type(freeze["frozen_monotonic_ns"]) is not int or freeze["frozen_monotonic_ns"] < 0
                or freeze["target_times_or_efficiency_used"] is not False):
            raise ValueError("Independent reference statistics were not frozen before target measurement")
    for role, native in native_sources(model_source or folder).items():
        receipt = sealed_read(native, "receipt.json")
        native_plan = sealed_read(native, "service/native-plan.json")
        native_timings = sealed_read(native, "service/client/timings.json")
        if (validate_native_plan(native_plan) != capture
                or receipt["formal_repetitions"] != len(repetitions(plan))
                or [r["repetition"] for r in native_timings["repetitions"]] != list(repetitions(plan))):
            raise ValueError("Matched statistics require all predeclared native repetitions")
        weights = sealed_read(native, "model-weights.json")
        work_proofs = necessary_audit["arms"][role].get("repetitions", [necessary_audit["arms"][role]])
        graph_proofs = graphs_audited["arms"][role].get("repetitions", [graphs_audited["arms"][role]])
        if len(work_proofs) != len(repetitions(plan)) or len(graph_proofs) != len(repetitions(plan)):
            raise ValueError("Every repetition needs its own arithmetic and graph proofs")
        rows, previous_end, request_ids = [], None, set()
        for repetition, work_proof, graph_proof in zip(repetitions(plan), work_proofs, graph_proofs, strict=True):
            prefix = repetition_prefix(role, repetition)
            geometry = sealed_read(native, f"service/client/formal-{repetition}/public-output-geometry.json")
            identity = dict(weights=weights, input_hashes=[r["input_ids_sha256"] for r in geometry["requests"]])
            if pair_identity and identity != next(iter(pair_identity.values())):
                raise ValueError("Native B0/B1 weights or ordered prompt population differ")
            pair_identity[role, repetition] = identity
            work = json.loads((folder / (prefix + "-necessary-work.json")).read_bytes())
            graph = json.loads((folder / (prefix + "-reference-lower-graph.json")).read_bytes())
            graph_audit_path = folder / (prefix + "-reference-graph-audit.json")
            graph_audit = json.loads(graph_audit_path.read_bytes())
            if (sha(folder / (prefix + "-necessary-work.json")) != work_proof["necessary_work_sha256"]
                    or work["native_checksums_sha256"] != sha(native / "checksums.json")
                    or sha(folder / (prefix + "-reference-lower-graph.json")) != graph_proof["graph_sha256"]
                    or graph["geometry_sha256"] != work["geometry_sha256"]
                    or sha(folder / (prefix + "-reference-graph-audit.json")) != graph_proof["independent_audit_sha256"]
                    or graph_audit["graph_integrity_passed"] is not True
                    or graph_audit["reference_model_lower_seconds"] != graph["reference_seconds_lower"]
                    or graph["reference_seconds_lower"] != graph_proof["reference_seconds_lower"]
                    or graph["source_runtime_applicability_certified"] is not True
                    or sha(folder / (prefix + "-dense-source.json")) != graph["source_proof_sha256"]
                    or graph["source_proof_sha256"] != work_proof["dense_source_sha256"]):
                raise ValueError("Necessary arithmetic, graph and native source populations differ")
            raw = sealed_read(native, f"service/client/formal-{repetition}/raw-public-output-frames.json")
            timing = sealed_read(native, f"service/client/formal-{repetition}-timing.json")
            audit = sealed_read(native, f"service/client/formal-{repetition}-pre-audit.json")
            seconds = validate_native_timing(raw, timing, audit, work, repetition=repetition)
            if freeze and raw["start_ns"] < freeze["frozen_monotonic_ns"]:
                raise ValueError("Formal target clock predates its frozen independent calibration")
            if (timing != native_timings["repetitions"][repetition - 1]
                    or previous_end is not None and raw["start_ns"] < previous_end
                    or len(set(raw["request_ids"])) != work["requests"]
                    or request_ids.intersection(raw["request_ids"])):
                raise ValueError("Reused, overlapping or mismatched formal repetition")
            previous_end = raw["end_ns"]
            request_ids.update(raw["request_ids"])
            row = arm_statistics(work["F_interval"], seconds, graph["reference_seconds_lower"])
            row.update(repetition=repetition, geometry_sha256=work["geometry_sha256"],
                native_source_seal_sha256=sha(native / "checksums.json"),
                timing_sha256=sha(native / f"service/client/formal-{repetition}-timing.json"),
                work_sha256=work_proof["necessary_work_sha256"], graph_sha256=graph_proof["graph_sha256"])
            rows.append(row)
        summaries[role] = aggregate(rows, expected_repetitions=capture["formal_repetitions"])
        summaries[role].update(full_model_certificate=False, source_and_timing_qualified=True,
            native_checksums_sha256=sha(native / "checksums.json"))
    result = pair_statistics(summaries["B0"], summaries["B1"])
    # This provider only compiles the dense decode tail. Complete F and a clean
    # clock do not turn that projection into the agreed complete reference.
    # Keep its useful source/timing diagnostics, but do not emit a certificate
    # or a milestone verdict, regardless of the numeric value of the bound.
    for summary in summaries.values():
        summary["partial_dense_diagnostic"] = {
            key: summary.pop(key) for key in (
                "reference_seconds_lower", "reference_seconds_exact",
                "reference_tflops_per_card_upper", "efficiency_lower_percent",
                "efficiency_exact_percent", "statistic")}
        for repetition in summary["repetitions"]:
            repetition["partial_dense_diagnostic"] = {
                key: repetition.pop(key) for key in (
                    "reference_seconds_lower", "reference_seconds_exact",
                    "reference_tflops_per_card_upper", "efficiency_lower_percent",
                    "efficiency_exact_percent", "statistic")}
    result.update(status="incomplete", full_model_certificate=False,
        certificate_kind="partial_dense_projection_diagnostic",
        capture_profile=capture, native_formal_repetitions_per_arm=capture["formal_repetitions"],
        clean_native_clock_source_qualified=True, independent_canonical_reference_lower_bound_qualified=True,
        efficiency_exact_percent=None, current_efficiency=None,
        efficiency_lower_observed_percent=None, efficiency_lower_observed_percent_decimal=None,
        milestone_met=None, final_numeric_met=None, milestone_decision=None, final_decision=None,
        blockers=["完整参考未补齐：当前提供器只有部分 dense 末段；必要的 prefill、末层、词表、MTP、MoE、GDN、标量、搬运、通信、同步和主机处理不能由此替代。"])
    save(folder / "matched-reference-pair.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("necessary", "calibration", "graphs", "matched"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--model-source", type=Path)
    parser.add_argument("--http-source", type=Path)
    parser.add_argument("--calibration-source", type=Path)
    args = parser.parse_args()
    if sha(args.plan) != args.sha256:
        raise ValueError("Final reference provider plan differs from the automatic admission")
    plan = json.loads(args.plan.read_bytes())
    for name, expected in plan["files"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Final reference provider source changed: " + name)
    full_reuse = plan.get("capture_profile", {}).get("id") == "submission-v2"
    if args.stage == "calibration" and args.calibration_source is None and not full_reuse:
        result = calibration(plan, args.folder, args.calibration_source)
    else:
        # Large CPU raw-record reads must not compete with another model's
        # cgroup/driver allocation. Release these leases before calibration,
        # whose own child independently owns its device/container guards.
        from resource_guard import acquire, release

        os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = ",".join(map(str, plan["physical_cards"]))
        guard = args.folder / ("cpu-guard-" + args.stage)
        guard.mkdir()
        with contextlib.ExitStack() as cleanup:
            leases = acquire(guard, 86400)
            cleanup.callback(release, leases)
            if args.stage == "calibration":
                if full_reuse:
                    from full_reference_reuse import prepare

                    result = prepare(args.folder)
                else:
                    result = calibration(plan, args.folder, args.calibration_source)
            elif args.stage == "necessary":
                result = necessary(plan, args.folder, args.model_source, args.http_source)
                release_owned_model_artifact_cache(args.folder, args.model_source, args.http_source)
            elif args.stage == "graphs":
                result = graphs(plan, args.folder)
            else:
                result = matched_pair(plan, args.folder, args.model_source)
    save(args.folder / ("final-stage-" + args.stage + ".json"), dict(completed=True, result=result))


if __name__ == "__main__":
    main()
