"""Recompute the original complete reference denominator, then match a fresh F/T.

The original reference has its own F and Tref. Different current necessary work
never cancels against that F. Target endpoint elapsed time does not select,
fit, scale, or otherwise modify the frozen reference denominator.
"""

from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import shutil

from dataset_contract import save, sha
from reference_statistics import statistics

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "protocols/m01-full-reference-reuse-domain-v1.json"
DOMAIN_SHA256 = "b7a5fbf468e0b93add821f545efc015a35860de616c904e02a0139ec3599ae9d"
KIND = "qualified_full_reference_denominator_reuse/v1"
REPORT = "reports/m01-qualified-pair-20261103"


def domain():
    if sha(DOMAIN) != DOMAIN_SHA256:
        raise ValueError("Original full-reference applicability protocol changed")
    return json.loads(DOMAIN.read_bytes())


def prepare(folder):
    """Resolve the already shipped, independently verified original evidence."""
    from project_evidence import ensure, load_module, verify

    folder = Path(folder)
    source = ensure()
    identity = verify(source)
    with localcontext() as context:
        context.prec = 40
        measured = load_module(source, "recompute_m01_qualified_pair").calculate()
    if measured != json.loads((source / REPORT / "summary.json").read_bytes()):
        raise ValueError("Original complete reference does not reproduce")
    if measured["reference_kind"] != domain()["reference_kind"]:
        raise ValueError("Only the qualified original complete-reference method is reusable")
    protocol = json.loads((source / REPORT / "evidence/qualification-protocol.json").read_bytes())
    expected = domain()
    original_input = protocol["input"]
    if any(original_input[key] != value for key, value in {
        "token_sha256": expected["prompt_tokens_sha256"], "requests": expected["requests"],
        "input_tokens": expected["input_tokens"], "max_output_tokens": expected["max_tokens"],
        **expected["sampling"],
    }.items()):
        raise ValueError("Original reference workload differs from its qualified reuse domain")
    result = statistics(dict(m01=measured, benchmark_commit=identity["commit"],
        manifest_sha256=sha(source / "source-readback.json")))
    # Archive the pinned verifier and all of its evidence, not just a scalar or
    # an old percentage. Its own manifest is validated again after the copy.
    destination = folder / "original-full-reference"
    shutil.copytree(source, destination)
    verify(destination)
    result.update(reference_kind=measured["reference_kind"],
        applicability_protocol_sha256=DOMAIN_SHA256,
        source_root=str(destination.resolve()),
        original_method_frozen_unix=protocol["frozen_unix"],
        original_efficiency_used=False, target_times_used=False,
        source_sha256={name: sha(destination / name) for name in (
            "source-readback.json", "scripts/recompute_m01_qualified_pair.py",
            REPORT + "/summary.json", REPORT + "/evidence/evidence-manifest.json",
            REPORT + "/evidence/qualification-protocol.json")})
    save(folder / "qualified-full-reference-statistics.json", result)
    return result


def read_statistics(folder):
    from project_evidence import verify

    folder = Path(folder)
    result = json.loads((folder / "qualified-full-reference-statistics.json").read_bytes())
    if (result["applicability_protocol_sha256"] != DOMAIN_SHA256
            or result["reference_kind"] != domain()["reference_kind"]
            or result["original_efficiency_used"] is not False or result["target_times_used"] is not False):
        raise ValueError("Full-reference identity or independence differs")
    source = Path(result["source_root"])
    if source.resolve() != (folder / "original-full-reference").resolve():
        raise ValueError("Original full-reference evidence escaped this run")
    identity = verify(source)
    original_protocol = json.loads((source / REPORT / "evidence/qualification-protocol.json").read_bytes())
    if (result["benchmark_commit"] != identity["commit"]
            or result["original_method_frozen_unix"] != original_protocol["frozen_unix"]):
        raise ValueError("Original reference publication or method freeze differs")
    for name, pin in result["source_sha256"].items():
        if sha(source / name) != pin:
            raise ValueError("Original full-reference source changed: " + name)
    reproduced = statistics(dict(m01=json.loads((source / REPORT / "summary.json").read_bytes()),
        benchmark_commit=result["benchmark_commit"], manifest_sha256=sha(source / "source-readback.json")))
    if any(result.get(key) != value for key, value in reproduced.items()):
        raise ValueError("Original reference F, Tref or per-card statistics changed")
    return result


def qualify_mapping(mapping, accelerator):
    physical, logical = mapping["physical_cards"], mapping["logical_cards"]
    if (len(physical) != 2 or len(logical) != 2
            or len(set(physical)) != 2 or len(set(logical)) != 2
            or any(type(value) is not int or value < 0 for value in physical + logical)):
        raise ValueError("Original full-reference two-card 910B2 mapping differs")
    selected = []
    for line in mapping["npu_smi_mapping"].splitlines():
        words = line.split()
        if (len(words) >= 4 and words[0].isdigit() and words[1] == "0"
                and int(words[0]) in physical):
            observed = "".join(words[3:]).removeprefix("Ascend")
            expected = accelerator.replace(" ", "").removeprefix("Ascend")
            if not words[2].isdigit() or observed != expected:
                raise ValueError("Original full-reference selected accelerator mapping differs")
            selected.append((int(words[0]), int(words[2])))
    if len(selected) != 2 or set(selected) != set(zip(physical, logical)):
        raise ValueError("Original full-reference two-card 910B2 mapping differs")


def qualify_native(native, role, capture):
    """Exact prompt, weight, hardware, precision, output and native scope gates."""
    from qualify_necessary_work import sealed_read
    from reference_capture_profile import profile, validate_native_plan

    expected = domain()
    if capture != profile("submission-v2"):
        raise ValueError("Original out256/native reference cannot qualify another capture profile")
    native = Path(native)
    plan = sealed_read(native, "service/native-plan.json")
    if validate_native_plan(plan) != capture:
        raise ValueError("Native plan and full-reference capture profile differ")
    work = sealed_read(native, "service/token-workload.json")
    arguments = sealed_read(native, "service/client/effective-engine-args.json")
    mapping = sealed_read(native, "device-mapping.json")
    if (plan["max_tokens"] != expected["max_tokens"]
            or plan["formal_repetitions"] != expected["formal_repetitions"]
            or len(work["request_token_ids"]) != expected["requests"]
            or sum(map(len, work["request_token_ids"])) != expected["input_tokens"]):
        raise ValueError("Original full-reference request population or budget differs")
    prompt_sha = hashlib.sha256((json.dumps(work["request_token_ids"],
        separators=(",", ":")) + "\n").encode()).hexdigest()
    if prompt_sha != expected["prompt_tokens_sha256"]:
        raise ValueError("Original full-reference ordered prompt token identities differ")
    if any(plan["sampling"].get(key) != value for key, value in expected["sampling"].items()):
        raise ValueError("Original full-reference sampling differs")
    weights = sealed_read(native, "model-weights.json")
    if {name: {key: item[key] for key in ("bytes", "sha256")}
            for name, item in weights.items()} != expected["weights"]:
        raise ValueError("Original full-reference checkpoint weights differ")
    if sha(Path(plan["model"]) / "config.json") != expected["model_config_sha256"]:
        raise ValueError("Original full-reference model configuration differs")
    argument_domain = dict(dtype=expected["dtype"], tensor_parallel_size=expected["cards"],
        max_num_seqs=expected["max_num_seqs"][role],
        max_num_batched_tokens=expected["max_num_batched_tokens"],
        speculative_config=expected["speculative_config"], quantization=None,
        async_scheduling=True)
    if any(type(arguments.get(key)) is not type(value) or arguments.get(key) != value
            for key, value in argument_domain.items()):
        raise ValueError("Original full-reference precision, capacity or execution domain differs")
    qualify_mapping(mapping, expected["accelerator"])
    return dict(role=role, scope="same-fixed200-out256-native-BF16-TP2-910B2",
        prompt_tokens_sha256=prompt_sha, checkpoint_shards=14,
        original_reference_F_retained=True, current_F_is_independently_recomputed=True,
        different_F_may_cancel=False, native_checksums_sha256=sha(native / "checksums.json"),
        applicability_protocol_sha256=DOMAIN_SHA256)


def arm_statistics(rows, reference):
    """Compute a new numerator divided by the original per-card denominator."""
    if ([row.get("repetition") for row in rows] != list(range(1, len(rows) + 1))
            or any(len(row["F_interval"]) != 2
                or any(type(value) is not int or value <= 0 for value in row["F_interval"])
                or row["F_interval"][0] > row["F_interval"][1]
                or not Decimal(row["T_seconds"]).is_finite() or Decimal(row["T_seconds"]) <= 0
                for row in rows)):
        raise ValueError("Every predeclared current work/time repetition must be valid")
    with localcontext() as context:
        context.prec = 40
        work = [sum(row["F_interval"][index] for row in rows) for index in (0, 1)]
        elapsed = sum(Decimal(row["T_seconds"]) for row in rows)
        reference_work = reference["effective_flops_interval"]
        reference_time = Decimal(reference["sum_reference_seconds_lower"])
        if (not rows or len(rows) != len(reference["reference_repetitions"])
                or work[0] <= 0 or work[1] < work[0] or elapsed <= 0
                or reference_work[0] <= 0 or reference_work[1] < reference_work[0]
                or not elapsed.is_finite() or not reference_time.is_finite() or reference_time <= 0):
            raise ValueError("Invalid current work, timing or original reference population")
        achieved = [Decimal(value) / (2 * elapsed * 10**12) for value in work]
        denominator = Decimal(reference_work[1]) / (2 * reference_time * 10**12)
        if denominator != Decimal(reference["reference_tflops_per_card_upper"]):
            raise ValueError("Original per-card reference denominator does not reproduce")
        eta = 100 * achieved[0] / denominator
    return dict(F_interval=work, T_seconds=str(elapsed), formal_repetitions=len(rows),
        achieved_tflops_per_card_interval=list(map(str, achieved)),
        reference_F_interval=reference_work, reference_seconds_lower=str(reference_time),
        reference_seconds_exact=None, reference_tflops_per_card_upper=str(denominator),
        efficiency_lower_percent=str(eta), efficiency_exact_percent=None,
        original_reference_repetitions=reference["reference_repetitions"],
        repetitions=rows, shared_work_cancellation_used=False,
        hardware_peak_utilization_percent=None, source_and_timing_qualified=True,
        full_model_certificate=True,
        statistic="100 * C_current_lower / C_reference_upper; original F_reference and T_reference retained; sum all current repetitions before division")


def audit(plan, folder, native_sources):
    """Reuse the independently rechecked complete original graph, with domain gates."""
    from reference_capture_profile import profile

    reference = read_statistics(folder)
    capture = plan.get("capture_profile") or profile("full-answer-v1")
    applicability = {role: qualify_native(source, role, capture)
        for role, source in native_sources.items()}
    result = dict(completed=True, reference_kind=KIND, arms=applicability,
        qualified_full_reference_statistics_sha256=sha(Path(folder) / "qualified-full-reference-statistics.json"),
        original_graph_recomputed=True, source_and_workload_applicability_qualified=True,
        original_method_frozen_unix=reference["original_method_frozen_unix"],
        current_graph_time_not_substituted=True, partial_dense_used=False)
    save(Path(folder) / "reference-graph-audits.json", result)
    return result


def matched(plan, folder, native_sources, validate_timing):
    from qualify_necessary_work import sealed_read
    from reference_capture_profile import profile
    from reference_pair_statistics import pair

    folder = Path(folder)
    reference = read_statistics(folder)
    capture = plan.get("capture_profile") or profile("full-answer-v1")
    graph_audit = json.loads((folder / "reference-graph-audits.json").read_bytes())
    necessary = json.loads((folder / "necessary-work-audit.json").read_bytes())
    if (graph_audit["completed"] is not True or necessary["completed"] is not True
            or graph_audit["reference_kind"] != KIND
            or graph_audit["qualified_full_reference_statistics_sha256"]
                != sha(folder / "qualified-full-reference-statistics.json")):
        raise ValueError("Qualified full-reference applicability or necessary-work audit missing")
    summaries = {}
    for role, source in native_sources.items():
        source = Path(source)
        if qualify_native(source, role, capture) != graph_audit["arms"][role]:
            raise ValueError("Full-reference source applicability changed")
        receipt = sealed_read(source, "receipt.json")
        if (receipt["completed"] is not True or receipt["quiescent"] is not True
                or receipt.get("failure") or receipt["formal_repetitions"] != 3
                or reference["original_method_frozen_unix"] >= receipt["started"]):
            raise ValueError("Clean native run must follow the frozen original independent reference")
        timings = sealed_read(source, "service/client/timings.json")["repetitions"]
        proofs = necessary["arms"][role]["repetitions"]
        if len(timings) != 3 or len(proofs) != 3:
            raise ValueError("Full-reference reuse requires all three current repetitions")
        rows, previous_end, identifiers = [], None, set()
        for index, proof in enumerate(proofs, 1):
            prefix = proof["prefix"]
            path = folder / (prefix + "-necessary-work.json")
            work = json.loads(path.read_bytes())
            if (sha(path) != proof["necessary_work_sha256"]
                    or work["native_checksums_sha256"] != sha(source / "checksums.json")):
                raise ValueError("Current numerator work or source changed")
            geometry = sealed_read(source, f"service/client/formal-{index}/public-output-geometry.json")
            if (len(geometry["requests"]) != 200
                    or any(row["output_tokens"] != 256 or row["finish_reason"] != "length"
                        for row in geometry["requests"])
                    or work["geometry_sha256"] != geometry["geometry_sha256"]):
                raise ValueError("Current completion population differs from fixed-output reference scope")
            raw = sealed_read(source, f"service/client/formal-{index}/raw-public-output-frames.json")
            timing = sealed_read(source, f"service/client/formal-{index}-timing.json")
            workers = sealed_read(source, f"service/client/formal-{index}-pre-audit.json")
            seconds = validate_timing(raw, timing, workers, work, repetition=index)
            if (timing != timings[index-1] or previous_end is not None and raw["start_ns"] < previous_end
                    or identifiers.intersection(raw["request_ids"])):
                raise ValueError("Duplicate, overlapping or changed native timing repetition")
            previous_end = raw["end_ns"]
            identifiers.update(raw["request_ids"])
            rows.append(dict(repetition=index, F_interval=work["F_interval"],
                T_seconds=str(seconds), geometry_sha256=work["geometry_sha256"],
                timing_sha256=sha(source / f"service/client/formal-{index}-timing.json"),
                work_sha256=sha(path), native_source_seal_sha256=sha(source / "checksums.json")))
        summaries[role] = arm_statistics(rows, reference["arms"][role])
    result = pair(summaries["B0"], summaries["B1"])
    result.update(status="measured", full_model_certificate=True, certificate_kind=KIND,
        capture_profile=capture, native_formal_repetitions_per_arm=3,
        clean_native_clock_source_qualified=True, full_reference_denominator_qualified=True,
        qualified_full_reference_statistics_sha256=sha(folder / "qualified-full-reference-statistics.json"),
        original_reference_work_and_time_retained=True, original_efficiency_used=False,
        historical_counts_or_times_reused=True, reference_reuse_scope="original qualified same-condition reference denominator only; all current F/T freshly audited",
        shared_work_cancellation_used=False, efficiency_exact_percent=None, current_efficiency=None)
    save(folder / "matched-reference-pair.json", result)
    return result
