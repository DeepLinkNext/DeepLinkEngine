"""Recompute the independently calibrated full-chain reference from raw records."""

from decimal import Decimal
import json
import re
import statistics


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_float=Decimal)


def integer(value, label, minimum=0):
    check(type(value) is int and value >= minimum, f"{label}: expected integer >= {minimum}")
    return value


def nonnegative(value, label):
    check(type(value) in (int, Decimal), f"{label}: expected numeric value")
    result = Decimal(value)
    check(result.is_finite() and result >= 0, f"{label}: expected finite nonnegative value")
    return result


def validate_method(method):
    check(method["estimator"] in ("critical_path", "whole_chain_median"), "Unknown reference estimator")
    check(isinstance(method["reference_id"], str) and method["reference_id"].strip(),
          "Freeze the independent reference implementation identity")
    check(re.fullmatch(r"[0-9a-f]{64}", method["workload_sha256"]), "Freeze the workload SHA-256")
    integer(method["min_calibration_samples"], "min_calibration_samples", 1)
    integer(method["min_validation_samples"], "min_validation_samples", 1)
    nonnegative(method["max_calibration_spread_ratio"], "max_calibration_spread_ratio")
    nonnegative(method["max_validation_error_ratio"], "max_validation_error_ratio")


def intervals(rows, count, seen, cards=None):
    values = []
    for row in rows:
        run_id = row["measurement_id"]
        check(isinstance(run_id, str) and run_id and run_id not in seen,
              "Measurement ID reused across timed, calibration or validation records")
        seen.add(run_id)
        start = integer(row["start_ns"], "start_ns")
        end = integer(row["end_ns"], "end_ns")
        if cards is not None:
            check(row["synchronized"] is True and sorted(row["cards"]) == cards,
                  "Whole-chain observations must use synchronized, matching physical cards")
        check(end > start, "Measurement duration must be positive")
        values.append(Decimal(end - start))
    check(len(values) >= count, "Insufficient independent measurements")
    return values


def calibrated_median(rows, method, seen, cards=None):
    values = intervals(rows, method["min_calibration_samples"], seen, cards)
    center = statistics.median(values)
    spread = (max(values) - min(values)) / center
    check(spread <= method["max_calibration_spread_ratio"], "Calibration spread exceeds frozen limit")
    return center, spread


def critical_path(groups, graph, method, seen):
    estimates = {}
    diagnostics = {}
    for group in groups:
        name = group["operation"]
        check(isinstance(name, str) and name and name not in estimates, "Duplicate calibration operation")
        check(group["kind"] in ("compute", "memory", "communication", "control"),
              "Unsupported calibrated operation kind")
        center, spread = calibrated_median(group["observations"], method, seen)
        estimates[name] = center
        diagnostics[name] = {"kind": group["kind"], "samples": len(group["observations"]),
                             "median_ns": float(center), "spread_ratio": float(spread)}
    finish = {}
    resource_tail = {}
    used = set()
    for node in graph:
        key, operation = node["id"], node["operation"]
        parents, resources = node["predecessors"], node["resources"]
        check(isinstance(key, str) and key and key not in finish and operation in estimates,
              "Unknown calibration operation or duplicate whole-chain node")
        check(isinstance(parents, list) and len(set(parents)) == len(parents)
              and all(parent in finish for parent in parents), "Reference graph must be acyclic and in topological order")
        check(isinstance(resources, list) and resources and len(set(resources)) == len(resources)
              and all(isinstance(resource, str) and resource for resource in resources),
              "Each reference node needs its occupied resources")
        start = max([Decimal(0), *(finish[parent] for parent in parents),
                     *(resource_tail.get(resource, Decimal(0)) for resource in resources)])
        finish[key] = start + estimates[operation]
        for resource in resources:
            resource_tail[resource] = finish[key]
        used.add(operation)
    check(bool(finish) and used == set(estimates), "Whole-chain graph and calibration operations differ")
    check({"compute", "memory", "communication"} <= {diagnostics[key]["kind"] for key in used},
          "The reference graph must cover compute, memory and communication")
    return max(finish.values()), {"graph_nodes": len(finish), "calibrated_operations": diagnostics}


def recompute(evidence, method, run_id, cards, shared_evidence=None):
    validate_method(method)
    work = read(evidence["work_ledger"])
    timing = read(evidence["timing"])
    calibration = read(evidence["calibration"])
    chain = read(evidence["whole_chain"])
    review = read(evidence["reference_review"])
    for item in (work, timing, calibration, chain, review):
        check(item["workload_sha256"] == method["workload_sha256"], "Reference workload identity differs")
        check(re.fullmatch(r"[0-9a-f]{64}", item["work_signature_sha256"])
              and item["work_signature_sha256"] == work["work_signature_sha256"],
              "Necessary-work signature differs across reference inputs")
    for item in (calibration, chain, review):
        check(item["reference_id"] == method["reference_id"], "Reference implementation differs from frozen method")
        check(sorted(item["cards"]) == sorted(cards), "Reference physical cards differ")
    check(work["run_id"] == run_id and timing["run_id"] == run_id,
          "Effective work and synchronized time must bind to this run")
    check(type(review["passed"]) is bool and review["passed"] and review["basis"],
          "The reference requires an independent applicability and whole-chain evidence review")
    components = work["components"]
    check(components and len({row["component"] for row in components}) == len(components),
          "Effective work components are missing or duplicated")
    flops = sum(integer(row["effective_flops"], "effective_flops") for row in components)
    check(flops > 0, "No positive effective work")
    for item in (calibration, chain):
        check(integer(item["effective_flops"], "effective_flops") == flops,
              "Reference and timed run must cover the same effective work")
    check(timing["measurement_method"], "Record the measurement method")
    seen = {run_id}
    seconds_ns, = intervals([timing], 1, seen, sorted(cards))
    if shared_evidence is not None:
        another_calibration = read(shared_evidence["calibration"])
        another_chain = read(shared_evidence["whole_chain"])
        another_timing = read(shared_evidence["timing"])
        for item in (another_calibration, another_chain):
            for key in ("workload_sha256", "work_signature_sha256", "reference_id", "effective_flops"):
                check(item[key] == chain[key], f"Shared reference differs: {key}")
            check(sorted(item["cards"]) == sorted(cards), "Shared reference physical cards differ")
        seen.add(another_timing["measurement_id"])
        chain["validation_observations"] += another_chain["validation_observations"]
        if method["estimator"] == "critical_path":
            check(chain["nodes"] == another_chain["nodes"], "Shared reference dependency graph differs")
            groups = {group["operation"]: group for group in another_calibration["operations"]}
            check(set(groups) == {group["operation"] for group in calibration["operations"]},
                  "Shared reference operations differ")
            for group in calibration["operations"]:
                other = groups[group["operation"]]
                check(group["kind"] == other["kind"], "Shared reference operation kind differs")
                group["observations"] += other["observations"]
        else:
            calibration["whole_chain_observations"] += another_calibration["whole_chain_observations"]
    if method["estimator"] == "critical_path":
        reference_ns, diagnostics = critical_path(calibration["operations"], chain["nodes"], method, seen)
    else:
        reference_ns, spread = calibrated_median(calibration["whole_chain_observations"], method, seen, sorted(cards))
        diagnostics = {"calibration_samples": len(calibration["whole_chain_observations"]),
                       "calibration_spread_ratio": float(spread)}
    validation = intervals(chain["validation_observations"], method["min_validation_samples"], seen, sorted(cards))
    worst_error = max(abs(reference_ns - observed) / observed for observed in validation)
    check(worst_error <= method["max_validation_error_ratio"],
          "Reference disagrees with independent synchronized whole-chain validation")
    return {
        "effective_flops": flops,
        "synchronized_s": float(seconds_ns / 10**9),
        "tflops_per_card": float(Decimal(flops) / (len(cards) * seconds_ns * 10**3)),
        "reference_tflops_per_card": float(Decimal(flops) / (len(cards) * reference_ns * 10**3)),
        "reference_s": float(reference_ns / 10**9),
        "measurement_method": timing["measurement_method"],
        "reference_time_kind": method["estimator"],
        "efficiency_pct": float(100 * reference_ns / seconds_ns),
        "reference_qualified": True,
        "reference_id": method["reference_id"],
        "work_signature_sha256": work["work_signature_sha256"],
        "reference_diagnostics": dict(diagnostics, validation_samples=len(validation),
                                      max_validation_error_ratio=float(worst_error),
                                      **({"paired_live_calibration": True} if shared_evidence else {})),
    }

