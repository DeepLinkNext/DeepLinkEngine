"""Reuse only sealed unchanged operator qualifications, never model timings."""

import json
import os
from pathlib import Path

from dataset_contract import sha
from reference_evidence import transient_cache, http_scope_fix_only, post_capture_fix_only, sealed_file_matches

STAGES = {
    "gdn-decode-independent-reference": ("gdn-decode", "cases"),
    "gdn-prefill-independent-reference": ("gdn-prefill", "prefill_cases"),
    "raw-route-recorder-qualification": ("route-recorder", "route_cases"),
}
# These files neither execute the reused operator nor supply its controls.
# draft_producer_geometry is a downstream CPU observer-record consumer; the
# qualification import-closure regression forbids it entering any of the three
# reused operator drivers. Every other old plan source stays pinned.
ORCHESTRATORS = {"scripts/automatic_reference.py", "scripts/run_native_reference.py", "scripts/third_party.py",
                 "scripts/live_console.py",
                 "scripts/reference_evidence.py",
                 "scripts/third_party_report.py",
                 "third-party-test-kit/draft_producer_geometry.py", "scripts/compute_reference.py",
                 "scripts/reference_qualification_reuse.py", "protocols/automatic-compute-reference-v1.json"}


# These model-capture and CPU result consumers do not execute an operator.
# A transitive static-import regression also checks all declared dynamic roots.
OPERATOR_INDEPENDENT = {
    "third-party-test-kit/draft_state_closure.py",
    "scripts/audit_http_reference.py", "scripts/resolved_http_inputs.py",
    "scripts/native_timing_entry.py", "scripts/baseline_native_entry.py", "scripts/reference_observation_reuse.py",
    "scripts/native_timing_prepare.py", "scripts/baseline_native_prepare.py",
    "scripts/native_timing_session.py", "scripts/reference_capture_profile.py",
    "scripts/reference_model_reuse.py", "scripts/reference_final_stages.py",
    "scripts/plot_compute_reference.py", "scripts/third_party_report.py",
    "third-party-test-kit/public_output_binding.py",
    "third-party-test-kit/qualify_native_capacity.py",
    "third-party-test-kit/qualify_baseline_capacity.py",
    "third-party-test-kit/qualify_dense_source.py",
    "third-party-test-kit/qualify_necessary_work.py",
    "third-party-test-kit/reference_pair_statistics.py",
}


def normalized(value, old_root, current_root):
    if isinstance(value, dict):
        return {k: normalized(v, old_root, current_root) for k, v in value.items()}
    if isinstance(value, list):
        return [normalized(v, old_root, current_root) for v in value]
    if isinstance(value, str) and value.startswith(str(old_root) + "/"):
        return str(current_root / Path(value).relative_to(old_root))
    return value


def controls(plan):
    value = {k: v for k, v in plan.items() if k not in (
        "files", "qualification_reuse", "stage_resume", "canonical_reuse", "observation_reuse", "protocol_sha256",
        "capture_profile", "reference_capture_binding")}
    # The model-clock budget is not an operator calibration control. Keep its
    # non-budget annotation fields fixed while admitting the exact corrected
    # native annotation (or the one known legacy HTTP annotation).
    legacy = dict(frontend="HTTP", dtype="BF16", tp=2,
        max_completion_tokens=8192, profiles=["mmlu-fixed200-v1"])
    capture = plan.get("capture_profile")
    current = dict(frontend="native_inproc", dtype="BF16", tp=2,
        max_completion_tokens=capture["max_tokens"], formal_repetitions=capture["formal_repetitions"],
        profiles=["mmlu-fixed200-v1"]) if capture else None
    if ((value.get("measurement_domain") == legacy and value.get("observation_domain_only") is True)
            or (value.get("measurement_domain") == current and value.get("observation_domain_only") is False)):
        value["measurement_domain"] = dict(dtype="BF16", tp=2, profiles=["mmlu-fixed200-v1"])
        value["observation_domain_only"] = False
    # Later model/CPU providers do not execute these qualified operators.
    # Preserve all three operator stage descriptors, every numeric control,
    # input binding, runtime and the separately hash-bound policy authority.
    value["stages"] = [row for row in plan["stages"] if row["id"] in STAGES]
    return value


def guarded_load(source, expected_seal, current_plan, current_root, guard_directory):
    """Keep whole-envelope CPU seal reads under the same container leases.

    A qualification source may also contain large completed model traces.
    Guard its admission even though the three reused operators do not run.
    """
    from resource_guard import acquire, release, verify_mapping

    guard_directory = Path(guard_directory)
    guard_directory.mkdir()
    os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = ",".join(map(str, current_plan["physical_cards"]))
    previous = os.environ.get("VLLMHUST_ENGINE_LOCKS")
    leases = None
    try:
        leases = acquire(guard_directory, 86400)
        verify_mapping(current_plan["logical_devices"])
        return load(source, expected_seal, current_plan, current_root)
    finally:
        if leases is not None and previous is None:
            release(leases)
        if previous is None:
            os.environ.pop("VLLMHUST_ENGINE_LOCKS", None)
        else:
            os.environ["VLLMHUST_ENGINE_LOCKS"] = previous


def load(source, expected_seal, current_plan, current_root):
    source, current_root = Path(source).resolve(), Path(current_root).resolve()
    seal_path = source / "checksums.json"
    if sha(seal_path) != expected_seal:
        raise ValueError("Prior operator qualification seal changed")
    seal = json.loads(seal_path.read_bytes())
    required = {"automatic-plan.json", "automatic-result.json"}
    if not required <= seal.keys():
        raise ValueError("Prior operator qualification seal is incomplete")
    view_context = {}
    print("REFERENCE_OPERATOR_SOURCE_VERIFY", 0, len(seal), flush=True)
    for index, (name, expected) in enumerate(seal.items(), 1):
        if transient_cache(name):
            continue
        path = source / name
        if not sealed_file_matches(source, name, expected, seal, view_context):
            raise ValueError("Prior reference artifact changed or escaped its seal: " + name)
        if index % 1000 == 0:
            print("REFERENCE_OPERATOR_SOURCE_VERIFY", index, len(seal), flush=True)
    print("REFERENCE_OPERATOR_SOURCE_VERIFIED", len(seal), flush=True)
    old_path = source / "automatic-plan.json"
    old = json.loads(old_path.read_bytes())
    result = json.loads((source / "automatic-result.json").read_bytes())
    if result["binding"] != current_plan["binding"] or result["plan_sha256"] != sha(old_path):
        raise ValueError("Prior reference workload or plan identity differs")
    old_root = Path(old["reference"]).resolve().parents[1]
    old_controls, current_controls = controls(old), controls(current_plan)
    if normalized(old_controls, old_root, current_root) != current_controls:
        raise ValueError("Prior qualification runtime, cases or numerical/statistical controls differ")
    for name, expected in old["files"].items():
        path = Path(name)
        relative = None
        if path.is_relative_to(old_root):
            relative = str(path.relative_to(old_root))
            if relative in ORCHESTRATORS | OPERATOR_INDEPENDENT:
                continue
            current = current_root / relative
        else:
            current = path
        current_pin = current_plan["files"].get(str(current))
        if current_pin != expected or sha(current) != expected:
            if (relative == "scripts/run_b1_r4.py" and current_pin == sha(current)
                    and http_scope_fix_only(path, current, expected)):
                continue
            if (relative == "scripts/reference_observation.py" and current_pin == sha(current)
                    and post_capture_fix_only(path, current, expected)):
                continue
            raise ValueError("Reused operator's source/dependency changed: " + str(current))
    stages = {}
    for row in result["stages"]:
        if row["id"] in STAGES and row["status"] == "completed":
            if row["id"] in stages:
                raise ValueError("Duplicate prior operator qualification")
            directory, _ = STAGES[row["id"]]
            path = source / directory / "receipt.json"
            if (str(path.relative_to(source)) not in seal
                    or sha(path) != row["receipt_sha256"]):
                raise ValueError("Completed operator receipt is outside its original seal")
            stages[row["id"]] = dict(receipt=json.loads(path.read_bytes()), path=path)
    if not stages:
        raise ValueError("No completed unchanged operator qualifications to reuse")
    return dict(plan=old, plan_path=old_path, stages=stages)
