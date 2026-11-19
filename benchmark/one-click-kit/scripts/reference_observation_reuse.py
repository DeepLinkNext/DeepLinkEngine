"""Admit an explicit sealed HTTP source-domain observation, never its F or T."""

import json
import os
from pathlib import Path

from dataset_contract import sha
from reference_evidence import transient_cache, sealed_file_matches

OBSERVATION_INDEPENDENT = {
    "scripts/automatic_reference.py", "scripts/compute_reference.py", "scripts/live_console.py",
    "scripts/reference_evidence.py", "scripts/reference_final_stages.py", "scripts/reference_model_reuse.py",
    "scripts/reference_qualification_reuse.py", "scripts/run_native_reference.py", "scripts/third_party.py",
    "scripts/third_party_report.py", "scripts/native_timing_entry.py", "scripts/baseline_native_entry.py",
    "scripts/native_timing_prepare.py", "scripts/baseline_native_prepare.py", "scripts/native_timing_session.py",
}
REQUIRED_RUNTIME = {
    "scripts/run_b1_r4.py", "scripts/engine_entry.py", "scripts/reference_observation.py",
    "scripts/resource_guard.py", "scripts/owned_processes.py", "scripts/serving_warmup.py",
    "scripts/checkpoint_cache.py", "scripts/qualification.py", "scripts/dataset_contract.py",
    "scripts/restore_b1_r4.py", "scripts/dependency_admission.py", "scripts/compile_boundary.py",
}


def guarded_load(source, expected_seal, plan, current_root, guard_directory):
    from resource_guard import acquire, release, verify_mapping

    guard_directory = Path(guard_directory)
    guard_directory.mkdir()
    os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = ",".join(map(str, plan["physical_cards"]))
    previous = os.environ.get("VLLMHUST_ENGINE_LOCKS")
    leases = None
    try:
        leases = acquire(guard_directory, 86400)
        verify_mapping(plan["logical_devices"])
        return load(source, expected_seal, plan, current_root)
    finally:
        if leases is not None and previous is None:
            release(leases)
        if previous is None:
            os.environ.pop("VLLMHUST_ENGINE_LOCKS", None)
        else:
            os.environ["VLLMHUST_ENGINE_LOCKS"] = previous


def load(source, expected_seal, plan, current_root):
    source, current_root = Path(source).resolve(), Path(current_root).resolve()
    receipt = json.loads((source / "receipt.json").read_bytes())
    if (receipt.get("completed") is not True or receipt.get("quiescent") is not True
            or receipt.get("role") != "B1" or receipt.get("raw_observation", {}).get("completed") is not True
            or receipt.get("planned_joint_requests") != 200):
        raise ValueError("Only a complete quiescent fixed200 HTTP observation may be reused")
    seal_path = source / "checksums.json"
    if sha(seal_path) != expected_seal:
        raise ValueError("HTTP observation source seal changed")
    seal = json.loads(seal_path.read_bytes())
    view_context = {}
    for name, pin in seal.items():
        if transient_cache(name):
            continue
        path = source / name
        if not sealed_file_matches(source, name, pin, seal, view_context):
            raise ValueError("HTTP observation artifact changed or escaped its seal: " + name)
    required = {"receipt.json", "deployment.json", "installed-files.json", "controller-source.json",
        "device-mapping.json", "model-weights.json", "service/raw-work/raw-partial-work.json"}
    if not required <= seal.keys():
        raise ValueError("HTTP observation lacks sealed runtime/source provenance")
    deployment = Path(plan["deployment"])
    if (json.loads((source / "deployment.json").read_bytes()) != json.loads((deployment / "deployment.json").read_bytes())
            or json.loads((source / "installed-files.json").read_bytes()) != json.loads((deployment / "provenance/installed-files.json").read_bytes())):
        raise ValueError("HTTP observation belongs to a different frozen SDK")
    mapping = json.loads((source / "device-mapping.json").read_bytes())
    if (mapping["physical_cards"] != plan["physical_cards"]
            or mapping["logical_cards"] != list(map(int, plan["logical_devices"].split(",")))):
        raise ValueError("HTTP observation belongs to a different device pair")
    controller = json.loads((source / "controller-source.json").read_bytes())
    if not REQUIRED_RUNTIME <= controller.keys():
        raise ValueError("HTTP observation runtime closure is incomplete")
    for name, pin in controller.items():
        if name in OBSERVATION_INDEPENDENT:
            continue
        path = current_root / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(current_root) or sha(path) != pin:
            raise ValueError("HTTP observation implementation differs: " + name)
    partial = json.loads((source / "service/raw-work/raw-partial-work.json").read_bytes())
    if (partial.get("effective_flops") is not None or partial.get("worker_schedule_matches") is not True
            or partial.get("moe_routes", {}).get("peer_routes_match") is not True
            or len(partial.get("requests", [])) != 200):
        raise ValueError("HTTP observation is incomplete or has inconsistent raw schedules/routes")
    return dict(source=str(source), checksums_sha256=expected_seal, receipt_sha256=sha(source / "receipt.json"),
        observation_only=True, old_counts_or_times_used_for_native_work=False,
        independent_source_and_producer_audit_still_required=True)
