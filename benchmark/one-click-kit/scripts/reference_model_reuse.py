"""Resume only sealed completed unchanged native/observation model stages."""

import json
import os
from pathlib import Path

from dataset_contract import sha
from reference_evidence import transient_cache, http_scope_fix_only, post_capture_fix_only, sealed_file_matches
from reference_qualification_reuse import normalized

MODEL_STAGES = {
    "current-native-synchronized-timing": "native-current-timing",
    "baseline-native-synchronized-timing": "native-baseline-timing",
    "current-http-work-observation": "http-current-work",
}
ORCHESTRATORS = {"scripts/automatic_reference.py", "scripts/compute_reference.py", "scripts/third_party.py",
    "scripts/live_console.py",
    "scripts/reference_evidence.py",
    "scripts/third_party_report.py",
    "scripts/plot_compute_reference.py",
    "scripts/reference_qualification_reuse.py", "scripts/reference_model_reuse.py",
    "scripts/reference_observation_reuse.py",
    "protocols/automatic-compute-reference-v1.json"}

# These consumers run only after original capture/model services close. They
# do not supply native timing, calibration, input or numerical controls.
CPU_DERIVATIONS = {"third-party-test-kit/public_output_binding.py", "third-party-test-kit/native_public_client_domain.py","scripts/audit_http_reference.py", "scripts/resolved_http_inputs.py",
    "scripts/reference_final_stages.py", "third-party-test-kit/draft_state_closure.py",
    "third-party-test-kit/qualify_necessary_work.py", "third-party-test-kit/qualify_baseline_capacity.py", "third-party-test-kit/draft_producer_geometry.py"}



def guarded_load(source, expected_seal, current_plan, current_root, guard_directory):
    from resource_guard import acquire, release, verify_mapping

    guard_directory = Path(guard_directory)
    guard_directory.mkdir()
    os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = ",".join(map(str, current_plan["physical_cards"]))
    previous = os.environ.get("VLLMHUST_ENGINE_LOCKS")
    leases = None
    try:
        leases = acquire(guard_directory, 86400)
        verify_mapping(current_plan["logical_devices"])
        return load(source, expected_seal, current_plan, current_root, guard_directory / "recovered-http-audit")
    finally:
        # acquire publishes descriptor numbers for child inheritance. A CPU
        # admission ends in this process, so never pass closed descriptors to
        # the next admission or model child. An inherited lease is borrowed:
        # its caller retains ownership of those descriptors.
        if leases is not None and previous is None:
            release(leases)
        if previous is None:
            os.environ.pop("VLLMHUST_ENGINE_LOCKS", None)
        else:
            os.environ["VLLMHUST_ENGINE_LOCKS"] = previous


def load(source, expected_seal, current_plan, current_root, recovery_directory=None):
    source, current_root = Path(source).resolve(), Path(current_root).resolve()
    seal_path = source / "checksums.json"
    if sha(seal_path) != expected_seal:
        raise ValueError("Completed model-stage source seal changed")
    seal = json.loads(seal_path.read_bytes())
    ignored_caches = []
    view_context = {}
    print("REFERENCE_MODEL_SOURCE_VERIFY", 0, len(seal), flush=True)
    for index, (name, pin) in enumerate(seal.items(), 1):
        if transient_cache(name):
            ignored_caches.append(name)
            continue
        path = source / name
        if not sealed_file_matches(source, name, pin, seal, view_context):
            raise ValueError("Completed model-stage artifact changed or escaped its seal: " + name)
        if index % 1000 == 0:
            print("REFERENCE_MODEL_SOURCE_VERIFY", index, len(seal), flush=True)
    print("REFERENCE_MODEL_SOURCE_VERIFIED", len(seal), flush=True)
    for name in ("automatic-plan.json", "automatic-result.json"):
        if name not in seal:
            raise ValueError("Model continuation lacks its sealed plan/result")
    old = json.loads((source / "automatic-plan.json").read_bytes())
    result = json.loads((source / "automatic-result.json").read_bytes())
    tracked_ids = set(MODEL_STAGES) | {"gdn-decode-independent-reference",
        "gdn-prefill-independent-reference", "raw-route-recorder-qualification"}
    def tracked(plan):
        rows = [row for row in plan["stages"] if row["id"] in tracked_ids]
        if len(rows) != len(tracked_ids) or {row["id"] for row in rows} != tracked_ids:
            raise ValueError("Missing or duplicate original operator/model stage controls")
        return {row["id"]: row for row in rows}
    if (result["binding"] != current_plan["binding"]
            or result["plan_sha256"] != sha(source / "automatic-plan.json")
            or tracked(old) != tracked(current_plan)):
        raise ValueError("Model continuation workload, plan identity or any operator/model stage control differs")
    old_root = Path(old["reference"]).resolve().parents[1]
    def controls(plan):
        result = {k: v for k, v in plan.items() if k not in (
            "files", "stages", "protocol_sha256", "qualification_reuse", "stage_resume", "canonical_reuse")}
        # Old coordinators described the HTTP companion here even though the
        # separately frozen capture profile and native plans governed clocks.
        # Correct only that exact annotation. Sampling, actual native plans,
        # every runtime control and source pin remain independently checked.
        legacy = dict(frontend="HTTP", dtype="BF16", tp=2,
            max_completion_tokens=8192, profiles=["mmlu-fixed200-v1"])
        if result.get("measurement_domain") == legacy and result.get("observation_domain_only") is True:
            capture = result["capture_profile"]
            result["measurement_domain"] = dict(frontend="native_inproc", dtype="BF16", tp=2,
                max_completion_tokens=capture["max_tokens"], formal_repetitions=capture["formal_repetitions"],
                profiles=["mmlu-fixed200-v1"])
            result["observation_domain_only"] = False
        return result
    if normalized(controls(old), old_root, current_root) != controls(current_plan):
        raise ValueError("Model continuation runtime, workload or numerical controls differ")
    # Keep all original native client/fence/warmup/guard/observer sources exact.
    # Only the coordinator and its new post-model providers may change.
    for filename, pin in old["files"].items():
        path = Path(filename)
        relative = None
        if path.is_relative_to(old_root):
            relative = str(path.relative_to(old_root))
            if relative in ORCHESTRATORS | CPU_DERIVATIONS:
                continue
            current = current_root / relative
        else:
            current = path
        current_pin = current_plan["files"].get(str(current))
        if (current_pin != pin or sha(current) != pin):
            if (relative == "scripts/run_b1_r4.py" and current_pin == sha(current)
                    and http_scope_fix_only(path, current, pin)):
                continue
            if (relative == "scripts/reference_observation.py" and current_pin == sha(current)
                    and post_capture_fix_only(path, current, pin)):
                continue
            raise ValueError("Completed model-stage implementation/dependency differs: " + str(current))
    rows = {row["id"]: row for row in result["stages"]}
    admitted = {}
    for stage, directory in MODEL_STAGES.items():
        row = rows.get(stage, {})
        path = source / directory / "receipt.json"
        name = str(path.relative_to(source))
        # A failed HTTP admission that never started a model/client has no
        # observation to reuse. Preserve its sealed failure and execute it anew.
        if stage == "current-http-work-observation" and row.get("status") == "failed":
            if name not in seal or sha(path) != seal[name]:
                raise ValueError("Unsealed failed HTTP admission")
            failed = json.loads(path.read_bytes())
            if failed.get("completed") is not False or failed.get("quiescent") is not True:
                raise ValueError("Only a quiescent HTTP failure can be examined for continuation")
            if failed.get("model_service_started") is False and failed.get("client_started") is False:
                continue
            if (failed.get("failure") != "ValueError: Draft consumer reused or mismatched its latest actual producer"
                    or recovery_directory is None):
                raise ValueError("Closed HTTP postprocessing recovery requires an independent guarded audit")
            from audit_http_reference import run as audit_http

            recovered = audit_http(path.parent, current_plan["deployment"], recovery_directory)
            if recovered.get("completed") is not True or recovered.get("original_completed") is not False:
                raise ValueError("Independent closed HTTP recovery did not pass")
        elif row.get("status") != "completed" or name not in seal or row["receipt_sha256"] != sha(path):
            raise ValueError("Only complete sealed model stages may resume: " + stage)
        receipt = json.loads(path.read_bytes())
        recovered_http = stage == "current-http-work-observation" and row.get("status") == "failed"
        if (receipt.get("completed") is not True and not recovered_http) or receipt.get("quiescent") is not True:
            raise ValueError("Model continuation contains an unfinished/live stage: " + stage)
        nested = source / directory / "checksums.json"
        if (str(nested.relative_to(source)) not in seal
                or json.loads(nested.read_bytes()).get("receipt.json") != sha(path)):
            raise ValueError("Model receipt is outside its original owned-stage seal: " + stage)
        expected_role = "B0" if stage == "baseline-native-synchronized-timing" else "B1"
        if receipt.get("role") != expected_role:
            raise ValueError("Model continuation has the wrong frozen arm role: " + stage)
        if stage != "current-http-work-observation" and (
                receipt.get("preflight_only") is not False or receipt.get("formal_repetitions") != (current_plan.get("capture_profile") or {"formal_repetitions": 1})["formal_repetitions"]):
            raise ValueError("Only full formal native timing stages can resume")
        mapping_path = source / directory / "device-mapping.json"
        if str(mapping_path.relative_to(source)) not in seal:
            raise ValueError("Original model-stage device mapping is unsealed")
        mapping = json.loads(mapping_path.read_bytes())
        if (mapping["physical_cards"] != current_plan["physical_cards"]
                or mapping["logical_cards"] != list(map(int, current_plan["logical_devices"].split(",")))):
            raise ValueError("Original model-stage device pair differs")
        admitted[stage] = dict(source=str(path.parent), receipt_sha256=sha(path),
            original_plan_sha256=sha(source / "automatic-plan.json"))
        if recovered_http:
            admitted[stage].update(recovered_post_capture_cpu_derivation=True,
                original_receipt_completed=False, original_failure=receipt["failure"],
                independent_recovery_source=str(Path(recovery_directory).resolve()),
                independent_recovery_checksums_sha256=sha(Path(recovery_directory) / "checksums.json"))
    return dict(source=str(source), checksums_sha256=sha(seal_path), stages=admitted,
        ignored_transient_cache_entries=ignored_caches,
        operator_qualification_source=old.get("qualification_reuse") or dict(
            source=str(source), checksums_sha256=sha(seal_path)),
        model_inference_rerun=False)
