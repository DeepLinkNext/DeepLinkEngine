"""Qualify original018 public delivery and hybrid capacity from sealed evidence."""

import ast
import hashlib
import json
from pathlib import Path
import re

from baseline_capacity_bounds import envelope
from public_output_binding import exact
from native_public_client_domain import bind as bind_clients

PROTOCOL = Path(__file__).resolve().parents[1] / "protocols/original018-native-capacity-source-domain-v2.json"
PROTOCOL_SHA256 = "40dac8ba575e932910165a5d4bf6e5a4712aa5c9c641bc928ba6e66a14662a6d"
SOURCE_MAP_SHA256 = "a357ad8cf1ff6f8797fb2d6476e71a0deef5ee0899a55ea32ed65f07fe7773bc"
RESET_SHA256 = "177d4e7fb324f107cc7d8fd882e0545af37b162abf8f300dcef4a1e8fa68ec89"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_client(plan, capture, protocol):
    """Whole-file entry admission, with unchanged reviewed cold-reset code."""
    clients = bind_clients(plan, capture)
    entries = [(Path(name), digest) for name, digest in plan["files"].items()
               if Path(name).name == "baseline_native_entry.py"]
    if len(entries) != 1:
        raise ValueError("One reviewed original018 native entry is required")
    entry, pin = entries[0]
    admitted = protocol["baseline_entrypoints"].get(pin)
    if admitted is None or capture["id"] not in admitted["profiles"] or sha(entry) != pin:
        raise ValueError("Original018 native entry is outside its reviewed capture profile")
    source = entry.read_text()
    reset = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name == "reset_prefix_cache"]
    if (len(reset) != 1 or hashlib.sha256(ast.get_source_segment(source, reset[0]).encode()).hexdigest()
            != protocol["reset_prefix_cache_function_source_sha256"]):
        raise ValueError("Reviewed original018 cold-reset implementation changed")
    if "budget_validator_sha256" in admitted:
        validators = [Path(name) for name, digest in plan["files"].items()
                      if Path(name).name == "reference_capture_profile.py"
                      and digest == admitted["budget_validator_sha256"]]
        if len(validators) != 1 or sha(validators[0]) != admitted["budget_validator_sha256"]:
            raise ValueError("Original018 versioned entry lacks its reviewed budget validator")
        policy = validators[0].resolve().parents[1] / "protocols/compute-capture-profiles-v2.json"
        if sha(policy) != admitted["budget_protocol_sha256"]:
            raise ValueError("Original018 capture policy differs from the reviewed validator")
        clients["reference_capture_profile.py"] = admitted["budget_validator_sha256"]
        clients["compute-capture-profiles-v2.json"] = admitted["budget_protocol_sha256"]
    clients["baseline_native_entry.py"] = pin
    return clients


def validate_artifacts(arguments, reset, workload, log, protocol):
    for key, value in protocol["arguments"].items():
        if key not in arguments or not exact(arguments[key], value):
            raise ValueError("Unreviewed original018 capacity argument: " + key)
    if not exact(reset, dict(schema="baseline-native-prefix-reset/v1", reset=True,
                            b1_state_cache_present=False, shared_request_cache_salts_unique=True)):
        raise ValueError("Original018 formal prefix reset did not complete")
    tokens = workload["request_token_ids"]
    if (len(tokens) != 200 or sum(map(len, tokens)) != 282265
            or any(not row or any(type(t) is not int or t < 0 for t in row) for row in tokens)):
        raise ValueError("Original018 capacity inputs differ from fixed200")
    capacities = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", log)
    if not capacities or len(set(capacities)) != 1:
        raise ValueError("Missing or conflicting actual original018 hybrid pool")
    result = envelope(list(map(len, tokens)), reported_tokens=int(capacities[0].replace(",", "")),
        maximum_output=protocol["formal_max_tokens"], execution_seats=arguments["max_num_seqs"],
        maximum_block_size=protocol["maximum_minimum_block_size"],
        in_flight_batches=protocol["in_flight_batches"], speculative_tokens=protocol["speculative_tokens"],
        lookahead_tokens=protocol["lookahead_tokens"])
    if not result["sufficient_page_capacity"]:
        raise ValueError("Full8192 hybrid ownership envelope does not fit every permitted grouping")
    return result


def qualify(baseline_deployment, native, *, repetition=1):
    root, native = Path(baseline_deployment).resolve(), Path(native).resolve()
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Reviewed original018 source-domain proof changed")
    protocol = json.loads(PROTOCOL.read_bytes())
    seal_path = native / "checksums.json"
    seal = json.loads(seal_path.read_bytes())

    def read(name, text=False):
        path = native / name
        if path.is_symlink() or not path.resolve().is_relative_to(native) or seal.get(name) != sha(path):
            raise ValueError("Original018 evidence changed or is unsealed: " + name)
        return path.read_text() if text else json.loads(path.read_bytes())

    receipt = read("receipt.json")
    if (receipt["role"] != "B0" or receipt["completed"] is not True or receipt["quiescent"] is not True
            or receipt["preflight_only"] is not False or receipt.get("failure")
            or type(receipt["formal_repetitions"]) is not int or not 1 <= repetition <= receipt["formal_repetitions"]):
        raise ValueError("Only a completed quiescent formal original018 run can be qualified")
    plan = read("service/native-plan.json")
    arguments = read("service/client/effective-engine-args.json")
    reset = read(f"service/client/formal-{repetition}-reset.json")
    workload = read("service/token-workload.json")
    admitted = read("service/baseline-source-admission.json")
    composition = read("service/client/composition.json")
    from reference_capture_profile import validate_native_plan

    capture = validate_native_plan(plan)
    if (plan["schema"] != "baseline-native-m01-timing/v1" or plan["role"] != "B0"
            or plan["engine_release"] != "original018-cann8.5.1"
            or plan["paired_engine_release"] != "full-answer-b1-20261117-r8"
            or Path(plan["baseline_deployment"]).resolve() != root
):
        raise ValueError("Original018 native plan or formal sampling differs")
    if (admitted["source_map_sha256"] != SOURCE_MAP_SHA256 or admitted["variant"] != "text"
            or Path(admitted["deployment"]).resolve() != root
            or composition["identity"]["role"] != "B0"):
        raise ValueError("Frozen original018 text SDK admission differs")
    for package, relative in (("vllm", "text-only-overlay/vllm/vllm/__init__.py"),
                              ("vllm_ascend", "text-only-overlay/vllm-ascend/vllm_ascend/__init__.py")):
        observed = "vllm_source" if package == "vllm" else "ascend_source"
        if (Path(admitted["origins"][package]).resolve() != root / relative
                or Path(composition["identity"][observed]).resolve() != root / relative):
            raise ValueError("Original018 actual import origin differs: " + package)
    for relative, pin in protocol["source_sha256"].items():
        if admitted["files"].get(relative, {}).get("sha256") != pin or sha(root / relative) != pin:
            raise ValueError("Reviewed original018 implementation differs: " + relative)
    client_binding = bind_client(plan, capture, protocol)
    workload_path = native / "service/token-workload.json"
    if (Path(plan["workload"]).resolve() != workload_path
            or plan["files"].get(str(workload_path)) != sha(workload_path)
            or workload["current_http_token_hashes_matched"] is not True
            or workload["formal_questions_used_for_warmup"] is not False):
        raise ValueError("Original018 prompt population is outside its native plan")
    quality = Path(plan["deployment"]) / "inputs/mmlu-quality018-r2/quality-request-plan.json"
    pin = "f4ed4ce5956c9ed08f360400b5a996ef92a217ce295d50648d4dc06443db768e"
    if workload["source_files"].get(str(quality)) != pin or sha(quality) != pin:
        raise ValueError("Original018 fixed200 quality source differs")
    result = validate_artifacts(arguments, reset, workload, read("service.log", text=True), protocol)
    result.update(source_ownership_qualified=True, no_preemption_certified=True,
        no_external_prefix_reuse_certified=True, uncoalesced_public_delivery_source_qualified=True,
        protocol_sha256=PROTOCOL_SHA256, native_checksums_sha256=sha(seal_path),
        reviewed_source_claims=protocol["claims"], native_client_binding=client_binding,
        observed_capture_profile=capture,
        capacity_proof_maximum_output_tokens=protocol["formal_max_tokens"],
        scope="Original018 fresh plain-token prefixes, uncoalesced public delivery, and no capacity preemption/recomputation; producer arithmetic and reference qualification remain separate.")
    return result
