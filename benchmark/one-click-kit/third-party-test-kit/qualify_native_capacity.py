"""Bind a reviewed finite ownership proof to one completed sealed r8 run."""

import hashlib
import json
from pathlib import Path
import re

from native_capacity_bounds import envelope
from public_output_binding import bind as publication_binding, exact

PROTOCOL = Path(__file__).resolve().parents[1] / "protocols/r8-native-capacity-source-domain-v1.json"
PROTOCOL_SHA256 = "9e5c089a37bec45fde053808bec28eba2093c7e3ccaf50e39e199509ccc88309"
RESET_SOURCE_SHA256 = "c8837c4ef8a0d46e5df9b7c1893403b3807f42713011ec538e27aeab1da59fdf"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_artifacts(arguments, reset, workload, service_log, protocol):
    for key, expected in protocol["arguments"].items():
        if key not in arguments or not exact(arguments[key], expected):
            raise ValueError("Unreviewed capacity argument: " + key)
    for key, expected in protocol["additional"].items():
        if key not in arguments["additional_config"] or not exact(arguments["additional_config"][key], expected):
            raise ValueError("Unreviewed State capacity argument: " + key)
    snapshot = reset["last_snapshot"]
    if (reset["reset"] is not True or reset.get("error") or snapshot["pending"] or snapshot["host"]
            or len(snapshot["seats"]) != 54 or any(s["owner"] is not None for s in snapshot["seats"])
            or (snapshot.get("partial_reclaim") or {}).get("paused")):
        raise ValueError("Formal State/prefix cache was not drained and reset")
    tokens = workload["request_token_ids"]
    if (len(tokens) != 200 or sum(map(len, tokens)) != 282265
            or any(not row or any(type(t) is not int or t < 0 for t in row) for row in tokens)):
        raise ValueError("Capacity proof belongs to another fixed200 prompt population")
    matches = re.findall(r"GPU KV cache size: ([0-9,]+) tokens, Maximum concurrency for ([0-9,]+) tokens", service_log)
    if (not matches or any(int(limit.replace(",", "")) != 65536 for _, limit in matches)
            or len({int(count.replace(",", "")) for count, _ in matches}) != 1):
        raise ValueError("Missing or ambiguous actual single-group page pool capacity")
    capacity = int(matches[0][0].replace(",", ""))
    if capacity <= 0 or capacity % protocol["block_size"]:
        raise ValueError("Resolved pool capacity is not aligned to source paged128")
    result = envelope(list(map(len, tokens)), maximum_output=protocol["formal_max_tokens"],
        resident_seats=54, execution_seats=36, block_size=protocol["block_size"],
        pool_blocks=capacity // protocol["block_size"], in_flight_batches=protocol["in_flight_batches"],
        speculative_tokens=protocol["speculative_tokens"], lookahead_tokens=protocol["lookahead_tokens"],
        transfer_directions=protocol["transfer_directions"])
    if not result["sufficient_page_capacity"]:
        raise ValueError("Full8192 ownership envelope does not fit the actual pool")
    return result


def qualify(deployment, native, *, repetition=1):
    deployment, native = Path(deployment).resolve(), Path(native).resolve()
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Reviewed native capacity source-domain proof changed")
    protocol = json.loads(PROTOCOL.read_bytes())
    seal_path = native / "checksums.json"
    seal = json.loads(seal_path.read_bytes())
    def read(name, text=False):
        path = native / name
        if (path.is_symlink() or not path.resolve().is_relative_to(native)
                or seal.get(name) != sha(path)):
            raise ValueError("Native capacity evidence changed or is unsealed: " + name)
        return path.read_text() if text else json.loads(path.read_bytes())
    receipt = read("receipt.json")
    if (receipt["role"] != "B1" or receipt["completed"] is not True
            or receipt["quiescent"] is not True or receipt["preflight_only"] is not False
            or receipt.get("failure") or type(receipt["formal_repetitions"]) is not int or not 1 <= repetition <= receipt["formal_repetitions"]):
        raise ValueError("Only a completed quiescent formal native B1 run can be qualified")
    plan = read("service/native-plan.json")
    args = read("service/client/effective-engine-args.json")
    reset = read(f"service/client/formal-{repetition}-reset.json")
    work = read("service/token-workload.json")
    log = read("service.log", text=True)
    workload_path = native / "service/token-workload.json"
    if (Path(plan["workload"]).resolve() != workload_path
            or plan["files"].get(str(workload_path)) != sha(workload_path)):
        raise ValueError("Formal prompt workload is outside the measured native plan")
    sources = work["source_files"]
    quality = deployment / "inputs/mmlu-quality018-r2/quality-request-plan.json"
    expected_quality = "f4ed4ce5956c9ed08f360400b5a996ef92a217ce295d50648d4dc06443db768e"
    if (sources.get(str(quality)) != expected_quality or sha(quality) != expected_quality
            or work["current_http_token_hashes_matched"] is not True
            or work["formal_questions_used_for_warmup"] is not False):
        raise ValueError("Native capacity workload is not bound to the fixed current MMLU200 inputs")
    installed = json.loads((deployment / "provenance/installed-files.json").read_bytes())
    for name, expected in protocol["source_sha256"].items():
        if installed.get(name) != expected or sha(deployment / name) != expected:
            raise ValueError("Reviewed ownership implementation differs: " + name)
    resets = [Path(name) for name, value in plan["files"].items()
              if Path(name).name == "native_state_cache.py" and value == RESET_SOURCE_SHA256]
    if len(resets) != 1 or sha(resets[0]) != RESET_SOURCE_SHA256:
        raise ValueError("Native cold-reset source is outside the reviewed domain")
    publication = publication_binding(deployment, native / "service/native-plan.json",
                                      native / "service/client/effective-engine-args.json")
    result = validate_artifacts(args, reset, work, log, protocol)
    result.update(source_ownership_qualified=True, no_preemption_certified=True,
        no_external_prefix_reuse_certified=True, protocol_sha256=PROTOCOL_SHA256,
        reviewed_source_claims=protocol["claims"], native_checksums_sha256=sha(seal_path),
        publication_binding=publication, scope="Fresh text prefixes and no active capacity preemption/recomputation for this sealed native r8 run; arithmetic/producer/reference qualifications remain separate.")
    return result
