"""Bind canonical dense-tail deletion to reviewed frozen B0/B1 sources."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from public_decode_tail import domain, dense_shapes
from public_output_binding import exact
from qualify_native_capacity import qualify as qualify_current_capacity
from qualify_baseline_capacity import qualify as qualify_original_capacity

PROTOCOL = Path(__file__).resolve().parents[1] / "protocols/canonical-dense-tail-source-domain-v1.json"
PROTOCOL_SHA256 = "1f6206448384534edc49291d977043e401f09e7e82a95e69dbe0912e5f41a8eb"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_domain(text, arguments, geometry, capacity, protocol):
    for key, expected in protocol["model_text_fields"].items():
        if key not in text or not exact(text[key], expected):
            raise ValueError("Unreviewed canonical dense model geometry: " + key)
    for key, expected in protocol["arguments"].items():
        if key not in arguments or not exact(arguments[key], expected):
            raise ValueError("Unreviewed canonical dense runtime: " + key)
    if (capacity["source_ownership_qualified"] is not True
            or capacity["no_preemption_certified"] is not True
            or capacity["no_external_prefix_reuse_certified"] is not True):
        raise ValueError("Native prefix ownership/source qualification must pass first")
    signature = {key: geometry[key] for key in ("requests", "maximum_output_tokens", "speculative_tokens", "public_steps")}
    if (geometry["schema"] != "native-public-output-geometry/v1"
            or geometry["geometry_sha256"] != digest(signature)
            or geometry["maximum_output_tokens"] not in (256, 8192) or geometry["speculative_tokens"] != 2
            or len(geometry["requests"]) != 200
            or sum(r["input_tokens"] for r in geometry["requests"]) != 282265):
        raise ValueError("Canonical tail is outside its complete current formal geometry")
    tail = domain(geometry, execution_seats=arguments["max_num_seqs"])
    if not tail["waves"] or any(not 1 <= w["necessary_target_rows"] <= 108 for w in tail["waves"]):
        raise ValueError("No nonempty canonical tail in the reviewed unsplit domain")
    model = SimpleNamespace(**text)
    return tail, dense_shapes(tail, model)


def qualify(deployment, native, *, baseline_deployment=None, repetition=1):
    deployment, native = Path(deployment).resolve(), Path(native).resolve()
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Reviewed dense-tail source-domain proof changed")
    protocol = json.loads(PROTOCOL.read_bytes())
    seal_path = native / "checksums.json"
    seal = json.loads(seal_path.read_bytes())

    def read(name):
        path = native / name
        if path.is_symlink() or not path.resolve().is_relative_to(native) or seal.get(name) != sha(path):
            raise ValueError("Native dense applicability evidence changed or is unsealed: " + name)
        return json.loads(path.read_bytes())

    role = "B0" if baseline_deployment else "B1"
    if baseline_deployment:
        source_root = Path(baseline_deployment).resolve()
        capacity = qualify_original_capacity(source_root, native, repetition=repetition)
        inventory = read("service/baseline-source-admission.json")["files"]
    else:
        source_root = deployment
        capacity = qualify_current_capacity(deployment, native, repetition=repetition)
        inventory = json.loads((deployment / "provenance/installed-files.json").read_bytes())
    for name, pin in protocol["sources"][role].items():
        registered = inventory.get(name)
        if role == "B0":
            registered = (registered or {}).get("sha256")
        if registered != pin or sha(source_root / name) != pin:
            raise ValueError("Reviewed dense dependency implementation differs: " + name)
    config = deployment / "inputs/model-metadata/config.json"
    if sha(config) != protocol["model_config_sha256"]:
        raise ValueError("Dense canonical architecture metadata differs")
    native_config = Path(read("service/native-plan.json")["model"]) / "config.json"
    if sha(native_config) != sha(config):
        raise ValueError("Native checkpoint architecture differs from frozen dense metadata")
    text = json.loads(config.read_bytes())["text_config"]
    arguments = read("service/client/effective-engine-args.json")
    geometry = read(f"service/client/formal-{repetition}/public-output-geometry.json")
    # Rebuild from the ordinary raw public records, independently of the
    # stage's already-derived geometry file. No private worker data is used.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from native_output_geometry import geometry as rebuild_geometry

    workload = read("service/token-workload.json")
    raw = read(f"service/client/formal-{repetition}/raw-public-output-frames.json")
    plan = read("service/native-plan.json")
    from reference_capture_profile import validate_native_plan

    capture = validate_native_plan(plan)
    if geometry != rebuild_geometry(raw, workload["request_token_ids"], maximum=capture["max_tokens"]):
        raise ValueError("Sealed native geometry differs from independent public-output replay")
    tail, cases = validate_domain(text, arguments, geometry, capacity, protocol)
    return dict(schema="qualified-canonical-dense-tail-source-domain/v1", role=role,
        native_checksums_sha256=sha(seal_path), protocol_sha256=PROTOCOL_SHA256,
        capacity=capacity, tail=tail, cases=cases, model_text_config=text,
        retained_source_sha256=protocol["sources"][role], reviewed_source_claims=protocol["claims"],
        source_runtime_applicability_certified=True, reference_is_lower_bound=True,
        calibration_and_independent_audit_required=True, full_model_certificate=False,
        efficiency_percent=None)
