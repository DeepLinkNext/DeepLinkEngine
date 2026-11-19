"""Qualify a clean native run's own complete necessary-arithmetic interval."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from public_work_ledger import ledger
from qualify_dense_source import qualify as qualify_dense_domain
from audit_loaded_work_inventory import validate as validate_inventory

PROTOCOL = Path(__file__).resolve().parents[1] / "protocols/necessary-arithmetic-source-domain-v1.json"
PROTOCOL_SHA256 = "02ad91a7de2926214f255d66697beeef18b4b6f6d4ab07263a563051e9621429"
KIT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_read(root, name):
    root = Path(root).resolve()
    seal = json.loads((root / "checksums.json").read_bytes())
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root) or seal.get(name) != sha(path):
        raise ValueError("Necessary-work evidence changed or is unsealed: " + name)
    return json.loads(path.read_bytes())


def companion_binding(deployment, companion, protocol):
    deployment, companion = Path(deployment).resolve(), Path(companion).resolve()
    receipt = sealed_read(companion, "receipt.json")
    provenance = sealed_read(companion, "source-inputs.json")
    correspondence = sealed_read(companion, "public-producer-correspondence.json")
    loaded = sealed_read(companion, "loaded-inventory-audit.json")
    if (receipt["schema"] != "sealed-http-reference-independent-recomputation/v1"
            or receipt["completed"] is not True or receipt["model_inference_executed"] is not False
            or receipt["requests"] != 200 or correspondence["all_request_closures_contained"] is not True
            or len(correspondence["requests"]) != 200 or set(loaded) != {"0", "1"}
            or any(r["loaded_semantic_dimensions_and_dispatch_passed"] is not True for r in loaded.values())
            or provenance["model_config_sha256"] != protocol["model_config_sha256"]):
        raise ValueError("Complete independently audited companion source evidence required")
    original = Path(provenance["source"]).resolve()
    if (Path(receipt["source"]).resolve() != original
            or sha(original / "checksums.json") != provenance["source_checksums_sha256"]):
        raise ValueError("Companion original source seal changed")
    expected_auditors = {name: sha(KIT.parent / name) for name in (
        "scripts/audit_http_reference.py", "third-party-test-kit/proc_capture/m01_observer_work.py",
        "third-party-test-kit/audit_loaded_work_inventory.py",
        "third-party-test-kit/audit_public_work_correspondence.py",
        "third-party-test-kit/draft_producer_geometry.py", "third-party-test-kit/draft_state_closure.py",
        "scripts/resolved_http_inputs.py",
        "protocols/async-http-device-input-source-domain-v1.json")}
    if provenance["audit_source_sha256"] != expected_auditors:
        raise ValueError("Companion independent derivation used different audit sources")
    installed = json.loads((deployment / "provenance/installed-files.json").read_bytes())
    if sealed_read(original, "installed-files.json") != installed:
        raise ValueError("Companion source inventory differs from the native deployment")
    # Bind the actual loaded method/class source observations too. The two
    # ordinary Torch module containers and the shipped pure route observer are
    # the only source records outside this admitted SDK in this fixed family.
    observed_sources = {}
    text = json.loads((deployment / "inputs/model-metadata/config.json").read_bytes())["text_config"]
    for rank in (0, 1):
        inventory = sealed_read(original, f"service/raw-work/rank-{rank}-model-inventory.json")
        if validate_inventory(inventory, text) != loaded[str(rank)]:
            raise ValueError("Loaded semantic inventory audit does not independently reproduce")
        for role in ("target", "draft"):
            for filename, pin in inventory[role]["source_sha256"].items():
                path = Path(filename).resolve()
                if path.is_relative_to(deployment):
                    if installed.get(str(path.relative_to(deployment))) != pin:
                        raise ValueError("Loaded semantic source is outside the frozen SDK: " + filename)
                elif path.name == "m01_route_capture.py":
                    if sha(KIT / "proc_capture/m01_route_capture.py") != pin:
                        raise ValueError("Loaded route observer differs from the reviewed companion")
                elif not (path.name in ("module.py", "container.py")
                          and path.parent.parts[-3:] == ("torch", "nn", "modules")):
                    raise ValueError("Unreviewed external loaded semantic source: " + filename)
                if sha(path) != pin:
                    raise ValueError("Observed loaded semantic source changed: " + filename)
                observed_sources[filename] = pin
    return dict(companion_checksums_sha256=sha(companion / "checksums.json"),
        original_checksums_sha256=sha(original / "checksums.json"),
        loaded_source_sha256=observed_sources, all_request_closures_contained=True,
        companion_counts_or_times_used_for_native_work=False)


def qualify(deployment, native, companion, *, baseline_deployment=None, repetition=1):
    deployment, native = Path(deployment).resolve(), Path(native).resolve()
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Reviewed complete arithmetic source-domain proof changed")
    protocol = json.loads(PROTOCOL.read_bytes())
    for name, pin in protocol["formula_sha256"].items():
        if sha(KIT / name) != pin:
            raise ValueError("Reviewed necessary-work algorithm differs: " + name)
    dense = qualify_dense_domain(deployment, native, baseline_deployment=baseline_deployment, repetition=repetition)
    role = dense["role"]
    if role == "B0":
        source_root = Path(baseline_deployment).resolve()
        admitted = sealed_read(native, "service/baseline-source-admission.json")["files"]
    else:
        source_root = deployment
        admitted = json.loads((deployment / "provenance/installed-files.json").read_bytes())
    for name, pin in protocol["sources"][role].items():
        registered = admitted.get(name)
        if role == "B0":
            registered = (registered or {}).get("sha256")
        if registered != pin or sha(source_root / name) != pin:
            raise ValueError("Reviewed arithmetic/producer source differs: " + name)
    companion_proof = companion_binding(deployment, companion, protocol)
    geometry = sealed_read(native, f"service/client/formal-{repetition}/public-output-geometry.json")
    result = ledger(SimpleNamespace(**dense["model_text_config"]), geometry,
                    allow_recurrent_prompt=(role == "B0"))
    result.update(role=role, necessary_work_certified=True, source_runtime_applicability_certified=True,
        native_checksums_sha256=dense["native_checksums_sha256"], protocol_sha256=PROTOCOL_SHA256,
        reviewed_source_claims=protocol["claims"], source_sha256=protocol["sources"][role],
        formula_sha256=protocol["formula_sha256"], capacity=dense["capacity"], companion=companion_proof,
        interpretation="Complete canonical necessary basic-arithmetic interval for this native run's own outputs; terminal MTP and prompt segmentation uncertainty are retained. Reference time qualification is separate.")
    return result
