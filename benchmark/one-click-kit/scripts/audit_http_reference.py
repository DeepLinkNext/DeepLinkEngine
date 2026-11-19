"""Re-audit sealed HTTP raw work after all original model owners exited.

This creates a new derivation, preserving the original receipt, including a
possible producer-filename comparison failure. It performs no inference and
imports no device runtime. Every source raw file remains in its original seal.
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from dataset_contract import save, sha
from reference_observation import response_counts
from resolved_http_inputs import prepare, remove_input_aliases

KIT = Path(__file__).resolve().parents[1] / "third-party-test-kit"


def run(source, deployment, out):
    source, deployment, out = Path(source).resolve(), Path(deployment).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    seal_path = source / "checksums.json"
    seal = json.loads(seal_path.read_bytes())

    def read(name):
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source) or seal.get(name) != sha(path):
            raise ValueError("HTTP original evidence changed or is unsealed: " + name)
        return json.loads(path.read_bytes())

    receipt = read("receipt.json")
    recoverable = {"ValueError: The two TP ranks disagree on real draft ownership/geometry",
                   "ValueError: Draft consumer reused or mismatched its latest actual producer"}
    if (receipt.get("quiescent") is not True or receipt.get("role") != "B1"
            or not (receipt.get("completed") is True or receipt.get("failure") in recoverable)):
        raise ValueError("Only complete or specifically recoverable quiescent HTTP observations are admitted")
    if read("deployment.json") != json.loads((deployment / "deployment.json").read_bytes()):
        raise ValueError("HTTP observation belongs to another frozen deployment")
    if read("installed-files.json") != json.loads((deployment / "provenance/installed-files.json").read_bytes()):
        raise ValueError("HTTP observation SDK source inventory differs")
    begin = read("service/raw-work/capture-begin.json")
    stop = read("service/raw-work/capture-stop.json")
    if (begin["completed"] is not True or stop["completed"] is not True
            or begin["window"] != stop["window"] or stop.get("error")
            or {w["rank"] for w in stop["workers"]} != {0, 1}
            or any(w["completed"] is not True or w["window"] != stop["window"] for w in stop["workers"])
            or any(w["sealed"] is not True or w["failure"] is not None for w in stop["compile_after"])):
        raise ValueError("HTTP capture did not close on both workers and scheduler under the compile boundary")
    for name in sorted(seal):
        if name.startswith("service/case-") and (name.endswith(".json") or name.endswith(".sse.jsonl.gz")):
            if sha(source / name) != seal[name]:
                raise ValueError("Original HTTP responses changed: " + name)
    requests = response_counts(source / "service")
    if len(requests) != 200 or sum(pair[0] for pair in requests.values()) != 282265:
        raise ValueError("Companion prompt population differs from current fixed200")
    inputs = {}
    for name, expected in seal.items():
        path = Path(name)
        if path.parent != Path("service/raw-work"):
            continue
        if not (path.name.startswith("rank-") and (
                path.name.endswith("-waves.jsonl") or path.name.endswith("-drafts.jsonl")
                or path.name.endswith("-routes.npz")) or path.name == "scheduler-waves.jsonl"):
            continue
        target = source / path
        if target.is_symlink() or not target.resolve().is_relative_to(source):
            raise ValueError("Unsafe original raw-work input: " + name)
        # Route arrays are independently hashed against the admitted metadata
        # by the route auditor. Bind the record streams before parsing them.
        if path.suffix != ".npz" and sha(target) != expected:
            raise ValueError("Original HTTP observer record changed: " + name)
        inputs[name] = expected
    sys.path[:0] = [str(KIT / "proc_capture"), str(KIT)]
    from m01_observer_work import collect
    from audit_loaded_work_inventory import validate
    from audit_public_work_correspondence import audit as audit_public

    for rank in (0, 1):
        read(f"service/raw-work/rank-{rank}-proposer-source.json")
    resolved = out / "resolved-inputs"
    normalization = prepare(source / "service/raw-work", deployment, resolved)
    text = json.loads((deployment / "inputs/model-metadata/config.json").read_bytes())["text_config"]
    counts = dict(scheduler=stop["counts"]["scheduler"])
    counts.update({rank: stop["counts"][str(rank)] for rank in (0, 1)})
    partial = collect(resolved, stop["window"], counts, {}, [], SimpleNamespace(**text),
        request_counts=requests, draft_counts={w["rank"]: w["draft_calls"] for w in stop["workers"]})
    inventories = {str(rank): validate(read(f"service/raw-work/rank-{rank}-model-inventory.json"), text) for rank in (0, 1)}
    def lines(name):
        with (resolved / name).open() as stream:
            return [json.loads(line) for line in stream]
    correspondence = audit_public(partial["requests"], lines("scheduler-waves.jsonl"),
        lines("rank-0-drafts.jsonl"), lines("rank-1-drafts.jsonl"))
    save(out / "public-producer-correspondence.json", correspondence)
    save(out / "loaded-inventory-audit.json", inventories)
    save(out / "raw-partial-work.json", partial)
    save(out / "source-inputs.json", dict(source=str(source), source_checksums_sha256=sha(seal_path),
        original_receipt_sha256=sha(source / "receipt.json"), raw_inputs=inputs,
        audit_source_sha256={name: sha(Path(__file__).resolve().parents[1] / name) for name in (
            "scripts/audit_http_reference.py", "third-party-test-kit/proc_capture/m01_observer_work.py",
            "third-party-test-kit/audit_loaded_work_inventory.py",
            "third-party-test-kit/audit_public_work_correspondence.py",
            "third-party-test-kit/draft_producer_geometry.py", "third-party-test-kit/draft_state_closure.py",
        "scripts/resolved_http_inputs.py",
            "protocols/async-http-device-input-source-domain-v1.json")},
        model_config_sha256=sha(deployment / "inputs/model-metadata/config.json")))
    result = dict(schema="sealed-http-reference-independent-recomputation/v1", completed=True,
        model_inference_executed=False, source=str(source), original_receipt_preserved=True,
        original_completed=receipt["completed"], original_failure=receipt.get("failure"),
        requests=len(requests), all_request_closures_contained=True,
        async_input_derivation_sha256=sha(resolved / "async-input-derivation.json"),
        resolved_placeholder_consumers=normalization["resolved_placeholder_consumers"],
        loaded_semantic_dimensions_and_dispatch_passed=True, necessary_work_certified=False,
        full_model_certificate=False, efficiency_percent=None)
    save(out / "receipt.json", result)
    # The input view contains only references to original sealed raw bytes;
    # derived files are copied into this new seal without changing originals.
    remove_input_aliases(resolved)
    for path in sorted(resolved.iterdir()):
        if not path.is_symlink():
            if path.name not in ("scheduler-waves.jsonl", "rank-0-waves.jsonl", "rank-1-waves.jsonl", "rank-0-drafts.jsonl", "rank-1-drafts.jsonl", "async-input-derivation.json"):
                path.rename(out / path.name)
    save(out / "checksums.json", {str(p.relative_to(out)): sha(p) for p in out.rglob("*")
         if p.is_file() and not p.is_symlink() and p.name != "checksums.json"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "deployment", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.deployment, args.out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
