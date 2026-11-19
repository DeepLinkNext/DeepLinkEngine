"""Capture native NPU operations and synchronized inference for one live M01 phase."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from uuid import uuid4


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def load_backend(path):
    spec = importlib.util.spec_from_file_location("live_m01_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fence(backend, cards):
    # The backend must fence every real worker rank, including async copies and collectives.
    result = backend.fence()
    if sorted(result) != sorted(cards):
        raise ValueError("Fence did not cover the two leased physical cards")


def measure(backend, cards, action):
    fence(backend, cards)
    start = time.monotonic_ns()
    payload = action()
    fence(backend, cards)
    end = time.monotonic_ns()
    return {
        "measurement_id": uuid4().hex, "start_ns": start, "end_ns": end,
        "synchronized": True, "cards": cards,
    }, payload


def account(raw):
    components = raw["components"]
    if not isinstance(components, list) or not components:
        raise ValueError("Native backend did not capture effective work")
    names = set()
    flops = 0
    for item in components:
        name, count = item["component"], item["effective_flops"]
        if not isinstance(name, str) or not name or name in names or type(count) is not int or count < 0:
            raise ValueError("Invalid or duplicate native effective-work component")
        names.add(name)
        flops += count
    if flops <= 0 or not raw["work_identity"]:
        raise ValueError("No positive, identifiable effective work")
    return {"components": sorted(components, key=lambda row: row["component"]), "effective_flops": flops,
            "work_signature_sha256": canonical_sha(raw["work_identity"]),
            "work_identity": raw["work_identity"]}


def run(request, out):
    phase = request["phase"]
    cards = request["cards"]
    backend = load_backend(Path(request["backend"])).open_backend(
        config=request["config"], workload=Path(request["workload"]),
        output=out, phase=phase, cards=cards,
    )
    try:
        identity = backend.identity()
        if (phase != "observe" or request["runtime_identity"] is not None) and identity != request["runtime_identity"]:
            raise ValueError("Imported native runtime identity differs from the frozen identity")
        write(out / "runtime-identity.json", identity)
        backend.warmup()
        if phase == "operators":
            observations = []
            for operation in request["operations"]:
                samples = []
                for _ in range(request["samples"]):
                    interval, _ = measure(
                        backend, cards, lambda: backend.run_operation(operation["operation"])
                    )
                    samples.append(interval)
                observations.append(dict(operation, observations=samples))
            data = {"operations": observations}
        else:
            interval, payload = measure(backend, cards, backend.run_batch)
            data = {"interval": interval, "request_identity": backend.request_identity(payload)}
            if phase == "observe":
                data["partial_work"] = backend.account(payload)
            elif phase == "timing":
                data["work"] = account(backend.account_timing(payload))
                describe = getattr(backend, "measurement_method", None)
                if describe:
                    data["measurement_method"] = describe()
            else:
                data["work"] = account(backend.account(payload))
            graph_phases = ["target", "reference-work"]
            if not request.get("separate_reference_trace"):
                graph_phases.append("validation")
            if phase in graph_phases and request["estimator"] == "critical_path":
                data["nodes"] = backend.dependency_graph(payload)
    finally:
        backend.close()
    write(out / "observation.json", dict(data, phase=phase, capture_id=request["capture_id"],
                                        workload_sha256=request["workload_sha256"], runtime_identity=identity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(json.loads(args.request.read_bytes()), args.output)

