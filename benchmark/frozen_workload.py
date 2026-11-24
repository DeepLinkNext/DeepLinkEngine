"""Load the frozen cases and measure requests through the selected inference endpoints."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from benchmark import profile_streaming_transport as transport
from benchmark.benchmark_io import load, request, save


def canonical_mmlu(inputs):
    ids = load(inputs / "verified-mmlu-quality-inputs.json")["cases"]
    requests = load(inputs / "mmlu-quality018-r2/quality-request-plan.json")["requests"]
    return "".join(
        json.dumps({"case_id": case["case_id"], "body": body}, ensure_ascii=False, separators=(",", ":")) + "\n"
        for case, body in zip(ids, requests, strict=True)
    ).encode()


def load_cases(config, site):
    if "requests" in site:
        raw = Path(site["requests"]).read_bytes()
    else:
        from benchmark.dataset_bundle import prepare
        if config.get("dataset") != "mmlu-pro":
            raise ValueError("Supply the frozen request file for this workload")
        assets = prepare(Path(site["data_cache"]), ["mmlu"])
        raw = canonical_mmlu(assets / "mmlu/inputs")
    if hashlib.sha256(raw).hexdigest() != config["request_sha256"]:
        raise ValueError("Input file digest differs")
    cases = [json.loads(line) for line in raw.splitlines()]
    if len(cases) != config["requests"] or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Input population differs")
    return cases


def measure(cases, config, folder, endpoints, program_release):
    def send(item):
        index, case = item
        endpoint = endpoints[index % len(endpoints)]
        identity = f'{folder.name}-{case["case_id"]}'
        body = dict(case["body"])
        body.pop("max_completion_tokens", None)
        body.update(max_tokens=config["max_tokens"], cache_salt=identity)
        if program_release:
            body["program_id"] = identity
        path = folder / hashlib.sha256(identity.encode()).hexdigest()
        headers = {"x-delivery-request-id": identity, "x-session-id": identity}
        row = transport.call({"base_url": endpoint, "timeout": None, "trace_headers": headers}, body, path)
        row.update(case_id=case["case_id"], endpoint=endpoint)
        save(path / "result.json", row)
        if program_release:
            release = request(endpoint + "/programs/release", {"program_id": identity}, timeout=None)
            save(path / "release.json", release)
            if release.get("released") is not True:
                raise RuntimeError("Program release failed: " + identity)
        return row["status"] == "ok"
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
        successful = sum(pool.map(send, enumerate(cases)))
    summary = dict(planned=len(cases), successful=successful, elapsed_seconds=time.monotonic()-started)
    save(folder / "measurement.json", summary)
    if successful != len(cases):
        raise RuntimeError("Incomplete workload")
