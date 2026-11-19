"""Validate the published benchmark copies, replay slices, and r8 archive identity."""

import argparse
import hashlib
import json
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_bytes())


def verify_hash(path, record):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != record["sha256"] or (
        "bytes" in record and path.stat().st_size != record["bytes"]
    ):
        raise SystemExit(f"Checksum or size mismatch: {path}")


def verify_replays():
    protocol_path = ROOT / "datasets/public-agent-trace-fixed32-v1.json"
    protocol = read_json(protocol_path)
    workloads = {
        "swe": ("swe-bench-pro-fixed32.json", 64),
        "terminal": ("terminal-bench-2.1-fixed32.json", 48),
    }
    for name, (filename, request_count) in workloads.items():
        path = ROOT / "datasets/individual" / filename
        manifest = read_json(path)
        if (path.parent / manifest["shared_protocol"]).resolve() != protocol_path:
            raise SystemExit(f"Replay protocol location mismatch: {filename}")
        verify_hash(protocol_path, {"sha256": manifest["shared_protocol_sha256"]})
        selection = protocol["datasets"][name]
        if (
            manifest["selection"] != selection
            or manifest["source"] != protocol["provenance"][name]
            or len(selection["sources"]) != 32
            or manifest["planned_tasks"] != 32
            or len(selection["requests"]) != request_count
            or manifest["planned_requests"] != request_count
        ):
            raise SystemExit(f"Replay selection mismatch: {filename}")


def verify_window():
    data = read_json(ROOT / "evidence/mmlu-r8-window.json")
    baseline, optimized = (data["arms"][name] for name in ("B0", "B1"))

    def card_seconds_per_token(arm):
        return arm["formal_window_seconds"] * arm["occupied_cards"] / arm["output_tokens"]

    reduction = 100 * (1 - card_seconds_per_token(optimized) / card_seconds_per_token(baseline))
    gain = optimized["ai_core_weighted_mean_percent"] - baseline["ai_core_weighted_mean_percent"]
    if not isclose(data["same_rate_estimated_unit_cost_reduction_percent"], reduction):
        raise SystemExit("MMLU-Pro same-rate reduction mismatch")
    if not isclose(data["utilization_gain_percentage_points"], gain):
        raise SystemExit("MMLU-Pro utilization change mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, help="Local frozen r8 source archive")
    args = parser.parse_args()
    manifest = read_json(ROOT / "benchmark/provenance.json")
    for item in manifest["files"]:
        verify_hash(ROOT / item["path"], item)

    lock = read_json(ROOT / "engine/r8-lock.json")
    assets = {item["name"]: item for item in lock["assets"]}
    requirements = "r8-pip-requirements-post2.txt"
    verify_hash(ROOT / "engine" / requirements, assets[requirements])
    if args.engine:
        archive = "vllm_hust_engine-2026.11.17.8.post2.tar.gz"
        verify_hash(args.engine, assets[archive])

    verify_replays()
    verify_window()
    print("PASS: benchmark copies, replay selections, frozen requirements, and MMLU-Pro window arithmetic")
    if args.engine:
        print("PASS: frozen r8 archive size and SHA-256")


if __name__ == "__main__":
    main()
