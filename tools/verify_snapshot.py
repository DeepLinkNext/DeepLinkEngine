"""Validate the published benchmark copies, replay slices, and r8 archive identity."""

import argparse
import hashlib
import json
import tarfile
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


def verify_engine(path, assets):
    archive = "vllm_hust_engine-2026.11.17.8.post2.tar.gz"
    verify_hash(path, assets[archive])
    package_root = archive.removesuffix(".tar.gz") + "/src/vllm_hust_engine/payload/"
    with tarfile.open(path) as source:
        metadata = json.load(source.extractfile(package_root + "frozen.json"))
        if (
            metadata["release"] != "full-answer-b1-20261117-r8"
            or metadata["version"] != "2026.11.17.8.post2"
            or metadata["runtime_files"] != 9235
        ):
            raise SystemExit("Unexpected frozen r8 payload identity")
        for filename, record in metadata["archives"].items():
            member = source.getmember(package_root + filename)
            with source.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if member.size != record["bytes"] or digest != record["sha256"]:
                raise SystemExit(f"Internal r8 archive differs: {filename}")
        with source.extractfile(package_root + "tools-r4.tar.gz") as stream:
            verify_tool_protocols(stream, metadata["tools_manifest_sha256"])


def verify_tool_protocols(stream, expected_manifest):
    protocols = {
        "mmlu-fixed200.json",
        "hle-fixed200.json",
        "public-agent-trace-fixed32-v1.json",
        "public-agent-trace-source-selection-v1.json",
        "swe-agent-v110-tools.json",
    }
    root = "vllm-hust-engine/"
    with tarfile.open(fileobj=stream, mode="r:gz") as tools:
        manifest = tools.extractfile(root + "DEPLOYMENT-TOOLS.json").read()
        if hashlib.sha256(manifest).hexdigest() != expected_manifest:
            raise SystemExit("Frozen tool manifest differs")
        for filename in sorted(protocols):
            expected = (ROOT / "datasets" / filename).read_bytes()
            actual = tools.extractfile(root + "protocols/" + filename).read()
            if expected != actual:
                raise SystemExit(f"Packaged assessment protocol differs: {filename}")


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
    parser.add_argument("--tools", type=Path, help="Standalone r8 evaluation tools archive")
    args = parser.parse_args()
    manifest = read_json(ROOT / "benchmark/provenance.json")
    for item in manifest["files"]:
        verify_hash(ROOT / item["path"], item)

    lock = read_json(ROOT / "engine/r8-lock.json")
    assets = {item["name"]: item for item in lock["assets"]}
    requirements = "r8-pip-requirements-post2.txt"
    source_requirements = ROOT / "engine" / requirements
    verify_hash(source_requirements, assets[requirements])
    mirror_requirements = ROOT / "engine/r8-pip-requirements-deeplinkengine.txt"
    upstream = b"https://github.com/vLLM-HUST/vllm-hust-website/releases/expanded_assets/engine-wheels-r8-v1"
    mirror = b"https://github.com/DeepLinkNext/DeepLinkEngine/releases/expanded_assets/r8-frozen-20261117"
    if mirror_requirements.read_bytes() != source_requirements.read_bytes().replace(upstream, mirror):
        raise SystemExit("DeepLinkEngine installation requirements differ from the frozen requirements")
    if args.engine:
        verify_engine(args.engine, assets)
    if args.tools:
        tools = lock["evaluator"]
        verify_hash(args.tools, tools)
        with args.tools.open("rb") as stream:
            verify_tool_protocols(stream, tools["manifest_sha256"])

    verify_replays()
    verify_window()
    print("PASS: benchmark copies, replay selections, frozen requirements, and MMLU-Pro window arithmetic")
    if args.engine:
        print("PASS: frozen r8 SDK, evaluator, and five shared protocol files")
    if args.tools:
        print("PASS: standalone evaluator size, SHA-256, and shared protocol files")


if __name__ == "__main__":
    main()
