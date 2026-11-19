#!/usr/bin/env python3
"""One command for the original B1 C8 prerequisite and joint200, with reports."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def reproduce(args):
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "vllm-hust-engine-b1-campaign/v1",
        "scope": "Original B1 C8 and joint200 only; not the complete four-target acceptance suite",
        "planned": ["continuous", "joint"],
        "attempted": [],
        "completed": False,
        "formal_complete": False,
    }
    code = 2
    try:
        for kind in receipt["planned"]:
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_b1_r4.py")),
                "--deployment",
                str(args.deployment.resolve()),
                "--out",
                str(root / kind),
                "--devices",
                args.devices,
                "--model",
                str(args.model.resolve()),
                "--kind",
                kind,
            ]
            if args.report_python:
                command += ["--report-python", str(args.report_python.absolute())]
            if kind == "joint":
                command += ["--qualified-continuous", str(root / "continuous")]
            result = subprocess.run(command, check=False)
            receipt["attempted"].append({"kind": kind, "returncode": result.returncode})
            if result.returncode:
                return 2
        receipt["completed"] = True
        code = 0
    finally:
        receipt["unexecuted"] = receipt["planned"][len(receipt["attempted"]) :]
        receipt["reports"] = [
            str(path.relative_to(root)) for path in root.glob("*-report/report.html")
        ]
        (root / "campaign.json").write_text(json.dumps(receipt, indent=2) + "\n")
        # Bind immutable child inventories, not writable compiler caches.
        bound = [root / "campaign.json"]
        for kind in receipt["planned"]:
            bound += [
                root / kind / "checksums.json",
                root / (kind + "-report") / "checksums.json",
            ]
        hashes = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in bound
            if p.is_file()
        }
        (root / "campaign-checksums.json").write_text(
            json.dumps(hashes, indent=2) + "\n"
        )
        print("CAMPAIGN", root / "campaign.json", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--model", type=Path, default=Path("/models/Qwen3.5-35B-A3B"))
    parser.add_argument("--report-python", type=Path)
    return reproduce(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
