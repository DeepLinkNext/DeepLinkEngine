#!/usr/bin/env python3
"""Verify the actual archives pinned by the checked-in downloaders; CPU only."""

import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
from evaluation_contract import load, verify  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = []
    with tempfile.TemporaryDirectory(prefix="engine-pinned-tools-") as tmp:
        tmp = Path(tmp)
        for path, profile in (
            ("download_and_run.py", "r6"),
            ("runners/r8/download_and_run.py", "r8"),
        ):
            module = load(ROOT / path, "pinned_bootstrap")
            tools = module.ensure(tmp / "cache")
            # The independent current contract also validates older published
            # bundles that predate the explicit contract file.
            reports.append(
                {
                    "runner": path,
                    "source_commit": module.COMMIT,
                    "asset_id": module.ASSET_ID,
                    "sha256": module.ASSET_SHA256,
                    "cases": verify(tools, ROOT / path, profile),
                }
            )
    out = ROOT / "reports/pinned-release-contract.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {"passed": True, "model_loaded": False, "archives": reports}, indent=2
        )
        + "\n"
    )
    print("PINNED_RELEASE_CONTRACT_OK: both downloaded archives and all entry routes")


if __name__ == "__main__":
    main()
