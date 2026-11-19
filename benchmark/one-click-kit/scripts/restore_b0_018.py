"""Apply the original text-only backports to the pinned018 image SDK."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from baseline_image_sdk import verify_image, probe, file_map, map_digest
from dataset_contract import sha

PATCHES = {
    "vllm-text-only-backports.patch": "4affc33e774ee06428479c15bc67fe9b3301930f169208e7577d8df6702e4ed1",
    "vllm-ascend-text-only-backport.patch": "2f108fb1fe36cd6263c1b1971a6bb8fc69eeecfb016a2c717fd616efbac672b8",
}


def prepare(root):
    root = root.resolve()
    verify_image(root)
    overlay = root / "text-only-overlay"
    if not overlay.exists():
        overlay.mkdir(exist_ok=False)
        for name in ("vllm", "vllm-ascend"):
            package = "vllm" if name == "vllm" else "vllm_ascend"
            shutil.copytree(
                root / "rootfs/vllm-workspace" / name / package,
                overlay / name / package,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        sources = Path(__file__).resolve().parents[1] / "baseline-sources"
        for name, expected in PATCHES.items():
            patch = sources / name
            if sha(patch) != expected:
                raise ValueError("Original text-only patch changed: " + name)
            target = overlay / (
                "vllm-ascend" if name.startswith("vllm-ascend") else "vllm"
            )
            subprocess.run(
                ["git", "apply", "--check", str(patch)],
                cwd=target,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "apply", str(patch)],
                cwd=target,
                check=True,
                capture_output=True,
            )
    variants = {name: probe(root, name) for name in ("text", "multimodal")}
    files = file_map(root)
    digest = map_digest(files)
    (root / "sdk-files.json").write_text(
        json.dumps(files, sort_keys=True, separators=(",", ":")) + "\n"
    )
    profile = {
        "schema": "vllm-hust-original018-sdk/v1",
        "variants": variants,
        "patches": PATCHES,
        "source_map_sha256": digest,
        "file_count": len(files),
        "native_qualification": False,
        "scope": "Original image Python/Torch/Torch-NPU/Triton/CANN8.5.1 restored. Text-only backports are the original benchmark patches; HLE uses original unmodified multimodal source. Host OS/driver/compiler remain separately observed.",
    }
    (root / "baseline.json").write_text(json.dumps(profile, indent=2) + "\n")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    args = parser.parse_args()
    data = prepare(args.deployment)
    print("ORIGINAL018_SDK_PREPARED", data["file_count"], data["source_map_sha256"])


if __name__ == "__main__":
    main()
