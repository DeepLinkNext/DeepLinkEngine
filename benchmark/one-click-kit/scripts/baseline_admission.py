"""Admit the original018 image SDK and exact text/MM variant before NPU imports."""

import json
from pathlib import Path
from baseline_image_sdk import (
    verify_image,
    file_map,
    map_digest,
    probe,
    python,
    environment,
)

PINNED_SOURCE_MAP_SHA256 = (
    "a357ad8cf1ff6f8797fb2d6476e71a0deef5ee0899a55ea32ed65f07fe7773bc"
)
VERSIONS = {
    "vllm": "0.18.0+empty",
    "vllm-ascend": "0.18.0",
    "torch": "2.9.0+cpu",
    "torch-npu": "2.9.0.post1+gitee7ba04",
    "transformers": "4.57.6",
    "triton-ascend": "3.2.0.dev20260322",
}


def admit(deployment, out, variant="text"):
    root = Path(deployment).resolve()
    verify_image(root)
    stored = json.loads((root / "sdk-files.json").read_bytes())
    if map_digest(stored) != PINNED_SOURCE_MAP_SHA256:
        raise ValueError(
            "Original018 SDK inventory differs from the externally pinned image/backport recipe"
        )
    actual = file_map(root)
    if actual != stored:
        changed = sorted(
            name
            for name in set(actual) | set(stored)
            if actual.get(name) != stored.get(name)
        )
        raise ValueError("Original018 SDK source changed: " + repr(changed[:8]))
    identity = probe(root, variant)
    if identity["versions"] != VERSIONS or identity["no_site"] != 1:
        raise ValueError(
            "Actual original018 image package family differs: "
            + repr(identity["versions"])
        )
    env = environment(root, variant)
    result = {
        "role": "B0",
        "deployment": str(root),
        "variant": variant,
        "python": str(python(root)),
        "versions": identity["versions"],
        "origins": identity["origins"],
        "cann_home": env["ASCEND_HOME_PATH"],
        "source_map_sha256": PINNED_SOURCE_MAP_SHA256,
        "source_files_verified": len(actual),
        "files": actual,
        "adaptations": [
            "Verified image path relocation; original text-only backports for text profile; original unmodified multimodal profile for HLE"
        ],
        "historical_sdk_byte_equivalence": "Pinned image SDK plus explicitly original backports; host OS/driver/compiler are separately observed",
    }
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result


def validate_identity(actual, admitted):
    if not (
        actual.get("role") == "B0"
        and actual.get("vllm_version") == actual.get("ascend_version") == "0.18.0"
    ):
        raise ValueError("Actual B0 family differs from admitted018 SDK")
    for observed, package in (
        ("vllm_source", "vllm"),
        ("ascend_source", "vllm_ascend"),
    ):
        if (
            not actual.get(observed)
            or Path(actual[observed]).resolve()
            != Path(admitted["origins"][package]).resolve()
        ):
            raise ValueError(
                "Actual B0 import origin differs from admitted source: " + package
            )
    if (
        admitted.get("cann_home")
        and Path(actual.get("cann_home", "/missing")).resolve()
        != Path(admitted["cann_home"]).resolve()
    ):
        raise ValueError("Actual B0 CANN differs from original8.5.1 image")
