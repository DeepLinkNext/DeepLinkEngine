"""Load the fixed B0 profile and cross-check it against the restored originals."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "engines/b0-vllm-v018.json"
HLE_PROFILE = ROOT / "baseline-sources/hle-original018-serving.json"
HLE_PROFILE_SHA256 = "e230b515259d752f4d180f8c3d3808bebcfdba1df31434edaf1483847e5348f1"
VARIANTS = ("text", "multimodal")
ADMISSION = "b0-profile-admission.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(argv):
    """Canonical argv digest; the profile records this value per variant."""
    return hashlib.sha256(
        json.dumps(list(argv), separators=(",", ":")).encode()
    ).hexdigest()


def load():
    raw = PROFILE.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def expected(variant, kind="continuous"):
    """Reconstruct the argv the restored original must produce for this run."""
    data, _ = load()
    if variant not in VARIANTS:
        raise ValueError("Unknown B0 variant: " + repr(variant))
    if variant == "multimodal":
        if sha(HLE_PROFILE) != HLE_PROFILE_SHA256:
            raise ValueError("Original018 HLE serving profile changed")
        return list(json.loads(HLE_PROFILE.read_bytes())["argv"])
    argv = list(data["serving"]["variants"][variant]["argv"])
    if kind != "continuous":
        index = argv.index("--reasoning-parser")
        del argv[index : index + 2]
    return argv


def _manifests(deployment):
    if deployment is None:
        return None
    path = Path(deployment) / "deployment.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_bytes()).get("original_manifest_sha256")
    return value or None


def verify(
    argv,
    variant,
    kind="continuous",
    deployment=None,
    source_sha256=None,
    entry_sha256=None,
):
    """Refuse to start when the restored originals disagree with the profile.

    Returns the record written next to the service plan; it binds the profile
    sha256, the selected variant and the exact argv this run will use.
    """
    data, profile_sha = load()
    if data.get("role") != "B0":
        raise ValueError("B0 profile role changed")
    if variant not in VARIANTS:
        raise ValueError("Unknown B0 variant: " + repr(variant))
    from restore_b0_image import MANIFEST as IMAGE_MANIFEST

    if data["image"]["manifest_digest"] != IMAGE_MANIFEST:
        raise ValueError("B0 profile image digest differs from the restore pin")
    from baseline_admission import PINNED_SOURCE_MAP_SHA256, VERSIONS

    if data["sdk"]["versions"] != VERSIONS:
        raise ValueError("B0 profile SDK versions differ from the admission pins")
    if data["sdk"]["source_map_sha256"] != PINNED_SOURCE_MAP_SHA256:
        raise ValueError("B0 profile source map differs from the admission pin")
    from restore_b0_018 import PATCHES

    if data["sdk"]["text_only_overlay"]["patches"] != PATCHES:
        raise ValueError("B0 profile text-only patches differ from the restore pins")
    if sha(HLE_PROFILE) != HLE_PROFILE_SHA256:
        raise ValueError("Original018 HLE serving profile changed")
    multimodal = data["serving"]["variants"]["multimodal"]
    if multimodal["profile_sha256"] != HLE_PROFILE_SHA256:
        raise ValueError("B0 profile HLE reference differs from the frozen profile")
    if digest(expected("multimodal")) != multimodal["argv_sha256"]:
        raise ValueError("B0 profile HLE argv digest differs from its own argv")
    text = data["serving"]["variants"]["text"]
    if text["argv_count"] != len(text["argv"]):
        raise ValueError("B0 profile text argv count is inconsistent")
    if digest(text["argv"]) != text["argv_sha256"]:
        raise ValueError("B0 profile text argv digest differs from its own argv")
    actual = list(argv)
    required = expected(variant, kind)
    if actual != required:
        difference = next(
            (
                index
                for index, (left, right) in enumerate(zip(actual, required))
                if left != right
            ),
            min(len(actual), len(required)),
        )
        raise ValueError(
            "Restored B0 argv differs from "
            + str(PROFILE.relative_to(ROOT))
            + " at index "
            + str(difference)
            + ": "
            + repr(actual[difference : difference + 2])
            + " != "
            + repr(required[difference : difference + 2])
        )
    if "--enforce-eager" in actual:
        raise ValueError("B0 cannot use eager mode")
    originals = data["serving"]["originals"]["files"]
    if variant == "text" and source_sha256 is not None:
        if source_sha256 != originals["runtime/entrypoints/prepare_retrieval_b0.py"][
            "restored_sha256"
        ]:
            raise ValueError("Restored B0 text argv source differs from the profile")
    if entry_sha256 is not None:
        if entry_sha256 != originals["runtime/entrypoints/serve_b0_r2.py"][
            "restored_sha256"
        ]:
            raise ValueError("Restored B0 serving entry differs from the profile")
    archives = data["serving"]["originals"]["archives"]
    for item in archives:
        if item["asset_sha256"] == item["manifest_sha256"]:
            raise ValueError("B0 profile must separate asset and manifest hashes")
        if not item["manifest_member"].endswith(".json"):
            raise ValueError("B0 profile manifest member is not a JSON manifest")
        if item["asset"] not in item["url"]:
            raise ValueError("B0 profile archive URL does not name its asset")
    manifests = _manifests(deployment)
    recorded = {item["asset"]: item["manifest_sha256"] for item in archives}
    if manifests is not None and manifests != recorded:
        raise ValueError("Restored original manifests differ from the B0 profile")
    return {
        "schema": "vllm-hust-b0-profile-admission/v1",
        "profile": str(PROFILE.relative_to(ROOT)),
        "profile_sha256": profile_sha,
        "variant": variant,
        "kind": kind,
        "argv_count": len(actual),
        "argv_sha256": digest(actual),
        "argv": actual,
        "original_manifests": manifests,
        "original_entry": data["serving"]["originals"]["files"][
            "runtime/entrypoints/serve_b0_r2.py"
        ],
        "argv_source_sha256": source_sha256,
        "serving_entry_sha256": entry_sha256,
        "checks": [
            "image manifest digest",
            "SDK package versions",
            "SDK file map sha256",
            "text-only patch sha256",
            "HLE profile and its argv digest",
            "text argv and its digest",
            "restored original manifests",
            "restored argv source sha256",
            "restored serving entry sha256",
            "no eager mode",
        ],
        "scope": "Cross-check only; this record does not promote B0 or reinterpret historical results.",
    }


def save(out, value):
    path = Path(out) / ADMISSION
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def attach(receipt, service_out):
    """Add the profile identity to the stage receipt when the service sealed one."""
    folder = Path(service_out)
    admission = folder / ADMISSION
    plan = folder / "plan.json"
    if not admission.is_file() or not plan.is_file():
        return receipt
    value = json.loads(admission.read_bytes())
    final = json.loads(plan.read_bytes())["serve_args"]
    receipt["b0_profile"] = {
        "admission": "service/" + ADMISSION,
        "admission_sha256": sha(admission),
        "profile": value["profile"],
        "profile_sha256": value["profile_sha256"],
        "variant": value["variant"],
        "kind": value["kind"],
        "argv_count": value["argv_count"],
        "argv_sha256": value["argv_sha256"],
        "final_argv_count": len(final),
        "final_argv_sha256": digest(final),
    }
    return receipt
