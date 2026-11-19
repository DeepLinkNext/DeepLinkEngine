"""Explicit r5 overlay; never mutate or relabel an r4 deployment."""

import argparse
import json
from pathlib import Path
import shutil

RELEASE = "full-answer-b1-20261113-r5"
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "engines/b1-r5-overlay.json"
HOST = "mods/microflow/hust_prefill_microflow/school028/host-contract.json"


LATEST_RELEASE = "full-answer-b1-20261113-r6"
R7_RELEASE = "full-answer-b1-20261116-r7"
R8_RELEASE = "full-answer-b1-20261117-r8"
RELEASES = {"r4": "full-answer-b1-20261030-r4", "r5": RELEASE, "r6": LATEST_RELEASE, "r7": R7_RELEASE, "r8": R8_RELEASE}


def manifest_path(release):
    if release == RELEASE:
        return MANIFEST
    if release == LATEST_RELEASE:
        return ROOT / "engines/b1-r6-overlay.json"
    if release == R7_RELEASE:
        return ROOT / "engines/b1-r7-overlay.json"
    if release == R8_RELEASE:
        return ROOT / "engines/b1-r8-overlay.json"
    raise ValueError("Unknown runtime overlay release")


def overlay_root(release):
    names = {RELEASE: "b1-r5", LATEST_RELEASE: "b1-r6", R7_RELEASE: "b1-r7", R8_RELEASE: "b1-r8"}
    return ROOT / "overlays" / names[release]


def manifest(release=RELEASE):
    from restore_b1_r4 import sha

    value = json.loads(manifest_path(release).read_bytes())
    if value["release"] != release or len(value["files"]) != (10 if release == R8_RELEASE else 9 if release == R7_RELEASE else 5):
        raise ValueError("Unknown runtime overlay")
    for name, pin in value["files"].items():
        p = (overlay_root(release) / name).resolve()
        if not p.is_relative_to(overlay_root(release).resolve()):
            raise ValueError("Unsafe overlay path")
        if sha(p) != pin["after_sha256"]:
            raise ValueError("Frozen overlay source changed: " + name)
    return value


def original_host(raw, value):
    """Undo only declared source pins; original date-label check still applies."""
    if (
        __import__("hashlib").sha256(raw).hexdigest()
        != value["files"][HOST]["after_sha256"]
    ):
        raise ValueError("r5 host contract differs from frozen overlay")
    data = json.loads(raw)
    pins = data["sources"]["betterscale"]
    for name, pin in value["files"].items():
        if not name.startswith("mods/betterscale/betterscale/"):
            continue
        key = name.removeprefix("mods/betterscale/betterscale/")
        if pin["before_sha256"] is None:
            del pins[key]
        else:
            pins[key] = pin["before_sha256"]
    return (json.dumps(data, indent=2) + "\n").encode()


def apply(out, release=RELEASE):
    from restore_b1_r4 import sha

    value = manifest(release)
    profile = json.loads((out / "deployment.json").read_bytes())
    source_release = profile["release"]
    prior = manifest(source_release) if release in (R7_RELEASE, R8_RELEASE) and source_release in (LATEST_RELEASE, R7_RELEASE) and source_release != release else None
    if source_release != value["base_release"] and prior is None:
        raise ValueError("Overlay requires the admitted r4 base, or r6 for the r7 upgrade")
    bindings = json.loads((out / "provenance/installed-files.json").read_bytes())
    for name, pin in value["files"].items():
        expected_base = prior["files"][name]["after_sha256"] if prior and name in prior["files"] else pin["before_sha256"]
        if bindings.get(name) != expected_base:
            raise ValueError("Overlay base binding differs: " + name)
        if (out / name).exists() and sha(out / name) != expected_base:
            raise ValueError("Overlay base file differs: " + name)
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(overlay_root(release) / name, out / name)
        bindings[name] = pin["after_sha256"]
    profile.update(
        release=release,
        runtime_overlay_sha256=sha(manifest_path(release)),
        model_started=False,
        scope="r4 composition plus explicitly pinned warmup fixes; fresh native qualification required",
    )
    if release == R8_RELEASE:
        from runtime_geometry import bind_profile

        bind_profile(profile)
        profile["scope"] = "C15 measured runtime: native guarded async copy, E36/R54, existing short-prefill-tail scheduling; same kernels and4GiB/rank budget"
    if prior is None:
        profile["host_contract_qualification"]["host_source_pins_verified"] += 2
    changes = json.loads((out / "provenance/deployment-changes.json").read_bytes())
    changes.append(value)
    (out / "provenance/deployment-changes.json").write_text(
        json.dumps(changes, indent=2) + "\n"
    )
    (out / "deployment.json").write_text(json.dumps(profile, indent=2) + "\n")
    (out / "provenance/installed-files.json").write_text(
        json.dumps(bindings, indent=2) + "\n"
    )
    shutil.copyfile(manifest_path(release), out / "provenance/runtime-overlay.json")
    from restore_b1_r4 import verify_installed_sources

    verify_installed_sources(out)
    return profile


def clone(source, out, release=RELEASE):
    from restore_b1_r4 import verify_installed_sources, PYTHONPATH_LAYOUT

    source, out = source.resolve(), out.resolve()
    verify_installed_sources(source)
    if out.exists():
        raise FileExistsError(out)
    shutil.copytree(
        source, out, ignore=shutil.ignore_patterns("__pycache__", "launch-host")
    )
    profile = json.loads((out / "deployment.json").read_bytes())
    profile.update(root=str(out), pythonpath=[str(out / n) for n in PYTHONPATH_LAYOUT])
    (out / "deployment.json").write_text(json.dumps(profile, indent=2) + "\n")
    return apply(out, release)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r4-deployment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine-profile", choices=["r5", "r6", "r7", "r8"], default="r6")
    args = parser.parse_args()
    print(json.dumps(clone(args.r4_deployment, args.out, RELEASES[args.engine_profile]), indent=2))
