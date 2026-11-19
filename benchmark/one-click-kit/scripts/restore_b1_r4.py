#!/usr/bin/env python3
"""Restore the pinned Engine release into a clean, independently named tree."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile

from fetch_b1_r4 import fetch


FROZEN = "full-answer-frozen-sources-20261030-r4.tar.gz"
COMPONENTS = "b1-r4-recovered-components.tar.gz"
ORIGINAL_PLAN = "243c495d294e9a1a03c2cc4c3fca15d13ad5ead8a555b5f081f437f18e95a96f"
SOURCE_ROUTES = {
    "/root/gennova-evidence/20261019-midterm-aligned/": "inputs/",
    "/root/joint-fullanswer-20261030/": "mods/packed-host/",
    "/root/m01-storage-timing-runtime-20261024/": "mods/correction/",
    "/root/long-context-fixed018-20261029/": "runtime/entrypoints/",
    "/root/vllm-hust-benchmark/src/vllm_hust_benchmark/": "scorers/",
    "/root/fixed-b0-cost-20261029/": "config/",
    "/models/Qwen3.5-35B-A3B/": "inputs/model-metadata/",
}
RUNTIME_PREFIX = "workspace/m01-qualification-20261101/runtime-b1/"
COMPONENT_ROUTES = {
    "root/modcost-runtime-028-20261017/vllm-workspace/vllm/": "runtime/engine-core/",
    "root/extension-manager/src/": "manager/",
    "root/frontierscience-recovery-20261022/filtered-site/": "runtime/filtered-site/",
    "root/gennova-runtime/betterscale-dist/extracted/vllm_betterscale-0.5.1/src/": "runtime/betterscale-distribution/",
    "workspace/m01-qualification-20261101/b1/correction-plugin-installed-r1/": "mods/correction/correction-plugin-installed-r1/",
    RUNTIME_PREFIX + "runtime/": "runtime/ascend/",
    RUNTIME_PREFIX + "microflow-recovery-payload-r2/": "mods/microflow/",
    RUNTIME_PREFIX + "kvmat-managed-payload-r1/": "mods/kv-materialization/",
    RUNTIME_PREFIX + "recovery-model-payload-r1/": "mods/betterscale/",
    RUNTIME_PREFIX + "compat/": "runtime/compat/",
    RUNTIME_PREFIX + "build/": "native-libs/",
    RUNTIME_PREFIX + "recovery-fia-build-r1/artifact/": "native-libs/context-parallel/",
    RUNTIME_PREFIX + "manager-recovery-r2/": "config/",
    RUNTIME_PREFIX + "mmlu-b1-microflow-fa-on-r10/": "provenance/base-launch/",
    RUNTIME_PREFIX: "provenance/sdk/",
}

MANIFEST_SHA256 = {
    "source-manifest.json": "f357230c7ab23fe482de6b97e99eceefa4da23eec794731d1fa577c81c821cdb",
    "component-manifest.json": "8ae289c1a65d9b28ffed4c516d8f2f4d23edd3f061773bca8b188dc5d807319d",
}
ABI_SOURCE = "abi-sources/vllm_ascend/worker/model_runner_v1.py.source"
ABI_SHA256 = "ceabfc9db30c9d8bf8dc7c9bddf523c4a8d5ef04165994635dca3b94a8e929f4"
ABI_OLD = "SOURCE = Path('/root/modcost-runtime-028-20261017/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')"
ABI_NEW = "SOURCE = Path(__file__).resolve().parents[2] / 'abi-sources/vllm_ascend/worker/model_runner_v1.py.source'"
PYTHONPATH_LAYOUT = (
    "mods/packed-host",
    "mods/correction",
    "mods/correction/correction-plugin-installed-r1",
    "mods/microflow",
    "manager",
    "mods/kv-materialization",
    "mods/betterscale",
    "runtime/compat",
    "runtime/ascend",
    "runtime/engine-core",
    "runtime/filtered-site",
)


def verify_installed_sources(out):
    """Bind admission to pinned original manifests, not a self-declared file map."""
    out = out.resolve()
    profile = json.loads((out / "deployment.json").read_bytes())
    overlay = None
    if profile["release"] in ("full-answer-b1-20261113-r5", "full-answer-b1-20261113-r6", "full-answer-b1-20261116-r7", "full-answer-b1-20261117-r8"):
        from runtime_overlay import manifest, manifest_path

        overlay = manifest(profile["release"])
        if profile.get("runtime_overlay_sha256") != sha(manifest_path(profile["release"])):
            raise ValueError("Runtime overlay identity differs")
    if (
        profile["root"] != str(out)
        or profile["pythonpath"]
        != [str((out / name).resolve()) for name in PYTHONPATH_LAYOUT]
        or profile["release"]
        not in ("full-answer-b1-20261030-r4", "full-answer-b1-20261113-r5", "full-answer-b1-20261113-r6", "full-answer-b1-20261116-r7", "full-answer-b1-20261117-r8")
    ):
        raise ValueError("Deployment paths or release identity differ")
    from runtime_geometry import validate_profile

    validate_profile(profile)
    original = json.loads((out / "provenance/original-plan.json").read_bytes())
    if (
        hashlib.sha256(json.dumps(original, indent=2).encode()).hexdigest()
        != ORIGINAL_PLAN
    ):
        raise ValueError("Pinned original run plan changed")
    expected = {}
    for name, routes in (
        ("source-manifest.json", SOURCE_ROUTES),
        ("component-manifest.json", COMPONENT_ROUTES),
    ):
        path = out / "provenance" / name
        if sha(path) != MANIFEST_SHA256[name]:
            raise ValueError("Pinned source manifest changed: " + name)
        manifest = json.loads(path.read_bytes())
        files = manifest if name == "source-manifest.json" else manifest["files"]
        for original, digest in files.items():
            target = route(original, routes)
            if target in expected:
                raise ValueError("Duplicate original source route: " + target)
            expected[target] = digest
    helper = "mods/correction/pinned_metadata_signature.py"
    raw = (out / helper).read_bytes()
    if (
        raw.count(ABI_NEW.encode()) != 1
        or hashlib.sha256(raw.replace(ABI_NEW.encode(), ABI_OLD.encode())).hexdigest()
        != expected[helper]
    ):
        raise ValueError("ABI helper differs beyond its allowed path relocation")
    expected[helper] = hashlib.sha256(raw).hexdigest()
    contract = "mods/microflow/hust_prefill_microflow/school028/host-contract.json"
    raw = (out / contract).read_bytes()
    deployed_host_raw = raw
    if overlay:
        from runtime_overlay import original_host

        raw = original_host(raw, overlay)
    if hashlib.sha256(raw).hexdigest() != expected[contract]:
        value = json.loads(raw)
        historical = "0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909"
        if value["versions"]["vllm-ascend"] != historical.removesuffix(".d20260909"):
            raise ValueError("Unexpected host-contract version adaptation")
        value["versions"]["vllm-ascend"] = historical
        restored = (json.dumps(value, indent=2) + "\n").encode()
        if hashlib.sha256(restored).hexdigest() != expected[contract]:
            raise ValueError(
                "Host contract differs beyond its allowed date-label adaptation"
            )
        expected[contract] = hashlib.sha256(raw).hexdigest()
    expected[ABI_SOURCE] = ABI_SHA256
    if overlay:
        for name, pin in overlay["files"].items():
            if name == contract:
                if sha_bytes(deployed_host_raw) != pin["after_sha256"]:
                    raise ValueError("Overlay host contract differs")
            elif expected.get(name) != pin["before_sha256"]:
                raise ValueError("Overlay original source binding differs: " + name)
            expected[name] = pin["after_sha256"]
    claimed = json.loads((out / "provenance/installed-files.json").read_bytes())
    if claimed != expected:
        raise ValueError(
            "Installed file map does not derive from the pinned original manifests"
        )
    for name, digest in expected.items():
        path = (out / name).resolve()
        if not path.is_relative_to(out) or sha(path) != digest:
            raise ValueError("Installed source changed: " + name)
    return expected


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def unpack(path, destination):
    with tarfile.open(path) as archive:
        seen = set()
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or member.name in seen
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Unsafe archive member: " + member.name)
            seen.add(member.name)
        archive.extractall(destination, filter="data")


def route(name, routes):
    for prefix, target in routes.items():
        if name.startswith(prefix):
            return target + name[len(prefix) :]
    raise ValueError("No explicit deployment route for " + name)


def qualify_host_contract(out, python, bindings):
    """Retain every source pin; disclose the packaging-date-only label change."""
    name = "mods/microflow/hust_prefill_microflow/school028/host-contract.json"
    path = out / name
    contract = json.loads(path.read_bytes())
    roots = {
        "betterscale": out / "mods/betterscale/betterscale",
        "vllm": out / "runtime/engine-core/vllm",
        "vllm_ascend": out / "runtime/ascend/vllm_ascend",
    }
    checked = 0
    for package, files in contract["sources"].items():
        for relative, expected in files.items():
            target = (roots[package] / relative).resolve()
            if (
                not target.is_relative_to(roots[package].resolve())
                or sha(target) != expected
            ):
                raise ValueError(
                    "Microflow host source pin differs: " + package + "/" + relative
                )
            checked += 1
    code = (
        "import importlib.metadata,json;print(json.dumps({n:importlib.metadata.version(n) for n in "
        + repr(list(contract["versions"]))
        + "}))"
    )
    result = subprocess.run(
        [str(python), "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    observed = json.loads(result.stdout)
    changes = []
    for package, expected in contract["versions"].items():
        actual = observed[package]
        if actual == expected:
            continue
        if package != "vllm-ascend" or re.sub(r"[.]d[0-9]{8}$", "", expected) != actual:
            raise ValueError("Unqualified dependency metadata: " + package)
        changes.append(
            {
                "path": name,
                "kind": "packaging-date-label-only",
                "original_sha256": bindings[name],
                "package": package,
                "original_version": expected,
                "installed_version": actual,
                "host_source_pins_verified": checked,
                "scope": "Keep the real installed metadata; do not invent a build-date suffix. All original host source checks remain unchanged.",
            }
        )
        contract["versions"][package] = actual
    if changes:
        path.write_text(json.dumps(contract, indent=2) + "\n")
        bindings[name] = sha(path)
        for change in changes:
            change["deployed_sha256"] = bindings[name]
    return {
        "host_source_pins_verified": checked,
        "installed_metadata": observed,
        "changes": changes,
    }


def relocate_signature_source(out, bindings):
    name = ABI_SOURCE
    source = Path(__file__).resolve().parents[1] / name
    expected = ABI_SHA256
    if sha(source) != expected:
        raise ValueError(
            "Canonical Ascend ABI source differs from the frozen requirement"
        )
    destination = out / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    bindings[name] = expected
    helper_name = "mods/correction/pinned_metadata_signature.py"
    helper = out / helper_name
    before = bindings[helper_name]
    text = helper.read_text()
    old, new = ABI_OLD, ABI_NEW
    if text.count(old) != 1:
        raise ValueError("Unexpected frozen ABI helper layout")
    helper.write_text(text.replace(old, new))
    bindings[helper_name] = sha(helper)
    return {
        "kind": "canonical-abi-source-and-path-relocation",
        "path": helper_name,
        "original_sha256": before,
        "deployed_sha256": bindings[helper_name],
        "canonical_source": name,
        "canonical_source_sha256": expected,
        "canonical_repository": "vLLM-HUST/vllm-ascend-hust",
        "canonical_commit": "74f0c0a272376412b51e1c1864803d5f3a0f1b5f",
        "scope": "Read the exact original canonical ABI source at a deployment-relative path; preserve the original SHA check and the separately loaded runtime override.",
    }


def restore(assets, out, python, env_file=None):
    fetch(assets, env_file=env_file)
    if out.exists():
        raise FileExistsError("Deployment must use a new directory: " + str(out))
    out.mkdir(parents=True)
    bindings = {}
    origins = {}
    manifest_hashes = {}
    with tempfile.TemporaryDirectory(prefix="vllm-hust-engine-unpack-") as scratch:
        scratch = Path(scratch)
        for archive_name, manifest_name, prefix, routes in (
            (FROZEN, "source-manifest.json", "original-paths/", SOURCE_ROUTES),
            (COMPONENTS, "component-manifest.json", "source/", COMPONENT_ROUTES),
        ):
            extracted = scratch / archive_name
            unpack(assets / archive_name, extracted)
            manifest_path = extracted / manifest_name
            manifest = json.loads(manifest_path.read_bytes())
            files = manifest if archive_name == FROZEN else manifest["files"]
            expected_count = 255 if archive_name == FROZEN else 8975
            if len(files) != expected_count:
                raise ValueError("Unexpected original manifest size")
            manifest_hashes[archive_name] = sha(manifest_path)
            for original, expected in files.items():
                source = extracted / prefix / original.lstrip("/")
                if not source.is_file() or sha(source) != expected:
                    raise ValueError(
                        "Original file failed hash verification: " + original
                    )
                name = route(original, routes)
                destination = out / name
                if name in bindings:
                    raise ValueError("Deployment path collision: " + name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                destination.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)
                bindings[name] = expected
                origins[original] = {
                    "deployment_path": name,
                    "sha256": expected,
                    "archive": archive_name,
                }
            target = out / "provenance" / manifest_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(manifest_path, target)
        with zipfile.ZipFile(assets / "joint-b1-r4-runtime-config.zip") as archive:
            if set(archive.namelist()) != {
                "plan.json",
                "launch-command.json",
                "runtime-identity.json",
            }:
                raise ValueError("Unexpected runtime configuration members")
            for name in archive.namelist():
                (out / "provenance" / ("original-" + name)).write_bytes(
                    archive.read(name)
                )
        plan_path = out / "provenance/original-plan.json"
        plan = json.loads(plan_path.read_bytes())
        canonical = json.dumps(plan, indent=2).encode()
        if hashlib.sha256(canonical).hexdigest() != ORIGINAL_PLAN:
            raise ValueError(
                "Exported plan does not reproduce the original plan digest"
            )
        if len(plan["source_sha256"]) != 246:
            raise ValueError("Expected 246 original plan references")
        for path, expected in plan["source_sha256"].items():
            entry = origins[path]
            if entry["archive"] != FROZEN or entry["sha256"] != expected:
                raise ValueError("Original plan binding failed: " + path)
        (out / "provenance/canonical-original-plan.json").write_bytes(canonical)
    host = qualify_host_contract(out, python, bindings)
    host["changes"].append(relocate_signature_source(out, bindings))
    pythonpath = PYTHONPATH_LAYOUT
    profile = {
        "schema": "vllm-hust-engine-deployment/v1",
        "release": "full-answer-b1-20261030-r4",
        "root": str(out.resolve()),
        "python": str(python.expanduser().absolute()),
        "pythonpath": [str((out / name).resolve()) for name in pythonpath],
        "original_plan_sha256": ORIGINAL_PLAN,
        "original_plan_export_sha256": sha(out / "provenance/original-plan.json"),
        "plan_export_note": "The ZIP export adds a terminal newline; original json.dumps(indent=2) bytes reproduce the recorded digest. Both are preserved.",
        "frozen_files_verified": 255,
        "plan_references_verified": 246,
        "recovered_files_verified": 8975,
        "original_manifest_sha256": manifest_hashes,
        "historical_dependency_byte_equivalence": False,
        "scope": "Original r4 launch/worker/inputs plus pinned recovered component snapshot; independent startup and inference remain to be verified.",
        "model_started": False,
        "host_contract_qualification": host,
    }
    for name, data in (
        ("deployment.json", profile),
        ("provenance/installed-files.json", bindings),
        ("provenance/original-path-map.json", origins),
        ("provenance/deployment-changes.json", host["changes"]),
    ):
        (out / name).write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: profile[k]
                for k in (
                    "release",
                    "root",
                    "frozen_files_verified",
                    "plan_references_verified",
                    "recovered_files_verified",
                    "model_started",
                )
            }
        )
    )
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    restore(args.assets, args.out, args.python, args.env_file)


if __name__ == "__main__":
    main()
