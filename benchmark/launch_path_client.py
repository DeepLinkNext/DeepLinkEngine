"""Keep the legacy runtime path rejection without rejecting the active SDK."""
import json
from pathlib import Path
import tempfile

from benchmark import evaluation_client as client
from benchmark.host_memory_client import RELEASE, SDIST_SHA256

ENGINE_ENTRY = "scripts/engine_entry.py"
ENGINE_ENTRY_SHA256 = "e11c1be5561f74acfb36df4d4f0f9782b0a2136df892042f905da7da8a5c8bf4"
PATH_CHECK = "scripts/launch_path_check.py"

ORIGINAL_GUARD = """    if any(
        "gennova" in v.lower()
        for k, v in audit_env.items()
        if k == "PYTHONPATH" or k.startswith("BETTERSCALE")
    ):
        raise ValueError("Legacy project path leaked into active deployment")
"""
REVISED_GUARD = """    from launch_path_check import verify as verify_launch_paths
    verify_launch_paths(audit_env, root, Path(__file__).resolve().parent)
"""
PATH_CHECK_SOURCE = b'''"""Check path ownership for the selected launch environment."""
import os
from pathlib import Path


DIRECTORIES = {"BETTERSCALE_QWEN35_RUNTIME": "runtime/ascend"}
LIBRARIES = {
    "BETTERSCALE_GDN_LIBRARY": "runtime/betterscale-distribution/betterscale/patches/qwen_gdn/libbs_gdn.so",
    "BETTERSCALE_GDN_HOST_LIBRARY": "native-libs/libbs_gdn_host_cann91.so",
    "BETTERSCALE_CP_LIBRARY": "native-libs/context-parallel/libbs_fia_cp.so",
}


def verify(environment, deployment, client_scripts):
    deployment, client_scripts = deployment.resolve(strict=True), client_scripts.resolve(strict=True)
    for key, value in environment.items():
        if key != "PYTHONPATH" and not key.startswith("BETTERSCALE"):
            continue
        for text in value.split(os.pathsep) if key == "PYTHONPATH" else [value]:
            if "gennova" not in text.lower():
                continue
            path = Path(text)
            if path.is_absolute():
                resolved = path.resolve(strict=True)
                if key == "PYTHONPATH" and resolved.is_dir() and (
                        resolved.is_relative_to(deployment) or resolved == client_scripts):
                    continue
                if key in DIRECTORIES and resolved.is_dir() and resolved.is_relative_to(deployment):
                    if resolved == (deployment / DIRECTORIES[key]).resolve(strict=True):
                        continue
                if key in LIBRARIES and resolved.is_file() and resolved.is_relative_to(deployment):
                    if resolved == (deployment / LIBRARIES[key]).resolve(strict=True):
                        continue
            raise ValueError("Legacy project path leaked into active deployment: " + key)
'''


def derive(raw):
    if client.sha(raw) != ENGINE_ENTRY_SHA256:
        raise ValueError("Active launch path admission requires controller source review")
    adapted = client.once(raw.decode(), ORIGINAL_GUARD, REVISED_GUARD).encode()
    compile(adapted, ENGINE_ENTRY, "exec")
    compile(PATH_CHECK_SOURCE, PATH_CHECK, "exec")
    return {ENGINE_ENTRY: adapted, PATH_CHECK: PATH_CHECK_SOURCE}


def prepare(cache, selection, record):
    if (record is None or selection.get("release") != RELEASE or selection.get("version") != "2026.11.24.10"
            or selection.get("sdist", {}).get("sha256") != SDIST_SHA256):
        return record
    parent = client.verify(record)
    if admission := parent.get("launch_path_admission"):
        if admission["adapted_sources_sha256"].get(PATH_CHECK) == client.sha(PATH_CHECK_SOURCE):
            return record
        record = admission["parent_client"]
        parent = client.verify(record)
    source = Path(record["path"])
    files = {name: (source / name).read_bytes() for name in parent["files"]}
    updated = derive(files[ENGINE_ENTRY])
    files.update(updated)
    manifest = dict(parent, launch_path_admission={
        "parent_client": record,
        "engine_entry_before_sha256": ENGINE_ENTRY_SHA256,
        "adapted_sources_sha256": {name: client.sha(raw) for name, raw in updated.items()},
    }, files={name: client.sha(raw) for name, raw in files.items()})
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    digest = client.sha(raw)
    folder = Path(cache) / "assessment-clients"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / digest
    result = dict(path=str(output), manifest_sha256=digest, release=record["release"])
    if not output.exists():
        with tempfile.TemporaryDirectory(prefix="launch-path-", dir=folder) as temporary:
            candidate = Path(temporary) / "client"
            for name, body in files.items():
                target = candidate / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
            (candidate / "CLIENT-MANIFEST.json").write_bytes(raw)
            try:
                candidate.rename(output)
            except FileExistsError:
                pass
    client.verify(result)
    return result
