"""Keep the legacy runtime path rejection without rejecting the active SDK."""
import json
from pathlib import Path
import tempfile

from benchmark import evaluation_client as client
from benchmark.host_memory_client import RELEASE, SDIST_SHA256

ENGINE_ENTRY = "scripts/engine_entry.py"
ENGINE_ENTRY_SHA256 = "e11c1be5561f74acfb36df4d4f0f9782b0a2136df892042f905da7da8a5c8bf4"
PATH_CHECK = "scripts/launch_path_check.py"
PREVIOUS_ENTRY_SHA256 = "34094741fb699a8d48abf09fe02ca197598fe892f8ea1c136bfbe9622d8640e4"
PREVIOUS_PATH_CHECK_SHA256 = "1d7fada05a272d482e03a7e1a7f28830290a00a4d64647815596ec13c6b8d8cf"
REFERENCE_CONTROLS = "scripts/automatic_reference.py"
REFERENCE_CONTROLS_SHA256 = "cac72e36064607bb63fb20451c9692a10f498d30360d569e035c06c907037cd0"
REFERENCE_REUSE_SOURCES = {
    "scripts/reference_model_reuse.py": (
        "531d2765ec5e2857d6c7c79e7f8c623ec18ccabfb48696ecb30e8ecb5ac2639b",
        "        if (current_pin != pin or sha(current) != pin):\n", "pin"),
    "scripts/reference_qualification_reuse.py": (
        "0a692b1ac062744aa90772124d225d40e0ffa99d9c3ba3dbf67b5f2f0563d95d",
        "        if current_pin != expected or sha(current) != expected:\n", "expected"),
}
REFERENCE_PATH_COMPATIBILITY = "scripts/reference_path_compatibility.py"

ORIGINAL_GUARD = """    if any(
        "gennova" in v.lower()
        for k, v in audit_env.items()
        if k == "PYTHONPATH" or k.startswith("BETTERSCALE")
    ):
        raise ValueError("Legacy project path leaked into active deployment")
"""
REVISED_GUARD = """    from launch_path_check import verify as verify_launch_paths
    verify_launch_paths(audit_env, root, Path(__file__).resolve().parent,
                        reference_paths=reference_paths)
"""
PATH_CHECK_SOURCE = b'''"""Check path ownership for the selected launch environment."""
import hashlib
import json
import os
from pathlib import Path

DIRECTORIES = {"BETTERSCALE_QWEN35_RUNTIME": "runtime/ascend"}
LIBRARIES = {
    "BETTERSCALE_GDN_LIBRARY": "runtime/betterscale-distribution/betterscale/patches/qwen_gdn/libbs_gdn.so",
    "BETTERSCALE_GDN_HOST_LIBRARY": "native-libs/libbs_gdn_host_cann91.so",
    "BETTERSCALE_CP_LIBRARY": "native-libs/context-parallel/libbs_fia_cp.so",
}


def reference_import_paths(client_scripts):
    kit = client_scripts.parent / "third-party-test-kit"
    return (kit, kit / "proc_capture")


def verify_reference_kit(client_scripts):
    client_root = client_scripts.resolve(strict=True).parent
    kit, capture = reference_import_paths(client_root / "scripts")
    manifest_path = client_root / "CLIENT-MANIFEST.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Reference observation requires a sealed assessment client")
    manifest_raw = manifest_path.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != client_root.name:
        raise ValueError("Reference observation assessment-client manifest differs")
    manifest = json.loads(manifest_raw)
    files = {name: digest for name, digest in manifest["files"].items()
             if name.startswith("third-party-test-kit/")}
    if (not files or kit.resolve(strict=True) != kit or capture.resolve(strict=True) != capture
            or not kit.is_dir() or not capture.is_dir()):
        raise ValueError("Reference observation assessment-kit paths differ")
    actual = {path.relative_to(client_root).as_posix(): path for path in kit.rglob("*")
              if path.is_file() or path.is_symlink()}
    if actual.keys() != files.keys():
        raise ValueError("Reference observation assessment-kit inventory differs")
    for name, path in actual.items():
        if (path.is_symlink() or not path.resolve(strict=True).is_relative_to(kit)
                or hashlib.sha256(path.read_bytes()).hexdigest() != files[name]):
            raise ValueError("Reference observation assessment-kit file differs: " + name)
    return (kit, capture)


def verify(environment, deployment, client_scripts, *, reference_paths=()):
    deployment, client_scripts = deployment.resolve(strict=True), client_scripts.resolve(strict=True)
    if reference_paths and reference_paths != reference_import_paths(client_scripts):
        raise ValueError("Reference observation import paths differ from its sealed client")
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
                        resolved.is_relative_to(deployment) or resolved == client_scripts
                        or resolved in reference_paths):
                    continue
                if key in DIRECTORIES and resolved.is_dir() and resolved.is_relative_to(deployment):
                    if resolved == (deployment / DIRECTORIES[key]).resolve(strict=True):
                        continue
                if key in LIBRARIES and resolved.is_file() and resolved.is_relative_to(deployment):
                    if resolved == (deployment / LIBRARIES[key]).resolve(strict=True):
                        continue
            raise ValueError("Legacy project path leaked into active deployment: " + key)
'''


REFERENCE_COMPATIBILITY_SOURCE = b'''"""Keep sealed native and operator evidence across observer import admission."""
import json

from dataset_contract import sha


def matches(relative, old, current, previous_pin, current_pin):
    pins = PINNED_CHANGES.get(relative)
    return (pins == (previous_pin, current_pin) and sha(old) == previous_pin
            and sha(current) == current_pin)


def rejected_observation_import(failure, source, directory, seal):
    if (failure.get("model_service_started") is not True
            or failure.get("client_started") is not False
            or failure.get("warmup_started") is True
            or failure.get("failure") != "RuntimeError: Model service exited with 1"):
        return False
    log_name, plan_name = directory + "/service.log", directory + "/service/plan.json"
    if log_name not in seal or plan_name not in seal:
        return False
    log, plan = source / log_name, source / plan_name
    if sha(log) != seal[log_name] or sha(plan) != seal[plan_name]:
        return False
    flags = json.loads(plan.read_bytes())
    traceback = log.read_text().splitlines()
    return (flags.get("reference_observation") is True
            and flags.get("timing_eligible_for_performance_comparison") is False
            and any("launch_path_check.py" in line for line in traceback)
            and traceback[-1:] == ["ValueError: Legacy project path leaked into active deployment: PYTHONPATH"])


def preserve_baseline_policy(plan, source):
    if "baseline_execution_policy" in plan:
        return
    original = source / "automatic-plan.json"
    sealed = json.loads((source / "checksums.json").read_bytes())
    if sha(original) != sealed.get(original.name):
        raise ValueError("Prior reference baseline execution policy is unsealed")
    previous = json.loads(original.read_bytes())
    for key in ("baseline_execution_policy", "baseline_reuse"):
        if key in previous:
            plan[key] = previous[key]
'''


def derive(files):
    raw = files[ENGINE_ENTRY]
    if client.sha(raw) != ENGINE_ENTRY_SHA256:
        raise ValueError("Active launch path admission requires controller source review")
    entry = client.once(raw.decode(), ORIGINAL_GUARD, REVISED_GUARD)
    observation = ('    if args.reference_observation:\n'
                   '        if args.kind != "joint" or not selected(profile):\n')
    entry = client.once(entry, observation, '    reference_paths = ()\n' + observation)
    entry = client.once(entry, '        from reference_observation import configure\n',
        '        from launch_path_check import verify_reference_kit\n'
        '        from reference_observation import configure\n\n'
        '        reference_paths = verify_reference_kit(Path(__file__).resolve().parent)\n')
    adapted = entry.encode()
    pins = {ENGINE_ENTRY: (PREVIOUS_ENTRY_SHA256, client.sha(adapted)),
            PATH_CHECK: (PREVIOUS_PATH_CHECK_SHA256, client.sha(PATH_CHECK_SOURCE))}
    updated = {ENGINE_ENTRY: adapted, PATH_CHECK: PATH_CHECK_SOURCE,
               REFERENCE_PATH_COMPATIBILITY: REFERENCE_COMPATIBILITY_SOURCE
               + ("\n\nPINNED_CHANGES = " + repr(pins) + "\n").encode()}
    for name, (expected, condition, original_pin) in REFERENCE_REUSE_SOURCES.items():
        source = files[name]
        if client.sha(source) != expected:
            raise ValueError("Reference continuation path admission requires source review: " + name)
        text = client.once(source.decode(), condition,
            condition + '            from reference_path_compatibility import matches\n'
            '            if matches(relative, path, current, ' + original_pin + ', current_pin):\n'
            '                continue\n')
        if name.endswith("reference_model_reuse.py"):
            anchor = ('            if failed.get("model_service_started") is False and '
                      'failed.get("client_started") is False:\n                continue\n')
            text = client.once(text, anchor,
                anchor + '            from reference_path_compatibility import rejected_observation_import\n\n'
                '            if rejected_observation_import(failed, source, directory, seal):\n'
                '                continue\n')
        updated[name] = text.encode()
    source = files[REFERENCE_CONTROLS]
    if client.sha(source) != REFERENCE_CONTROLS_SHA256:
        raise ValueError("Reference continuation policy requires source review")
    text = client.once(source.decode(),
        '            resume = load_models(resume_source, plan["stage_resume"]["checksums_sha256"], plan, ROOT,\n',
        '            from reference_path_compatibility import preserve_baseline_policy\n\n'
        '            preserve_baseline_policy(plan, resume_source)\n'
        '            resume = load_models(resume_source, plan["stage_resume"]["checksums_sha256"], plan, ROOT,\n')
    updated[REFERENCE_CONTROLS] = text.encode()
    for name, raw in updated.items():
        compile(raw, name, "exec")
    return updated


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
    updated = derive(files)
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
