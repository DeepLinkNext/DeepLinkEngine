"""Bind the reviewed live cgroup-v1 memory guard to the revised r10 client."""
import ast
import json
from pathlib import Path
import sys
import tempfile

from benchmark import evaluation_client as client

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "full-answer-b1-20261124-r10"
PROVIDER_SHA256 = "0f5ab9de2c8e1aa8e6e091ecd88c80c7136307b79cc326b2f0a73c81d5e682b7"
SDIST_SHA256 = "b11bcc9c2064268112b5386de97d103a45eab5f6956ea35477f047d2e0725c0d"
SHARED_GUARD = "scripts/official_memory.py"
CONTROLLERS = {
    "scripts/run_b1_r4.py": "b2d6abc4a107d12cb6b1fb6b7bdf9ddbb74c1506a513e3ac0101008e67d85d7f",
    "scripts/run_native_reference.py": "d7d5276ae8582ed9673f9424b944d6a7f40f4d289b5488430dfd009755956f73",
    "scripts/run_canonical_reference.py": "cdc44427cd16331208f6463065485e1f5522062e5fda93ae08d55fe4896100e7",
}


def derive(files):
    for name, expected in CONTROLLERS.items():
        if client.sha(files[name]) != expected:
            raise ValueError("Cgroup v1 memory admission requires source review: " + name)

    provider = (ROOT / "official/official_memory.py").read_bytes()
    if client.sha(provider) != PROVIDER_SHA256:
        raise ValueError("Reviewed live cgroup v1 memory source differs")
    baseline_path = ROOT / "official/official_baselines.py"
    baseline = baseline_path.read_text(encoding="utf-8")
    original_save = next(node for node in ast.parse(baseline).body
                         if isinstance(node, ast.FunctionDef) and node.name == "save")
    # Stage the same atomic receipt writer beside the guard; the timing
    # children need no repository path or side effect from the official CLI.
    text = client.once(provider.decode(), "from official_baselines import save",
                       "import json\n\n\n" + ast.get_source_segment(baseline, original_save))
    updates = {SHARED_GUARD: text.encode()}

    runner, native, canonical = CONTROLLERS
    text = files[runner].decode()
    text = client.once(text, "    initial_memory = admit_startup_memory(memory_guard, args.model, weights, root)",
        "    from official_memory import admit_startup, memory_guard as live_memory_guard\n"
        "    memory_guard = live_memory_guard(memory_guard)\n"
        "    initial_memory = admit_startup(memory_guard, args.model, weights, root)")
    updates[runner] = text.encode()

    text = files[native].decode()
    text = client.once(text, "            spec.loader.exec_module(memory_guard)\n",
        "            spec.loader.exec_module(memory_guard)\n"
        "            from official_memory import admit_startup, memory_guard as live_memory_guard\n"
        "            memory_guard = live_memory_guard(memory_guard)\n")
    text = client.once(text, "initial = admit_startup_memory(memory_guard, model, weights, root)",
                       "initial = admit_startup(memory_guard, model, weights, root)")
    updates[native] = text.encode()

    text = files[canonical].decode()
    text = client.once(text, "            spec.loader.exec_module(guard)\n",
        "            spec.loader.exec_module(guard)\n"
        "            from official_memory import memory_guard as live_memory_guard\n"
        "            guard = live_memory_guard(guard)\n")
    updates[canonical] = text.encode()
    for name, raw in updates.items():
        compile(raw, name, "exec")
    return updates


def prepare(cache, selection, record):
    if (sys.platform != "linux" or record is None or selection.get("release") != RELEASE
            or selection.get("version") != "2026.11.24.10"
            or selection.get("sdist", {}).get("sha256") != SDIST_SHA256
            or Path("/sys/fs/cgroup/memory.stat").is_file()):
        return record
    from official import enable_module_imports

    enable_module_imports()
    from official.official_memory import memory_scopes

    memory_scopes(Path("/proc"))
    parent = client.verify(record)
    if parent.get("host_memory_admission"):
        return record
    source = Path(record["path"])
    files = {name: (source / name).read_bytes() for name in parent["files"]}
    updates = derive(files)
    files.update(updates)
    manifest = dict(parent, host_memory_admission={
        "cgroup_version": 1,
        "provider_sha256": PROVIDER_SHA256,
        "parent_client": record,
        "original_controller_sha256": CONTROLLERS,
        "adapted_sources_sha256": {name: client.sha(raw) for name, raw in updates.items()},
        "live_kernel_counters": True,
        "raw_startup_headroom_bytes": 36 * 2**30,
        "working_set_reserve_bytes": 4 * 2**30,
    }, files={name: client.sha(raw) for name, raw in files.items()})
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    digest = client.sha(raw)
    folder = Path(cache) / "assessment-clients"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / digest
    result = dict(path=str(output), manifest_sha256=digest, release=record["release"])
    if not output.exists():
        with tempfile.TemporaryDirectory(prefix="host-memory-", dir=folder) as temporary:
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
