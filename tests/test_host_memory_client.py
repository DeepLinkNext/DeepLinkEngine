"""The fixed-cgroup admission adapters retain the sealed execution steps."""
import pytest

from benchmark import host_memory_client as host
from benchmark.evaluation_client import sha


def test_v1_derivation_preserves_all_other_controller_statements(monkeypatch):
    run, native, canonical = host.CONTROLLERS
    original = {
        run: b"""def arm():
    initial_memory = admit_startup_memory(memory_guard, args.model, weights, root)
    prepare_startup_reserve(memory_guard, initial_memory, args.model, weights, root)
""",
        native: b"""def native():
    with controller():
        if True:
            spec.loader.exec_module(memory_guard)
            initial = admit_startup_memory(memory_guard, model, weights, root)
            seal_native_work(initial)
""",
        canonical: b"""def canonical():
    with controller():
        if True:
            spec.loader.exec_module(guard)
            initial = guard.sample()
            seal_canonical_work(initial)
""",
    }
    monkeypatch.setattr(host, "CONTROLLERS", {name: sha(data) for name, data in original.items()})
    derived = host.derive(original)
    assert b"prepare_startup_reserve(memory_guard, initial_memory, args.model, weights, root)" in derived[run]
    assert b"seal_native_work(initial)" in derived[native]
    assert b"seal_canonical_work(initial)" in derived[canonical]
    assert b"def memory_scopes(proc):" in derived[host.SHARED_GUARD]
    assert b"required = 36 * 2**30" in derived[host.SHARED_GUARD]
    assert b"reserve = 4 * 2**30" in derived[host.SHARED_GUARD]
    assert b"temporary.replace(path)" in derived[host.SHARED_GUARD]
    with pytest.raises(ValueError, match="requires source review"):
        host.derive(dict(original, **{run: original[run] + b"\nadded()\n"}))


def test_no_cgroup_client_derivation_for_a_different_engine(tmp_path):
    original = {"path": "unchanged", "release": "other"}
    assert host.prepare(tmp_path, {"release": "other"}, original) is original
