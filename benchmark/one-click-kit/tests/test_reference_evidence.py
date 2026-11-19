import ast
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import automatic_reference as automatic
import reference_evidence as evidence
import reference_model_reuse as models
import reference_qualification_reuse as operators
from dataset_contract import sha
from test_reference_qualification_reuse import prior
from test_automatic_reference import fake_stage


def input_view(root, *, complete=True):
    source = root / "http-current-work"
    raw = source / "service/raw-work"
    raw.mkdir(parents=True)
    for rank in (0, 1):
        (raw / f"rank-{rank}-waves.jsonl").write_text(json.dumps(dict(test_only=True, rank=rank)) + "\n")
    (source / "receipt.json").write_text(json.dumps(dict(completed=complete, quiescent=complete,
        role="B1", planned_joint_requests=200)))
    (source / "checksums.json").write_text(json.dumps(evidence.evidence_manifest(source)))
    audit = root / "http-independent-audit"
    (audit / "raw-view").mkdir(parents=True)
    inputs = {}
    for path in raw.iterdir():
        (audit / "raw-view" / path.name).symlink_to(path)
        inputs[str(path.relative_to(source))] = sha(path)
    (audit / "source-inputs.json").write_text(json.dumps(dict(source=str(source),
        source_checksums_sha256=sha(source / "checksums.json"), raw_inputs=inputs)))
    seal = evidence.evidence_manifest(root)
    return source, audit, seal


def test_sealed_input_view_keeps_each_raw_content_check_with_shared_metadata(tmp_path):
    root = tmp_path / "run"
    source, audit, seal = input_view(root)
    context = {}
    names = [str((audit / "raw-view" / f"rank-{rank}-waves.jsonl").relative_to(root)) for rank in (0, 1)]
    assert evidence.sealed_file_matches(root, names[0], seal[names[0]], seal, context)
    assert evidence.sealed_file_matches(root, names[1], seal[names[1]], seal, context)
    (source / "service/raw-work/rank-1-waves.jsonl").write_text("changed raw measurement")
    assert not evidence.sealed_file_matches(root, names[1], seal[names[1]], seal, context)


@pytest.mark.parametrize("complete", [False, True])
def test_input_view_rejects_resealed_incomplete_source_or_equal_foreign_bytes(tmp_path, complete):
    root = tmp_path / "run"
    source, audit, seal = input_view(root, complete=complete)
    alias = audit / "raw-view/rank-0-waves.jsonl"
    if complete:
        foreign = tmp_path / "foreign"
        foreign.write_bytes(alias.read_bytes())
        alias.unlink()
        alias.symlink_to(foreign)
    name = str(alias.relative_to(root))
    assert not evidence.sealed_file_matches(root, name, seal[name], seal, {})


@pytest.mark.parametrize("directory", ["native-baseline-timing/compiler-cache", "http-current-work-compiler-cache"])
def test_manifest_never_traverses_compiler_cache_or_external_weights(tmp_path, monkeypatch, directory):
    root = tmp_path / "run"
    cache = root / directory / "model-text-only"
    cache.mkdir(parents=True)
    external = tmp_path / "weights.safetensors"
    external.write_bytes(b"external checkpoint")
    (cache / external.name).symlink_to(external)
    (root / "receipt.json").write_text("{}")
    (root / "checksums.json").write_text("old")
    real = evidence.sha
    def guarded(path):
        assert not evidence.transient_cache(str(path.relative_to(root)))
        return real(path)
    monkeypatch.setattr(evidence, "sha", guarded)
    assert evidence.evidence_manifest(root) == {"receipt.json": real(root / "receipt.json")}


@pytest.mark.parametrize("loader", [models.load, operators.load])
@pytest.mark.parametrize("directory", ["native-baseline-timing/compiler-cache", "http-current-work-compiler-cache"])
def test_legacy_outer_cache_entries_are_ignored_without_dereferencing(tmp_path, monkeypatch, loader, directory):
    _, _, _, source = prior(tmp_path, monkeypatch)
    name = directory + "/model-text-only/weights.safetensors"
    path = source / name
    path.parent.mkdir(parents=True)
    path.symlink_to(tmp_path / "missing-external-checkpoint")
    seal = json.loads((source / "checksums.json").read_bytes())
    seal[name] = "0" * 64
    (source / "checksums.json").write_text(json.dumps(seal))
    plan = json.loads((source / "automatic-plan.json").read_bytes())
    loader(source, sha(source / "checksums.json"), plan, automatic.ROOT)


@pytest.mark.parametrize("name", ["receipt-link.json", "../compiler-cache/escaped.json"])
@pytest.mark.parametrize("loader", [models.load, operators.load])
def test_noncache_links_and_escaped_cache_names_still_fail(tmp_path, monkeypatch, name, loader):
    _, _, _, source = prior(tmp_path, monkeypatch)
    external = tmp_path / "external.json"
    external.write_text("{}")
    path = source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(external)
    seal = json.loads((source / "checksums.json").read_bytes())
    seal[name] = sha(external)
    (source / "checksums.json").write_text(json.dumps(seal))
    plan = json.loads((source / "automatic-plan.json").read_bytes())
    with pytest.raises(ValueError, match="escaped"):
        loader(source, sha(source / "checksums.json"), plan, automatic.ROOT)


def test_http_driver_has_no_local_sys_shadow_before_service_start():
    path = automatic.ROOT / "scripts/run_b1_r4.py"
    function = next(n for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "admitted_run")
    assert not any(isinstance(n, ast.Import) and any(a.name == "sys" for a in n.names)
        for n in ast.walk(function))
    code = compile(ast.parse(path.read_text()), str(path), "exec")
    function_code = next(c for c in code.co_consts if getattr(c, "co_name", None) == "admitted_run")
    assert "sys" not in function_code.co_varnames


def test_only_exact_http_scope_fix_admitted_and_native_helpers_unchanged(tmp_path):
    current = automatic.ROOT / "scripts/run_b1_r4.py"
    original = tmp_path / "old.py"
    source = current.read_text()
    marker = '        if getattr(args, "reference_observation", False):\n'
    assert source.count(marker) == 1
    original.write_text(source.replace(marker, marker + "            import sys\n"))
    assert evidence.http_scope_fix_only(original, current, sha(original))
    changed = tmp_path / "changed.py"
    changed.write_text(source.replace("def check_memory(", "def changed_check_memory("))
    assert not evidence.http_scope_fix_only(original, changed, sha(original))
    assert not evidence.http_scope_fix_only(original, current, "0" * 64)


def failed_http(source, **changes):
    path = source / "http-current-work/receipt.json"
    value = dict(completed=False, quiescent=True, model_service_started=False,
        client_started=False, role="B1", failure="Pre-model HTTP admission failed")
    value.update(changes)
    path.write_text(json.dumps(value))
    nested = source / "http-current-work/checksums.json"
    seal = json.loads(nested.read_bytes()); seal["receipt.json"] = sha(path)
    nested.write_text(json.dumps(seal))
    result_path = source / "automatic-result.json"
    result = json.loads(result_path.read_bytes())
    row = next(r for r in result["stages"] if r["id"] == "current-http-work-observation")
    row.update(status="failed", receipt_sha256=sha(path))
    result_path.write_text(json.dumps(result))
    (source / "checksums.json").write_text(json.dumps(evidence.evidence_manifest(source)))


def test_unstarted_http_recovery_runs_only_missing_observation_and_providers(tmp_path, monkeypatch):
    args, frozen, out, source = prior(tmp_path, monkeypatch)
    failed_http(source)
    args.reference_resume_source = source
    monkeypatch.setattr(models, "guarded_load", lambda s, pin, plan, root, guard: models.load(s, pin, plan, root))
    from compute_reference import execute, prepare
    admission = prepare(args, out, frozen)
    assert admission["status"] == "automatic_ready"
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        return fake_stage(command, **kwargs)
    monkeypatch.setattr(automatic, "run_stage", launch)
    result = execute(out, admission)
    assert result["status"] == "measured"
    assert len(calls) == 5
    assert sum(any(p.endswith("run_b1_r4.py") for p in c) for c in calls) == 1
    assert not any(any(p.endswith("run_native_reference.py") for p in c) for c in calls)
    providers = [c for c in calls if any(p.endswith("reference_final_stages.py") for p in c)]
    assert len(providers) == 4 and all("--http-source" in c for c in providers)


@pytest.mark.parametrize("changes", [dict(quiescent=False), dict(model_service_started=True), dict(client_started=True)])
def test_partial_resume_rejects_live_or_started_http_attempt(tmp_path, monkeypatch, changes):
    _, _, _, source = prior(tmp_path, monkeypatch)
    failed_http(source, **changes)
    plan = json.loads((source / "automatic-plan.json").read_bytes())
    with pytest.raises(ValueError, match="quiescent" if changes.get("quiescent") is False else "independent guarded audit"):
        models.load(source, sha(source / "checksums.json"), plan, automatic.ROOT)


@pytest.mark.parametrize("fails", [False, True])
def test_sequential_model_admissions_never_inherit_closed_descriptors(tmp_path, monkeypatch, fails):
    import resource_guard
    monkeypatch.delenv("VLLMHUST_ENGINE_LOCKS", raising=False)
    descriptors = []
    def acquire(*args):
        assert "VLLMHUST_ENGINE_LOCKS" not in os.environ
        fd = os.open(tmp_path / "lease", os.O_CREAT | os.O_RDWR, 0o600)
        descriptors.append(fd)
        entries = [dict(fd=fd)]
        os.environ["VLLMHUST_ENGINE_LOCKS"] = json.dumps(entries)
        return entries
    monkeypatch.setattr(resource_guard, "acquire", acquire)
    monkeypatch.setattr(resource_guard, "verify_mapping", lambda value: None)
    def load(*args):
        if fails:
            raise ValueError("bad seal")
        return "admitted"
    monkeypatch.setattr(models, "load", load)
    for index in range(2):
        args = (tmp_path, "pin", dict(physical_cards=[1, 6], logical_devices="0,1"), tmp_path, tmp_path / str(index))
        if fails:
            with pytest.raises(ValueError, match="bad seal"):
                models.guarded_load(*args)
        else:
            assert models.guarded_load(*args) == "admitted"
        assert "VLLMHUST_ENGINE_LOCKS" not in os.environ
        with pytest.raises(OSError):
            os.fstat(descriptors[-1])


def test_model_admission_borrows_inherited_descriptor_without_closing_it(tmp_path, monkeypatch):
    import resource_guard
    fd = os.open(tmp_path / "lease", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        entries = [dict(fd=fd)]
        original = json.dumps(entries)
        monkeypatch.setenv("VLLMHUST_ENGINE_LOCKS", original)
        monkeypatch.setattr(resource_guard, "acquire", lambda *a: entries)
        monkeypatch.setattr(resource_guard, "verify_mapping", lambda value: None)
        monkeypatch.setattr(models, "load", lambda *a: "admitted")
        assert models.guarded_load(tmp_path, "pin", dict(physical_cards=[1,6], logical_devices="0,1"), tmp_path, tmp_path / "guard") == "admitted"
        os.fstat(fd)
        assert os.environ["VLLMHUST_ENGINE_LOCKS"] == original
    finally:
        os.close(fd)
