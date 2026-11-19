import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import automatic_reference as automatic
from compute_reference import execute, prepare
from dataset_contract import save, sha
from test_automatic_reference import fixture, fake_stage


def prior(tmp_path, monkeypatch):
    import reference_qualification_reuse as reuse
    # CPU orchestration fixtures do not own real Ascend devices. Keep the
    # seal/source validator real while replacing just the shared lease layer.
    monkeypatch.setattr(reuse, "guarded_load", lambda source, pin, plan, root, guard:
                        reuse.load(source, pin, plan, root))
    args, frozen, out = fixture(tmp_path)
    monkeypatch.setattr(automatic, "run_stage", fake_stage)
    import engine_entry
    monkeypatch.setattr(engine_entry, "environment", lambda *a: {})
    result = execute(out, prepare(args, out, frozen))
    assert result["status"] == "measured"
    source = out / "compute-reference"
    if not (source / "checksums.json").exists():
        save(source / "checksums.json", {str(p.relative_to(source)): sha(p)
             for p in source.rglob("*") if p.is_file()})
    next_out = tmp_path / "continued"
    next_out.mkdir()
    save(next_out / "frozen-stages.json", json.loads((out / "frozen-stages.json").read_bytes()))
    args.reference_qualification_source = source
    return args, frozen, next_out, source


def test_reuses_only_passed_sealed_operators_and_still_runs_both_native_arms(tmp_path, monkeypatch):
    args, frozen, out, _ = prior(tmp_path, monkeypatch)
    admission = prepare(args, out, frozen)
    assert admission["status"] == "automatic_ready"
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        return fake_stage(command, **kwargs)
    monkeypatch.setattr(automatic, "run_stage", launch)
    result = execute(out, admission)
    assert result["status"] == "measured"
    assert len(calls) == 7
    assert all(row["reused_qualification"] for row in result["stages"][:3])
    assert sum(any(part.endswith("run_native_reference.py") for part in call) for call in calls) == 2
    assert all("qualify_" not in " ".join(call) for call in calls)


@pytest.mark.parametrize("change", ["artifact", "controls", "workload"])
def test_modified_or_incompatible_qualification_cannot_be_reused(tmp_path, monkeypatch, change):
    args, frozen, out, source = prior(tmp_path, monkeypatch)
    if change == "artifact":
        (source / "gdn-decode/receipt.json").write_text("{}")
    elif change == "controls":
        original = automatic.prefill_cases
        monkeypatch.setattr(automatic, "prefill_cases", lambda: original()[:-1])
    else:
        frozen["mmlu_inputs"]["quality_plan_sha256"] = "b" * 64
    result = prepare(args, out, frozen)
    assert result["status"] == "blocked"


def test_short_three_round_capture_can_reuse_identical_operator_qualifications(tmp_path, monkeypatch):
    args, frozen, out, _ = prior(tmp_path, monkeypatch)
    # Qualification uses unchanged operator cases and numeric/statistical controls.
    args.reference_profile = "full-answer-v1"
    admission = prepare(args, out, frozen)
    assert admission["status"] == "automatic_ready"
    assert admission["capture_profile"]["max_tokens"] == 8192
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        return fake_stage(command, **kwargs)
    monkeypatch.setattr(automatic, "run_stage", launch)
    result = execute(out, admission)
    assert result["status"] == "measured"
    assert len(calls) == 7
    assert all(row["reused_qualification"] for row in result["stages"][:3])
    native = [c for c in calls if any(p.endswith("run_native_reference.py") for p in c)]
    assert len(native) == 2
    assert all(c[c.index("--formal-repetitions") + 1] == "1" for c in native)


def test_operator_annotation_migration_accepts_only_exact_known_domains():
    from copy import deepcopy
    from reference_qualification_reuse import controls

    old = dict(stages=[], capture_profile=dict(max_tokens=256, formal_repetitions=3),
        measurement_domain=dict(frontend="HTTP", dtype="BF16", tp=2,
            max_completion_tokens=8192, profiles=["mmlu-fixed200-v1"]),
        observation_domain_only=True)
    current = deepcopy(old)
    current["measurement_domain"] = dict(frontend="native_inproc", dtype="BF16", tp=2,
        max_completion_tokens=256, formal_repetitions=3, profiles=["mmlu-fixed200-v1"])
    current["observation_domain_only"] = False
    assert controls(old) == controls(current)
    for key, value in (("dtype", "FP16"), ("tp", 4),
                       ("profiles", ["other-workload"]), ("max_completion_tokens", 512)):
        changed = deepcopy(current)
        changed["measurement_domain"][key] = value
        assert controls(changed) != controls(current)


@pytest.mark.parametrize('borrowed,fails', [(False, False), (False, True), (True, False), (True, True)])
def test_operator_envelope_admission_owns_or_borrows_lease_without_stale_environment(tmp_path, monkeypatch, borrowed, fails):
    import os
    import resource_guard
    import reference_qualification_reuse as reuse
    fd = os.open(tmp_path / 'lease', os.O_CREAT | os.O_RDWR, 0o600)
    entries = [dict(fd=fd)]
    original = json.dumps(entries) if borrowed else None
    if borrowed:
        monkeypatch.setenv('VLLMHUST_ENGINE_LOCKS', original)
    else:
        monkeypatch.delenv('VLLMHUST_ENGINE_LOCKS', raising=False)
    def acquire(*args):
        os.environ['VLLMHUST_ENGINE_LOCKS'] = json.dumps(entries)
        return entries
    def load(*args):
        if fails:
            raise ValueError('bad original seal')
        return 'qualified'
    monkeypatch.setattr(resource_guard, 'acquire', acquire)
    monkeypatch.setattr(resource_guard, 'verify_mapping', lambda value: None)
    monkeypatch.setattr(reuse, 'load', load)
    try:
        args = (tmp_path, 'pin', dict(physical_cards=[1, 6], logical_devices='0,1'), tmp_path, tmp_path / 'guard')
        if fails:
            with pytest.raises(ValueError, match='bad original seal'):
                reuse.guarded_load(*args)
        else:
            assert reuse.guarded_load(*args) == 'qualified'
        assert os.environ.get('VLLMHUST_ENGINE_LOCKS') == original
        if borrowed:
            os.fstat(fd)
        else:
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        if borrowed:
            os.close(fd)
