"""A dead root must not turn a live, unsealed or unreleased child into reusable evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmark/one-click-kit/scripts")]
import recovery_source
import run_all


def fixture(tmp_path):
    old = tmp_path / "old"
    data = dict(model="/model", physical_devices=[1, 6], cache="/cache", out=str(tmp_path / "new"))
    prior = dict(status="running", configuration=dict(data, out=str(old)), stages=[
        dict(name="prepare-datasets", status="completed"), dict(name="run-fixed-five", status="running")])
    rows = {
        "results.json": prior,
        "fixed-five-logs/exit.json": dict(returncode=2, controller_reaped=True),
        "fixed-five/campaign.json": dict(status="failed", stages=[
            dict(name="continuous-B0", status="reused_verified"),
            dict(name="mmlu-pair", status="completed"), dict(name="hle-solver-B1", status="failed")]),
        "fixed-five/hle-solver-B1/receipt.json": dict(status="failed", quiescent=True),
        "fixed-five/hle-solver-B1/cleanup.json": dict(quiescent=True),
    }
    for name, value in rows.items():
        path = old / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    seal(old / "fixed-five/hle-solver-B1")
    (tmp_path / "root-process.json").write_text(json.dumps(dict(pid=12345, start_ticks="12", out=str(old))))
    return old, prior, data


def seal(stage):
    (stage / "checksums.json").write_text(json.dumps({
        str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in stage.rglob("*") if p.is_file() and p.name != "checksums.json"}))


def change(path, **values):
    record = json.loads(path.read_bytes())
    record.update(values)
    path.write_text(json.dumps(record))


def test_dead_root_keeps_source_bytes_and_requires_the_original_completed_stage_audit(tmp_path):
    old, prior, data = fixture(tmp_path)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*.json")}
    campaign = json.loads((old / "fixed-five/campaign.json").read_bytes())
    with patch.object(recovery_source, "alive", return_value=False), \
            patch("reuse_campaign.source", return_value=(old, {}, campaign, {})), \
            patch("reuse_campaign.audit_stage") as audit, patch("release_selection.active", return_value=None):
        result = run_all.completed_run_source(data, old)
    audit.assert_called_once_with(old / "fixed-five", "mmlu-pair")
    assert result["completed_stages"] == ["mmlu-pair"]
    assert result["interruption_recovery"]["originals_modified"] is False
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*.json")}


@pytest.mark.parametrize("fault", ["live-root", "missing-owner", "other-output", "not-reaped", "exit-success",
                                   "campaign-running", "unreleased-receipt", "unreleased-cleanup", "future-root-stage"])
def test_incomplete_shutdown_cannot_be_reused(tmp_path, fault):
    old, prior, _ = fixture(tmp_path)
    if fault == "missing-owner":
        (tmp_path / "root-process.json").unlink()
    elif fault == "other-output":
        change(tmp_path / "root-process.json", out=str(tmp_path / "other"))
    elif fault in ("not-reaped", "exit-success"):
        change(old / "fixed-five-logs/exit.json", **({"controller_reaped": False} if fault == "not-reaped" else {"returncode": 0}))
    elif fault == "campaign-running":
        change(old / "fixed-five/campaign.json", status="running")
    elif fault.startswith("unreleased"):
        change(old / "fixed-five/hle-solver-B1" / ("receipt.json" if fault.endswith("receipt") else "cleanup.json"), quiescent=False)
    elif fault == "future-root-stage":
        prior["stages"].append(dict(name="run-context-tenant", status="running"))
    seal(old / "fixed-five/hle-solver-B1")
    with patch.object(recovery_source, "alive", return_value=fault == "live-root"):
        with pytest.raises((ValueError, FileNotFoundError)):
            recovery_source.stopped(old, prior)


def test_modified_cleanup_cannot_supply_a_shutdown_proof(tmp_path):
    old, prior, _ = fixture(tmp_path)
    change(old / "fixed-five/hle-solver-B1/cleanup.json", actions=[dict(pid=1, start_ticks="1")])
    with patch.object(recovery_source, "alive", return_value=False):
        with pytest.raises(ValueError, match="inventory or file hash"):
            recovery_source.stopped(old, prior)


def test_live_descendant_is_rejected_even_with_a_quiescent_receipt(tmp_path):
    old, prior, _ = fixture(tmp_path)
    change(old / "fixed-five/hle-solver-B1/cleanup.json", actions=[dict(pid=1, start_ticks="1")])
    seal(old / "fixed-five/hle-solver-B1")
    with patch.object(recovery_source, "alive", side_effect=[False, True]):
        with pytest.raises(ValueError, match="descendant is still alive"):
            recovery_source.stopped(old, prior)


def test_current_process_is_live_but_recycled_pid_is_not():
    owner = recovery_source.process_identity()
    assert owner["pid"] == os.getpid()
    assert recovery_source.alive(owner)
    assert not recovery_source.alive(dict(owner, start_ticks="0"))


def test_recovery_from_a_fresh_public_process_loads_its_own_audit_modules(tmp_path):
    old, _, data = fixture(tmp_path)
    change(tmp_path / "root-process.json", boot_id="previous-boot")
    code = f"""
import sys,json,types
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
import run_all
from unittest.mock import patch
source = Path({str(old)!r})
module = types.ModuleType('reuse_campaign')
module.source = lambda path: (source, {{}}, json.loads((source/'fixed-five/campaign.json').read_bytes()), {{}})
module.audit_stage = lambda root, name: None
with patch.dict(sys.modules, reuse_campaign=module), patch('release_selection.active', return_value=None):
    result = run_all.completed_run_source({data!r}, source)
assert result['completed_stages'] == ['mmlu-pair']
assert sys.dont_write_bytecode and __import__('os').environ['PYTHONDONTWRITEBYTECODE'] == '1'
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_changed_devices_still_rejected_before_raw_audit(tmp_path):
    old, _, data = fixture(tmp_path)
    with patch("reuse_campaign.source") as audit:
        with pytest.raises(ValueError, match=r"original \[1, 6\], current \[0, 3\]"):
            run_all.completed_run_source(dict(data, physical_devices=[0, 3]), old)
    audit.assert_not_called()
