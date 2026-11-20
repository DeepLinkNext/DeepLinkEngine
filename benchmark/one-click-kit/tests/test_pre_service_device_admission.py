"""A foreign owner arriving during checkpoint hashing must not race model startup."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_b1_r4 as runner


def test_waits_for_foreign_owner_without_spawning_or_stopping_it(tmp_path, monkeypatch):
    outputs = iter(["| 1 0 | 123 | python | 123 |", "No running processes found in NPU 1\nNo running processes found in NPU 6"])
    calls = []
    def sample(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=next(outputs))
    monkeypatch.setattr(runner, "cards", lambda: [1, 6])
    monkeypatch.setattr(runner.subprocess, "run", sample)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("No model spawn in device wait"))
    runner.wait_before_service(tmp_path, 10)
    assert calls == [["npu-smi", "info"], ["npu-smi", "info"]]
    import json
    receipt = json.loads((tmp_path / "pre-service-device-admission.json").read_bytes())
    assert receipt["clear"] is True
    assert receipt["model_service_started"] is False
    assert receipt["foreign_processes_stopped"] is False


def test_foreign_owner_timeout_preserves_no_model_start(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "cards", lambda: [1, 6])
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="another process"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("No model spawn"))
    with pytest.raises(TimeoutError, match="no model service spawned"):
        runner.wait_before_service(tmp_path, 0)


def progress(path, **overrides):
    import json
    value = dict(schema="baseline-source-preparation/v1", status="verifying",
                 started_monotonic=101, model_service_started=False)
    value.update(overrides)
    (path / "source-preparation.json").write_text(json.dumps(value))


def test_slow_cpu_verification_is_not_a_device_queue_timeout(tmp_path):
    progress(tmp_path)
    assert runner.service_admission_phase(tmp_path, "B0", 100, 2500, 1800) == "verifying-baseline"


def test_source_verification_has_its_own_finite_bound(tmp_path):
    progress(tmp_path)
    with pytest.raises(TimeoutError, match="source verification exceeded7200s"):
        runner.service_admission_phase(tmp_path, "B0", 100, 7302, 1800)


def test_source_completion_starts_the_original_lease_wait(tmp_path):
    progress(tmp_path, status="completed", completed_monotonic=2500)
    assert runner.service_admission_phase(tmp_path, "B0", 100, 4330, 1800) == "admitting"
    with pytest.raises(TimeoutError, match="Device admission wait expired"):
        runner.service_admission_phase(tmp_path, "B0", 100, 4331, 1800)


@pytest.mark.parametrize("role", ["B0", "B1"])
def test_missing_progress_keeps_original_wait_timeout(tmp_path, role):
    with pytest.raises(TimeoutError, match="Device admission wait expired"):
        runner.service_admission_phase(tmp_path, role, 100, 1931, 1800)


def test_b1_cannot_use_a_baseline_progress_marker_to_extend_wait(tmp_path):
    progress(tmp_path)
    with pytest.raises(TimeoutError, match="Device admission wait expired"):
        runner.service_admission_phase(tmp_path, "B1", 100, 2500, 1800)


@pytest.mark.parametrize("invalid", [
    {"started_monotonic": 99}, {"started_monotonic": True},
    {"started_monotonic": float("nan")}, {"started_monotonic": 3000},
    {"schema": "unknown"}, {"status": "unknown"},
    {"model_service_started": True},
    {"status": "completed", "completed_monotonic": 100},
    {"status": "completed", "completed_monotonic": 3000},
])
def test_malformed_or_stale_preparation_progress_is_rejected(tmp_path, invalid):
    progress(tmp_path, **invalid)
    with pytest.raises(ValueError, match="baseline source preparation"):
        runner.service_admission_phase(tmp_path, "B0", 100, 2500, 1800)
