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
