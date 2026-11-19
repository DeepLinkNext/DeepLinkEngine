"""Capture must close before the ordinary clean-runtime post-warmup audit."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reference_observation
import serving_warmup
from reference_http_client import requests_for_capture
from reference_capture_profile import profile
from run_b1_r4 import finish_client_window


def test_capture_closes_before_clean_identity_and_compile_snapshot(monkeypatch):
    state = dict(accounting_active=True)
    order = []
    def finish(*args):
        order.append("capture closed")
        state["accounting_active"] = False
        return dict(effective_flops=None)
    def verify(*args):
        if state["accounting_active"]:
            raise ValueError("Diagnostic worker accounting is active")
        order.append("clean runtime verified")
    monkeypatch.setattr(reference_observation, "finish", finish)
    monkeypatch.setattr(serving_warmup, "verify_after", verify)
    assert finish_client_window(Path("unused"), Path("model"), observation=True, observed_warmup=True) == dict(effective_flops=None)
    assert order == ["capture closed", "clean runtime verified"]


def test_ordinary_full_answer_completion_keeps_its_clean_runtime_audit(monkeypatch):
    calls = []
    monkeypatch.setattr(reference_observation, "finish", lambda *args: pytest.fail("Ordinary performance run enabled capture"))
    monkeypatch.setattr(serving_warmup, "verify_after", lambda *args: calls.append("verified"))
    assert finish_client_window(Path("unused"), Path("model"), observation=False, observed_warmup=True) is None
    assert calls == ["verified"]


def test_failed_capture_close_cannot_be_marked_complete(monkeypatch):
    def fail(*args):
        raise ValueError("Capture did not close")
    monkeypatch.setattr(reference_observation, "finish", fail)
    monkeypatch.setattr(serving_warmup, "verify_after", lambda *args: pytest.fail("Unclosed capture passed ordinary audit"))
    with pytest.raises(ValueError, match="did not close"):
        finish_client_window(Path("unused"), Path("model"), observation=True, observed_warmup=True)


def test_v2_changes_only_the_explicit_reference_copy_of_original_requests():
    original = dict(max_completion_tokens=8192, temperature=0, seed=0,
        requests=[dict(max_completion_tokens=8192, temperature=0, seed=0, messages=[dict(role="user", content=str(i))]) for i in range(200)])
    requests = requests_for_capture(original, profile("submission-v2"))
    assert all(row["max_completion_tokens"] == 256 for row in requests)
    assert all(row["max_completion_tokens"] == 8192 for row in original["requests"])
    assert [row["messages"] for row in requests] == [row["messages"] for row in original["requests"]]
    assert all(row["temperature"] == row["seed"] == 0 for row in requests)
    assert all(row["max_completion_tokens"] == 8192 for row in requests_for_capture(original, profile("full-answer-v1")))
