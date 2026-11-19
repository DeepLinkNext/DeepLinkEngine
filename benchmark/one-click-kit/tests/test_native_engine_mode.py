"""Reject the unsupported in-process State-cache seam before loading weights."""

import importlib.util
import os
from pathlib import Path
import sys

import pytest

KIT = Path(__file__).resolve().parents[1] / "third-party-test-kit"
sys.path.insert(0, str(KIT))
SPEC = importlib.util.spec_from_file_location(
    "mode_accounting", KIT / "native_m01_accounting.py"
)
accounting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(accounting)


def test_frozen_b1_cannot_silently_disable_its_maintenance_process(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "unchanged-on-rejection")
    config = dict(
        m01_in_process_engine=True,
        llm=dict(
            scheduler_cls="betterscale.models.qwen35.seat_scheduler.LiveStateScheduler",
            additional_config=dict(state_cache_host_bytes=4 * 2**30),
        ),
    )
    with pytest.raises(ValueError, match="EngineCoreProc"):
        accounting.prepare(config, "observe")
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "unchanged-on-rejection"
