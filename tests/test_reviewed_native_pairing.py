"""A sealed original018 B0 arm may pair with every reviewed B1 reference release."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmark/one-click-kit/scripts")]

from test_context_baseline_reuse import combined


def test_reviewed_releases_reach_the_native_plan_gate():
    text = combined()["third-party-test-kit/qualify_baseline_capacity.py"].decode()
    assert 'plan["paired_engine_release"] not in (' in text
    for release in ("full-answer-b1-20261117-r8", "full-answer-b1-20261119-r9"):
        assert release in text


def test_the_native_plan_gate_no_longer_pins_a_single_release():
    text = combined()["third-party-test-kit/qualify_baseline_capacity.py"].decode()
    assert 'plan["paired_engine_release"] != "full-answer-b1-20261117-r8"' not in text


def test_the_kit_source_stays_the_reviewed_original():
    source = (ROOT / "benchmark/one-click-kit/third-party-test-kit/qualify_baseline_capacity.py").read_text()
    assert 'plan["paired_engine_release"] != "full-answer-b1-20261117-r8"' in source
