"""Exercise both real entry admission functions before any model import."""

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_capture_profile import profile


def entry(name, monkeypatch):
    monkeypatch.delenv("VLLMHUST_NATIVE_OBSERVER_PLAN", raising=False)
    spec = importlib.util.spec_from_file_location("entry_admission_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan(name, baseline):
    capture = profile(name)
    return dict(schema="baseline-native-m01-timing/v1" if baseline else "current-native-m01-timing/v1",
        role="B0" if baseline else "B1", reference_capture_profile=name,
        max_tokens=capture["max_tokens"], formal_repetitions=capture["formal_repetitions"],
        sampling=dict(max_tokens=capture["max_tokens"], temperature=0, seed=0, ignore_eos=False,
            top_p=1.0, top_k=-1, presence_penalty=0.0, detokenize=False))


@pytest.mark.parametrize("name", ["submission-v2", "full-answer-v1"])
@pytest.mark.parametrize("baseline", [False, True])
def test_actual_native_entries_accept_only_the_declared_capture(name, baseline, monkeypatch):
    module = entry("baseline_native_entry" if baseline else "native_timing_entry", monkeypatch)
    kwargs = dict(no_site=True) if baseline else {}
    value = plan(name, baseline)
    assert module.validate_config(value, **kwargs) == profile(name)
    value["max_tokens"] = 8192 if name == "submission-v2" else 256
    with pytest.raises(ValueError):
        module.validate_config(value, **kwargs)


@pytest.mark.parametrize("baseline", [False, True])
@pytest.mark.parametrize("field,value", [("formal_repetitions", 1), ("formal_repetitions", 3.0), ("max_tokens", 256.0)])
def test_native_entries_reject_wrong_or_noninteger_submission_controls(baseline, field, value, monkeypatch):
    module = entry("baseline_native_entry" if baseline else "native_timing_entry", monkeypatch)
    document = plan("submission-v2", baseline)
    document[field] = value
    with pytest.raises(ValueError):
        module.validate_config(document, **(dict(no_site=True) if baseline else {}))


def test_original018_still_requires_the_isolated_interpreter(monkeypatch):
    module = entry("baseline_native_entry", monkeypatch)
    with pytest.raises(ValueError, match="isolated interpreter"):
        module.validate_config(plan("submission-v2", True), no_site=False)
