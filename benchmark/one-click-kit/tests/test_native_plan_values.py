from enum import Enum
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from native_plan_values import encode


def test_native_args_sets_and_enums_have_stable_lossless_type_tags():
    class Mode(Enum):
        FULL = "FULL"
    value = dict(captures={48, 3, 12}, mode=Mode.FULL, path=Path("/fixture"), fixed=(1, 2))
    actual = encode(value)
    assert json.loads(json.dumps(actual)) == actual
    assert actual["captures"] == dict(native_type="set", values=[12, 3, 48])
    assert actual["mode"]["value"] == "FULL"
    assert actual["mode"]["native_type"].endswith("Mode")
    assert actual["fixed"] == dict(native_type="tuple", values=[1, 2])
    assert encode(dict(captures={12, 48, 3}))["captures"] == actual["captures"]


@pytest.mark.parametrize("value", [object(), lambda: None, {1: "unsupported key"}])
def test_unknown_effective_config_is_not_hidden_by_repr(value):
    with pytest.raises(TypeError):
        encode(value)


def test_nonfinite_effective_values_are_rejected():
    with pytest.raises(ValueError):
        encode(float("nan"))


def test_engine_compilation_counters_retain_key_types_and_counts():
    actual = encode(Counter({32: 4, "compile": 2}))
    assert actual == dict(native_type="collections.Counter", entries=[["compile", 2], [32, 4]])
    assert encode(Counter()) == dict(native_type="collections.Counter", entries=[])


def test_nested_dataclass_does_not_reconstruct_counter_as_an_iterable():
    @dataclass
    class Configuration:
        counters: Counter
    original = Configuration(Counter({"compile": 7}))
    actual = encode(original)
    assert actual["fields"]["counters"] == dict(native_type="collections.Counter", entries=[["compile", 7]])
    assert original.counters == {"compile": 7}
