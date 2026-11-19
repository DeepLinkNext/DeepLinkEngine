"""Lossless type tags for non-JSON EngineArgs values in audit receipts."""

from enum import Enum
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path


def encode(value):
    if is_dataclass(value) and not isinstance(value, type):
        return dict(native_type=type(value).__module__ + "." + type(value).__qualname__,
                    fields={field.name: encode(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, Enum):
        return dict(native_type=type(value).__module__ + "." + type(value).__qualname__, value=encode(value.value))
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Nonfinite native EngineArgs value")
        return value
    if isinstance(value, Path):
        return dict(native_type="pathlib.Path", value=str(value))
    if type(value) in (set, frozenset):
        items = [encode(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True))
        return dict(native_type=type(value).__name__, values=items)
    if type(value) in (list, tuple):
        items = [encode(item) for item in value]
        return items if type(value) is list else dict(native_type="tuple", values=items)
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("Non-string EngineArgs mapping key")
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        items = [[encode(key), encode(item)] for key, item in value.items()]
        items.sort(key=lambda item: json.dumps(item[0], sort_keys=True))
        return dict(native_type=type(value).__module__ + "." + type(value).__qualname__, entries=items)
    raise TypeError("Unsupported native EngineArgs audit value: " + type(value).__module__ + "." + type(value).__qualname__)
