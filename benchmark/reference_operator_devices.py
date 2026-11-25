"""Bind verified host device IDs to process-local Ascend ordinals."""

import re


def device_bindings(mapping, visible_devices, device_count):
    if not isinstance(visible_devices, str) or not re.fullmatch(r"[0-9]+,[0-9]+", visible_devices):
        raise ValueError("Reference operators require an explicit two-device Ascend visibility mask")
    logical = tuple(map(int, visible_devices.split(",")))
    physical = mapping["physical_cards"]
    if logical != tuple(mapping["logical_cards"]) or logical[0] >= logical[1]:
        raise ValueError("Reference operator visibility order differs from the admitted host mapping")
    if len(physical) != 2 or len(set(physical)) != 2:
        raise ValueError("Reference operators require two distinct leased physical cards")
    if type(device_count) is not int or device_count != len(logical):
        raise ValueError("Reference operator Torch device count differs from the admitted mask")
    return list(zip(range(device_count), logical, physical, strict=True))
