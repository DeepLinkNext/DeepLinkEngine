"""Validate and disclose sealed evidence carried between Ascend 910B2 pairs."""

import json
from pathlib import Path


def read_mapping(root):
    return json.loads((Path(root) / "device-mapping.json").read_bytes())


def compare(original, current, *, allowed):
    """Return both real mappings for an explicitly authorized device change."""
    if original["physical_cards"] == current["physical_cards"]:
        return None
    if allowed is not True:
        raise ValueError("Reuse uses another physical card pair; explicit device-change admission is required")
    from full_reference_reuse import qualify_mapping

    for mapping in (original, current):
        qualify_mapping(mapping, "Ascend 910B2")
    if original["logical_cards"] != current["logical_cards"]:
        raise ValueError("Device-change reuse requires the original logical rank mapping")
    return {"original": original, "current": current}


def for_campaign(root, previous, current):
    if previous["physical_cards"] == current["physical_cards"]:
        return None
    original = read_mapping(root)
    actual = current.get("device_reuse_mapping")
    proof = compare(original, actual or {"physical_cards": current["physical_cards"]},
                    allowed=actual is not None)
    for mapping, plan in ((original, previous), (actual, current)):
        if (mapping["physical_cards"] != plan["physical_cards"]
                or mapping["logical_cards"] != list(map(int, plan["logical_devices"].split(",")))):
            raise ValueError("Device-change mapping differs from the frozen plan")
    return proof


def for_continuous(root, qualification, logical, *, allowed):
    from resource_guard import cards, verify_mapping

    if qualification["physical_cards"] == cards():
        return None
    if allowed is not True:
        raise ValueError("Continuous qualification belongs to another physical NPU pair")
    original = read_mapping(root)
    if original["physical_cards"] != qualification["physical_cards"]:
        raise ValueError("Continuous qualification and sealed device mapping differ")
    return compare(original, verify_mapping(logical), allowed=True)
