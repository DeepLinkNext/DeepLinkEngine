"""Keep operator timing bound to mask-local Torch ordinals."""

from pathlib import Path

import pytest

from benchmark.evaluation_client import REFERENCE_OPERATOR_SOURCES, reference_operator_devices_bridge
from benchmark.reference_operator_devices import device_bindings


@pytest.mark.parametrize("visible,logical", [("0,1", [0, 1]), ("2,3", [2, 3]), ("2,7", [2, 7])])
def test_maps_verified_host_ids_to_local_ordinals(visible, logical):
    assert device_bindings({"logical_cards": logical, "physical_cards": [6, 4]}, visible, 2) == [
        (0, logical[0], 6),
        (1, logical[1], 4),
    ]


@pytest.mark.parametrize(
    "logical,physical,visible,count,diagnostic",
    [
        ([2, 3], [4, 6], None, 2, "explicit two-device"),
        ([2, 3], [4, 6], "2,3,4", 3, "explicit two-device"),
        ([2, 3], [4, 6], "1,3", 2, "visibility order"),
        ([3, 2], [4, 6], "3,2", 2, "visibility order"),
        ([2, 2], [4, 6], "2,2", 2, "visibility order"),
        ([2, 3], [4, 4], "2,3", 2, "distinct leased physical cards"),
        ([2, 3], [4, 6], "2,3", 8, "Torch device count"),
        ([2, 3], [4, 6], "2,3", True, "Torch device count"),
    ],
)
def test_rejects_missing_or_inconsistent_device_admission(logical, physical, visible, count, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        device_bindings({"logical_cards": logical, "physical_cards": physical}, visible, count)


def test_revised_reference_binds_all_three_probes():
    root = Path(__file__).resolve().parents[1]
    base = root / "benchmark/one-click-kit"
    original = {
        "third-party-test-kit/" + name: (base / "third-party-test-kit" / name).read_bytes()
        for name in REFERENCE_OPERATOR_SOURCES
    }
    adapted = dict(original)
    reference_operator_devices_bridge(adapted)
    helper = adapted.pop("third-party-test-kit/reference_operator_devices.py")
    assert helper == (root / "benchmark/reference_operator_devices.py").read_bytes()
    assert adapted.keys() == original.keys()
    for name, after in adapted.items():
        text = after.decode()
        assert 'mapping = guard.verify_mapping(plan["logical_devices"])' in text
        assert "for device, logical, card in bindings:" in text
        assert "torch.npu.set_device(device)" in text
        assert "logical_device=logical, torch_device_ordinal=device" in text
        assert 'zip(map(int, plan["logical_devices"]' not in text
    changed = dict(original)
    key = "third-party-test-kit/qualify_route_capture.py"
    changed[key] += b"\n# changed source\n"
    with pytest.raises(ValueError, match="qualify_route_capture.py"):
        reference_operator_devices_bridge(changed)
