import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from audit_loaded_work_inventory import validate


def fixture():
    # Shape/scalar metadata only, extracted from a real loaded r8 model.
    # Two target layers retain both families; tensor values and paths excluded.
    return json.loads((Path(__file__).parent / "fixtures/loaded-work-dimensions-two-layer.json").read_bytes())


def test_real_loaded_projection_and_scalar_metadata_match_the_semantic_inventory():
    result = validate(fixture(), dict(layer_types=["linear_attention", "full_attention"]))
    assert result["loaded_semantic_dimensions_and_dispatch_passed"]
    assert result["checked_modules"]["target"] > result["checked_modules"]["draft"] > 15


@pytest.mark.parametrize("change", ["quantization", "transposed_experts", "routing", "rotary", "mtp_fc", "missing_layer", "duplicate"])
def test_subtle_loaded_dispatch_or_geometry_changes_are_rejected(change):
    data = fixture()
    target = {r["name"]: r for r in data["target"]["modules"]}
    draft = {r["name"]: r for r in data["draft"]["modules"]}
    if change == "quantization":
        target["language_model.model.layers.0.linear_attn.in_proj_ba"]["objects"]["quant_method"]["type"] = "fp8"
    elif change == "transposed_experts":
        target["language_model.model.layers.1.mlp.experts.routed_experts"]["parameters"]["w2_weight"]["shape"] = [256, 2048, 256]
    elif change == "routing":
        target["language_model.model.layers.0.mlp.experts.routed_experts"]["scalars"]["renormalize"] = False
    elif change == "rotary":
        target["language_model.model.layers.1.self_attn.rotary_emb"]["scalars"]["rotary_dim"] = 128
    elif change == "mtp_fc":
        draft["model.fc"]["parameters"]["weight"]["shape"] = [2048, 2048]
    elif change == "missing_layer":
        data["target"]["modules"] = [r for r in data["target"]["modules"] if "layers.1." not in r["name"]]
    else:
        data["draft"]["modules"].append(data["draft"]["modules"][-1])
    with pytest.raises(ValueError):
        validate(data, dict(layer_types=["linear_attention", "full_attention"]))
