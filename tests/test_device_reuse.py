"""Test the opt-in recovery scope and selected-client device admission."""

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

import run_all
from device_reuse import compare, for_campaign
from test_multi_source_reuse import bridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark/one-click-kit/scripts"))


def mapping(physical, logical=(0, 1), accelerator="Ascend 910B2"):
    return dict(physical_cards=list(physical), logical_cards=list(logical),
                npu_smi_mapping="\n".join(
                    f"{card} 0 {rank} {accelerator}" for card, rank in zip(physical, logical)
                ))


def test_device_change_preserves_original_and_requires_matching_model_and_ranks():
    original = mapping((1, 6))
    current = mapping((0, 3))
    with pytest.raises(ValueError, match="explicit device-change"):
        compare(original, current, allowed=False)
    assert compare(original, current, allowed=True) == {"original": original, "current": current}
    with pytest.raises(ValueError, match="accelerator"):
        compare(original, mapping((0, 3), accelerator="Ascend 910C"), allowed=True)
    with pytest.raises(ValueError, match="logical rank"):
        compare(original, mapping((0, 3), logical=(1, 0)), allowed=True)


def test_campaign_reuse_binds_both_mappings_to_their_plans(tmp_path):
    original = mapping((1, 6))
    current = mapping((0, 3))
    (tmp_path / "device-mapping.json").write_text(json.dumps(original))
    previous = dict(physical_cards=[1, 6], logical_devices="0,1")
    plan = dict(physical_cards=[0, 3], logical_devices="0,1")
    with pytest.raises(ValueError, match="explicit device-change"):
        for_campaign(tmp_path, previous, plan)
    with pytest.raises(ValueError, match="frozen plan"):
        for_campaign(tmp_path, previous, dict(plan, device_reuse_mapping=mapping((4, 5))))
    proof = for_campaign(tmp_path, previous, dict(plan, device_reuse_mapping=current))
    assert proof == {"original": original, "current": current}


def test_full_entry_admits_new_cache_but_still_requires_device_admission(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    previous = dict(model="/model", cache="/cache", physical_devices=[1, 6], out=str(source))
    ledger = source / "results.json"
    ledger.write_text(json.dumps(dict(configuration=previous)))
    current = dict(previous, physical_devices=[0, 3], out=str(tmp_path / "new"),
                   allow_device_change=True)
    with patch("recovery_source.stopped"), patch("reuse_campaign.audit_stage"), patch(
        "reuse_campaign.source", return_value=(source, {}, dict(status="failed", stages=[
            dict(name="frontier-solver-B1", status="completed")]), {})
    ), patch("release_selection.active", return_value=None):
        result = run_all.completed_run_source(current, source)
        assert result["physical_device_change"] == dict(original=[1, 6], current=[0, 3])
        relocated = run_all.completed_run_source(dict(current, cache="/other"), source)
        assert relocated["physical_device_change"] == result["physical_device_change"]
        with pytest.raises(ValueError, match="configuration differs: model"):
            run_all.completed_run_source(dict(current, model="/other-model"), source)
    assert json.loads(ledger.read_bytes())["configuration"] == previous


def test_whole_mmlu_pair_survives_a_device_change_with_the_same_runtime(tmp_path):
    _, files = bridge()
    module = {}
    exec(files["scripts/reuse_campaign.py"], module)
    original = mapping((1, 6))
    current = mapping((0, 3))
    (tmp_path / "device-mapping.json").write_text(json.dumps(original))
    previous = dict(physical_cards=[1, 6], logical_devices="0,1", model="/model",
                    baseline_source_map_sha256="pin", optimized_deployment="/sdk",
                    optimized_dependencies={"runtime": "r9"},
                    mmlu_inputs=dict(quality_plan_sha256="plan", scorer_sha256="score"))
    plan = dict(previous, physical_cards=[0, 3], device_reuse_mapping=current)
    module["compatible"]("mmlu-pair", previous, plan, tmp_path)
    with pytest.raises(ValueError, match="runtime/model"):
        module["compatible"]("mmlu-pair", previous, dict(plan, optimized_deployment="/other"), tmp_path)
    with pytest.raises(ValueError, match="scorer"):
        module["compatible"]("mmlu-pair", previous, dict(plan, mmlu_inputs=dict(
            quality_plan_sha256="plan", scorer_sha256="modified")), tmp_path)

def test_completed_run_carries_only_sealed_native_progress(tmp_path):
    source = tmp_path / "source"
    compute = source / "fixed-five/compute-reference"
    compute.mkdir(parents=True)
    configured = dict(model="/model", cache="/cache", physical_devices=[0, 3], out=str(source))
    (source / "results.json").write_text(json.dumps(dict(status="failed", configuration=configured)))
    (compute / "automatic-result.json").write_text(json.dumps(dict(stages=[
        dict(id="current-native-synchronized-timing", status="completed")
    ])))
    current = dict(configured, out=str(tmp_path / "new"), allow_device_change=True)
    campaign = dict(status="completed", selected_execution_complete=True, stages=[
        dict(name="hle-solver-B1", status="completed")
    ])
    bindings = {"compute-reference/checksums.json": "sealed",
                "compute-reference/native-current-timing/checksums.json": "sealed"}
    with patch("reuse_campaign.audit_stage"), patch("reuse_campaign.source",
            return_value=(compute.parent, {}, campaign, bindings)), patch("release_selection.active", return_value=None):
        assert run_all.completed_run_source(current, source)["compute_resume_source"] == str(compute)
        bindings.pop("compute-reference/native-current-timing/checksums.json")
        assert "compute_resume_source" not in run_all.completed_run_source(current, source)


def test_context_pair_renderer_replays_cross_card_proof(tmp_path):
    from types import ModuleType

    _, files = bridge()
    kit = Path(__file__).resolve().parents[1] / "benchmark/one-click-kit"
    suite = ModuleType("acceptance_suite")
    suite.__file__ = str(kit / "scripts/acceptance_suite.py")
    exec(files["scripts/acceptance_suite.py"], suite.__dict__)
    renderer = {"__file__": str(kit / "scripts/acceptance_report.py")}
    exec(files["scripts/acceptance_report.py"], renderer)

    arms = {role: tmp_path / role for role in ("B0", "B1")}
    original, current = mapping((1, 6)), mapping((0, 3))
    for role, device_mapping in (("B0", original), ("B1", current)):
        arm = arms[role]
        arm.mkdir()
        (arm / "device-mapping.json").write_text(json.dumps(device_mapping))
        (arm / "resource-admission.json").write_text(json.dumps(device_mapping))
        (arm / "model-weights.json").write_text('{"weights": "fixture"}')
        (arm / "checksums.json").write_text("{}")
    (arms["B1"] / "service").mkdir()
    (arms["B1"] / "service/plan.json").write_text(json.dumps({
        "acceptance_suite": {"baseline_run": str(arms["B0"].resolve())},
    }))

    def inspect(arm):
        return dict(role=arm.name, execution_complete=True, contract={}, phases=[])

    with patch.dict(sys.modules, acceptance_suite=suite), patch.object(suite, "inspect", inspect):
        summary = suite.paired(arms, device_change=compare(original, current, allowed=True))
        source = tmp_path / "paired-summary.json"
        source.write_text(json.dumps(summary))
        # The destination check is reached only after the summary matches replay.
        with pytest.raises(FileExistsError):
            renderer["report_pair"](source, tmp_path)

        summary["device_change"]["current"]["physical_cards"] = [2, 4]
        source.write_text(json.dumps(summary))
        with pytest.raises(ValueError, match="Paired context device provenance differs"):
            renderer["report_pair"](source, tmp_path / "new-report")
        assert not (tmp_path / "new-report").exists()
