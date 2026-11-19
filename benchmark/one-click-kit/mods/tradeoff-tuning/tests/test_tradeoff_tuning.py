import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "tradeoff_tuning", Path(__file__).parents[1] / "tradeoff_tuning.py"
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def row(trial, speed=1000, correct=150):
    return dict(
        id=trial,
        completed=True,
        quality_measured=True,
        planned=200,
        recorded=200,
        successful=200,
        unreturned=0,
        correct=correct,
        truncated=20,
        unparsed=0,
        output_tokens=600000,
        duration_seconds=600,
        output_tokens_per_second=speed,
        mean_tpot_ms=20,
        mean_ttft_ms=200000,
        tpot_p95_ms=22,
        ttft_p95_ms=400000,
        seal_sha256="a" * 64,
    )


def catalog(tmp_path, groups=None):
    data = dict(
        schema="vllm-hust-capacity-analysis/v2",
        protocol="test",
        all_repetitions_included=True,
        rows=[row("baseline"), row("candidate1", 1200), row("candidate2", 900, 130)],
    )
    source = tmp_path / "analysis.json"
    source.write_text(json.dumps(data))
    manifest = dict(
        schema="vllm-hust-tradeoff-catalog/v1",
        scope={},
        sources=[
            dict(
                path="analysis.json",
                sha256=mod.digest(source.read_bytes()),
                trial_ids=[r["id"] for r in data["rows"]],
            )
        ],
        configurations=groups
        or [
            dict(id="all-off-baseline", family="capacity", trial_ids=["baseline"]),
            dict(
                id="candidate",
                family="capacity",
                trial_ids=["candidate1", "candidate2"],
            ),
        ],
    )
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, source, data


def budgets():
    return dict(max_tpot_ms=21, max_ttft_ms=220000, min_correct=145, max_truncated=25)


def test_keep_all_repetitions_and_all_off(tmp_path):
    path, _, _, _ = catalog(tmp_path)
    evidence = mod.load_catalog(path, tmp_path)
    result = mod.shortlist(evidence, budgets())
    assert evidence["physical_trials"] == 3
    assert result["results"][0]["all_observed_repetitions_meet_constraints"]
    candidate = result["results"][1]
    assert not candidate["all_observed_repetitions_meet_constraints"]
    assert candidate["reasons"][0]["trial"] == "candidate2"
    assert not any(r["deployable"] for r in result["results"])


def test_deduplicate_only_exact_physical_references(tmp_path):
    path, manifest, _, _ = catalog(tmp_path)
    manifest["sources"].append(dict(manifest["sources"][0]))
    path.write_text(json.dumps(manifest))
    data = mod.load_catalog(path, tmp_path)
    assert data["physical_trials"] == 3
    assert data["source_row_references"] == 6
    assert data["rows"][0]["source_indices"] == [0, 1]


def test_reject_conflicting_reused_measurement(tmp_path):
    path, manifest, _, data = catalog(tmp_path)
    data["rows"][0]["mean_tpot_ms"] += 1
    other = tmp_path / "other.json"
    other.write_text(json.dumps(data))
    manifest["sources"].append(
        dict(
            manifest["sources"][0],
            path="other.json",
            sha256=mod.digest(other.read_bytes()),
        )
    )
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Conflicting reuse"):
        mod.load_catalog(path, tmp_path)


def test_reject_changed_hash_and_dropped_repetition(tmp_path):
    path, manifest, source, data = catalog(tmp_path)
    source.write_text("{}")
    with pytest.raises(ValueError, match="source changed"):
        mod.load_catalog(path, tmp_path)
    source.write_text(json.dumps(data))
    manifest["sources"][0]["trial_ids"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="every declared outcome"):
        mod.load_catalog(path, tmp_path)


def test_no_failed_zero_scores_or_timings(tmp_path):
    path, manifest, source, data = catalog(tmp_path)
    data["rows"][2].update(
        completed=False,
        quality_measured=False,
        timing_eligible=False,
        state="sealed-failed",
        failure="NPU graph failure",
    )
    source.write_text(json.dumps(data))
    manifest["sources"][0]["sha256"] = mod.digest(source.read_bytes())
    path.write_text(json.dumps(manifest))
    evidence = mod.load_catalog(path, tmp_path)
    failed = evidence["rows"][2]
    assert failed["correct"] is None and failed["output_tokens_per_second"] is None
    assert failed["failure"] == "NPU graph failure"
    assert evidence["physical_trials"] == 3 and evidence["admitted_trials"] == 2
    assert not mod.shortlist(evidence, budgets())["results"][1][
        "all_observed_repetitions_meet_constraints"
    ]


def test_unknown_schema_and_incomplete200_are_rejected(tmp_path):
    r = row("incomplete")
    r["recorded"] = 199
    with pytest.raises(ValueError, match="Incomplete"):
        mod.normalize(r, "vllm-hust-capacity-analysis/v2")
    path, manifest, source, data = catalog(tmp_path)
    data["schema"] = "unknown"
    source.write_text(json.dumps(data))
    manifest["sources"][0]["sha256"] = mod.digest(source.read_bytes())
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported"):
        mod.load_catalog(path, tmp_path)


def test_filters_need_finite_budgets(tmp_path):
    path, _, _, _ = catalog(tmp_path)
    data = mod.load_catalog(path, tmp_path)
    with pytest.raises(ValueError, match="Finite"):
        mod.shortlist(data, dict(budgets(), max_tpot_ms=float("nan")))
    with pytest.raises(ValueError, match="integer"):
        mod.shortlist(data, dict(budgets(), min_correct=200.1))


def test_every_physical_trial_assigned_once(tmp_path):
    path, manifest, _, _ = catalog(tmp_path)
    manifest["configurations"][1]["trial_ids"].remove("candidate2")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Assign every"):
        mod.load_catalog(path, tmp_path)
