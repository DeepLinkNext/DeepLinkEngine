import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from public_decode_tail import dense_shapes, domain
from dense_lower_graph import compile_graph
from test_public_request_work import row
from test_semantic_work import model


def geometry():
    a, b = row(7, [1, 3, 2, 1]), row(9, [1, 2, 3])
    b["request_ordinal"] = 1
    for event in b["ordered_public_emissions"]:
        event["public_step"] += 2
    return dict(requests=[a, b], geometry_sha256="a" * 64)


def test_cut_is_based_only_on_all_first_outputs_and_preserves_every_later_delivery():
    source = geometry()
    original = copy.deepcopy(source)
    tail = domain(source, execution_seats=36)
    assert source == original
    assert tail["cutoff_public_step"] == 2
    assert [w["public_step"] for w in tail["waves"]] == [4, 6]
    assert [w["necessary_target_rows"] for w in tail["waves"]] == [4, 4]
    assert tail["target_rows"] == 8
    source["T_seconds"] = 999
    assert domain(source, execution_seats=36) == tail
    assert not tail["prompt_partition_assumed"]


def inputs():
    tail = domain(geometry(), execution_seats=36)
    cases = dense_shapes(tail, model())
    calibration = dict(family="dense", complete=True, plan_sha256="b" * 64,
        ranks=[dict(rank=rank, timings=[dict(case=c["case"], seconds=.01 * (rank + 1))
                                      for c in cases]) for rank in (0, 1)])
    return tail, cases, calibration


def test_only_exact_calibrated_dense_leaves_and_explicit_dependency_joins_are_weighted():
    tail, cases, calibration = inputs()
    graph = compile_graph(tail, model(), cases, calibration)
    # Per wave/rank:30 GDN layers *3 projections +9 FA layers *2.
    leaves = [n for n in graph["nodes"] if n["kind"] == "calibrated_dense"]
    joins = [n for n in graph["nodes"] if n["kind"] == "dependency_join"]
    assert len(leaves) == 2 * 2 * (30 * 3 + 9 * 2)
    assert len(joins) == 2 * 39
    assert all(n["seconds"] > 0 and n["shape"]["m"] == 4 for n in leaves)
    assert all(n["seconds"] == 0 and len(n["parents"]) == 2 and not n["resources"] for n in joins)
    assert all(p < n["id"] for n in graph["nodes"] for p in n["parents"])
    assert not graph["missing_duration_assigned_zero"]
    assert graph["reference_seconds"] is None


def test_missing_rank_or_shape_never_becomes_zero():
    tail, cases, calibration = inputs()
    calibration["ranks"][0]["timings"].pop()
    with pytest.raises(ValueError, match="no qualified rank duration"):
        compile_graph(tail, model(), cases, calibration)
    with pytest.raises(ValueError, match="missing exact"):
        compile_graph(tail, model(), cases[:-1], calibration)


def test_empty_tail_remains_empty_without_invented_positive_reference():
    source = dict(requests=[row(7, [1])], geometry_sha256="a" * 64)
    tail = domain(source, execution_seats=36)
    assert tail["waves"] == []
    assert dense_shapes(tail, model()) == []


def test_foreign_architecture_is_not_silently_assigned_reviewed_projection_dimensions():
    tail, _, _ = inputs()
    changed = model()
    changed.hidden_size = 4096
    with pytest.raises(ValueError, match="projection dimensions"):
        dense_shapes(tail, changed)
