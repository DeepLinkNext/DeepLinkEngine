import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from audit_dense_lower_graph import audit
from dense_lower_graph import compile_graph
from test_public_decode_tail import inputs
from test_semantic_work import model


def test_independent_inventory_and_decimal_critical_path_replay():
    tail, cases, calibration = inputs()
    graph = compile_graph(tail, model(), cases, calibration)
    result = audit(graph, tail, model(), cases, calibration)
    assert result["graph_integrity_passed"]
    assert result["dense_leaves"] == 432
    assert result["dependency_joins"] == 78
    assert result["reference_model_lower_seconds"] == "4.32"
    assert result["critical_path"]
    assert not result["source_applicability_certified"]


@pytest.mark.parametrize("change", ["resource_edge", "join_edge", "duration", "shape", "zero", "missing_leaf"])
def test_corrupt_graph_cannot_produce_a_lower_bound(change):
    tail, cases, calibration = inputs()
    graph = copy.deepcopy(compile_graph(tail, model(), cases, calibration))
    leaf = next(n for n in graph["nodes"] if n["kind"] == "calibrated_dense" and n["layer"] == 2)
    if change == "resource_edge":
        leaf["parents"] = []
    elif change == "join_edge":
        next(n for n in graph["nodes"] if n["kind"] == "dependency_join")["parents"].pop()
    elif change == "duration":
        leaf["seconds"] *= 10
    elif change == "shape":
        leaf["shape"]["m"] += 1
    elif change == "zero":
        leaf["seconds"] = 0
    else:
        graph["nodes"].remove(leaf)
    with pytest.raises(ValueError):
        audit(graph, tail, model(), cases, calibration)
