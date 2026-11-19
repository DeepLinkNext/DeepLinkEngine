import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third-party-test-kit"))
sys.path.insert(0, str(ROOT / "scripts"))
from canonical_protocol import current_plan
from canonical_results import audit
from run_canonical_reference import verify_device_owners


def sample(rep, replays=100):
    return dict(repetition=rep, replays=replays, device_ms=(1, 100, 2, 4, 3)[rep],
                host_seconds=.2, start_wall=100., end_wall=100.2)


def dense_fixture(tmp_path):
    case = dict(case=0, m=3, n=512, k=2048)
    plan, _ = current_plan("dense", [case], {"/sdk/source.py": "a"*64})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    for rank in range(2):
        row = dict(case=0, shape=case, check=dict(finite=True, close=True),
                   timings=[sample(rep) for rep in range(5)])
        record = dict(rank=rank, status="qualified_component_timings",
                      plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(), cases=[row])
        (tmp_path / f"rank-{rank}.json").write_text(json.dumps(record))
    return plan_path


def test_all_repetitions_and_physical_cards_survive_reduction(tmp_path):
    plan = dense_fixture(tmp_path)
    result = audit(tmp_path, plan, [1, 6])
    assert result["complete"] and not result["full_model_certificate"]
    assert result["efficiency_percent"] is None
    assert [r["physical_card"] for r in result["ranks"]] == [1, 6]
    timing = result["ranks"][0]["timings"][0]
    assert timing["samples_seconds"] == [.00001, .001, .00002, .00004, .00003]
    assert timing["seconds"] == .00003


@pytest.mark.parametrize("mutation", ["missing_repeat", "duplicate_repeat", "nan", "wrong_replays", "failed_oracle", "missing_oracle", "wrong_geometry", "wrong_rank", "foreign_plan"])
def test_bad_or_partial_canonical_results_cannot_become_component_timings(tmp_path, mutation):
    plan = dense_fixture(tmp_path)
    path = tmp_path / "rank-1.json"
    record = json.loads(path.read_bytes())
    row = record["cases"][0]
    if mutation == "missing_repeat": row["timings"].pop()
    if mutation == "duplicate_repeat": row["timings"][1]["repetition"] = 0
    if mutation == "nan": row["timings"][0]["device_ms"] = float("nan")
    if mutation == "wrong_replays": row["timings"][0]["replays"] = 10
    if mutation == "failed_oracle": row["check"]["close"] = False
    if mutation == "missing_oracle": row["check"] = {}
    if mutation == "wrong_geometry": row["shape"]["m"] = 4
    if mutation == "wrong_rank": record["rank"] = 0
    if mutation == "foreign_plan": record["plan_sha256"] = "b"*64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        audit(tmp_path, plan, [1, 6])


def test_foreign_container_owner_is_rejected_but_other_cards_are_not_owned():
    data = "| 1 0 | 999 | Worker | 25 | 123 |\n| 6 0 | 124 | Worker | 25 | 124 |\n| 5 0 | 125 | Other | 25 | 125 |"
    verify_device_owners(data, [1, 6], {123, 124})
    with pytest.raises(RuntimeError, match="Foreign"):
        verify_device_owners(data, [1, 6], {999, 124})


def test_moe_requires_every_operator_bucket_and_exact_checks_and_samples(tmp_path):
    cases = [dict(case=0, rows=24, group_counts=[3]*8+[0]*248)]
    plan, _ = current_plan("moe", cases, {"/sdk/source.py":"a"*64})
    path = tmp_path / "plan.json"; path.write_text(json.dumps(plan))
    for rank in range(2):
        buckets = [dict(operator=op, rows=24, cases=1, timed_cases=3,
                       weight_format=2, weight_stride=[op["k"]*op["n"],op["n"],1],
                       checks=[dict(case=0,samples=[dict(finite=True,close=True)])]) for op in plan["operators"]]
        record = dict(rank=rank,status="qualified_component_timings", buckets=buckets,
                      plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (tmp_path/f"rank-{rank}.json").write_text(json.dumps(record))
        timings = [dict(case=0,operator=op["name"],**sample(rep,1)) for op in plan["operators"] for rep in range(3)]
        (tmp_path/f"timings-{rank}.jsonl").write_text("".join(json.dumps(t)+"\n" for t in timings))
    result = audit(tmp_path,path,[1,6])
    assert len(result["ranks"][0]["timings"]) == 2
    record["buckets"].pop()
    (tmp_path/"rank-1.json").write_text(json.dumps(record))
    with pytest.raises(ValueError,match="Missing"):
        audit(tmp_path,path,[1,6])
