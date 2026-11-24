"""Validate revised r10 partition evidence without accepting partial ranks."""
import hashlib
import json
from pathlib import Path

import pytest

from benchmark import evaluation_client as client


@pytest.fixture
def audit():
    source = Path(__file__).resolve().parents[1] / "benchmark/one-click-kit"
    names = ("scripts/original_audit.py", "scripts/qualification.py", "scripts/report_execution.py")
    files = {name: (source / name).read_bytes() for name in names}
    client.revised_microflow_partition_bridge(files)
    module = {}
    exec(compile(files[names[0]], names[0], "exec"), module)
    return module


def evidence(tokens, change=None):
    def lines(rows):
        return "\n".join(json.dumps(row) for row in rows).encode()

    salt = "one"
    files = {
        "kvmat-policy.jsonl": lines([dict(runtime_effective_decision="recompute",
            runtime_control_path="request_scoped_prefix_cache_bypass", reused_tokens=0,
            runtime_decision_supported=True, runtime_cache_salt=salt, prompt_tokens=4096)]),
        "kvmat-engine.jsonl": lines([dict(event="school028_request_salt_admitted",
            cache_salt_sha256=hashlib.sha256(salt.encode()).hexdigest(), prompt_tokens=4096)]),
        "runtime-identity.json": json.dumps(dict(worker="corrected_serving_runtime.CorrectedWorker")).encode(),
    }
    for rank in (0, 1):
        records = [dict(event="partition_equivalence", rank=rank, tokens=count, exact=[True] * 5) for count in tokens]
        if rank == 1 and change is not None:
            change(records)
        files[f"microflow-{rank}.jsonl"] = lines([
            dict(event="model_loaded", rank=rank, active=True), *records,
            dict(event="prefix", split=True),
            dict(event="graph_replay", active=True, target=True, speculative_enabled=True),
        ])
    return files


def test_revised_profile_keeps_old_check_and_pins_the_new_source(audit):
    partition = audit["partition_tokens_for_release"]
    check = audit["audit_activation"]
    profile = {"release": client.R10_3072_REFERENCE_RELEASE}
    sizes = partition(profile, {client.R10_3072_MICROFLOW: client.R10_3072_MICROFLOW_SHA256})
    assert sizes == {1024, 2048, 3072, 4096}
    assert partition({"release": "full-answer-b1-20261121-r10"}, {}) is None
    assert check(evidence([1024, 2048, 4096]), prefix="", expected_requests=1)["microflow"]
    assert check(evidence(sorted(sizes)), prefix="", expected_requests=1,
                 expected_partition_tokens=sizes)["microflow"]
    with pytest.raises(ValueError, match="partition qualification failed"):
        check(evidence(sorted(sizes)), prefix="", expected_requests=1)
    with pytest.raises(ValueError, match="partition source differs"):
        partition(profile, {client.R10_3072_MICROFLOW: "changed"})


@pytest.mark.parametrize("change", [
    lambda rows: rows.pop(2),
    lambda rows: rows[2]["exact"].__setitem__(3, False),
    lambda rows: rows[2]["exact"].__setitem__(3, 1),
    lambda rows: rows[2].__setitem__("rank", 0),
    lambda rows: rows.append(dict(rows[2])),
    lambda rows: rows[2].__setitem__("tokens", 3584),
])
def test_revised_profile_rejects_bad_partition_rows(audit, change):
    sizes = {1024, 2048, 3072, 4096}
    with pytest.raises(ValueError, match="[Pp]artition"):
        audit["audit_activation"](evidence(sorted(sizes), change), prefix="", expected_requests=1,
                                 expected_partition_tokens=sizes)
