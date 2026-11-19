"""Describe and verify the original fixed200 subset; never select new questions."""

import hashlib
import json
from pathlib import Path

PROTOCOL = Path(__file__).resolve().parents[1] / "protocols/mmlu-fixed200.json"
PROTOCOL_SHA256 = "8124492d6c59f53817116e27b32cd86c7b06c5ce59a45d1d77df56c080772c87"  # pragma: allowlist secret


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol():
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Fixed200 scope differs from its published digest")
    return json.loads(PROTOCOL.read_bytes())


def verify_cases(cases, expected):
    # A different set of200, reordered requests, or changed prompt/source bytes
    # must not inherit this workload's name or its comparison results.
    keys = ("case_id", "category", "source_row_sha256", "prompt_sha256")
    rows = [{key: case[key] for key in keys} for case in cases]
    if rows != expected["cases"]:
        raise ValueError(
            "Fixed200 ordered case identities or prompt/source binding differ"
        )


def verify(deployment):
    data = protocol()
    for name, digest in data["frozen_files_sha256"].items():
        if sha(deployment / name) != digest:
            raise ValueError("Original fixed200 deployment input differs: " + name)
    verified = json.loads(
        (deployment / "inputs/verified-mmlu-quality-inputs.json").read_bytes()
    )
    verify_cases(verified["cases"], data)
    return data


def description():
    data = protocol()
    return {
        key: data[key]
        for key in (
            "workload_name",
            "tasks_per_arm",
            "full_test_tasks",
            "category_count",
            "category_counts",
            "ordered_case_ids_sha256",
            "selection_policy",
            "generation",
            "result_scope",
        )
    } | {"protocol_sha256": PROTOCOL_SHA256}
