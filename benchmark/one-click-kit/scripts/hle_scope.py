"""Prospectively fixed HLE subset; preserve original prompts and scoring budgets."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/hle-fixed200.json"
PROTOCOL_SHA256 = "ffcdd5e27b0822937b0f60aea1ca7c88eac82a39fd9d55c50f41c797ba676282"  # pragma: allowlist secret -- public protocol SHA256
STRATA = (
    ("gold", "text"),
    ("gold", "image"),
    ("revision", "text"),
    ("revision", "image"),
)
POPULATION = (575, 93, 1025, 118)


def allocate(total=200):
    """Largest-remainder proportional quotas, with fixed stratum-order ties."""
    quotas = [total * n // sum(POPULATION) for n in POPULATION]
    order = sorted(
        range(4), key=lambda i: (-(total * POPULATION[i] % sum(POPULATION)), i)
    )
    for i in order[: total - sum(quotas)]:
        quotas[i] += 1
    return quotas


def select(tasks):
    if len(tasks) != sum(POPULATION) or len({t["case_id"] for t in tasks}) != len(
        tasks
    ):
        raise ValueError("HLE source must contain exactly1811 distinct original tasks")
    selected = set()
    for (track, modality), count, quota in zip(
        STRATA, POPULATION, allocate(), strict=True
    ):
        rows = [
            t
            for t in tasks
            if t["track"] == track and bool(t["image"]) == (modality == "image")
        ]
        if len(rows) != count:
            raise ValueError("HLE source track/modality population changed")
        selected.update(rows[i * count // quota]["case_id"] for i in range(quota))
    return [t for t in tasks if t["case_id"] in selected]


def entries(tasks):
    result = []
    for task in tasks:
        raw = json.dumps(
            task, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        result.append(
            {
                "case_id": task["case_id"],
                "track": task["track"],
                "modality": "image" if task["image"] else "text",
                "source_row_index": task["row_index"],
                "task_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return result


def protocol():
    raw = PROTOCOL.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("HLE sampling protocol differs from its source pin")
    return json.loads(raw)


def verify_entries(tasks, expected):
    if entries(tasks) != expected:
        raise ValueError("HLE fixed200 IDs/order/original task bytes differ")


def verify_selected(plan, contract, tasks):
    scope = contract.get("execution_scope", {})
    if scope.get("profile") != "fixed200-v1":
        if len(tasks) != 1811:
            raise ValueError("Unrecognised HLE subset; full1811 required")
        return
    expected = protocol()
    if scope.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("HLE contract sampling identity differs")
    if (Path(plan) / "sample-scope.json").read_bytes() != PROTOCOL.read_bytes():
        raise ValueError("Archived HLE sampling protocol changed")
    verify_entries(tasks, expected["cases"])


def apply(plan, contract, profile):
    """Change the task scope before its final contract hash or any model startup."""
    from dataset_contract import save, sha

    plan = Path(plan)
    tasks = json.loads((plan / "tasks.json").read_bytes())
    if profile == "full1811":
        if len(tasks) != 1811 or sum(bool(t["image"]) for t in tasks) != 211:
            raise ValueError("Original full HLE scope changed")
        scope = {
            "profile": profile,
            "source_tasks": 1811,
            "selected_tasks": 1811,
            "image_tasks": 211,
        }
        contract["execution_scope"] = scope
        return scope
    if profile != "fixed200-v1":
        raise ValueError("Unknown explicit HLE execution profile")
    selected = select(tasks)
    expected = protocol()
    verify_entries(selected, expected["cases"])
    ids = {t["case_id"] for t in selected}
    save(plan / "tasks.json", selected)
    save(
        plan / "excluded.json",
        [
            {
                "case_id": t["case_id"],
                "reason": "Outside prospectively fixed stratified200 scope",
            }
            for t in tasks
            if t["case_id"] not in ids
        ],
    )
    (plan / "sample-scope.json").write_bytes(PROTOCOL.read_bytes())
    scope = {
        "profile": profile,
        "source_tasks": 1811,
        "selected_tasks": 200,
        "image_tasks": 23,
        "protocol_sha256": PROTOCOL_SHA256,
        "strata": expected["strata"],
        "selection_method": expected["selection_method"],
        "scope_note": "New fixed subset, not historical611 short-output benchmark or official full-set quality",
    }
    contract["execution_scope"] = scope
    contract["selection"].update(profile=profile, selected=200, excluded=1611)
    for name in ("tasks", "excluded"):
        contract[name + "_sha256"] = sha(plan / (name + ".json"))
    return scope
