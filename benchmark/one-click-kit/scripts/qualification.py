"""Re-audit a sealed current C8 run before admitting the joint B1 arm."""

import hashlib
import json
import math
import re

from original_audit import audit_activation, reconstruct


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_files(root):
    root = root.resolve()
    manifest = json.loads((root / "checksums.json").read_bytes())
    found = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Sealed run contains a symlink")
        if path.is_file() and path != root / "checksums.json":
            found[str(path.relative_to(root))] = digest(path)
    if not manifest or manifest != found:
        raise ValueError("Sealed run inventory or file hash differs")
    return manifest


def qualify(root, deployment):
    root, deployment = root.resolve(), deployment.resolve()
    manifest = sealed_files(root)
    from serving_warmup import audit as audit_warmup

    warmup_evidence = audit_warmup(root)

    def read(name):
        return json.loads((root / name).read_bytes())

    receipt = read("receipt.json")
    if not (
        receipt.get("status") == "completed"
        and receipt.get("completed") is True
        and receipt.get("quiescent") is True
        and receipt.get("kind") == "continuous"
        and read("cleanup.json").get("quiescent") is True
    ):
        raise ValueError("A completed and released current continuous run is required")
    if read("installed-files.json") != json.loads(
        (deployment / "provenance/installed-files.json").read_bytes()
    ):
        raise ValueError("Qualification deployment source bindings differ")
    if read("deployment.json") != json.loads(
        (deployment / "deployment.json").read_bytes()
    ):
        raise ValueError("Qualification deployment identity differs")
    plan = read("service/plan.json")
    env = read("service/runtime-environment.json")
    if (
        plan["host_budget_per_rank_bytes"] != 4 * 2**30
        or env["BETTERSCALE_STATE_CACHE_HOST_BYTES"] != str(4 * 2**30)
        or len(plan["physical_cards"]) != 2
        or len(set(plan["physical_cards"])) != 2
        or plan["physical_cards"] != read("resource-admission.json")["physical_cards"]
    ):
        raise ValueError("Qualification budget or card pair differs")
    inputs = read("service/request-plan.json")
    if digest(root / "service/request-plan.json") != plan["input_sha256"]:
        raise ValueError("Qualification input digest differs")
    if len(inputs) != 2 or any(len(cases) != 8 for cases in inputs.values()):
        raise ValueError("Both C8 input cohorts are required")
    for cases, minimum in zip(inputs.values(), (4096, 16384)):
        if not all(c["input_tokens"] >= minimum for c in cases):
            raise ValueError("Qualification context length differs")
    expected_tokens = [
        {key: case[key] for key in ("input_tokens", "token_ids_sha256")}
        for cases in inputs.values()
        for case in cases
    ]
    if read("service/tokenize-preflight.json") != expected_tokens:
        raise ValueError("Qualification server token hashes differ")
    phases = read("service/continuous.json")
    expected_phases = [
        (name + "-" + suffix, duration)
        for name in inputs
        for suffix, duration in (("warmup", 60), ("formal", 300))
    ]
    if (
        phases.get("status") != "completed"
        or [(p["name"], p["seconds"]) for p in phases["phases"]] != expected_phases
    ):
        raise ValueError("Both original 60s warmup and 300s formal phases are required")
    rows = []
    for name in manifest:
        if not name.startswith("service/") or not name.endswith(".sse.jsonl.gz"):
            continue
        row = read(name.removesuffix(".sse.jsonl.gz") + ".json")
        first, last, usage, finish = reconstruct((root / name).read_bytes())
        start, end = row["start"]["monotonic"], row["end"]["monotonic"]
        if not (
            row["success"] is True
            and row["http_status"] == 200
            and first == row["first_token"]["monotonic"]
            and last == row["last_token"]["monotonic"]
            and usage == row["usage"]
            and finish == row["finish_reason"] == "length"
            and usage["completion_tokens"] == 256
            and usage["prompt_tokens"] == row["input_tokens"]
            and usage.get("prompt_tokens_details", {}).get("cached_tokens") in (None, 0)
            and all(math.isfinite(t) for t in (start, first, last, end))
            and start <= first <= last <= end
        ):
            raise ValueError("Qualification raw SSE/scalar mismatch: " + name)
        rows.append(row)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Qualification requests missing or duplicated")
    for phase in phases["phases"]:
        cohort = [r for r in rows if r["phase"] == phase["name"]]
        start, end = phase["start"]["monotonic"], phase["end"]["monotonic"]
        deadline = phase["submission_deadline_monotonic"]
        lanes = {int(re.search(r"-lane([0-7])-\d+$", r["id"])[1]) for r in cohort}
        if not (
            phase["failed"] == 0
            and phase["requests"] == len(cohort)
            and lanes == set(range(8))
            and math.isclose(deadline - start, phase["seconds"], abs_tol=1e-6)
            and end >= deadline
            and all(
                start <= r["start"]["monotonic"] < deadline
                and r["end"]["monotonic"] <= end
                for r in cohort
            )
        ):
            raise ValueError("Qualification phase cohort or clock differs")
    if sum(p["requests"] for p in phases["phases"]) != len(rows):
        raise ValueError("Qualification has requests outside the declared phases")
    activation_files = {
        name.removeprefix("service/"): (root / name).read_bytes()
        for name in manifest
        if name.startswith("service/")
        and (name.endswith(".jsonl") or name.endswith("runtime-identity.json"))
    }
    activation = audit_activation(
        activation_files,
        prefix="",
        expected_requests=len(rows)
        + (warmup_evidence["requests"] if warmup_evidence else 0),
    )
    packed = [read(n) for n in manifest if n.startswith("service/packed-host-install-")]
    expected_packed = read("installed-files.json")[
        "mods/packed-host/packed_host_state.py"
    ]
    if len(packed) < 2 or any(p["adapter_sha256"] != expected_packed for p in packed):
        raise ValueError("Packed-host activation missing or changed")
    return {
        "scope": "Current original-protocol C8 qualification only; formal four-target acceptance remains separate",
        "run": str(root),
        "seal_sha256": digest(root / "checksums.json"),
        "requests": len(rows),
        "physical_cards": plan["physical_cards"],
        "activation": activation,
        "serving_warmup": warmup_evidence,
        "weights": read("model-weights.json"),
    }
