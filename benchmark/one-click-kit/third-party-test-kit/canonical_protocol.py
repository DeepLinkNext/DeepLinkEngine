"""Prepare fresh exact-shape runs using the original public sampler protocols.

This module performs no imports of torch, no device initialization and no model
work. It preserves original numerical/statistical controls and sampler bytes;
current source/shape applicability and device admission remain separate checks.
"""

import ast
import copy
import hashlib
import json
from pathlib import Path
import re

SOURCES = Path(__file__).with_name("canonical-reference-sources")
REGISTRY_SHA256 = "348fe0c67013387af703be233534eace56d93690952693039afa9622aec97ecb"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def protocol(family, source_root=SOURCES):
    source_root = Path(source_root)
    raw = (source_root / "protocols.json").read_bytes()
    if sha(raw) != REGISTRY_SHA256:
        raise ValueError("Original canonical protocol registry changed")
    registry = json.loads(raw)
    if family not in registry["families"]:
        raise ValueError("Canonical source/protocol is not restored for family: " + family)
    row = registry["families"][family]
    path = source_root / row["sampler"]
    if path.parent != source_root or sha(path.read_bytes()) != row["sampler_sha256"]:
        raise ValueError("Original canonical sampler source changed")
    return copy.deepcopy(row), path


def locations(source):
    result = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("PLAN", "OUT")):
            value = node.value
            if (not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Div)
                    or not isinstance(value.left, ast.Name) or value.left.id != "ROOT"
                    or not isinstance(value.right, ast.Constant) or type(value.right.value) is not str):
                raise ValueError("Unknown original sampler output/plan path expression")
            name = value.right.value
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Original sampler path is unsafe")
            result[node.targets[0].id] = name
    if set(result) != {"PLAN", "OUT"}:
        raise ValueError("Original sampler paths are incomplete")
    return result


def current_plan(family, cases, source_bindings, *, source_root=SOURCES):
    row, source = protocol(family, source_root)
    if (not isinstance(source_bindings, dict) or not source_bindings
            or any(not isinstance(name, str) or not name or not isinstance(value, str)
                   or not re.fullmatch(r"[a-f0-9]{64}", value) for name, value in source_bindings.items())):
        raise ValueError("Current independently admitted source SHA256 bindings required")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Current exact source shapes are missing")
    required = set(row["required_case_keys"])
    prepared = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != required or case.get("case") != index or type(case["case"]) is not int:
            raise ValueError("Current case fields/ordinals differ from the original source interface")
        for name, value in case.items():
            if name == "kind":
                if value not in row["case_kind_values"]:
                    raise ValueError("Unqualified canonical operation branch")
            elif family == "moe" and name == "group_counts":
                if (not isinstance(value, list) or len(value) != 256
                        or any(type(n) is not int or n < 0 for n in value)):
                    raise ValueError("MoE requires 256 explicit nonnegative expert row counts")
            elif family == "gdn_ho" and name in ("lengths", "initial_flags"):
                if (not isinstance(value, list) or not value
                        or (name == "lengths" and any(type(n) is not int or n < 1 for n in value))
                        or (name == "initial_flags" and any(type(n) is not bool for n in value))):
                    raise ValueError("GDN requires explicit positive lengths and Boolean state flags")
            elif name in ("mapped", "q_head_major"):
                if type(value) is not bool:
                    raise ValueError("Canonical branch switches must be explicit Booleans")
            elif type(value) is not int or value < (0 if name in ("case", "offset") else 1):
                raise ValueError("Canonical shape extents must be explicit integers")
        if family == "moe":
            rows, counts = case["rows"], case["group_counts"]
            if rows % 8 or rows < 24 or sum(counts) != rows or max(counts) > rows // 8:
                raise ValueError("MoE shape is outside the restored M>=3, top-8 expert-list domain")
        if family == "gdn_ho":
            lengths, flags = case["lengths"], case["initial_flags"]
            if len(lengths) != len(flags) or len(lengths) > case["requests"] or sum(lengths) > case["capacity"]:
                raise ValueError("GDN ragged state geometry exceeds owned capacity")
        prepared.append(copy.deepcopy(case))
    plan = copy.deepcopy(row["controls"])
    plan["cases"] = prepared
    if "source_sha256" in row["required_plan_keys"]:
        plan["source_sha256"] = copy.deepcopy(source_bindings)
    if set(row["required_plan_keys"]) - set(plan):
        raise ValueError("Restored canonical sampler plan controls are incomplete")
    plan.update(schema="current-exact-canonical-sampler-plan/v1", family=family,
        original_sampler_sha256=row["sampler_sha256"],
        original_protocol_plan_sha256=row["original_plan_sha256"],
        current_source_sha256=copy.deepcopy(source_bindings),
        numerical_or_statistical_thresholds_relaxed=False,
        historical_counts_or_timings_reused=False,
        source_runtime_applicability_certified=False, full_model_certificate=False)
    return plan, source


def prepare(directory, family, cases, source_bindings, *, source_root=SOURCES):
    """Write the unchanged original program and a fresh current-shape plan.

    A separately guarded parent may launch this program only after current
    source/shape admission. Original worker-ready barriers and two-rank spawn
    remain intact; this function never launches them.
    """
    plan, source = current_plan(family, cases, source_bindings, source_root=source_root)
    raw = source.read_bytes()
    paths = locations(raw)
    if paths["PLAN"] != protocol(family, source_root)[0]["original_plan"]:
        raise ValueError("Sampler PLAN filename differs from the protocol authority")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / source.name).write_bytes(raw)
    plan_path = directory / paths["PLAN"]
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    receipt = dict(schema="canonical-sampler-preparation/v1", family=family,
        sampler=source.name, sampler_sha256=sha(raw), plan=paths["PLAN"],
        plan_sha256=sha(plan_path.read_bytes()), output=paths["OUT"],
        requires_guarded_device_launch=True, device_initialized=False,
        measurement_executed=False, full_model_certificate=False)
    (directory / "preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
