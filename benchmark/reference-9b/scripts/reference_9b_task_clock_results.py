"""Validate independent 9B target matmul clocks from on-device task intervals."""

import csv
from decimal import Decimal
from hashlib import file_digest
import json
import math
from pathlib import Path
from statistics import median


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def inspect_numerics(record, shape, plan):
    check = record["check"]
    if (record["weight_stride"] != [shape["k"], 1] or check["finite"] is not True
            or check["close"] is not True or len(check["rows"]) != min(8, shape["m"])
            or len(check["columns"]) != min(8, shape["n"])):
        raise ValueError(f"Canonical 9B numeric or layout check failed at case {shape['case']}")
    left, right = check["actual"], check["cpu_fp32"]
    if len(left) != len(right) or len(left) != len(check["rows"]):
        raise ValueError(f"Invalid 9B numeric comparison rows at case {shape['case']}")
    for result, reference in zip(left, right, strict=True):
        if len(result) != len(reference) or len(result) != len(check["columns"]):
            raise ValueError(f"Invalid 9B numeric comparison columns at case {shape['case']}")
        for a, b in zip(result, reference, strict=True):
            if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > plan["atol"] + plan["rtol"] * abs(b):
                raise ValueError(f"Device and independent CPU matmul differ at case {shape['case']}")


def qualified_timing(path, card, plan, case):
    """The profiler sees 20 warmups then 100 measured tasks within each repeat."""
    scope = path / "ASCEND_PROFILER_OUTPUT/kernel_details.csv"
    with scope.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    warm, kept = plan["graph_warmups_per_repetition"], plan["timed_replays"]
    repeats = plan["repetitions"]
    width = warm + kept
    if len(records) != repeats * width:
        raise ValueError(f"Expected {repeats * width} canonical task records, got {len(records)} in {scope}")
    task = "aclnnMatmul_MatMulCommon_MatMulV2"
    if (case["m"], case["n"], case["k"]) == (1369, 4096, 6144):
        task = "aclnnMatmul_MatMulV3Common_MatMulV3"
    tasks = []
    for row in records:
        if row["Device_id"] != str(card) or row["Name"] != task:
            raise ValueError(f"Expected task {task} on card {card} in {scope}")
        start = Decimal(row["Start Time(us)"].strip())
        duration = Decimal(row["Duration(us)"])
        if not duration.is_finite() or duration <= 0 or not start.is_finite():
            raise ValueError(f"Invalid canonical device task timestamp in {scope}")
        tasks.append((start, duration))
    tasks.sort()
    if any(first[0] + first[1] > second[0] for first, second in zip(tasks[:-1], tasks[1:], strict=True)):
        raise ValueError(f"Overlapping device tasks for an isolated serial matmul in {scope}")
    medians = [median(task[1] for task in tasks[r * width + warm:(r + 1) * width]) for r in range(repeats)]
    return dict(task_trace=scope, task_trace_sha256=sha(scope), profiled_task_count=len(tasks),
        samples_seconds=[str(value / 10**6) for value in medians], seconds=str(median(medians) / 10**6))


def audit(directory, plan_path, physical_cards):
    directory, plan_path = Path(directory), Path(plan_path)
    root = directory.parent
    plan = json.loads(plan_path.read_bytes())
    if (plan["timing_domain"] != "cann-device-task-duration/v1" or physical_cards != [4, 5]
            or plan["repetitions"] != 5 or plan["timed_replays"] != 100
            or plan["graph_warmups_per_repetition"] != 20
            or [row["case"] for row in plan["cases"]] != list(range(len(plan["cases"])))):
        raise ValueError("9B device task plan or scope differs")
    result = []
    for rank, card in enumerate(physical_cards):
        raw_path = directory / f"rank-{rank}.json"
        raw = json.loads(raw_path.read_bytes())
        if (raw["rank"] != rank or raw["status"] != "qualified_component_timings"
                or raw["torch_version"] != "2.13.0+cpu" or raw["torch_npu_version"] != "2.13.0.rc1"
                or raw["plan_sha256"] != sha(plan_path) or len(raw["cases"]) != len(plan["cases"])):
            raise ValueError(f"9B independent sampler did not finish the exact cases at rank {rank}")
        timings = []
        for case, record in zip(plan["cases"], raw["cases"], strict=True):
            if record["case"] != case["case"] or record["shape"] != case or record["timings"]:
                raise ValueError(f"Wrong clock type or shape for case {case['case']} at rank {rank}")
            inspect_numerics(record, case, plan)
            raw_profile = root / "device-task-profiles" / f"rank-{rank}" / f"case-{case['case']:03d}"
            profile, = [item for item in raw_profile.iterdir() if item.name.endswith("_ascend_pt")]
            task = qualified_timing(profile, card, plan, case)
            task["task_trace"] = task["task_trace"].relative_to(root).as_posix()
            timings.append(dict(case=case["case"], **task))
        result.append(dict(rank=rank, physical_card=card, source_rank_sha256=sha(raw_path), timings=timings))
    return dict(schema="qwen35-9b-device-task-duration-clock-panel/v1", timing_domain=plan["timing_domain"],
        plan_sha256=sha(plan_path), witnessed_operands_sha256=plan["witnessed_operands_sha256"],
        ranks=result, includes_external_graph_submit_gaps=False, complete=True)
