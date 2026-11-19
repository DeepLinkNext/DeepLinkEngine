"""Audit every original sampler result and retain every planned timing sample."""

import hashlib
import json
import math
from pathlib import Path
import statistics


def checks_passed(checks):
    if isinstance(checks, list):
        return bool(checks) and all(checks_passed(item) for item in checks)
    if not isinstance(checks, dict) or not checks:
        return False
    if "finite" in checks or "close" in checks:
        return checks.get("finite") is True and checks.get("close") is True and checks.get("ids_exact", True) is True
    return all(checks_passed(value) for value in checks.values())


def samples(rows, plan):
    if not isinstance(rows, list) or len(rows) != plan["repetitions"]:
        raise ValueError("Canonical repetitions are missing or duplicated")
    seconds, seen = [], set()
    for row in rows:
        repetition = row["repetition"]
        if (type(repetition) is not int or repetition in seen
                or repetition not in range(plan["repetitions"])
                or type(row["replays"]) is not int or row["replays"] != plan["timed_replays"]):
            raise ValueError("Canonical repetition IDs or replay population differ")
        seen.add(repetition)
        for key in ("device_ms", "host_seconds", "start_wall", "end_wall"):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Nonfinite or nonpositive canonical timing")
        if row["end_wall"] <= row["start_wall"]:
            raise ValueError("Canonical timing wall clocks are reversed")
        seconds.append(row["device_ms"] / (1000 * row["replays"]))
    return dict(samples_seconds=seconds, seconds=statistics.median(seconds),
                reduction="median of every predeclared repetition; no fastest-run selection",
                minimum_seconds=min(seconds), maximum_seconds=max(seconds))


def audit(directory, plan_path, physical_cards):
    directory, plan_path = Path(directory), Path(plan_path)
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    plan_sha = hashlib.sha256(raw).hexdigest()
    results = []
    for rank, card in enumerate(physical_cards):
        record = json.loads((directory / f"rank-{rank}.json").read_bytes())
        if (type(record["rank"]) is not int or record["rank"] != rank
                or record["status"] != "qualified_component_timings" or record["plan_sha256"] != plan_sha):
            raise ValueError("Incomplete or foreign canonical rank/plan")
        timings = []
        if plan["family"] == "moe":
            buckets = {}
            for case in plan["cases"]:
                buckets.setdefault(case["rows"], []).append(case["case"])
            expected = {(op["name"], rows) for op in plan["operators"] for rows in buckets}
            seen = set()
            for bucket in record["buckets"]:
                key = (bucket["operator"]["name"], bucket["rows"])
                if key not in expected or key in seen or bucket["operator"] not in plan["operators"]:
                    raise ValueError("Foreign or duplicate canonical MoE bucket")
                seen.add(key)
                ids = buckets[key[1]]
                if (bucket["cases"] != len(ids) or bucket["timed_cases"] != len(ids) * plan["repetitions"]
                        or bucket["weight_format"] != 2
                        or bucket["weight_stride"] != [bucket["operator"]["k"] * bucket["operator"]["n"], bucket["operator"]["n"], 1]
                        or [check["case"] for check in bucket["checks"]] != sorted({ids[0], ids[-1]})
                        or any(not checks_passed(check["samples"]) for check in bucket["checks"])):
                    raise ValueError("MoE graph checks, layout or timing population differ")
            if seen != expected:
                raise ValueError("Missing canonical MoE operator bucket")
            groups = {(op["name"], case["case"]): [] for op in plan["operators"] for case in plan["cases"]}
            with (directory / f"timings-{rank}.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    key = (row["operator"], row["case"])
                    if key not in groups:
                        raise ValueError("Foreign canonical MoE timing case")
                    groups[key].append(row)
            for (operator, case), rows in groups.items():
                timings.append(dict(case=case, operator=operator, **samples(rows, plan)))
        else:
            rows = record["cases"]
            if len(rows) != len(plan["cases"]) or [row["case"] for row in rows] != list(range(len(rows))):
                raise ValueError("Canonical case population differs")
            for case, row in zip(plan["cases"], rows, strict=True):
                if row["shape"] != case or not checks_passed(row.get("checks", row.get("check"))):
                    raise ValueError("Canonical geometry or independent numerical oracle failed")
                timings.append(dict(case=case["case"], **samples(row["timings"], plan)))
        results.append(dict(rank=rank, physical_card=card, timings=timings))
    return dict(schema="fresh-original-canonical-component-timings/v1", family=plan["family"],
                plan_sha256=plan_sha, complete=True, ranks=results,
                historical_counts_or_timings_reused=False, full_model_certificate=False,
                source_program_applicability_certified=False, efficiency_percent=None)
