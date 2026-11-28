"""Resolve the six original-window r11 target-FIA clocks from qualified device tasks."""

import csv
from decimal import Decimal
from pathlib import Path
import re
import shutil
from statistics import median
import subprocess
from tempfile import mkdtemp

from child_runner import run_stage
from dataset_contract import save, sha
from reference_9b_archive import PINS, member, read


SCOPE = "078c33249dd45dd6bf20bcd31d4ec6078a7eaea07a45e9a45f9b2fcea197c644"
PANEL = "2d9d6f4dc2f9408d9eccbaddbc5f6ac139e4fd42966b720bc82f7075ec03385c"
PROJECTION = "27454d4729b444446eb3d0a5186badffd2b1ca34afcaa2e138dd002d3123cfb3"
BUNDLE = {
    "r11-first-target-fia-operands.json": SCOPE,
    "r11-native-first-kv-operands.json": "d0299418901078049ba4610deb91253fbded2b19ef526ff3929352fba07f834d",
    "r11-fia-checkpoint-config.json": "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05",
    "audit_r11_first_target_fia.py": "c8a993625b479dbf02defe4e0b682af742e4b298896c0e44356d90d12ff72250",
    "calibrate_r11_native_kv_task.py": "6d45f1ccc717869a07d5f0c26b80f204ebd2e05cf18441a25d0e907f6125111f",
    "calibrate_r11_first_target_fia.py": "289e86e41ea8ac6434baf44036a8c35a30591abe1afb169e0a7d5b56b60c5211",
}
EXPECTED = {(rank, rep): f"rank-{rank}/round-{rep}" for rank in (0, 1) for rep in (1, 2, 3)}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def task_clock(path, card):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        raw = list(csv.DictReader(stream))
    require(len(raw) == 230 and all(row["Name"] == "FusedInferAttentionScore" and row["Device_id"] == str(card)
                                   for row in raw), f"Invalid FIA device task scope: {path}")
    tasks = sorted((Decimal(row["Start Time(us)"]), Decimal(row["Duration(us)"])) for row in raw)
    require(all(a.is_finite() and d.is_finite() and d > 0 for a, d in tasks)
            and all(a + d <= b for (a, d), (b, _) in zip(tasks, tasks[1:])),
            f"FIA device tasks overlap or have invalid timestamps: {path}")
    samples = [median(d for _, d in tasks[i+3:i+23]) for i in range(0, 230, 23)]
    return str(median(samples[:7]) / 10**6)


def release(root):
    quota = read(root / "allocation/quota.json")
    at_end = (root / "allocation/npu-after.txt").read_text()
    locks = (root / "allocation/locks-after.txt").read_text()
    claims = quota["allocation"] + quota["admission"]["exclusive_leases"]
    require(quota["physical"] == quota["logical"] == [4, 5] and len(set(claims)) == 18
            and all(claim not in locks for claim in claims)
            and all(re.search(fr"No running processes found in NPU {card}\s", at_end) for card in (4, 5)),
            "The auto-calibrated FIA pair did not release all devices and leases")
    return {name: sha(root / "allocation" / name) for name in ("quota.json", "npu-after.txt", "locks-after.txt")}


def calibrate(archive, folder, bundle, needed, operands):
    run = Path(mkdtemp(prefix="qwen9b-fia-auto-sample.", dir=archive.parent))
    for name, source in bundle.items():
        shutil.copyfile(source, run / name)
    recipe = run / "calibrate_r11_first_target_fia.py"
    donor = archive.parent / ("r10-kernel-native-10204-20261124.6ke9QK/runtime/clean-cache-v2/"
                              "environments/dc75dd91be2a521c-9ebc8e5cb831/venv/bin/python")
    command = [str(donor), "-B", str(recipe), str(run), "launch"]
    command.extend(part for rank, rep in needed for part in ("--request", f"{rank}:{rep}"))
    dispatch = folder / "target-fia-dispatch.json"
    save(dispatch, dict(operands_sha256=SCOPE, missing_cases=[EXPECTED[key] for key in needed], command=command))
    with (folder / "target-fia-sampler.log").open("x") as log:
        result = run_stage(command, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0 and (run / "launch.exit").read_text().strip() == "0",
            f"Exact target FIA calibration failed at {run}")
    allocation = release(run)
    samples = {}
    for rank in sorted({key[0] for key in needed}):
        receipt = run / f"rank-{rank}.json"
        numeric = read(receipt)
        reps = [rep for r, rep in needed if r == rank]
        require(numeric["rank"] == rank and numeric["physical_card"] == 4 + rank
                and numeric["operands_sha256"] == SCOPE and numeric["model_inference"] is False
                and [case["repetition"] for case in numeric["cases"]] == reps,
                "The auto sampler did not provide exactly the requested rank and round")
        for case in numeric["cases"]:
            rep = case["repetition"]
            require(case["pages"] == operands["cases"][rep - 1]["page_operand_read_prefix"]
                    and case["numerical_passed"] is True
                    and case["query_shape"] == operands["query_shape"] and case["cache_shape"] == operands["cache_plane_shape"]
                    and case["table_shape"] == operands["block_table_shape"],
                    "The auto-calibrated FIA operand or numerical witness differs")
            target = run / "profiles" / f"rank-{rank}" / f"round-{rep}"
            trace, = sorted(target.glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
            samples[rank, rep] = dict(seconds=task_clock(trace, 4 + rank), task_sha256=sha(trace),
                                      numerical_sha256=sha(receipt), source="sampled", task_path=str(trace))
    return samples, dict(run=str(run), release_sha256=allocation, dispatch_sha256=sha(dispatch))


def resolve(archive, folder, cache, manifest_pin, clocks):
    require(cache is not None and manifest_pin is not None, "The typed FIA graph requires its exact clock cache and checksums SHA-256")
    manifest = read(member(cache, "checksums.json", manifest_pin))
    index = read(member(cache, "target-fia-cache.json", manifest["target-fia-cache.json"]))
    active_cases = index["available_cases"]
    require(index["schema"] == "m01-9b-r11-first-target-fia-clock-cache/v1"
            and index["operands_sha256"] == SCOPE and isinstance(active_cases, list)
            and len(active_cases) == len(set(active_cases)) and set(active_cases) <= set(EXPECTED.values()),
            "The target FIA clock index has a foreign operand")
    bundle = {}
    for name, checksum in BUNDLE.items():
        relative = "sampler/" + name
        require(manifest[relative] == checksum, "A source-pinned target FIA sampling input changed")
        bundle[name] = member(cache, relative, checksum)
    operands = read(bundle["r11-first-target-fia-operands.json"])
    panel_path = "r11-first-target-fia-clocks.json"
    require(manifest[panel_path] == PANEL and clocks["B1"]["checksums_sha256"] == PINS["native-clock/B1/checksums.json"],
            "The FIA clock panel belongs to different native timing windows")
    panel = read(member(cache, panel_path, PANEL))
    require([r["original_timing_sha256"] for r in operands["cases"]] == [r["source_sha256"] for r in clocks["B1"]["rounds"]],
            "FIA cache is paired to different original wall boundaries")
    available, needed = {}, []
    for key, name in EXPECTED.items():
        rank, rep = key
        if name not in active_cases:
            needed.append(key)
            continue
        case = panel["all_ranks"][rank]["cases"][rep - 1]
        trace_name = "traces/" + name + ".csv"
        checksum = manifest[trace_name]
        require(checksum == case["task_trace_sha256"], "A retained FIA component is outside its signed raw panel")
        trace = member(cache, trace_name, checksum)
        duration = task_clock(trace, 4 + rank)
        require(duration == case["calibration_median_seconds"], "Retained FIA raw timing does not reproduce its statistic")
        available[key] = dict(seconds=duration, task_sha256=checksum, source="retained", task_path=str(trace))
    selection = dict(operands_sha256=SCOPE, cache_sha256=manifest_pin, retained=len(available), sampled=len(needed),
                     sampler_invocations=int(bool(needed)), missing_cases=[EXPECTED[key] for key in needed])
    if needed:
        fresh, run = calibrate(archive, folder, bundle, needed, operands)
        selection.update(run)
        available.update(fresh)
    result_path = folder / "qualified-target-fia-clocks.json"
    save(result_path, dict(**selection, cases={EXPECTED[k]: v for k, v in available.items()}))
    return available, dict(**selection, index_sha256=sha(result_path), index=result_path.relative_to(folder.parent).as_posix())
