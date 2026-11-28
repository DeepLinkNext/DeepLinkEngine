"""Resolve independent device clocks for the exact first r11 GDN convolution."""
import csv
from decimal import Decimal
from pathlib import Path
import shutil
from statistics import median
import subprocess
from tempfile import mkdtemp

from child_runner import run_stage
from dataset_contract import save, sha
from reference_9b_archive import PINS, member, read
from reference_9b_target_fia import BUNDLE, release, require


OPERANDS = "r11-cold-gdn-conv-operands.json"
DEMAND = "202c14ce4d8911218d12ab32f50f612fb1df64380c943c090812991e166009f6"
SAMPLER = "reference_9b_r11_gdn_conv_sampler.py"
SAMPLER_SHA = "d9b1b8a3508724f0d035d3201f5f4a331190fe1f2a9fbd5a2c61b89464fc0a97"
HELPER = "calibrate_r11_native_kv_task.py"
INDEX = "r11-cold-gdn-conv-cache.json"
VERSIONS = {"torch": "2.13.0+cpu", "torch-npu": "2.13.0rc1"}
VENDOR = "d56523e9328794b811d3dc7a6bc72523373c556412c4cfc9f2622f1f5afe4bc2"
LAYERS = tuple(layer for layer in range(32) if layer % 4 != 3)
LEASE = ("allocation/quota.json", "allocation/npu-after.txt", "allocation/locks-after.txt")
SCHEMA = "m01-r11-cold-gdn-conv-clock-cache/v1"


def device_clock(path, card, *, expected=("causalconv1d",), block_field="Block Num"):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 230 and all(any(name in row["Name"].replace("_", "").casefold() for name in expected) and int(row["Device_id"]) == card for row in rows)
            and len({(row["Name"], row[block_field]) for row in rows}) == 1
            and (int(rows[0][block_field]) > 0 or (block_field == "Block Dim" and rows[0][block_field] == "0")),
            "Independent reference clock does not contain one consistent device task per replay")
    durations = sorted((Decimal(row["Start Time(us)"]), Decimal(row["Duration(us)"])) for row in rows)
    require(all(start.is_finite() and time.is_finite() and time > 0 for start, time in durations)
            and all(a + duration <= b for (a, duration), (b, _) in zip(durations, durations[1:])),
            "Independent reference CANN task timing is invalid or overlapping")
    samples = [median(time for _, time in durations[i + 3:i + 23]) for i in range(0, 230, 23)]
    return str(median(samples) / 10**6), rows[0]["Name"], int(rows[0][block_field]) or None


def read_clock(job, manifest, plan, case):
    for key, value in {OPERANDS: DEMAND, SAMPLER: SAMPLER_SHA, HELPER: BUNDLE[HELPER]}.items():
        require(manifest[key] == value, f"GDN sampler source changed: {key}")
        member(job, key, value)
    for name in LEASE:
        member(job, name, manifest[name])
    request = read(member(job, "request.json", manifest["request.json"]))
    slug = case["id"].replace(":", "-")
    numeric = f"{slug}/numeric.json"
    witness = read(member(job, numeric, manifest[numeric]))
    conv = plan["convolution"]
    require(request["operands_sha256"] == DEMAND and case["id"] in request["case_ids"]
            and witness["case"] == case and witness["operands_sha256"] == DEMAND
            and witness["vendor_api_sha256"] == VENDOR and witness["versions"] == VERSIONS
            and witness["sampler_sha256"] == SAMPLER_SHA and not witness["model_inference"]
            and all(witness[name] for name in ("numerical_passed", "convolution_suffix_verified",
                                               "inactive_state_unchanged", "graph_agreed"))
            and witness["input_shape"] == conv["input_shape"] and witness["weight_shape"] == conv["weight_shape"]
            and witness["state_shape"] == conv["state_backing"]["shape"]
            and witness["state_stride"] == conv["state_backing"]["stride"],
            "GDN clock numerical witness or state layout changed")
    traces = [name for name in manifest if name.startswith(slug + "/") and name.endswith("/kernel_details.csv")]
    require(len(traces) == 1, f"Missing or ambiguous convolution device trace: {slug}")
    path = member(job, traces[0], manifest[traces[0]])
    timing, kernel, blocks = device_clock(path, case["physical_card"])
    return dict(seconds=timing, kernel=kernel, blocks=blocks, task_sha256=manifest[traces[0]],
                numeric_sha256=manifest[numeric], case=case, timing_domain="cann-device-task-duration/v1",
                release_sha256=release(job))


def calibrate(archive, output, proof, sampler_cache, needed, plan):
    job = Path(mkdtemp(prefix="r11-cold-gdn-auto.", dir=archive.parent))
    script = member(Path(__file__).parent, SAMPLER, SAMPLER_SHA)
    helper = member(sampler_cache, "sampler/" + HELPER, BUNDLE[HELPER])
    for name, path in ((SAMPLER, script), (HELPER, helper), (OPERANDS, proof)):
        shutil.copyfile(path, job / name)
    save(job / "request.json", dict(operands_sha256=DEMAND, case_ids=list(needed)))
    donor_python = archive.parent / ("r10-kernel-native-10204-20261124.6ke9QK/runtime/clean-cache-v2/"
                                     "environments/dc75dd91be2a521c-9ebc8e5cb831/venv/bin/python")
    command = [str(donor_python), "-B", str(job / SAMPLER), str(job), "launch"]
    save(output / "r11-cold-gdn-conv-dispatch.json", dict(case_ids=list(needed), job=str(job), command=command))
    with (output / "r11-cold-gdn-conv-sampler.log").open("x") as log:
        child = run_stage(command, stdout=log, stderr=subprocess.STDOUT)
    require(child.returncode == 0, f"r11 cold GDN sampler exited {child.returncode} at {job}")
    numeric = sorted(job.glob("r11-cold-gdn-conv-*/numeric.json"))
    require({path.parent.name for path in numeric} == {name.replace(":", "-") for name in needed},
            "Independent GDN convolution sampler touched an unrequested case")
    traces = sorted(job.glob("r11-cold-gdn-conv-*/**/kernel_details.csv"))
    names = [SAMPLER, HELPER, OPERANDS, "request.json", *LEASE]
    names.extend(path.relative_to(job).as_posix() for path in (*numeric, *traces))
    manifest = {name: sha(job / name) for name in names}
    save(job / "readback-checksums.json", manifest)
    return ({name: read_clock(job, manifest, plan, case) for name, case in needed.items()},
            dict(source_root=str(job), raw_manifest_sha256=sha(job / "readback-checksums.json")))


def resolve(archive, output, cache, seal, target_cache, clocks):
    manifest = read(member(cache, "checksums.json", seal))
    require(manifest[OPERANDS] == DEMAND, "Convolution cache uses another operand demand")
    proof = member(cache, OPERANDS, DEMAND)
    plan = read(proof)
    require(plan["original_archive_sha256"] == clocks["B1"]["checksums_sha256"] == PINS["native-clock/B1/checksums.json"]
            and tuple(plan["layers"]) == LAYERS and not plan["companion_addresses_transferred"]
            and [(row["repetition"], row["wall_sha256"]) for row in plan["rounds"]]
            == [(row["repetition"], row["source_sha256"]) for row in clocks["B1"]["rounds"]],
            "GDN convolution does not refer to all three original r11 clocks")
    cases = {case["id"]: case for case in plan["missing_independent_clocks"]}
    index = read(member(cache, INDEX, manifest[INDEX]))
    require(index["schema"] == SCHEMA and index["operands_sha256"] == DEMAND
            and set(index["available"]) <= cases.keys(), "Unexpected convolution cache index")
    hits = {}
    for name, entry in index["available"].items():
        job = Path(entry["source_root"])
        bundle = read(member(job, "readback-checksums.json", entry["raw_manifest_sha256"]))
        hits[name] = read_clock(job, bundle, plan, cases[name])
    needed = {name: case for name, case in cases.items() if name not in hits}
    result = dict(operands_sha256=DEMAND, cache_sha256=seal, retained=len(hits), sampled=len(needed),
                  sampler_invocations=int(bool(needed)), exact_missing_cases=list(needed))
    if needed:
        fresh, result["device_run"] = calibrate(archive, output, proof, target_cache, needed, plan)
        hits.update(fresh)
    save(output / "r11-cold-gdn-conv-clocks.json", dict(**result, clocks=hits))
    return {case["rank"]: hits[name] for name, case in cases.items()}, result

