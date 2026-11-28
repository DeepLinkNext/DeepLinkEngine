"""Resolve independently qualified clocks for exact first-retained r11 draft work."""

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
from reference_9b_target_fia import BUNDLE as TARGET_BUNDLE, release, require, task_clock


DEMAND = "5cea6291d431044e325818a89ab2a3109e3d66651c546745a777ece764bec309"
SAMPLER = "b97f6f54d18c83924666c974d94e985cf9ebe3ea5bd28a6ff62509816a31223e"
WRITER = "151f03a3344f2651317db0c9d5503a272dcdd61ff8b5712cbef4e112f6e4a013"
EXECUTABLE = ("r10-kernel-native-10204-20261124.6ke9QK/runtime/clean-cache-v2/"
              "environments/dc75dd91be2a521c-9ebc8e5cb831/venv/bin/python")
DEMAND_FILE = "draft-clock-demand.json"
INDEX = "draft-clock-cache.json"
# These two completed standalone FIA cases were sealed before a later scatter
# operand failed. Their complete raw clocks and post-failure release were read back.
RECOVERED_FIA = {
    "draft-fia:0:1:0": ("1c453d0c82a0a2344cfd86db688f1aef49e9ea4552d95aa405bf1bcf7a259225",
                        "80520a7ed39a7ee47cd41d30c8b4eb3cac919ca69166450c6ff4309704dc043a"),
    "draft-fia:0:1:1": ("e2b6347de708e1d17b9222bcd4ad830b76681c76135e52357d30285389dcb88e",
                        "c7cc0f56ee044c2edeb3b60a4cf2ad36c5b62fdb27e6b5ef97e1def5f22065c4"),
}
PRIOR_FIA_BINDING = ("dc19a565b6b9b61643f5e37b67a3f854a8e1b80bcc6907fcb0e02b0c53c4b719",
                     "561a4ac85c93ca7601ab5c11a2e387549123ea1b73a8fe25ef716f206df83fa4")
SCATTER_KERNEL = "aclnnScatterPaKvCache_ScatterPaKvCache_ScatterPaKvCache"
FIA = "canonical-fused-infer-attention-score"
VERSIONS = {"torch": "2.13.0+cpu", "torch-npu": "2.13.0rc1"}
SAMPLE_SUPPORT = ("calibrate_r11_first_target_fia.py", "calibrate_r11_native_kv_task.py", "audit_r11_first_target_fia.py")


def scatter_clock(path, card, controls):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    warm, steps = controls["warmups_per_repetition"], controls["replays"]
    group = warm + steps
    require(len(rows) == group * controls["repetitions"]
            and all(row["Name"] == SCATTER_KERNEL and row["Device_id"] == str(card)
                    and row["Block Num"] == "1" for row in rows),
            f"Unexpected one-row scatter device task at {path}")
    tasks = sorted((Decimal(row["Start Time(us)"]), Decimal(row["Duration(us)"])) for row in rows)
    require(all(a.is_finite() and d.is_finite() and d > 0 for a, d in tasks)
            and all(a + d <= b for (a, d), (b, _) in zip(tasks, tasks[1:])),
            f"Invalid one-row scatter task timeline at {path}")
    sample = [median(d for _, d in tasks[i+warm:i+group]) for i in range(0, len(tasks), group)]
    return str(median(sample[:controls["retained_repetitions"]]) / 10**6)


def case_clock(case, trace, numeric, controls):
    result = read(numeric)
    pins = sha(trace), sha(numeric)
    binding = result["demand_sha256"], result["sampler_sha256"]
    admitted = binding == (DEMAND, SAMPLER) or (
        case["operator"] == FIA and binding == PRIOR_FIA_BINDING and RECOVERED_FIA.get(case["id"]) == pins)
    require(admitted and result["case"] == case and result["writer_sha256"] == WRITER
            and result["runtime"] == VERSIONS and result["numerical_passed"] is True
            and result["standalone_operator"] is True and result["model_inference"] is False,
            f"Numerical or runtime witness differs for {case['id']} at {numeric}")
    card = case["physical_card"]
    if case["operator"] == FIA:
        require(result["pages"] == case["actual_page_prefix"] and result["query_shape"] == case["query_shape"]
                and result["cache_shape"] == case["cache_plane_shape"]
                and result["table_shape"] == case["live_block_table_shape"]
                and result["kv_length"] == case["actual_kv_endpoints"][0]
                and case["actual_query_endpoints"] == [1],
                f"Preserved live FIA shape differs for {case['id']}")
        seconds = task_clock(trace, card)
    else:
        require(result["write_slot"] == case["write_slot"] and result["fenced_two_generation_bitwise"] is True
                and result["key_shape"] == [1, 2, 256] and result["plane_shape"] == case["cache_plane_shape"]
                and result["plane_stride"] == [65536, 512, 256, 1],
                f"One-row scatter geometry differs for {case['id']}")
        seconds = scatter_clock(trace, card, controls)
    return dict(seconds=seconds, task_sha256=pins[0], numeric_sha256=pins[1],
                task_path=str(trace), numeric_path=str(numeric), operand=case)


def admit(clocks, demand):
    role = clocks["B1"]
    active = {w["repetition"]: w["source_sha256"] for w in role["rounds"]}
    require(role["checksums_sha256"] == demand["native_seal_sha256"] == PINS["native-clock/B1/checksums.json"],
            "The first-draft clocks belong to a different native r11 archive")
    cases = {}
    for case in demand["cases"]:
        rank, rep, phase = (case[k] for k in ("rank", "repetition", "phase"))
        require(rank in (0, 1) and rep in active and case["physical_card"] == 4 + rank
                and case["original_timing_sha256"] == active[rep],
                f"First-draft operands do not match the native wall for {case['id']}")
        require(case["id"] not in cases and phase in (0, 1), "Ambiguous first-draft operand")
        cases[case["id"]] = case
    require(len(cases) == 18, "Incomplete first-draft case scope")
    return cases


def audit_cached(cache, manifest, index, cases, controls):
    reused = {}
    releases = {}
    for name, record in index["available_cases"].items():
        job = record["job"]
        for relative in (record["task"], record["numeric"]):
            require(relative.startswith(job + "/"), f"Cached path escaped its measurement job for {name}")
            member(cache, relative, manifest[relative])
        if job not in releases:
            for basename in ("quota.json", "npu-after.txt", "locks-after.txt"):
                relative = job + "/allocation/" + basename
                member(cache, relative, manifest[relative])
            releases[job] = release(cache / job)
        audited = case_clock(cases[name], cache / record["task"], cache / record["numeric"], controls)
        require(audited["seconds"] == record["seconds"], f"Cached draft task statistic differs for {name}")
        reused[name] = dict(**audited, source="retained", release_sha256=releases[job])
    return reused


def dispatch(archive, folder, demand_path, source_bundle, needed, controls):
    job = Path(mkdtemp(prefix="qwen9b-first-draft-task.", dir=archive.parent))
    shutil.copyfile(demand_path, job / DEMAND_FILE)
    for name, source in source_bundle.items():
        shutil.copyfile(source, job / name)
    request_path = job / "request.json"
    save(request_path, dict(demand_sha256=DEMAND, missing_cases=list(needed)))
    selection = folder / "first-draft-dispatch.json"
    command = [str(archive.parent / EXECUTABLE), "-B", str(job / "reference_9b_draft_sampler.py"), str(job), "launch"]
    save(selection, dict(job=str(job), request_sha256=sha(request_path), missing_cases=list(needed), command=command))
    with (folder / "first-draft-sampler.log").open("x", encoding="utf-8") as output:
        result = run_stage(command, stdout=output, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f"Exact first-draft device calibration failed at {job}")
    released = release(job)
    sampled = {}
    for name, case in needed.items():
        case_root = job / name.replace(":", "-")
        trace, = sorted(case_root.glob("**/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
        sample = case_clock(case, trace, case_root / "numeric.json", controls)
        sampled[name] = dict(**sample, source="sampled", release_sha256=released, job=str(job))
    require({p.parent.name for p in job.glob("draft-*/numeric.json")} == {n.replace(":", "-") for n in needed},
            "The standalone draft sampler touched cases outside the exact missing set")
    return sampled, dict(job=str(job), release_sha256=released, dispatch_sha256=sha(selection))


def resolve(archive, folder, cache, manifest_pin, clocks):
    require(cache is not None and manifest_pin is not None, "First-draft clocks require their exact cache seal")
    manifest = read(member(cache, "checksums.json", manifest_pin))
    index = read(member(cache, INDEX, manifest[INDEX]))
    require(index["schema"] == "m01-r11-first-retained-draft-clock-cache/v1" and index["demand_sha256"] == DEMAND,
            "The first-draft cache uses foreign source or case metadata")
    demand_path = member(cache, DEMAND_FILE, DEMAND)
    demand = read(demand_path)
    cases = admit(clocks, demand)
    require(isinstance(index["available_cases"], dict) and set(index["available_cases"]) <= set(cases),
            "The first-draft cache offers a foreign or duplicate case")
    recipe = Path(__file__).with_name("reference_9b_draft_sampler.py")
    require(sha(recipe) == SAMPLER, "The native first-draft calibration adapter changed")
    bundle = {"reference_9b_draft_sampler.py": recipe}
    for name in SAMPLE_SUPPORT:
        bundle[name] = member(cache, "sampler/" + name, TARGET_BUNDLE[name])
    controls = demand["scatter_calibration_controls"]
    available = audit_cached(cache, manifest, index, cases, controls)
    needed = {name: case for name, case in cases.items() if name not in available}
    result = dict(demand_sha256=DEMAND, cache_sha256=manifest_pin, retained=len(available),
                  sampled=len(needed), missing_cases=list(needed), sampler_invocations=int(bool(needed)))
    if needed:
        fresh, execution = dispatch(archive, folder, demand_path, bundle, needed, controls)
        result.update(execution)
        available.update(fresh)
    path = folder / "qualified-first-draft-clocks.json"
    save(path, dict(**result, cases=available))
    return available, dict(**result, index=path.relative_to(folder.parent).as_posix(), index_sha256=sha(path))

