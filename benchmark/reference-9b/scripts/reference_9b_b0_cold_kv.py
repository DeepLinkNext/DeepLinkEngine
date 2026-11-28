"""Resolve B0 cold KV movement clocks with a normalized page-label domain."""
from pathlib import Path
import shutil
import subprocess
from tempfile import mkdtemp

from child_runner import run_stage
from dataset_contract import save, sha
from reference_9b_archive import B0_SEAL, member, read
from reference_9b_b0_fia import BUNDLE, SCOPE as FIA_SCOPE
from reference_9b_r11_gdn_conv import device_clock
from reference_9b_target_fia import release, require


NOTE = "original018-cold-kv-operands.json"
DEMAND = "13f9931e48d20d8887a10009eca5783c5b6e9084496afea79a35069a98e23565"
INDEX = "original018-cold-kv-cache.json"
SCHEMA = "m01-original018-cold-kv-clock-cache/v1"
SAMPLER = "reference_9b_b0_cold_kv_sampler.py"
SAMPLER_SHA = "36b8434f2b592247c738b0b3fec4f8160343673afc050157c4848ab2bc397038"
HELPERS = ("calibrate_r11_native_kv_task.py", "prepare_b0_tp2_task_panel.py")
LEASE = ("allocation/quota.json", "allocation/npu-after.txt", "allocation/locks-after.txt")
VERSIONS = {"torch": "2.9.0+cpu", "torch-npu": "2.9.0.post1+gitee7ba04"}


def admitted_clock(job, entries, case, operand):
    for name, digest in {NOTE:DEMAND, SAMPLER:SAMPLER_SHA, **{file:BUNDLE[file] for file in HELPERS}}.items():
        require(entries[name] == digest, f"Wrong original018 sampler or proof member: {name}")
        member(job, name, digest)
    for name in LEASE:
        member(job, name, entries[name])
    request = read(member(job, "request.json", entries["request.json"]))
    require(request["operands_sha256"] == DEMAND and case["id"] in request["case_ids"],
            "The original018 cache writer was outside the signed sampler request")
    slug = case["id"].replace(":", "-")
    name = slug + "/numeric.json"
    witness = read(member(job, name, entries[name]))
    require(witness["case"] == case and witness["versions"] == VERSIONS and witness["sampler_sha256"] == SAMPLER_SHA
            and witness["case_has_no_model"] and witness["numerical_passed"] and witness["independent_fenced_generations"] == 2
            and witness["operands_sha256"] == DEMAND and witness["slots"] == operand["slots"]
            and witness["operand_shape"] == operand["key_value_shape"] and witness["plane_shape"] == operand["cache_plane_shape"]
            and witness["plane_stride"] == operand["cache_plane_stride"]
            and witness["slot_indices_dtype"] == operand["slot_indices_dtype"]
            and witness["timing_domain"] == "cann-device-task-duration/v1",
            "The original018 exact cache writer or two-generation witness changed")
    traces = [file for file in entries if file.startswith(slug+"/") and file.endswith("/kernel_details.csv")]
    require(len(traces) == 1, "Original018 reshape-and-cache device trace is absent or ambiguous")
    task = member(job, traces[0], entries[traces[0]])
    seconds, kernel, blocks = device_clock(task, case["physical_card"],
        expected=("reshapeandcachendkernel",), block_field="Block Dim")
    return dict(case=case, seconds=seconds, kernel=kernel, blocks=blocks, task_sha256=entries[traces[0]],
                witness_sha256=entries[name], release_sha256=release(job), page_normalized=True)


def calibrate(archive, output, proof, target_cache, missed, operand):
    job = Path(mkdtemp(prefix="original018-cold-kv-auto.", dir=archive.parent))
    files = {name:member(target_cache, "b0-sampler/"+name, BUNDLE[name]) for name in HELPERS}
    files.update({NOTE:proof, SAMPLER:member(Path(__file__).parent, SAMPLER, SAMPLER_SHA)})
    for name, path in files.items():
        shutil.copyfile(path, job/name)
    save(job/"request.json", dict(operands_sha256=DEMAND, case_ids=list(missed)))
    donor = archive.parent/("r10-kernel-native-10204-20261124.6ke9QK/runtime/clean-cache-v2/"
                             "environments/dc75dd91be2a521c-9ebc8e5cb831/venv/bin/python")
    cmd = [str(donor), "-B", str(job/SAMPLER), str(job), "launch"]
    save(output/"b0-cold-kv-dispatch.json", dict(case_ids=list(missed), job=str(job), command=cmd, inference=False))
    with (output/"b0-cold-kv-sampler.log").open("x") as log:
        child = run_stage(cmd, stdout=log, stderr=subprocess.STDOUT)
    require(child.returncode == 0, f"Original018 cold KV calibration failed at {job}, exit {child.returncode}")
    numeric = sorted(job.glob("b0-cold-kv-*/numeric.json"))
    require({path.parent.name for path in numeric} == {case.replace(":", "-") for case in missed},
            "Original018 cold KV sampler ran an unrequested case")
    profiles = sorted(job.glob("b0-cold-kv-*/**/kernel_details.csv"))
    keys = [*files, "request.json", *LEASE, *(path.relative_to(job).as_posix() for path in (*numeric, *profiles))]
    manifest = {name:sha(job/name) for name in keys}
    save(job/"readback-checksums.json", manifest)
    return ({name:admitted_clock(job, manifest, case, operand) for name, case in missed.items()},
            dict(source_root=str(job), raw_manifest_sha256=sha(job/"readback-checksums.json")))


def resolve(archive, output, cache, seal, target_cache, clocks):
    manifest = read(member(cache, "checksums.json", seal))
    proof = member(cache, NOTE, DEMAND)
    plan = read(proof)
    require(manifest[NOTE] == DEMAND and plan["schema"] == "m01-9b-original018-canonical-first-kv-write/v1"
            and clocks["B0"]["checksums_sha256"] == plan["original_native_sha256"] == B0_SEAL
            and plan["first_fia_operands_sha256"] == FIA_SCOPE and not plan["old_absolute_pages_used"]
            and [row["original_wall_sha256"] for row in plan["original_rounds"]]
                == [row["source_sha256"] for row in clocks["B0"]["rounds"]],
            "Original018 cold KV proof refers to different original first packets")
    operand = plan["operator"]
    cases = {row["id"]:row for row in plan["independent_case_ids"]}
    index = read(member(cache, INDEX, manifest[INDEX]))
    require(index["schema"] == SCHEMA and index["operands_sha256"] == DEMAND
            and set(index["available"]) <= cases.keys(), "Wrong original018 first-write cache")
    kept = {}
    for name, record in index["available"].items():
        root = Path(record["source_root"])
        entries = read(member(root,"readback-checksums.json",record["raw_manifest_sha256"]))
        kept[name] = admitted_clock(root, entries, cases[name], operand)
    missed = {name:case for name,case in cases.items() if name not in kept}
    result = dict(operands_sha256=DEMAND, cache_sha256=seal, retained=len(kept), sampled=len(missed),
                  sampler_invocations=int(bool(missed)), exact_missing_cases=list(missed))
    if missed:
        fresh, result["device_run"] = calibrate(archive, output, proof, target_cache, missed, operand)
        kept.update(fresh)
    save(output/"b0-cold-kv-clocks.json", dict(**result, clocks=kept))
    return {case["rank"]:kept[name] for name,case in cases.items()}, result
