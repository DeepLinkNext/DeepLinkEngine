"""Resolve the original018 clock for the first uncached target FIA on each rank."""

from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import mkdtemp

from child_runner import run_stage
from dataset_contract import save, sha
from reference_9b_archive import B0_SEAL, member, read
from reference_9b_target_fia import release, require, task_clock


SCOPE = "0aa542c5aba3101f121c7fbdb28ca953150495b7a8395960c5f120110a4c0be5"
PANEL = "18e4aeb3fc262ff8c0117f7dcd9a8a9445e1975afa218e5ed61ee24f10eb8e47"
PROJECTION = "77a98f23d3a1f6aa0497d10ff98dc31f7a49bbf7572708d012c5ee453503d0b9"
BUNDLE = {
    "b0-first-target-fia-operands.json": SCOPE,
    "b0-first-native-prefill.json": "30f7921a4e28ba9ea5b76c6a5b3473811e948bc6c4a0c85a8a6b089569190d0f",
    "b0-first-target-fia-source.tar": "d1de3a15e7592a74ae69719c524a31b652df3229f7c774e7cbb46e2a410bac74",
    "config.json": "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05",
    "audit_b0_first_target_fia.py": "2e4264512f72d494f0534209e8851eca5a5493fbf4930db432faf68fb7e3bb66",
    "calibrate_b0_first_target_fia.py": "13591775f0c1cd55f39b575fa72c05342fa7349e719ce4e4e6228da48fc62942",
    "calibrate_r11_native_kv_task.py": "6d45f1ccc717869a07d5f0c26b80f204ebd2e05cf18441a25d0e907f6125111f",
    "prepare_b0_tp2_task_panel.py": "0323174aea8c76108d85af0d0ffa0534486963263fc3e28dcdce04a06a066b27",
}
EXPECTED = {0: "rank-0", 1: "rank-1"}


def sample(archive, folder, bundle, missing, proof):
    run = Path(mkdtemp(prefix="qwen9b-b0-direct-fia-auto.", dir=archive.parent))
    for name, path in bundle.items():
        shutil.copyfile(path, run / name)
    with tarfile.open(run / "b0-first-target-fia-source.tar", "r:") as source:
        source.extractall(run / "sources", filter="data")
    donor = archive.parent / ("r10-kernel-native-10204-20261124.6ke9QK/runtime/clean-cache-v2/"
                              "environments/dc75dd91be2a521c-9ebc8e5cb831/venv/bin/python")
    command = [str(donor), "-B", str(run / "calibrate_b0_first_target_fia.py"), str(run), "launch", *map(str, missing)]
    dispatch = folder / "b0-direct-fia-dispatch.json"
    save(dispatch, dict(operands_sha256=SCOPE, missing_cases=[EXPECTED[rank] for rank in missing], command=command))
    with (folder / "b0-direct-fia-sampler.log").open("x") as log:
        result = run_stage(command, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0 and (run / "launcher.exit").read_text().strip() == "0",
            f"Exact original018 direct FIA calibration failed at {run}")
    allocation = release(run)
    clocks = {}
    for rank in missing:
        name = EXPECTED[rank]
        card = proof["physical_cards"][rank]
        result = run / f"{name}.json"
        numeric = read(result)
        require(numeric["rank"] == rank and numeric["physical_card"] == card and numeric["operands_sha256"] == SCOPE
                and numeric["eager_actual_fia"] is True and numeric["model_inference"] is False
                and numeric["numerical_passed"] is True and numeric["query_shape"] == proof["query_shape"]
                and numeric["key_value_shape"] == proof["key_value_shape"] and numeric["mask_shape"] == proof["mask_shape"],
                "The new original018 FIA clock does not match its admitted operand and numerical witness")
        trace, = (run / "profiles" / name).glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv")
        clocks[rank] = dict(seconds=task_clock(trace, card), task_sha256=sha(trace), numerical_sha256=sha(result),
                            source="sampled", task_path=str(trace))
    return clocks, dict(run=str(run), release_sha256=allocation, dispatch_sha256=sha(dispatch))


def resolve(archive, folder, cache, manifest_pin, clocks):
    require(cache is not None and manifest_pin is not None, "Original018 direct FIA requires its exact cache manifest")
    manifest = read(member(cache, "checksums.json", manifest_pin))
    index_path = "b0-direct-fia-cache.json"
    index = read(member(cache, index_path, manifest[index_path]))
    active = index["available_cases"]
    require(index["schema"] == "m01-9b-original018-direct-fia-clock-cache/v1" and index["operands_sha256"] == SCOPE
            and isinstance(active, list) and len(active) == len(set(active)) and set(active) <= set(EXPECTED.values()),
            "Invalid original018 direct FIA clock index")
    bundle = {}
    for name, pin in BUNDLE.items():
        entry = "b0-sampler/" + name
        require(manifest[entry] == pin, f"Original018 direct FIA sampling input changed: {name}")
        bundle[name] = member(cache, entry, pin)
    proof = read(bundle["b0-first-target-fia-operands.json"])
    panel_name = "b0-first-target-fia-clocks.json"
    require(manifest[panel_name] == PANEL and clocks["B0"]["checksums_sha256"] == B0_SEAL
            and [row["timing_sha256"] for row in proof["rounds"]] == [row["source_sha256"] for row in clocks["B0"]["rounds"]],
            "Original018 direct FIA is paired to different native wall boundaries")
    panel = read(member(cache, panel_name, PANEL))
    missing, available = [], {}
    for rank, name in EXPECTED.items():
        if name not in active:
            missing.append(rank)
            continue
        signed = panel["ranks"][rank]
        relative = "b0-traces/" + name + ".csv"
        pin = manifest[relative]
        require(pin == signed["task_sha256"], "A retained original018 FIA trace is outside its signed panel")
        trace = member(cache, relative, pin)
        seconds = task_clock(trace, signed["card"])
        require(seconds == signed["calibration_median_seconds"], "Original018 raw trace does not reproduce its statistic")
        available[rank] = dict(seconds=seconds, task_sha256=pin, source="retained", task_path=str(trace))
    selection = dict(operands_sha256=SCOPE, cache_sha256=manifest_pin, retained=len(available), sampled=len(missing),
                     sampler_invocations=int(bool(missing)), missing_cases=[EXPECTED[rank] for rank in missing])
    if missing:
        sampled, receipt = sample(archive, folder, bundle, missing, proof)
        available.update(sampled)
        selection.update(receipt)
    result = folder / "qualified-b0-direct-fia-clocks.json"
    save(result, dict(**selection, cases={EXPECTED[rank]: item for rank, item in available.items()}))
    return available, dict(**selection, index_sha256=sha(result), index=result.relative_to(folder.parent).as_posix())
