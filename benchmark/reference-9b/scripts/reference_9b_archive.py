"""Read the original018/r11 RC4 9B acceptance and native clocks."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import statistics

from dataset_contract import sha

RELEASE = "full-answer-b1-20261125-r11-rc4"
MODEL = "Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CARDS = [4, 5]
HLE = "hle-judge-recovery.VpNn7TUa"
PINS = {
    "acceptance-summary.json": "479009da417fdde21b1ac3151e4e5eb68cac2af238d50b27a9342b31e0c6b667",
    f"{HLE}/completion-admission.json": "6c7e7309788c1dc3b0b639614e2e1d2793da847c371ed7a4c8f35e3c4aa1f7c4",
    f"{HLE}/acceptance-summary.json": "905630dcda4818687478e707e0f2fa6439b1bacc4e90ac32c3fec543110718d6",
    f"{HLE}/remaining-state.json": "ee2d000db8735221b322103b4a75b7de5b38e3fcd4d2cb27c69b1cf78659c6c9",
    f"{HLE}/derived_summarize_campaign.py": "27acf33c224d226b01b6385621420e09a557669ed11d8502bde2d31108638675",
    "context/release-verification.json": "f91dfa919284807282ca859ab81a868f65b4040f963db243c97d36e4fb31f371",
    "native-clock/B1/checksums.json": "ff625e149052ce2e0964f17d46a6c9123bce11847857b0ea222edc1234ef0ead",
}
B0_SEAL = "cd1604a5fb808140bb2c55b496a2d7a03244c31ab2a6214965f2d5709a271f2d"
CHECKPOINT_SEAL = "5ef4a37eb81805e51a6014fccd314c1e09daedf52d29b9009913e64674644ffe"
WORKLOAD_SEAL = "6ed460a7f6c720a39b17b7084e7c040219c7bd97975cd1c5a1a3abb89dfe29a7"


def read(path):
    return json.loads(path.read_bytes())


def member(root, name, digest):
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha(path) != digest:
        raise ValueError(f"Archive SHA-256 or path differs for {name}")
    return path


def retained_sections(root):
    # The frozen assembler uses absolute paths for the retained B0 native
    # source. Reuse it with its original r11 evidence root.
    paths = {name: member(root, name, pin) for name, pin in PINS.items()}
    archived = read(paths[f"{HLE}/acceptance-summary.json"])
    state = read(paths[f"{HLE}/remaining-state.json"])
    completion = read(paths[f"{HLE}/completion-admission.json"])
    if (completion["work"] != str(root) or completion["physical_cards"] != CARDS
            or archived["engine"] != RELEASE or archived["model"] != MODEL
            or completion["completed_summary_sha256"] != PINS[f"{HLE}/acceptance-summary.json"]):
        raise ValueError("Retained r11 recovery or model domain differs")
    for item in state["evidence"].values():
        member(root, item["asset"], item["sha256"])
    context_seal = state["evidence"]["context-seal"]
    context = read(root / context_seal["asset"])
    for name, pin in context.items():
        member(root / "context", name, pin)
    assembler_spec = spec_from_file_location("sealed_r11_acceptance_assembly",
        paths[f"{HLE}/derived_summarize_campaign.py"])
    assembler = module_from_spec(assembler_spec)
    assembler_spec.loader.exec_module(assembler)
    if assembler.assemble(root, state) != archived:
        raise ValueError("Archived r11 acceptance does not reproduce")
    initial = read(paths["acceptance-summary.json"])
    sections = {key: archived[key] for key in ("4.2", "4.3", "4.4")}
    if any(sections[key] != initial[key] for key in ("4.3", "4.4")):
        raise ValueError("The judge recovery changed 4.3 or 4.4")
    return sections, dict(source=str(root), members_sha256=PINS,
        context_member_count=len(context), evidence=state["evidence"],
        context_worker_release=read(paths["context/release-verification.json"]))


def native_clocks(root):
    comparator = read(member(root, "native-b0-comparator.json",
        "32317b84851be104ba11282f8c94618b02bd87dd85388fc9c27d1feff5a19f02"))
    paths = {"B0": Path(comparator["path"]), "B1": root / "native-clock/B1"}
    seals = {"B0": B0_SEAL, "B1": PINS["native-clock/B1/checksums.json"]}
    timings = {}
    for role, directory in paths.items():
        files = read(member(directory, "checksums.json", seals[role]))
        config = read(member(directory, "native-plan.json", files["native-plan.json"]))
        profile = read(member(directory, "client/effective-engine-args.json", files["client/effective-engine-args.json"]))
        if (config["role"] != role or config["max_tokens"] != 256 or config["formal_repetitions"] != 3
                or profile["tensor_parallel_size"] != 2 or profile["dtype"] != "bfloat16"):
            raise ValueError(f"Native workload scope differs for {role}")
        checkpoint = Path(config["checkpoint_receipt"])
        if read(member(checkpoint.parent, checkpoint.name, CHECKPOINT_SEAL))["revision"] != REVISION:
            raise ValueError(f"The model checkpoint revision differs for {role}")
        workload = Path(config["workload"])
        if len(read(member(workload.parent, workload.name, WORKLOAD_SEAL))["request_token_ids"]) != 200:
            raise ValueError(f"The ordered request population differs for {role}")
        rows = []
        for repetition in (1, 2, 3):
            name = f"client/formal-{repetition}-timing.json"
            timing = read(member(directory, name, files[name]))
            if (timing["repetition"] != repetition or timing["output_tokens"] != 51200
                    or timing["ordinary_public_outputs_retained"] is not True
                    or len(timing["boundary"]) != 2
                    or {entry["rank"] for entry in timing["boundary"]} != {0, 1}
                    or any(entry["sealed"] is not True or entry["failure"] is not None
                           for entry in timing["boundary"])):
                raise ValueError(f"The original dual-card timing window failed for {role} repetition {repetition}")
            interval = timing["end_ns"] - timing["start_ns"]
            if interval <= 0 or interval / 10**9 != timing["T_seconds"]:
                raise ValueError(f"The monotonic duration differs for {role} repetition {repetition}")
            rows.append(dict(repetition=repetition, wall_seconds=timing["T_seconds"], source_sha256=files[name]))
        timings[role] = dict(archive=str(directory), checksums_sha256=seals[role],
            median_wall_seconds=statistics.median(row["wall_seconds"] for row in rows), rounds=rows)
    return timings
