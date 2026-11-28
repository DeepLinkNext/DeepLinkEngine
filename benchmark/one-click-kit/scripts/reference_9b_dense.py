"""Use the canonical dense cache through the guarded native 9B sampler."""

from pathlib import Path
import shutil
import subprocess
from tempfile import mkdtemp

from child_runner import run_stage
from dataset_contract import save, sha
from reference_9b_archive import CARDS, PINS, member, read

PLAN_PIN = "ef032d061ee10257cff2bde547a13112e95fd175a2b6ea97bd0effbb4549daa6"
CONTRACT = "mods/microflow/hust_prefill_microflow/school028/host-contract.json"
CONTRACT_PIN = "b14fed9456ed99d72eb45f3ff3d6690574d64dc3ed36fac8925001f6628e7b4d"
NAMESPACE_PIN = "56ff3086ba75eea4ba7a185ee7fd7c69a80a5ba12f1cd9c5cc077f5e299662f8"
DRIVER_PIN = "486b8d35e156ab2599bc2f65782cec0b9a11001b701ad9bb9f3e71142fce81cd"


def calibration(archive, independent, baseline_sdk, cache, cache_seal, folder, clocks):
    from reference_final_stages import resolve_dense_calibration

    cache_pin = sha(cache / "checksums.json")
    if cache_pin != cache_seal:
        raise ValueError("Canonical 9B cache seal SHA-256 differs")
    authority = read(member(independent, "dense-reference-plan-r2.json", PLAN_PIN))
    sdk = archive / "installation/profiles/qwen9b/sdk"
    role_sources = {"B0": baseline_sdk, "B1": sdk}
    for name, pin in authority["current_source_sha256"].items():
        role, relative = name.split("/", 1)
        member(role_sources[role], relative, pin)
    contract = read(member(sdk, CONTRACT, CONTRACT_PIN))
    native = archive / "native-clock/B1"
    seal = read(member(archive, "native-clock/B1/checksums.json", PINS["native-clock/B1/checksums.json"]))
    python = read(member(native, "launch-command.json", seal["launch-command.json"]))["command"][0]
    request = dict(schema="current-canonical-sampler-request/v1", family="dense",
        cases=authority["cases"], source_bindings=authority["current_source_sha256"])
    controller = folder / "controller-metadata"
    (controller / CONTRACT).parent.mkdir(parents=True)
    save(controller / "deployment.json", dict(python=python))
    save(controller / CONTRACT, contract)
    plan = dict(deployment=str(controller), physical_cards=CARDS, logical_devices="4,5",
                canonical_reuse=dict(source=str(cache), checksums_sha256=cache_pin))
    owner = Path(clocks["B0"]["archive"]).parents[1]

    def sampler(_, out, request_path, source):
        return sample(archive.parent, independent, baseline_sdk, sdk, owner, python, contract,
                      out, request_path, source)

    return resolve_dense_calibration(plan, folder, request, cache, run_sampler=sampler)


def sample(private_root, independent, baseline_sdk, sdk, owner, python, contract,
           folder, request_path, source):
    from canonical_protocol import prepare

    request = read(request_path)
    # Only these dimensions are established by the ordinary r11 target
    # projection route and equal replay/output row bounds in the original run.
    proven = {(m, n, k) for m in (3, 108) for n, k in ((12288, 4096), (4096, 6144))}
    for case in request["cases"]:
        shape = case["m"], case["n"], case["k"]
        if shape not in proven:
            raise ValueError(f"Missing 9B dense operand {shape} has no admitted exact native row witness")
    namespace = member(owner, "namespace-9b.sh", NAMESPACE_PIN)
    driver = member(independent, "run_9b_canonical_components.py", DRIVER_PIN)
    source.mkdir()
    prepared = prepare(source / "sampler", "dense", request["cases"], request["source_bindings"])
    job = Path(mkdtemp(prefix="qwen9b-one-click-dense.", dir=private_root))
    for name in (prepared["sampler"], prepared["plan"]):
        shutil.copyfile(source / "sampler" / name, job / name)
    kit = Path(__file__).resolve().parents[1] / "third-party-test-kit"
    shutil.copyfile(kit / "canonical_results.py", job / "canonical_results.py")
    body = driver.read_text(encoding="utf-8").replace(
        '"component_cases":648',
        '"component_cases":len(json.loads((root / "dense-reference-plan-r2.json").read_bytes())["cases"])',
    )
    controller = job / "controller.py"
    controller.write_text("import signal\n"
        "def cancel(signum, frame):\n"
        "    raise KeyboardInterrupt(f'controller cancellation {signum}')\n"
        "signal.signal(signal.SIGTERM, cancel)\n" + body, encoding="utf-8")
    command = ["unshare", "-m", "--propagation", "private", "bash", str(namespace),
               str(job), str(baseline_sdk.parents[1]), python, "-B", str(controller),
               str(job), str(owner), str(baseline_sdk), str(sdk)]
    save(folder / "actual-standalone-dispatch.json", dict(command=command, controller_sha256=sha(controller),
        original_controller_sha256=DRIVER_PIN, request_sha256=sha(request_path)))
    with (folder / "canonical-dense.log").open("x") as log:
        process = run_stage(command, stdout=log, stderr=subprocess.STDOUT)
    execution_path = job / "execution.json"
    if process.returncode:
        error = read(execution_path).get("error") if execution_path.is_file() else "see launcher log"
        raise ValueError(f"9B independent sampler exited {process.returncode}: {error}")
    execution = read(execution_path)
    allocation = job / "allocation"
    quota = read(allocation / "quota.json")
    previous_quota = read(independent / "allocation/quota.json")
    after = (allocation / "npu-after.txt").read_text()
    locks = (allocation / "locks-after.txt").read_text()
    if (execution["status"] != "completed_components" or execution["component_cases"] != len(request["cases"])
            or quota["physical"] != CARDS or quota["logical"] != CARDS
            or quota["allocation"] != previous_quota["allocation"]
            or any(f"No running processes found in NPU {card}" not in after for card in CARDS)
            or any(path in locks for path in quota["allocation"] + quota["admission"]["exclusive_leases"])):
        raise ValueError("9B independent sample did not close and release both cards and leases")
    raw = source / "sampler" / prepared["output"]
    raw.mkdir()
    for rank in (0, 1):
        name = f"rank-{rank}.json"
        shutil.copyfile(job / prepared["output"] / name, raw / name)
    for name in ("component-timings.json", "source-admission.json", "execution.json"):
        shutil.copyfile(job / name, source / name)
    shutil.copytree(allocation, source / "allocation")
    save(source / "mapping.json", dict(physical_cards=CARDS, logical_cards=CARDS))
    save(source / "dependencies.json", dict(python=python, metadata=contract["versions"], native_packages_imported=False))
    save(source / "request.json", request)
    save(source / "receipt.json", dict(schema="guarded-native-9b-component-sample/v1",
        complete=True, quiescent=True, model_loaded=False, request_sha256=sha(source / "request.json"),
        standalone_source=str(job), original_sampler_sha256=prepared["sampler_sha256"]))
    save(source / "checksums.json", {p.relative_to(source).as_posix(): sha(p) for p in sorted(source.rglob("*"))
         if p.is_file() and p.name != "checksums.json"})
