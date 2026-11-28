"""Run only absent 9B device task clocks under the original cluster lease guard."""

from pathlib import Path
import shutil
import subprocess
from tempfile import mkdtemp

from child_runner import run_stage
from dataset_contract import save, sha
from reference_9b_archive import PINS, member, read
from reference_9b_dense import DRIVER_PIN, NAMESPACE_PIN, PLAN_PIN

SAMPLER_PIN = "d363d6723599f3ca7afa9e3a0b92c2820d593103d7977ab80d8f838e81bd4cd5"
START = "            for repetition in range(plan['repetitions']):\n"
END = "            del graph, output, x\n"
TASK_PROFILE = """            profile = ROOT / 'device-task-profiles' / f'rank-{rank}' / f"case-{case['case']:03d}"
            barrier.wait(timeout=180)
            options = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0
            )
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile)),
                experimental_config=options,
            ) as recorder:
                for _ in range(plan['repetitions']):
                    for _ in range(plan['graph_warmups_per_repetition']):
                        graph.replay()
                    torch.npu.synchronize()
                    barrier.wait(timeout=180)
                    for _ in range(plan['timed_replays']):
                        graph.replay()
                    torch.npu.synchronize()
                recorder.step()
"""


def prepare(job, independent, missing, witness):
    from reference_9b_task_cache import PREFILL_PROOF, PREFILL_SHAPES, PREFILL_WITNESS, shape

    plan_path = job / "dense-reference-plan-r2.json"
    authority = read(member(independent, plan_path.name, PLAN_PIN))
    frozen = {shape(entry): entry["case"] for entry in authority["cases"]}
    chosen = []
    for row in missing:
        operand = shape(row)
        if operand in frozen:
            chosen.append(dict(row, original_case=frozen[operand]))
        elif witness == PREFILL_WITNESS and operand in PREFILL_SHAPES:
            chosen.append(dict(row, exact_native_operand_proof_sha256=PREFILL_PROOF))
        else:
            raise ValueError(f"No archived source admits the supplementary 9B device task {operand}")
    program = job / "dense_reference_repeat.py"
    legacy = member(independent, program.name, SAMPLER_PIN).read_text()
    if legacy.count(START) != 1 or legacy.count(END) != 1:
        raise ValueError("The sealed 9B device sampler does not have the expected timing scope")
    program.write_text(legacy[:legacy.index(START)] + TASK_PROFILE + legacy[legacy.index(END):], encoding="utf-8")
    compile(program.read_bytes(), str(program), "exec")
    save(plan_path, dict(authority, cases=chosen, original_sampler_sha256=sha(program),
        witnessed_operands_sha256=witness, timing_domain="cann-device-task-duration/v1"))
    implementation = member(independent, "run_9b_canonical_components.py", DRIVER_PIN).read_text()
    if (implementation.count("from canonical_results import audit") != 1
            or implementation.count('"component_cases":648') != 1):
        raise ValueError("The original device and quota guard bytes have a different entry")
    driver = job / "controller.py"
    implementation = implementation.replace("from canonical_results import audit",
        "from reference_9b_task_clock_results import audit").replace('"component_cases":648',
            f'"component_cases":{len(chosen)}')
    final_audit = ('        summary = audit(result, plan_path, quota["physical"])\n'
                   '        save(root / "component-timings.json", summary)\n')
    if implementation.count(final_audit) != 1:
        raise ValueError("The frozen controller does not have the expected post-sampling audit")
    implementation = implementation.replace(final_audit,
        "".join(line[4:] for line in final_audit.splitlines(keepends=True)))
    driver.write_text("import signal\n"
        "def cancel(signum, frame):\n"
        "    raise KeyboardInterrupt(f'controller cancellation {signum}')\n"
        "signal.signal(signal.SIGTERM, cancel)\n" + implementation, encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("reference_9b_task_clock_results.py"),
                    job / "reference_9b_task_clock_results.py")
    return driver


def calibrate(archive, independent, baseline_sdk, candidate_sdk, folder, clocks, missing, witness):
    """The caller has already checked exact native shape scope and cache visibility."""
    native = archive / "native-clock/B1"
    sealed = read(member(archive, "native-clock/B1/checksums.json", PINS["native-clock/B1/checksums.json"]))
    interpreter = read(member(native, "launch-command.json", sealed["launch-command.json"]))["command"][0]
    owner = Path(clocks["B0"]["archive"]).parents[1]
    namespace = member(owner, "namespace-9b.sh", NAMESPACE_PIN)
    job = Path(mkdtemp(prefix="qwen9b-task-auto-sample.", dir=archive.parent))
    driver = prepare(job, independent, missing, witness)
    command = ["unshare", "-m", "--propagation", "private", "bash", str(namespace), str(job),
               str(baseline_sdk.parents[1]), interpreter, "-B", str(driver), str(job),
               str(owner), str(baseline_sdk), str(candidate_sdk)]
    dispatch = folder / "device-task-clock-dispatch.json"
    save(dispatch, dict(command=command, missing_cases=missing, controller_sha256=sha(driver),
        native_launch_member_sha256=sealed["launch-command.json"], original_device_guard_sha256=DRIVER_PIN))
    with (folder / "device-task-clock-sampler.log").open("x") as log:
        result = run_stage(command, stdout=log, stderr=subprocess.STDOUT)
    state = read(job / "execution.json") if (job / "execution.json").is_file() else {}
    if result.returncode or state.get("status") != "completed_components" or state.get("component_cases") != len(missing):
        raise ValueError(f"9B task sampler failed in {job}: {state.get('error', f'exit {result.returncode}')}")
    files = [
        job / name for name in ("controller.py", "dense_reference_repeat.py", "dense-reference-plan-r2.json",
            "reference_9b_task_clock_results.py", "component-timings.json", "execution.json",
            "source-admission.json", "dense-reference-r2/rank-0.json", "dense-reference-r2/rank-1.json")
    ] + sorted((job / "allocation").glob("*")) + sorted((job / "device-task-profiles").rglob("kernel_details.csv"))
    manifest = job / "readback-checksums.json"
    save(manifest, {path.relative_to(job).as_posix(): sha(path) for path in files})
    return job, sha(manifest), dispatch
