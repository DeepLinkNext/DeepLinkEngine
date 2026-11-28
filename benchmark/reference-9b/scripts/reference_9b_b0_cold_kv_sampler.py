"""Measure the original018 canonical cold scatter in the frozen 8.5.1 runtime."""
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys

from calibrate_r11_native_kv_task import (
    BASE, DONOR, NAMESPACE, NAMESPACE_SHA256, ORIGINAL, PYTHON, read, require, save, sha,
)
from prepare_b0_tp2_task_panel import ADMISSION, ARCHIVED_TOOLS, B0, ENTRY_SHA, SDK


OPERANDS = "original018-cold-kv-operands.json"
DEMAND = "13f9931e48d20d8887a10009eca5783c5b6e9084496afea79a35069a98e23565"
VERSIONS = {"torch": "2.9.0+cpu", "torch-npu": "2.9.0.post1+gitee7ba04"}
OP = "vllm-ascend/vllm_ascend/device/device_op.py"


def proof(root):
    path = root / OPERANDS
    request = read(root / "request.json")
    require(sha(path) == request["operands_sha256"] == DEMAND, "Wrong original018 first-write operands")
    plan = read(path)
    cases = {row["id"]:row for row in plan["independent_case_ids"]}
    ids = request["case_ids"]
    require(ids and len(ids) == len(set(ids)) and set(ids) <= cases.keys(), "Unknown original018 first-write request")
    return plan, [cases[key] for key in ids]


def sample(root, rank):
    import torch
    import torch_npu

    plan, requested = proof(root)
    current, = [case for case in requested if case["rank"] == rank]
    versions = {name:version(name) for name in VERSIONS}
    require(versions == VERSIONS and os.environ["ASCEND_HOME_PATH"].endswith("/cann-8.5.1")
            and sha(SDK/"text-only-overlay"/OP) == plan["authoritative_sources_sha256"][OP],
            "Original018 cache producer or CANN/Torch changed")
    torch.set_num_threads(4)
    device = torch.device("npu", rank)
    torch.npu.set_device(device)
    operator = plan["operator"]
    shape = operator["key_value_shape"]
    case_root = root/current["id"].replace(":", "-")
    case_root.mkdir()
    with torch.inference_mode():
        rows = torch.arange(shape[0])
        require(all(0 <= slot < 2**31 for slot in operator["slots"]), "Canonical cache slot cannot be narrowed")
        slots = torch.tensor(operator["slots"], dtype=torch.int32, device=device)
        witness_indices = slots.to(dtype=torch.int64)
        wanted = (rows % 97).to(torch.bfloat16).reshape(-1, 1, 1).expand(shape).contiguous()
        key, value = torch.empty(shape, dtype=torch.bfloat16, device=device), torch.empty(shape, dtype=torch.bfloat16, device=device)
        cache = [torch.empty(operator["cache_plane_shape"], dtype=torch.bfloat16, device=device) for _ in range(2)]
        owner = torch.npu.current_stream(device)
        consumer = torch.npu.Stream(device=device)
        def write():
            torch_npu._npu_reshape_and_cache(key=key, value=value, key_cache=cache[0], value_cache=cache[1],
                                             slot_indices=slots)
        for marker in (31, 0):
            expected = wanted + marker
            key.copy_(expected.to(device))
            value.copy_((-expected-4).to(device))
            write()
            done = torch.npu.Event()
            done.record(owner)
            with torch.npu.stream(consumer):
                consumer.wait_event(done)
                for plane, truth in zip(cache, (expected, -expected-4), strict=True):
                    seen = plane.view(-1, 2, 256).index_select(0, witness_indices).cpu()
                    torch.testing.assert_close(seen, truth, atol=0, rtol=0)
            torch.npu.synchronize(device)
        exp = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level0)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(case_root/"profile")),
            experimental_config=exp
        ) as recorder:
            for _ in range(10):
                for _ in range(3):
                    write()
                torch.npu.synchronize(device)
                for _ in range(20):
                    write()
                torch.npu.synchronize(device)
            recorder.step()
        save(case_root/"numeric.json", dict(case=current, operands_sha256=DEMAND,
            versions=versions, case_has_no_model=True, sampler_sha256=sha(Path(__file__)),
            numerical_passed=True, independent_fenced_generations=2, slots=operator["slots"],
            operand_shape=list(key.shape), plane_shape=list(cache[0].shape), plane_stride=list(cache[0].stride()),
            slot_indices_dtype="int32", timing_domain="cann-device-task-duration/v1"))


def guard(root):
    _, requested = proof(root)
    sys.path.insert(0, str(ORIGINAL))
    from candidate_campaign import resource_guard, screen_campaign
    from run_revised_context import prepare

    config, *_ = prepare()
    require(sha(ARCHIVED_TOOLS/"baseline_entry.py") == ENTRY_SHA, "Frozen original018 launcher changed")
    sys.path.insert(0, str(ARCHIVED_TOOLS))
    from baseline_entry import baseline_environment
    from baseline_image_sdk import python as native_python

    allocation = root/"allocation"
    allocation.mkdir()
    with screen_campaign.reserve(config, allocation, resource_guard, expected_pair=[4,5]) as (quota, fds):
        env = baseline_environment(SDK, root, root/"original018-cache", ",".join(map(str, quota["logical"])))
        for item in requested:
            command = [str(B0/"oci-libc-v1/with-image-libc.sh"), str(native_python(SDK)), "-S", "-B", "-u",
                       __file__, str(root), "sample", str(item["rank"])]
            subprocess.run(command, env=env, cwd=root, pass_fds=fds, timeout=240, check=True)


def main():
    root, mode = Path(sys.argv[1]).resolve(), sys.argv[2]
    if mode == "sample":
        sample(root, int(sys.argv[3]))
    elif mode == "guard":
        guard(root)
    elif mode == "launch":
        proof(root)
        require(root.parent == BASE and sha(NAMESPACE) == NAMESPACE_SHA256, "Frozen B0 namespace differs")
        command = ["unshare", "-m", "--propagation", "private", "bash", str(NAMESPACE), str(root), str(DONOR),
                   str(PYTHON), "-B", __file__, str(root), "guard"]
        os.execvp(command[0], command)
    else:
        raise ValueError("Unknown original018 sampler mode")


if __name__ == "__main__":
    main()
