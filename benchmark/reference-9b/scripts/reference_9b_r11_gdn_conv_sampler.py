"""Collect CANN task clocks only for cache-missing normalized r11 GDN cases."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys

from calibrate_r11_native_kv_task import (
    B1, BASE, DONOR, NAMESPACE, NAMESPACE_SHA256, ORIGINAL, PYTHON, read, require, save, sha,
)


OPERANDS = "r11-cold-gdn-conv-operands.json"
DEMAND = "202c14ce4d8911218d12ab32f50f612fb1df64380c943c090812991e166009f6"
VERSIONS = {"torch": "2.13.0+cpu", "torch-npu": "2.13.0rc1"}
VENDOR = "d56523e9328794b811d3dc7a6bc72523373c556412c4cfc9f2622f1f5afe4bc2"
REPEATS, WARMUPS, CALLS = 10, 3, 20


def select(root):
    source = root / OPERANDS
    request = read(root / "request.json")
    require(sha(source) == DEMAND == request["operands_sha256"], "The exact r11 GDN convolution proof changed")
    plan = read(source)
    by_name = {case["id"]: case for case in plan["missing_independent_clocks"]}
    keys = request["case_ids"]
    require(keys and len(set(keys)) == len(keys) and set(keys) <= by_name.keys(), "Unknown GDN cache miss")
    return plan, [by_name[name] for name in keys]


def guard(root):
    sys.path.insert(0, str(ORIGINAL))
    from run_revised_context import prepare
    from candidate_campaign import resource_guard, screen_campaign

    _, missed = select(root)
    config, *_ = prepare()
    allocation = root / "allocation"
    allocation.mkdir()
    runtime = [B1 / path for path in ("runtime/engine-core", "runtime/ascend")]
    with screen_campaign.reserve(config, allocation, resource_guard, expected_pair=[4, 5]) as (quota, fds):
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, quota["logical"])),
                   TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=os.pathsep.join(map(str, runtime)))
        for case in missed:
            subprocess.run([sys.executable, "-B", __file__, str(root), "rank", str(case["rank"])],
                           cwd=root, env=env, pass_fds=fds, timeout=360, check=True)


def task_profile(torch, npu, device, graph, path):
    config = npu.profiler._ExperimentalConfig(profiler_level=npu.profiler.ProfilerLevel.Level0)
    with npu.profiler.profile(
            activities=[npu.profiler.ProfilerActivity.NPU],
            schedule=npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=npu.profiler.tensorboard_trace_handler(str(path)), experimental_config=config) as recorder:
        for _ in range(REPEATS):
            for _ in range(WARMUPS):
                graph.replay()
            torch.npu.synchronize(device)
            for _ in range(CALLS):
                graph.replay()
            torch.npu.synchronize(device)
        recorder.step()


def rank_run(root, rank):
    import torch
    import torch_npu
    import vllm_ascend
    from vllm_ascend.utils import enable_custom_op

    plan, selected = select(root)
    case, = [one for one in selected if one["rank"] == rank]
    versions = {name: version(name) for name in VERSIONS}
    require(versions == VERSIONS and os.environ["ASCEND_HOME_PATH"].endswith("/cann-9.1.0") and
            Path(vllm_ascend.__file__).resolve().is_relative_to(B1),
            "The convolution sampler's Torch, CANN or Ascend package differs from r11")
    torch.set_num_threads(4)
    device = torch.device("npu", rank)
    torch.npu.set_device(device)
    require(enable_custom_op(), "The sealed r11 custom operators could not be registered")
    op = torch.ops._C_ascend.npu_causal_conv1d_custom
    sample = plan["convolution"]
    generator = torch.Generator().manual_seed(42071 + rank)
    def randomize(shape):
        return (torch.randn(shape, generator=generator) * 0.15).bfloat16()
    with torch.inference_mode():
        x, weight = (randomize(sample[key]) for key in ("input_shape", "weight_shape"))
        prior = randomize(sample["state_backing"]["shape"])
        cpu = x.float().transpose(0, 1).unsqueeze(0)
        packed = weight.float().transpose(0, 1).unsqueeze(1).contiguous()
        expected = torch.nn.functional.silu(torch.nn.functional.conv1d(
            torch.nn.functional.pad(cpu, (3, 0)), packed, groups=4096))[0].transpose(0, 1).bfloat16()
        tokens, kernel = x.to(device), weight.to(device)
        out, state = torch.empty_like(tokens), prior.to(device)
        start = torch.tensor(sample["canonical_cu"], dtype=torch.int32, device=device)
        active = torch.tensor(sample["canonical_cache_indices"], dtype=torch.int32, device=device)
        init = torch.tensor(sample["canonical_initial_state"], dtype=torch.bool, device=device)
        def invoke():
            op(out, tokens, kernel, conv_state=state, bias_opt=None, query_start_loc_opt=start,
               cache_indices_opt=active, initial_state_mode_opt=init, num_accepted_tokens_opt=None,
               activation_mode=1, pad_slot_id=-1, run_mode=0)
        invoke()
        torch.npu.synchronize(device)
        observed, eager_state = out.cpu(), state.cpu()
        torch.testing.assert_close(observed, expected, **sample["tolerance"])
        torch.testing.assert_close(eager_state[0, :3], x[-3:], atol=0, rtol=0)
        torch.testing.assert_close(eager_state[1:], prior[1:], atol=0, rtol=0)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            invoke()
        graph.replay()
        torch.npu.synchronize(device)
        torch.testing.assert_close(out.cpu(), observed, atol=0, rtol=0)
        torch.testing.assert_close(state.cpu(), eager_state, atol=0, rtol=0)
        libraries = {Path(line.split()[-1]) for line in Path("/proc/self/maps").read_text().splitlines()
                     if line.endswith("/libcust_opapi.so")}
        require(libraries and {sha(path) for path in libraries} == {VENDOR}, "Wrong loaded convolution vendor API")
        case_root = root / case["id"].replace(":", "-")
        case_root.mkdir()
        task_profile(torch, torch_npu, device, graph, case_root / "profile")
        save(case_root / "numeric.json", dict(case=case, operands_sha256=DEMAND, versions=versions,
             input_shape=list(tokens.shape), weight_shape=list(kernel.shape), state_shape=list(state.shape),
             state_stride=list(state.stride()), vendor_api_sha256=VENDOR, numerical_passed=True,
             convolution_suffix_verified=True, inactive_state_unchanged=True, graph_agreed=True,
             max_abs=float((observed.float() - expected.float()).abs().max()), model_inference=False,
             sampler_sha256=sha(Path(__file__))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("mode", choices=("launch", "guard", "rank"))
    parser.add_argument("rank", nargs="?", type=int, choices=(0, 1))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "launch":
        select(root)
        require(root.parent == BASE and sha(NAMESPACE) == NAMESPACE_SHA256, "The r11 CANN namespace changed")
        command = ["unshare", "-m", "--propagation", "private", "bash", str(NAMESPACE), str(root),
                   str(DONOR), str(PYTHON), "-B", __file__, str(root), "guard"]
        os.execvp(command[0], command)
    elif args.mode == "guard":
        guard(root)
    else:
        rank_run(root, args.rank)


if __name__ == "__main__":
    main()

