"""Measure only selected r11 first-draft FIA and KV operations in the frozen namespace."""

import argparse
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys

from calibrate_r11_first_target_fia import sample_case as sample_fia_case
from calibrate_r11_native_kv_task import (
    BASE, DONOR, NAMESPACE, NAMESPACE_SHA256, ORIGINAL, PYTHON, WRITER, WRITER_SHA256,
    load_writer, read, require, save, sha,
)


DEMAND = "5cea6291d431044e325818a89ab2a3109e3d66651c546745a777ece764bec309"
VERSIONS = {"torch": "2.13.0+cpu", "torch-npu": "2.13.0rc1"}
FIA = "canonical-fused-infer-attention-score"


def selection(root):
    plan_path = root / "draft-clock-demand.json"
    require(sha(plan_path) == DEMAND, "The first-draft operand demand differs")
    plan = read(plan_path)
    request = read(root / "request.json")
    available = {case["id"]: case for case in plan["cases"]}
    ids = request["missing_cases"]
    require(request["demand_sha256"] == DEMAND and isinstance(ids, list)
            and ids and len(set(ids)) == len(ids) and set(ids) <= set(available),
            "The isolated sampler received unbound first-draft cases")
    return plan, [available[case] for case in ids]


def scatter(torch, torch_npu, case_root, case, pool, controls):
    k_cache, v_cache = pool
    device = k_cache.device
    slot = torch.tensor([case["write_slot"]], dtype=torch.int64, device=device)
    reader = torch.npu.Stream(device=device)
    writer = load_writer(torch_npu)
    for generation in (27, 3):
        expected = torch.full((1, 2, 256), generation, dtype=torch.bfloat16)
        k, v = expected.to(device), (-expected - 3).to(device)

        def call():
            writer(None, key=k, value=v, key_cache=k_cache, value_cache=v_cache, slot_mapping=slot)

        call()
        fence = torch.npu.Event()
        fence.record(torch.npu.current_stream(device))
        with torch.npu.stream(reader):
            reader.wait_event(fence)
            for plane, wanted in zip(pool, (expected, -expected - 3), strict=True):
                found = plane.view(-1, 2, 256).index_select(0, slot).cpu()
                torch.testing.assert_close(found, wanted, atol=0, rtol=0)
    torch.npu.synchronize(device)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    graph.replay()
    torch.npu.synchronize(device)
    profile = case_root / "scatter-profile"
    options = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level0)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile)),
        experimental_config=options,
    ) as recorder:
        for _ in range(controls["repetitions"]):
            for _ in range(controls["warmups_per_repetition"]):
                graph.replay()
            torch.npu.synchronize(device)
            for _ in range(controls["replays"]):
                graph.replay()
            torch.npu.synchronize(device)
        recorder.step()
    return dict(profile=profile.relative_to(case_root).as_posix(), numerical_passed=True,
                write_slot=case["write_slot"], key_shape=[1, 2, 256],
                plane_shape=list(k_cache.shape), plane_stride=list(k_cache.stride()),
                fenced_two_generation_bitwise=True)


def fia(torch, torch_npu, case_root, case, pool, plan):
    device = pool[0].device
    pages = case["actual_page_prefix"]
    n, = case["actual_kv_endpoints"]
    controls = plan["fia_calibration_controls"]
    random = torch.Generator().manual_seed(controls["seed"])

    def operand(shape):
        return (torch.randn(shape, generator=random) * 0.15).bfloat16()

    q = operand((1, 8, 256))
    key, value = [operand((len(pages), 128, 512)) for _ in range(2)]
    full_k, full_v = [x.reshape(-1, 2, 256)[:n].transpose(0, 1).float().unsqueeze(0).repeat_interleave(4, 1)
                      for x in (key, value)]
    expected = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1).float().unsqueeze(0),
        full_k, full_v, dropout_p=0.0, scale=plan["fia_parameters"]["scale"])[0].transpose(0, 1).contiguous()
    mask = torch.triu(torch.ones(plan["fia_parameters"]["mask_shape"]), diagonal=1).to(torch.int8).to(device)
    operand_plan = dict(plan["fia_parameters"], kv_endpoints=[n], query_endpoints=[1],
                        block_table_shape=case["live_block_table_shape"])
    operand_case = dict(repetition=case["repetition"], cached_blocks_observed_prefix=pages,
                        page_operand_read_prefix=pages)
    # The pure canonical sampler reproduces the necessary live query. The native
    # graph's dummy query/output rows are excluded from this operator.
    return sample_fia_case(torch, torch_npu, case_root, operand_plan, case["rank"], operand_case,
                           (q.to(device), *pool, key, value, expected, mask))


def rank_sample(root, rank):
    from calibrate_r11_first_target_fia import CONTROLS
    import torch
    import torch_npu

    plan, cases = selection(root)
    require(plan["fia_calibration_controls"] == CONTROLS and sha(WRITER) == WRITER_SHA256
            and {name: version(name) for name in VERSIONS} == VERSIONS
            and os.environ["ASCEND_HOME_PATH"].endswith("/cann-9.1.0"),
            "The first-draft operator runtime, writer or calibrated controls differ")
    torch.set_num_threads(4)
    device = torch.device("npu", rank)
    torch.npu.set_device(device)
    cases = [case for case in cases if case["rank"] == rank]
    require(cases, "The requested rank has no missing operations")
    with torch.inference_mode():
        plane = (7344, 128, 2, 256)
        pool = [torch.empty(plane, dtype=torch.bfloat16, device=device) for _ in range(2)]
        for case in cases:
            folder = root / case["id"].replace(":", "-")
            folder.mkdir()
            if case["operator"] == FIA:
                view = [item.view(case["cache_plane_shape"]) for item in pool]
                numeric = fia(torch, torch_npu, folder, case, view, plan)
            else:
                numeric = scatter(torch, torch_npu, folder, case, pool, plan["scatter_calibration_controls"])
            save(folder / "numeric.json", dict(case=case, demand_sha256=DEMAND,
                sampler_sha256=sha(Path(__file__)), runtime=VERSIONS, writer_sha256=WRITER_SHA256,
                standalone_operator=True, model_inference=False, **numeric))


def guard(root):
    sys.path.insert(0, str(ORIGINAL))
    from candidate_campaign import resource_guard, screen_campaign
    from run_revised_context import prepare

    _, selected = selection(root)
    config, *_ = prepare()
    allocation = root / "allocation"
    allocation.mkdir()
    try:
        with screen_campaign.reserve(config, allocation, resource_guard, expected_pair=[4, 5]) as (quota, fds):
            env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, quota["logical"])),
                       TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1")
            for rank in sorted({row["rank"] for row in selected}):
                subprocess.run([sys.executable, "-B", "-u", __file__, str(root), "rank", str(rank)],
                               cwd=root, env=env, pass_fds=fds, timeout=720, check=True)
    finally:
        # The frozen lease guard records these on normal exit. Preserve the
        # corresponding post-release observation on a failed child as well.
        if (allocation / "quota.json").is_file():
            for name, command in (
                ("npu-after.txt", ["npu-smi", "info"]),
                ("locks-after.txt", ["lslocks", "--noheadings", "--notruncate", "--output", "PATH"]),
            ):
                path = allocation / name
                if not path.exists():
                    with path.open("x", encoding="utf-8") as output:
                        subprocess.run(command, stdout=output, check=True)


def launch(root):
    selection(root)
    require(root.parent == BASE and sha(NAMESPACE) == NAMESPACE_SHA256,
            "The frozen first-draft namespace is unavailable")
    # exec preserves the one-click stage's cancellation and cleanup chain.
    command = ["unshare", "-m", "--propagation", "private", "bash", str(NAMESPACE), str(root),
               str(DONOR), str(PYTHON), "-B", __file__, str(root), "guard"]
    os.execvp(command[0], command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("mode", choices=("launch", "guard", "rank"))
    parser.add_argument("rank", nargs="?", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.mode == "launch":
        launch(args.root.resolve())
    elif args.mode == "guard":
        guard(args.root.resolve())
    else:
        rank_sample(args.root.resolve(), args.rank)

