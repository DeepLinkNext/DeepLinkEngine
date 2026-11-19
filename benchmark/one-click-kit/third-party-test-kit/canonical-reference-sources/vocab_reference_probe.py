"""Two independent local BF16 matmul partitions with CPU sampled-output checks."""
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "vocab-reference-r1"
PLAN = ROOT / "vocab-reference-plan-r1.json"


def worker(rank, barrier):
    torch.npu.set_device(rank)
    torch.set_num_threads(4)
    plan = json.loads(PLAN.read_text())
    torch.manual_seed(plan["seed"] + rank)
    result = {
        "rank": rank, "status": "initializing", "cases": [],
        "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "torch_version": torch.__version__, "torch_npu_version": torch_npu.__version__,
    }

    def save():
        (OUT / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    (OUT / f"ready-{rank}.json").write_text(json.dumps({"rank": rank, "wall": time.time()}))
    deadline = time.monotonic() + 180
    while not (OUT / "go").exists():
        assert time.monotonic() < deadline
        time.sleep(.1)
    with torch.inference_mode():
        weight = torch.randn(124160, 2048, dtype=torch.bfloat16, device=f"npu:{rank}") * .02
        columns = torch.linspace(0, 124159, 8).long()
        cpu_weight = weight[columns.to(weight.device)].float().cpu()
        for case in plan["cases"]:
            x = torch.randn(case["m"], 2048, dtype=torch.bfloat16, device=weight.device) * .02
            output = torch.empty(case["m"], 124160, dtype=torch.bfloat16, device=weight.device)

            def call():
                torch.matmul(x, weight.t(), out=output)

            for _ in range(plan["warmups"]):
                call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                call()
            graph.replay()
            torch.npu.synchronize()
            rows = torch.linspace(0, case["m"] - 1, min(8, case["m"])).long().unique()
            expected = x[rows.to(x.device)].float().cpu() @ cpu_weight.t()
            actual = output[rows.to(x.device)][:, columns.to(x.device)].float().cpu()
            check = {
                "rows": rows.tolist(), "columns": columns.tolist(),
                "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
                "close": bool(torch.allclose(actual, expected, atol=plan["atol"], rtol=plan["rtol"])),
                "max_abs": float((actual - expected).abs().max()),
                "actual": actual.tolist(), "cpu_fp32": expected.tolist(),
            }
            row = {"case": case["case"], "shape": case, "check": check, "timings": []}
            result["cases"].append(row)
            save()
            assert check["finite"] and check["close"], check
            for repetition in range(plan["repetitions"]):
                torch.npu.synchronize()
                barrier.wait(timeout=180)
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                wall, mono = time.time(), time.monotonic()
                start.record()
                for _ in range(plan["timed_replays"]):
                    graph.replay()
                end.record()
                end.synchronize()
                row["timings"].append({
                    "repetition": repetition, "replays": plan["timed_replays"],
                    "device_ms": start.elapsed_time(end), "host_seconds": time.monotonic() - mono,
                    "start_wall": wall, "end_wall": time.time(),
                })
                save()
            del graph
    result["status"] = "qualified_component_timings"
    save()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=False)
    mp.spawn(worker, args=(mp.get_context("spawn").Barrier(2),), nprocs=2, join=True)
