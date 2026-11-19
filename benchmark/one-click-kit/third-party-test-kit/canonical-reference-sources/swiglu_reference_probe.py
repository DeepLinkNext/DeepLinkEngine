"""Measure deployed SwiGLU on both ranks with sampled independent CPU checks."""
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch
import torch.multiprocessing as mp
import torch_npu
from vllm_ascend.ops.activation import AscendSiluAndMul

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "swiglu-reference-r1"
PLAN = ROOT / "swiglu-reference-plan-r1.json"


def worker(rank, barrier):
    torch.npu.set_device(rank)
    torch.set_num_threads(4)
    plan = json.loads(PLAN.read_text())
    for name, digest in plan["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    torch.manual_seed(plan["seed"] + rank)
    result = {"rank": rank, "status": "initializing", "cases": [],
              "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
              "torch_version": torch.__version__, "torch_npu_version": torch_npu.__version__,
              "activation_import_path": inspect.getfile(AscendSiluAndMul)}

    def save():
        (OUT / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    (OUT / f"ready-{rank}.json").write_text(json.dumps({"rank": rank, "wall": time.time()}))
    deadline = time.monotonic() + 180
    while not (OUT / "go").exists():
        assert time.monotonic() < deadline
        time.sleep(0.1)
    with torch.inference_mode():
        for case in plan["cases"]:
            rows, width = case["rows"], case["width"]
            x = torch.randn(rows, width * 2, device=f"npu:{rank}", dtype=torch.bfloat16) * 3
            selected = torch.linspace(0, rows - 1, min(9, rows)).long().unique()
            cpu = x.cpu()[selected].float()
            gate, up = cpu.chunk(2, dim=-1)
            expected = (gate / (1 + torch.exp(-gate)) * up).bfloat16().float()

            def call():
                if case["kind"] == "shared":
                    return AscendSiluAndMul.forward_oot(None, x)
                return torch_npu.npu_swiglu(x)

            for _ in range(plan["warmups"]):
                output = call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = call()
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(x.cpu()[selected].float(), cpu)
            actual = output.cpu()[selected].float()
            checks = {"finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
                      "close": bool(torch.allclose(actual, expected, atol=plan["atol"], rtol=plan["rtol"])),
                      "max_abs": float((actual - expected).abs().max()),
                      "rows": selected.tolist(), "channels": width,
                      "actual_first16": actual.flatten()[:16].tolist(),
                      "cpu_first16": expected.flatten()[:16].tolist()}
            row = {"case": case["case"], "shape": case, "checks": checks, "timings": []}
            result["cases"].append(row)
            save()
            assert checks["finite"] and checks["close"], str(checks)
            for rep in range(plan["repetitions"]):
                for _ in range(plan["graph_warmups"]):
                    graph.replay()
                torch.npu.synchronize()
                barrier.wait(timeout=180)
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                wall, mono = time.time(), time.monotonic()
                start.record()
                for _ in range(plan["timed_replays"]):
                    graph.replay()
                end.record()
                end.synchronize()
                row["timings"].append({"repetition": rep, "replays": plan["timed_replays"],
                                       "device_ms": start.elapsed_time(end),
                                       "host_seconds": time.monotonic() - mono,
                                       "start_wall": wall, "end_wall": time.time()})
                save()
            del graph, output, x
    result["status"] = "qualified_component_timings"
    save()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=False)
    context = mp.get_context("spawn")
    mp.spawn(worker, args=(context.Barrier(2),), nprocs=2, join=True)
