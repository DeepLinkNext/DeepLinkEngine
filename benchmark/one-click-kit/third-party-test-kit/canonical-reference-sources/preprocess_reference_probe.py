"""Graph-replay actual preprocessing, with independent sampled CPU equations."""

import hashlib
import json
from pathlib import Path
import time

import torch
import torch.multiprocessing as mp
import torch_npu
from betterscale.models.qwen35.preprocess import preprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "preprocess-reference-r1"
PLAN = ROOT / "preprocess-reference-plan-r1.json"


def worker(rank, barrier):
    torch.npu.set_device(rank)
    torch.set_num_threads(4)
    plan = json.loads(PLAN.read_text())
    torch.manual_seed(plan["seed"] + rank)
    result = {"rank": rank, "status": "initializing", "cases": [],
              "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
              "torch_version": torch.__version__, "torch_npu_version": torch_npu.__version__}

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
            s, t = case["source_rows"], case["output_rows"]
            device = f"npu:{rank}"
            x = torch.randn(s, 4096, device=device, dtype=torch.bfloat16)
            a = torch.linspace(-5, 25, s * 16, device=device).reshape(s, 16).bfloat16()
            b = torch.linspace(-8, 8, s * 16, device=device).reshape(s, 16).bfloat16()
            log = torch.linspace(-2, 0, 16, device=device)
            bias = torch.linspace(-0.5, 0.5, 16, device=device)
            mapping = (torch.arange(t, device=device, dtype=torch.int64) * 17) % s if case["mapped"] else None

            def call():
                return preprocess(x, a, b, log, bias, mapping, q_head_major=case["q_head_major"])

            for _ in range(plan["warmups"]):
                outputs = call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = call()
            graph.replay()
            torch.npu.synchronize()
            rows = torch.linspace(0, t - 1, min(5, t)).long().unique()
            sources = rows if mapping is None else mapping.cpu()[rows]
            cx = x.cpu()[sources].float()
            ca, cb = a.cpu()[sources].float(), b.cpu()[sources].float()
            q, k = cx[:, :1024].reshape(-1, 8, 128), cx[:, 1024:2048].reshape(-1, 8, 128)
            gate = ca + bias.cpu()
            expected = [
                (q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)).bfloat16().float(),
                (k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)).bfloat16().float(),
                cx[:, 2048:].reshape(-1, 16, 128),
                -torch.exp(log.cpu()) * torch.where(gate <= 20, torch.log1p(torch.exp(gate)), gate),
                torch.sigmoid(cb).bfloat16().float(),
            ]
            checks = []
            for name, output, oracle in zip(("q", "k", "v", "g", "beta"), outputs, expected):
                actual = output.cpu()[0, rows].float()
                atol, rtol = (2e-5, 2e-5) if name == "g" else (0.002, 0.02)
                flat_a, flat_o = actual.flatten(), oracle.flatten()
                check = {
                    "name": name, "rows": rows.tolist(), "source_rows": sources.tolist(),
                    "finite": bool(torch.isfinite(actual).all() and torch.isfinite(oracle).all()),
                    "close": bool(torch.allclose(actual, oracle, atol=atol, rtol=rtol)),
                    "max_abs": float((actual - oracle).abs().max()), "atol": atol, "rtol": rtol,
                    "actual_first16": flat_a[:16].tolist(), "cpu_first16": flat_o[:16].tolist(),
                }
                checks.append(check)
            row = {"case": case["case"], "shape": case, "checks": checks, "timings": [],
                   "sample_gate_le20": int((gate <= 20).sum()), "sample_gate_gt20": int((gate > 20).sum())}
            result["cases"].append(row)
            save()
            assert all(c["finite"] and c["close"] for c in checks), str(checks)
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
            del graph, outputs, x, a, b, mapping
    result["status"] = "qualified_component_timings"
    save()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=False)
    mp.spawn(worker, args=(mp.get_context("spawn").Barrier(2),), nprocs=2, join=True)
