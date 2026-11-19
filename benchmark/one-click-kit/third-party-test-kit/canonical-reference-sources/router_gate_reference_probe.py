"""Exercise the deployed fused router and full-width FA gate under NPU graphs."""
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch
import torch.multiprocessing as mp
import torch_npu
import vllm_ascend.ops
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.router.router_factory import create_ascend_fused_moe_router
from vllm_ascend.utils import enable_custom_op

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "router-gate-reference-r1"
PLAN = ROOT / "router-gate-reference-plan-r1.json"


def worker(rank, barrier):
    torch.npu.set_device(rank)
    torch.set_num_threads(4)
    assert enable_custom_op(), "Custom router unavailable; fallback forbidden"
    plan = json.loads(PLAN.read_text())
    for name, digest in plan["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    torch.manual_seed(plan["seed"] + rank)
    router = create_ascend_fused_moe_router(top_k=8, global_num_experts=256, renormalize=True)
    assert type(router).__name__ == "AscendFusedTopKRouter"
    calls = []
    original = DeviceOperator.moe_gating_top_k

    def recording_call(x, **kwargs):
        calls.append({"shape": list(x.shape), "dtype": str(x.dtype),
                      **{k: v for k, v in kwargs.items() if k != "bias_opt"},
                      "bias_is_none": kwargs.get("bias_opt") is None})
        return original(x, **kwargs)

    DeviceOperator.moe_gating_top_k = staticmethod(recording_call)
    result = {"rank": rank, "status": "initializing", "cases": [],
              "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
              "torch_version": torch.__version__, "torch_npu_version": torch_npu.__version__,
              "router_type": type(router).__name__, "router_import": inspect.getfile(type(router))}

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
            n, kind, dev = case["rows"], case["kind"], f"npu:{rank}"
            selected = torch.linspace(0, n - 1, min(9, n)).long().unique()
            offset = len(calls)
            hidden = torch.randn(n, 2048, device=dev, dtype=torch.bfloat16)
            if kind == "router":
                x = (torch.rand(n, 256, device=dev).argsort(-1).float() - 127.5) / 16
                x = x + torch.linspace(-32, 32, n, device=dev)[:, None]
                cx = x.cpu()[selected].double()
                cpu_scores = torch.exp(cx - cx.max(-1, keepdim=True).values)
                cpu_scores = cpu_scores / cpu_scores.sum(-1, keepdim=True)
                cpu_weights, cpu_ids = torch.topk(cpu_scores, 8, dim=-1)
                expected = (cpu_weights / (cpu_weights.sum(-1, keepdim=True) + 1e-20)).float()
                assert router.is_fused_supported(hidden)

                def call():
                    return router._compute_routing(hidden, x, torch.int32)
            else:
                x = torch.randn_like(hidden) * 3
                cx = x.cpu()[selected].float()
                ch = hidden.cpu()[selected].float()
                expected = (torch.sigmoid(cx).bfloat16().float() * ch).bfloat16().float()

                def call():
                    return (hidden * torch.sigmoid(x),)
            for _ in range(plan["warmups"]):
                outputs = call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = call()
            graph.replay()
            torch.npu.synchronize()
            actual = outputs[0].cpu()[selected].float()
            tolerance = plan[kind]
            checks = {"finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
                      "close": bool(torch.allclose(actual, expected, atol=tolerance["atol"], rtol=tolerance["rtol"])),
                      "max_abs": float((actual - expected).abs().max()),
                      "rows": selected.tolist(), "channels": actual.shape[1],
                      "actual_first16": actual.flatten()[:16].tolist(),
                      "cpu_first16": expected.flatten()[:16].tolist()}
            if kind == "router":
                checks["ids_exact"] = torch.equal(outputs[1].cpu()[selected].long(), cpu_ids)
                checks["max_weight_sum_error"] = float((actual.sum(-1) - 1).abs().max())
                assert len(calls) - offset == plan["warmups"] + 1
            else:
                assert len(calls) == offset
            row = {"case": case["case"], "shape": case, "checks": checks,
                   "router_calls": calls[offset:], "timings": []}
            result["cases"].append(row)
            save()
            assert checks["finite"] and checks["close"] and checks.get("ids_exact", True), str(checks)
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
            del graph, outputs, x, hidden
    result["status"] = "qualified_component_timings"
    save()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=False)
    context = mp.get_context("spawn")
    mp.spawn(worker, args=(context.Barrier(2),), nprocs=2, join=True)
