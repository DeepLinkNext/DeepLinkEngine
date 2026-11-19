"""Measure every observed device group distribution with graph-buffer reuse."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.multiprocessing as mp
import torch_npu

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "moe-reference-r2"
PLAN = ROOT / "moe-reference-plan-r2.json"


def check_output(x, weight, output, case, op, plan):
    active = [i for i, n in enumerate(case["group_counts"]) if n]
    chosen = sorted({active[i] for i in torch.linspace(
        0, len(active) - 1, min(8, len(active))).long().tolist()})
    columns = torch.linspace(0, op["n"] - 1, 8).long().to(x.device)
    offsets = [0]
    for n in case["group_counts"]:
        offsets.append(offsets[-1] + n)
    checks = []
    for expert in chosen:
        rows = sorted({offsets[expert], offsets[expert + 1] - 1})
        expected = x[rows].float().cpu() @ weight[expert][:, columns].float().cpu()
        actual = output[rows][:, columns].float().cpu()
        checks.append({
            "expert": expert, "rows": rows, "columns": columns.cpu().tolist(),
            "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
            "close": bool(torch.allclose(actual, expected, atol=plan["atol"], rtol=plan["rtol"])),
            "max_abs": float((actual - expected).abs().max()),
            "actual": actual.tolist(), "cpu_fp32": expected.tolist(),
        })
    return checks


def worker(rank, barrier):
    torch.npu.set_device(rank)
    torch.set_num_threads(4)
    plan = json.loads(PLAN.read_text())
    torch.manual_seed(plan["seed"] + rank)
    buckets = defaultdict(list)
    for case in plan["cases"]:
        buckets[case["rows"]].append(case)
    result = {
        "rank": rank, "status": "initializing", "buckets": [],
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
    with torch.inference_mode(), (OUT / f"timings-{rank}.jsonl").open("x") as timings:
        for op in plan["operators"]:
            weight = torch.randn(256, op["k"], op["n"], dtype=torch.bfloat16,
                                 device=f"npu:{rank}") * .02
            for rows, cases in sorted(buckets.items()):
                x = torch.randn(rows, op["k"], dtype=torch.bfloat16, device=weight.device) * .02
                all_groups = torch.tensor([c["group_counts"] for c in cases],
                                          dtype=torch.int64, device=x.device)
                groups = all_groups[0].clone()

                def call():
                    return torch_npu.npu_grouped_matmul(
                        x=[x], weight=[weight], bias=None, split_item=2,
                        group_list_type=1, group_type=0, group_list=groups)[0]

                for _ in range(plan["warmups"]):
                    output = call()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = call()
                record = {
                    "operator": op, "rows": rows, "cases": len(cases), "checks": [],
                    "weight_stride": list(weight.stride()),
                    "weight_format": int(torch_npu.get_npu_format(weight)),
                    "timed_cases": 0,
                }
                result["buckets"].append(record)
                for index in sorted({0, len(cases) - 1}):
                    groups.copy_(all_groups[index])
                    graph.replay()
                    torch.npu.synchronize()
                    checks = check_output(x, weight, output, cases[index], op, plan)
                    record["checks"].append({"case": cases[index]["case"], "samples": checks})
                    save()
                    assert checks and all(c["finite"] and c["close"] for c in checks)
                for repetition in range(plan["repetitions"]):
                    barrier.wait(timeout=300)
                    for index, case in enumerate(cases):
                        groups.copy_(all_groups[index])
                        start = torch.npu.Event(enable_timing=True)
                        end = torch.npu.Event(enable_timing=True)
                        wall, mono = time.time(), time.monotonic()
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        timing = {
                            "case": case["case"], "operator": op["name"],
                            "repetition": repetition, "replays": 1,
                            "device_ms": start.elapsed_time(end),
                            "host_seconds": time.monotonic() - mono,
                            "start_wall": wall, "end_wall": time.time(),
                        }
                        assert timing["device_ms"] > 0
                        timings.write(json.dumps(timing, separators=(",", ":")) + "\n")
                        record["timed_cases"] += 1
                    timings.flush()
                    save()
                del graph, output, x, groups, all_groups
            del weight
    result["status"] = "qualified_component_timings"
    save()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=False)
    mp.spawn(worker, args=(mp.get_context("spawn").Barrier(2),), nprocs=2, join=True)
