"""Check bounded route recording through real graph replay before model load."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from qualify_gdn_reference import load, sha, write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_bytes())
    for name, digest in plan["files"].items():
        if sha(name) != digest:
            raise ValueError("Frozen route probe source changed: " + name)
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    state = dict(status="admitting", complete=False, started=time.time(), plan_sha256=sha(args.plan),
                 cases=[], observer_only=True, full_model_certificate=False, efficiency_percent=None)
    guard = load("route_probe_guard", plan["resource_guard"])
    os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = ','.join(map(str, plan["physical_cards"]))
    leases = []
    try:
        leases = guard.acquire(out, plan["resource_wait_seconds"])
        write(out / "mapping.json", guard.verify_mapping(plan["logical_devices"]))
        import torch
        import torch_npu  # noqa: F401
        from vllm.platforms import current_platform  # noqa: F401
        import vllm_ascend.ops  # noqa: F401
        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
        sys.path.insert(0, str(Path(__file__).resolve().parent / "proc_capture"))
        from m01_route_device import record

        for device, card in zip(map(int, plan["logical_devices"].split(',')), plan["physical_cards"], strict=True):
            torch.npu.set_device(device)
            init_device_properties_triton()
            # torch_npu.graph caches one class-wide default capture stream.
            # A single-process two-device qualification must own one per card.
            capture_stream = torch.npu.Stream(device=device)
            for case in plan["route_cases"]:
                rows, dtype = case["rows"], getattr(torch, case["dtype"])
                expected = (torch.arange(rows * 8, dtype=dtype).reshape(rows, 8) + plan["seed"]) % 256
                ids = expected.to(f"npu:{device}")
                enabled = torch.zeros(1, dtype=torch.int32, device=ids.device)
                counter = torch.zeros_like(enabled)
                lengths = torch.zeros(4, dtype=torch.int32, device=ids.device)
                records = torch.full((4, 4096, 8), -1, dtype=torch.int32, device=ids.device)
                record(ids, enabled, counter, lengths, records)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=capture_stream):
                    record(ids, enabled, counter, lengths, records)
                graph.replay()
                torch.npu.synchronize()
                assert counter.item() == 0 and torch.all(records == -1).item()
                enabled.fill_(1)
                torch.npu.synchronize()
                for _ in range(4): graph.replay()
                torch.npu.synchronize()
                assert counter.item() == 4 and lengths.cpu().tolist() == [rows] * 4
                actual = records.cpu()
                assert all(torch.equal(actual[i, :rows], expected.int()) for i in range(4))
                assert torch.all(actual[:, rows:] == -1).item()
                graph.replay()  # Overflow increments the sentinel, never writes past capacity.
                torch.npu.synchronize()
                assert counter.item() == 5 and torch.equal(records.cpu(), actual)
                assert torch.equal(ids.cpu(), expected)
                enabled.zero_()
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                assert counter.item() == 5
                row = dict(case=case, physical_card=card, logical_device=device, passed=True,
                           graph_replays=7, disabled_untouched=True, overflow_bounded=True, input_unchanged=True)
                state["cases"].append(row)
                write(out / f"card-{card}-case-{len(state['cases']):02d}.json", row)
                del graph, records, counter, lengths, enabled, ids
            del capture_stream
        state.update(status="completed", complete=True)
    except BaseException as error:
        state.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        guard.release(leases)
        state["ended"] = time.time()
        write(out / "receipt.json", state)
    print(json.dumps(dict(complete=state["complete"], cases=len(state["cases"]), observer_only=True)), flush=True)


if __name__ == "__main__":
    main()
