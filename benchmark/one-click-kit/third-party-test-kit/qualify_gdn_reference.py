"""Lease both cards and validate/calibrate an independent GDN operator reference."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4



def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_bytes())
    for filename, expected in plan["files"].items():
        if sha(filename) != expected:
            raise ValueError("Frozen probe source changed: " + filename)
    output = args.out.resolve()
    output.mkdir(exist_ok=False)
    state = dict(status="admitting", complete=False, started=time.time(), plan_sha256=sha(args.plan), cases=[], full_model_certificate=False, efficiency_percent=None)
    guard = load("gdn_probe_guard", plan["resource_guard"])
    os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = ",".join(map(str, plan["physical_cards"]))
    leases = []
    try:
        leases = guard.acquire(output, 0)
        write(output / "mapping.json", guard.verify_mapping(plan["logical_devices"]))
        # Import the Ascend runtime only after exclusive device/model leases.
        import torch
        import torch_npu  # noqa: F401
        ref = load("independent_gdn_reference", plan["reference"])
        target = load("frozen_r8_gdn_target", plan["target"])
        state["status"] = "running"
        for device, card in zip(map(int, plan["logical_devices"].split(",")), plan["physical_cards"], strict=True):
            torch.npu.set_device(device)
            for case in plan["cases"]:
                n, width = case["batch"], case["width"]
                lengths = [width - (i % width) if case["ragged"] else width for i in range(n)]
                cumulative = [0]
                for length in lengths:
                    cumulative.append(cumulative[-1] + length)
                tokens = cumulative[-1]
                rng = torch.Generator(device="cpu").manual_seed(plan["seed"])
                def random(shape, factor=0.1):
                    return (torch.randn(shape, generator=rng) * factor).to(f"npu:{device}")
                q = random((1, tokens, 8, 128)).to(torch.bfloat16)
                k = random((1, tokens, 8, 128)).to(torch.bfloat16)
                v = random((1, tokens, 16, 128)).to(torch.bfloat16)
                g = (-random((1, tokens, 16)).abs() - 0.1).float()
                beta = random((1, tokens, 16)).sigmoid().to(torch.bfloat16)
                pool = random((n * width, 16, 128, 128), 0.02).float()
                slots_cpu = torch.arange(n * width, dtype=torch.int32).reshape(n, width)
                slots = slots_cpu.to(f"npu:{device}")
                accepted_cpu = torch.tensor([1 + i % width for i in range(n)], dtype=torch.int32)
                accepted = accepted_cpu.to(f"npu:{device}")
                cu = torch.tensor(cumulative, dtype=torch.int32, device=f"npu:{device}")
                expected_pool = pool.clone()
                expected_output = torch.empty((1, tokens, 16, 128), dtype=torch.float32, device=f"npu:{device}")
                scale = 128**-0.5
                for row, length in enumerate(lengths):
                    current = pool[int(slots_cpu[row, int(accepted_cpu[row]) - 1])].clone()
                    for token in range(length):
                        offset = cumulative[row] + token
                        value, current = ref.tensor_step(q[0, offset], k[0, offset], v[0, offset], g[0, offset], beta[0, offset], current, scale)
                        expected_output[0, offset] = value
                        expected_pool[int(slots_cpu[row, token])] = current
                actual_pool = pool.clone()
                kwargs = dict(cu_seqlens=cu, ssm_state_indices=slots if width > 1 else slots[:, 0].contiguous(), num_accepted_tokens=accepted if width > 1 else None)
                value, result_pool = target.fused_recurrent_gated_delta_rule_fwd(q, k, v, g, beta, scale, actual_pool, **kwargs)
                torch.npu.synchronize(device)
                torch.testing.assert_close(value, expected_output.to(torch.bfloat16), **plan["output_tolerance"])
                torch.testing.assert_close(result_pool, expected_pool, **plan["state_tolerance"])
                if result_pool.data_ptr() != actual_pool.data_ptr():
                    raise ValueError("Target stopped updating the owned pool in place")
                # Also check the tensor reference against an independent Python
                # oracle on one real BF16-input head, outside calibration.
                offset = 0
                oracle, oracle_state = ref.scalar_step(q[0, offset, 0].cpu().float().tolist(), k[0, offset, 0].cpu().float().tolist(), v[0, offset, 0].cpu().float().tolist(), float(g[0, offset, 0].cpu()), float(beta[0, offset, 0].cpu()), pool[int(slots_cpu[0, int(accepted_cpu[0]) - 1]), 0].cpu().tolist(), scale)
                torch.testing.assert_close(expected_output[0, 0, 0].cpu(), torch.tensor(oracle), atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(expected_pool[int(slots_cpu[0, 0]), 0].cpu(), torch.tensor(oracle_state), atol=2e-6, rtol=2e-5)
                row = dict(case=case, physical_card=card, logical_device=device, tokens=tokens, output_max_abs=float((value.float() - expected_output.to(torch.bfloat16).float()).abs().max().cpu()), state_max_abs=float((result_pool - expected_pool).abs().max().cpu()), numerical_passed=True, calibrated=False)
                # Calibrate a fixed-shape batch transition separately. Inputs,
                # output allocation and state reset do not use target timings.
                q1, k1, v1, g1, b1 = q[:, :1].squeeze(0), k[:, :1].squeeze(0), v[:, :1].squeeze(0), g[:, :1].squeeze(0), beta[:, :1].squeeze(0)
                s1 = pool[:1]
                for _ in range(plan["warmup_repetitions"]):
                    ref.tensor_step(q1, k1, v1, g1, b1, s1, scale)
                torch.npu.synchronize(device)
                samples = []
                for index in range(plan["calibration_repetitions"] + plan["validation_repetitions"]):
                    torch.npu.synchronize(device)
                    start = time.monotonic_ns()
                    ref.tensor_step(q1, k1, v1, g1, b1, s1, scale)
                    torch.npu.synchronize(device)
                    end = time.monotonic_ns()
                    samples.append(dict(measurement_id=uuid4().hex, start_ns=start, end_ns=end, physical_card=card, group="calibration" if index < plan["calibration_repetitions"] else "held_out_validation"))
                row.update(calibrated=True, calibration_scope="one batch row, one transition, 8 key/16 value heads, K=V=128; host monotonic and device fence", basic_operations=ref.operation_counts(128, 128)["total_basic"] * 16, samples=samples)
                state["cases"].append(row)
                write(output / f"card-{card}-case-{len(state['cases']):02d}.json", row)
        state.update(status="completed", complete=True)
    except BaseException as error:
        state.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        guard.release(leases)
        state["ended"] = time.time()
        write(output / "receipt.json", state)
    print(json.dumps(dict(complete=state["complete"], cases=len(state["cases"]), full_model_certificate=False)), flush=True)


if __name__ == "__main__":
    main()
