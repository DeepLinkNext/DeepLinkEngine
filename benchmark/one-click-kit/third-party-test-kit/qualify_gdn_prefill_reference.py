"""Validate the deployed segmented GDN prefill against independent recurrence."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from uuid import uuid4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_bytes())
    for name, digest in plan["files"].items():
        if sha(name) != digest:
            raise ValueError("Prefill source changed: " + name)
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    result = dict(complete=False, cases=[], plan_sha256=sha(args.plan),
                  full_model_certificate=False, started=time.time())
    guard = load("prefill_resource_guard", plan["resource_guard"])
    os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = ",".join(map(str, plan["physical_cards"]))
    leases = []
    try:
        leases = guard.acquire(out, plan.get("resource_wait_seconds", 0))
        for name, digest in plan["files"].items():
            if sha(name) != digest:
                raise ValueError("Prefill source changed while waiting for the device: " + name)
        write(out / "mapping.json", guard.verify_mapping(plan["logical_devices"]))
        os.environ["BETTERSCALE_GDN_HOST_LIBRARY"] = plan["gdn_host"]
        import torch
        import torch_npu  # noqa: F401
        # Resolve the platform before directly importing DeviceOperator through
        # the target wrapper. Its plugin imports MoE patches that also import
        # DeviceOperator; entering through that module first creates a cycle.
        from vllm.platforms import current_platform

        if current_platform.device_type != "npu":
            raise ValueError("The selected prefill runtime is not Ascend")
        import vllm_ascend.ops  # noqa: F401 -- enter operator registration before DeviceOperator

        torch.npu.matmul.allow_hf32 = False

        ref = load("independent_prefill", plan["prefill_reference"])
        scalar = load("independent_scalar_decode", plan["reference"])
        target = load("betterscale.models.qwen35.chunk_layout", plan["prefill_target"])
        runtime = load("prefill_native_runtime", plan["prefill_runtime"])
        for device, card in zip(map(int, plan["logical_devices"].split(",")), plan["physical_cards"], strict=True):
            torch.npu.set_device(device)
            from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

            init_device_properties_triton()
            for case in plan["prefill_cases"]:
                lengths = [sum(parts) for parts in case["segments"]]
                rng = torch.Generator().manual_seed(plan["seed"])
                def random(shape, factor=.1):
                    return (torch.randn(shape, generator=rng) * factor).to(f"npu:{device}")
                inputs = []
                for n in lengths:
                    q, k = [random((n, 8, 128)).to(torch.bfloat16) for _ in range(2)]
                    v = random((n, 16, 128)).to(torch.bfloat16)
                    magnitude = {"normal": .02, "strong": 1., "near_zero": 1e-5}[case["decay"]]
                    g = -(random((n, 16)).abs() + .01) * magnitude
                    beta = (random((n, 16)).sigmoid()).to(torch.bfloat16)
                    inputs.append((q, k, v, g, beta))
                bank = random((len(lengths) + 2, 16, 128, 128), .02).float()
                original = bank.clone()
                expected_bank = bank.clone()
                mathematical_bank = bank.clone()
                offsets = [0] * len(lengths)
                outputs = [[] for _ in lengths]
                output_error = state_error = 0.
                catalogs = []
                for turn in range(max(map(len, case["segments"]))):
                    packed = [[] for _ in range(5)]
                    cu, expected_outputs, state_meta = [0], [], []
                    for row, parts in enumerate(case["segments"]):
                        size = parts[turn] if turn < len(parts) else 0
                        cu.append(cu[-1] + size)
                        present = case["initial"][row] or turn > 0
                        state_meta.append([row, int(present)] if size else [-1, 0])
                        if not size:
                            continue
                        start, stop = offsets[row], offsets[row] + size
                        values = tuple(x[start:stop] for x in inputs[row])
                        for dest, x in zip(packed, values):
                            dest.append(x)
                        expected, state = ref.segmented(*values, expected_bank[row], 128**-.5,
                                                        [size], initial_state=present,
                                                        storage_dtype=torch.bfloat16)
                        expected_bank[row] = state
                        _, mathematical_bank[row] = ref.segmented(*values, mathematical_bank[row],
                            128**-.5, [size], initial_state=present)
                        expected_outputs.append(expected)
                        catalogs.append(dict(request=row, turn=turn,
                                             **ref.segment_arithmetic(size, present)))
                        offsets[row] = stop
                    # An empty suffix is real capacity padding: it must not
                    # create triangular work or overwrite any pool slot.
                    cu.append(cu[-1])
                    state_meta.append([-1, 0])
                    meta = SimpleNamespace(cu=torch.tensor(cu, dtype=torch.int64, device=bank.device))
                    meta.indices = {block: torch.tensor(
                        [[row, chunk] for row in range(len(cu) - 1)
                         for chunk in range((cu[row + 1] - cu[row] + block - 1) // block)],
                        dtype=torch.int64, device=bank.device).reshape(-1, 2)
                        for block in (64, 256, 1216)}
                    q, k, v, g, beta = [torch.cat(xs).unsqueeze(0) for xs in packed]
                    w, u, gh = target.chunk_wy(k, v, beta, g, meta)
                    kernels = runtime.Kernels(plan["gdn_kernel"], q.shape[1], len(cu) - 1,
                                              len(meta.indices[64]), state_pool=True,
                                              key_heads=8, value_heads=16)
                    value = kernels.pool_forward(q.transpose(1, 2).contiguous(),
                        k.transpose(1, 2).contiguous(), w.transpose(1, 2).contiguous(),
                        u.transpose(1, 2).contiguous(), gh, bank, meta.cu,
                        torch.tensor(state_meta, dtype=torch.int64, device=bank.device),
                        meta.indices[64]).transpose(1, 2).squeeze(0)
                    torch.npu.synchronize(device)
                    expected = torch.cat(expected_outputs).to(torch.bfloat16)
                    torch.testing.assert_close(value, expected, **plan["prefill_output_tolerance"])
                    torch.testing.assert_close(bank, expected_bank, **plan["prefill_state_tolerance"])
                    torch.testing.assert_close(bank[-2:], original[-2:], atol=0, rtol=0)
                    output_error = max(output_error, float((value.float() - expected.float()).abs().max().cpu()))
                    state_error = max(state_error, float((bank - expected_bank).abs().max().cpu()))
                    for row in range(len(lengths)):
                        if cu[row + 1] > cu[row]:
                            outputs[row].append(value[cu[row]:cu[row + 1]])
                # Check the independent chunk solution against the pure Python
                # single-head recurrence on the genuine BF16 input population.
                scalar_error = 0.
                for row, values in enumerate(inputs):
                    q, k, v, g, beta = [x.cpu().float() for x in values]
                    state = original[row, 0].cpu().tolist() if case["initial"][row] else [[0.] * 128 for _ in range(128)]
                    for token in range(lengths[row]):
                        _, state = scalar.scalar_step(q[token, 0].tolist(), k[token, 0].tolist(),
                            v[token, 0].tolist(), float(g[token, 0]), float(beta[token, 0]), state, 128**-.5)
                    expected = mathematical_bank[row, 0].cpu()
                    torch.testing.assert_close(expected, torch.tensor(state), atol=2e-6, rtol=2e-4)
                    scalar_error = max(scalar_error, float((expected - torch.tensor(state)).abs().max()))
                samples = []
                def reference_case():
                    return [ref.segmented(*values, original[row], 128**-.5, case["segments"][row],
                                           initial_state=case["initial"][row])
                            for row, values in enumerate(inputs)]
                for _ in range(plan["warmup_repetitions"]):
                    reference_case()
                for i in range(plan["calibration_repetitions"] + plan["validation_repetitions"]):
                    torch.npu.synchronize(device)
                    start = time.monotonic_ns()
                    reference_case()
                    torch.npu.synchronize(device)
                    samples.append(dict(measurement_id=uuid4().hex, start_ns=start, end_ns=time.monotonic_ns(),
                        physical_card=card, group="calibration" if i < plan["calibration_repetitions"] else "held_out_validation"))
                result["cases"].append(dict(case=case, physical_card=card, logical_device=device,
                    numerical_passed=True, output_max_abs=output_error, state_max_abs=state_error,
                    comparison_reference="independent chunk algebra with BF16 inverse/W/U/H/Vnew/score storage; FP32 state bank",
                    semantic_fp32_state_max_abs=float((bank - mathematical_bank).abs().max().cpu()),
                    scalar_oracle_max_abs=scalar_error, samples=samples, arithmetic=catalogs,
                    full_model_certificate=False))
                write(out / f"card-{card}-case-{len(result['cases']):02d}.json", result["cases"][-1])
                print(json.dumps(dict(card=card, case=case, numerical_passed=True)), flush=True)
        result.update(complete=True, status="completed")
    except BaseException as error:
        result.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        guard.release(leases)
        result["ended"] = time.time()
        write(out / "receipt.json", result)


if __name__ == "__main__":
    main()
