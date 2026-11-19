"""Run both actual fused kernels and compare to independent sampled CPU math."""
import hashlib
import inspect
import json
from pathlib import Path
import time
import torch
import torch.multiprocessing as mp
import torch_npu
from hust_prefill_microflow.runtime.microflow_rope import split_qkv_rmsnorm_mrope_kernel as partitioned_kernel
from vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_mrope import triton_split_qkv_rmsnorm_mrope
ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'rope-reference-plan-r2.json'
OUT = ROOT / 'rope-reference-r2'

def oracle(qkv, qw, kw, cos_sin):
    q_gate = qkv[:, :4096].reshape(-1, 8, 512)
    q, gate = (q_gate[:, :, :256], q_gate[:, :, 256:])
    k = qkv[:, 4096:4352].reshape(-1, 1, 256)
    v = qkv[:, 4352:]
    q = q / torch.sqrt(q.square().mean(-1, keepdim=True) + 1e-06) * qw
    k = k / torch.sqrt(k.square().mean(-1, keepdim=True) + 1e-06) * kw
    index = torch.arange(32)
    axes = torch.where(index % 3 == 1, 1, torch.where((index % 3 == 2) & (index <= 30), 2, 0))
    cos = cos_sin[axes, :, index].T
    sin = cos_sin[axes, :, index + 32].T

    def rotate(x):
        y = x.clone()
        a, b = (x[:, :, :32], x[:, :, 32:64])
        y[:, :, :32] = a * cos[:, None] - b * sin[:, None]
        y[:, :, 32:64] = b * cos[:, None] + a * sin[:, None]
        return y.flatten(1)
    return (rotate(q), rotate(k), v, gate.flatten(1))

def worker(rank, barrier):
    torch.npu.set_device(rank)
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton, get_vectorcore_num
    init_device_properties_triton()
    assert get_vectorcore_num() == 48
    torch.set_num_threads(4)
    plan = json.loads(PLAN.read_text())
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    torch.manual_seed(plan['seed'] + rank)
    result = {'rank': rank, 'status': 'initializing', 'cases': [], 'plan_sha256': hashlib.sha256(PLAN.read_bytes()).hexdigest(), 'torch_version': torch.__version__, 'torch_npu_version': torch_npu.__version__, 'contiguous_import': inspect.getfile(triton_split_qkv_rmsnorm_mrope), 'partitioned_import': partitioned_kernel.fn.__code__.co_filename}

    def save():
        (OUT / f'rank-{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
    save()
    (OUT / f'ready-{rank}.json').write_text(json.dumps({'rank': rank, 'wall': time.time()}))
    deadline = time.monotonic() + 180
    while not (OUT / 'go').exists():
        assert time.monotonic() < deadline
        time.sleep(0.1)
    with torch.inference_mode():
        for case in plan['cases']:
            n, total, offset = (case['rows'], case['cos_tokens'], case['offset'])
            device = f'npu:{rank}'
            qkv = torch.randn(n, 4608, device=device, dtype=torch.bfloat16)
            qw = torch.linspace(0.5, 1.5, 256, device=device).bfloat16()
            kw = torch.linspace(1.5, 0.5, 256, device=device).bfloat16()
            angles = torch.rand(3, total, 32, device=device) * 6
            full_cos = torch.cat((angles.cos(), angles.sin()), dim=-1).bfloat16()
            cos = full_cos[:, offset:offset + n]
            selected = torch.linspace(0, n - 1, min(n, 5)).long().unique()
            cpu_qkv, cpu_cos = (qkv.cpu()[selected].float(), cos.cpu()[:, selected].float())
            expected = oracle(cpu_qkv, qw.cpu().float(), kw.cpu().float(), cpu_cos)
            if case['kind'] == 'partitioned':
                buffers = tuple((torch.empty(n, width, device=device, dtype=torch.bfloat16) for width in (2048, 256, 256, 2048)))

            def call():
                if case['kind'] == 'contiguous':
                    return torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(qkv=qkv, q_weight=qw, k_weight=kw, cos_sin=cos, num_q_heads=8, num_kv_heads=1, head_size=256, eps=1e-06, mrope_section=[11, 11, 10], is_interleaved=True, rope_dim=64, has_gate=True)
                partitioned_kernel[min(n, 48),](qkv, qw, None, kw, None, cos, *buffers, n, total, n % 48 or 48, (n + 47) // 48, n // 48, 8, 1, 256, 2048, 256, 1e-06, 11, 11, 10, False, True, 64, 32, True, 2048)
                return buffers
            for _ in range(plan['warmups']):
                outputs = call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = call()
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(qkv.cpu()[selected].float(), cpu_qkv)
            checks = []
            for name, output, reference in zip(('q', 'k', 'v', 'gate'), outputs, expected, strict=True):
                actual, reference = (output.cpu()[selected].float(), reference.bfloat16().float())
                checks.append({'name': name, 'rows': selected.tolist(), 'atol': 0.002, 'rtol': 0.02, 'finite': bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()), 'close': bool(torch.allclose(actual, reference, atol=0.002, rtol=0.02)), 'max_abs': float((actual - reference).abs().max()), 'actual_first16': actual.flatten()[:16].tolist(), 'cpu_first16': reference.flatten()[:16].tolist()})
            row = {'case': case['case'], 'shape': case, 'cos_stride': list(cos.stride()), 'checks': checks, 'timings': []}
            result['cases'].append(row)
            save()
            assert all((c['finite'] and c['close'] for c in checks)), str(checks)
            assert all((c['max_abs'] == 0 for c in checks if c['name'] in ('v', 'gate')))
            for repetition in range(plan['repetitions']):
                for _ in range(plan['graph_warmups']):
                    graph.replay()
                torch.npu.synchronize()
                barrier.wait(timeout=180)
                start, end = (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
                wall, mono = (time.time(), time.monotonic())
                start.record()
                for _ in range(plan['timed_replays']):
                    graph.replay()
                end.record()
                end.synchronize()
                row['timings'].append({'repetition': repetition, 'replays': plan['timed_replays'], 'device_ms': start.elapsed_time(end), 'host_seconds': time.monotonic() - mono, 'start_wall': wall, 'end_wall': time.time()})
                save()
            del graph, outputs, qkv, full_cos, cos, angles
    result['status'] = 'qualified_component_timings'
    save()
if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    context = mp.get_context('spawn')
    mp.spawn(worker, args=(context.Barrier(2),), nprocs=2, join=True)
