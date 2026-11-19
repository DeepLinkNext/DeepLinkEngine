"""Exercise deployed norm entry points with independent sampled CPU equations."""
import hashlib
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch
import torch.multiprocessing as mp
import torch_npu
from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, AscendRMSNormGated
from vllm_ascend.utils import enable_custom_op
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'norm-reference-r2'
PLAN = ROOT / 'norm-reference-plan-r2.json'

def worker(rank, barrier):
    torch.npu.set_device(rank)
    torch.set_num_threads(4)
    assert enable_custom_op(), 'Custom norm ops unavailable; fallback forbidden'
    plan = json.loads(PLAN.read_text())
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    torch.manual_seed(plan['seed'] + rank)
    result = {'rank': rank, 'status': 'initializing', 'cases': [], 'plan_sha256': hashlib.sha256(PLAN.read_bytes()).hexdigest(), 'torch_version': torch.__version__, 'torch_npu_version': torch_npu.__version__, 'custom_ops_enabled': True, 'norm_import_path': inspect.getfile(AscendGemmaRMSNorm)}

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
            rows, width, kind = (case['rows'], case['width'], case['kind'])
            device = f'npu:{rank}'
            x = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
            extra = torch.randn_like(x)
            weight = torch.linspace(-0.5, 0.5, width, device=device).bfloat16()
            selected = torch.linspace(0, rows - 1, min(5, rows)).long().unique()
            cx, ce = (x.cpu()[selected].float(), extra.cpu()[selected].float())
            module = SimpleNamespace(weight=weight, variance_epsilon=plan['epsilon'], eps=plan['epsilon'], bias=None, group_size=None, norm_before_gate=True)

            def call():
                if kind == 'gdn_output':
                    return (AscendRMSNormGated.forward_oot(module, x, extra),)
                if kind == 'hidden_residual':
                    return AscendGemmaRMSNorm.forward_oot(module, x, extra)
                return (AscendGemmaRMSNorm.forward_oot(module, x),)
            for _ in range(plan['warmups']):
                outputs = call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = call()
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(x.cpu()[selected].float(), cx)
            assert torch.equal(extra.cpu()[selected].float(), ce)
            cw = weight.cpu().float()
            residual = cx + ce if kind == 'hidden_residual' else None
            values = cx if residual is None else residual
            normalized = values / torch.sqrt(values.square().mean(-1, keepdim=True) + plan['epsilon'])
            if kind == 'gdn_output':
                expected = [normalized * cw * (ce * torch.sigmoid(ce))]
            else:
                affine = (1 + cw).bfloat16().float() if residual is not None else 1 + cw
                expected = [normalized * affine]
                if residual is not None:
                    expected.append(residual)
            checks = []
            for index, (output, oracle) in enumerate(zip(outputs, expected)):
                actual = output.cpu()[selected].float()
                oracle = oracle.bfloat16().float()
                checks.append({'output_index': index, 'rows': selected.tolist(), 'finite': bool(torch.isfinite(actual).all() and torch.isfinite(oracle).all()), 'close': bool(torch.allclose(actual, oracle, atol=0.02, rtol=0.02)), 'atol': 0.02, 'rtol': 0.02, 'max_abs': float((actual - oracle).abs().max()), 'actual_first16': actual.flatten()[:16].tolist(), 'cpu_first16': oracle.flatten()[:16].tolist()})
            assert len(outputs) == len(expected)
            row = {'case': case['case'], 'shape': case, 'checks': checks, 'timings': []}
            result['cases'].append(row)
            save()
            assert all((c['finite'] and c['close'] for c in checks)), str(checks)
            for rep in range(plan['repetitions']):
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
                row['timings'].append({'repetition': rep, 'replays': plan['timed_replays'], 'device_ms': start.elapsed_time(end), 'host_seconds': time.monotonic() - mono, 'start_wall': wall, 'end_wall': time.time()})
                save()
            del graph, outputs, x, extra
    result['status'] = 'qualified_component_timings'
    save()
if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    context = mp.get_context('spawn')
    mp.spawn(worker, args=(context.Barrier(2),), nprocs=2, join=True)
