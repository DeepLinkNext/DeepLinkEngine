"""Qualify current 35B H/O state-pool graphs, then measure fixed hot replay cases."""
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import torch.multiprocessing as mp
import torch_npu
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gdn-reference-r3'
PLAN = ROOT / 'gdn-reference-plan-r3.json'

def compare(actual, expected, plan):
    actual, expected = (actual.float().cpu(), expected.float().cpu())
    difference = actual - expected
    return {'shape': list(actual.shape), 'finite': bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()), 'close': bool(torch.allclose(actual, expected, atol=plan['atol'], rtol=plan['rtol'])), 'max_abs': float(difference.abs().max()), 'rms': float(difference.square().mean().sqrt())}

def worker(rank, barrier):
    from vllm_ascend.utils import enable_custom_op
    from betterscale.patches.qwen_gdn.metadata import Metadata, chunk_rows
    assert enable_custom_op()
    torch.npu.set_device(rank)
    torch.set_num_threads(4)
    plan = json.loads(PLAN.read_text())
    torch.manual_seed(plan['seed'] + rank)
    library = Path(os.environ['BETTERSCALE_GDN_LIBRARY'])
    library_sha = hashlib.sha256(library.read_bytes()).hexdigest()
    assert library_sha == '1c522f04a701d3423798b4b7f157737ef41956be25cf46842b6ba33c7ec41bea'
    record = {'rank': rank, 'status': 'initializing', 'cases': [], 'plan_sha256': hashlib.sha256(PLAN.read_bytes()).hexdigest(), 'library_sha256': library_sha, 'host_library_sha256': hashlib.sha256(Path(os.environ['BETTERSCALE_GDN_HOST_LIBRARY']).read_bytes()).hexdigest()}

    def save():
        (OUT / f'rank-{rank}.json').write_text(json.dumps(record, indent=2) + '\n')
    save()
    (OUT / f'ready-{rank}.json').write_text(json.dumps({'rank': rank, 'wall': time.time()}))
    deadline = time.monotonic() + 180
    while not (OUT / 'go').exists():
        assert time.monotonic() < deadline, 'Controller did not admit probe'
        time.sleep(0.1)
    with torch.inference_mode():
        for case in plan['cases']:
            capacity, requests, lengths = (case['capacity'], case['requests'], case['lengths'])
            n, total = (len(lengths), sum(lengths))
            meta = Metadata(capacity, False, f'npu:{rank}', requests=requests, key_heads=8, value_heads=16)
            ends = [0]
            for length in lengths:
                ends.append(ends[-1] + length)
            padded = ends + [total] * (requests + 1 - n)
            meta.cu.copy_(torch.tensor(padded, dtype=torch.int64))
            states = torch.zeros(requests + 1, 2, dtype=torch.int64)
            states[:n, 0] = torch.arange(n)
            states[:n, 1] = torch.tensor(case['initial_flags'], dtype=torch.int64)
            meta.state.copy_(states)
            meta.indices[64].copy_(torch.tensor(chunk_rows(lengths, 64, capacity, requests)))
            bf = {'device': f'npu:{rank}', 'dtype': torch.bfloat16}
            q = (torch.randn(1, 8, capacity, 128, **bf) * 0.05).contiguous()
            k = (torch.randn_like(q) * 0.05).contiguous()
            w = (torch.randn(1, 16, capacity, 128, **bf) * 0.01).contiguous()
            u = (torch.randn_like(w) * 0.1).contiguous()
            g = -torch.rand(1, 16, capacity, device=f'npu:{rank}') * 0.1
            for start, end in zip(ends, ends[1:]):
                for offset in range(start, end, 64):
                    stop = min(offset + 64, end)
                    g[:, :, offset:stop] = g[:, :, offset:stop].cumsum(-1)
            bank = torch.randn(requests + 1, 16, 128, 128, device=f'npu:{rank}') * 0.01
            seed = bank.clone()
            for i, flag in enumerate(case['initial_flags']):
                if not flag:
                    seed[i] = float('nan')
            initial = torch.where(torch.tensor(case['initial_flags'], device=f'npu:{rank}')[:, None, None, None], seed[:n], 0)
            rows = [(i, j) for i, length in enumerate(lengths) for j in range((length + 63) // 64)]
            flat = tuple((value for row in rows for value in row))
            active = [value[:, :, :total].contiguous() for value in (q, k, w, u, g)]
            aq, ak, aw, au, ag = active
            h, v, final = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(ak, aw, au, g=ag, gk=None, initial_state=initial, output_final_state=True, chunk_size=64, save_new_value=True, cu_seqlens=tuple(ends), chunk_indices=flat, use_exp2=False, transpose_state_layout=False)
            expected = torch.ops._C_ascend.chunk_fwd_o(aq, ak, v, h, g=ag, scale=128 ** (-0.5), g_gamma=None, cu_seqlens=tuple(ends), chunk_indices=flat, chunk_size=64, transpose_state_layout=False)
            expected_bank = seed.clone()
            expected_bank[:n] = final

            def call():
                return meta.engine.pool_forward(q, k, w, u, g, bank, meta.cu, meta.state, meta.indices[64])
            for _ in range(plan['warmups']):
                bank.copy_(seed)
                call()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = call()
            bank.copy_(seed)
            graph.replay()
            torch.npu.synchronize()
            checks = {'output': compare(output[:, :, :total], expected, plan), 'state': compare(bank, expected_bank, plan)}
            row = {'case': case['case'], 'shape': case, 'checks': checks, 'timings': []}
            record['cases'].append(row)
            save()
            assert all((value['finite'] and value['close'] for value in checks.values())), checks
            for repetition in range(plan['repetitions']):
                bank.copy_(seed)
                torch.npu.synchronize()
                barrier.wait(timeout=180)
                start, end = (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
                wall = time.time()
                mono = time.monotonic()
                start.record()
                for _ in range(plan['timed_replays']):
                    graph.replay()
                end.record()
                end.synchronize()
                row['timings'].append({'repetition': repetition, 'replays': plan['timed_replays'], 'start_wall': wall, 'end_wall': time.time(), 'host_seconds': time.monotonic() - mono, 'device_ms': start.elapsed_time(end)})
                save()
            del graph
    record['status'] = 'qualified_component_timings'
    save()
if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    barrier = mp.get_context('spawn').Barrier(2)
    mp.spawn(worker, args=(barrier,), nprocs=2, join=True)
