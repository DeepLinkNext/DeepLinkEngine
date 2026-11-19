"""Bounded replay-visible route recording, adapted from archived MoE capture.

Benchmark 18257625f75799577b81d8e2e71e1eae7904985b, r18 configuration
archive, token-observer-v16/hust_token_observer/moe_deferred_capture.py.
Record per-token expert IDs so necessary subsets can be reconstructed later.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _record(IDS, ENABLED, COUNTER, LENGTHS, RECORDS, ROWS,
            TOPK: tl.constexpr, MAX_ROWS: tl.constexpr, CAPACITY: tl.constexpr,
            BLOCK: tl.constexpr):
    active = tl.load(ENABLED)
    if active != 0:
        slot = tl.atomic_add(COUNTER, 1)
        if (slot >= 0) & (slot < CAPACITY):
            tl.store(LENGTHS + slot, ROWS)
            for block in range(tl.cdiv(ROWS * TOPK, BLOCK)):
                index = block * BLOCK + tl.arange(0, BLOCK)
                values = tl.load(IDS + index, index < ROWS * TOPK, other=-1)
                tl.store(RECORDS + slot * MAX_ROWS * TOPK + index, values,
                         index < ROWS * TOPK)


@torch.library.custom_op("hust_m01_routes::record", mutates_args=("counter", "lengths", "records"))
def record(ids: torch.Tensor, enabled: torch.Tensor, counter: torch.Tensor,
           lengths: torch.Tensor, records: torch.Tensor) -> None:
    if (ids.ndim != 2 or ids.shape[1] != 8 or ids.shape[0] > records.shape[1]
            or not ids.is_contiguous() or ids.dtype not in (torch.int32, torch.int64)
            or records.ndim != 3 or records.shape[2] != 8 or records.dtype != torch.int32
            or not records.is_contiguous() or counter.numel() != 1 or enabled.numel() != 1
            or counter.dtype != torch.int32 or enabled.dtype != torch.int32
            or lengths.dtype != torch.int32 or lengths.numel() != records.shape[0]):
        raise ValueError("Unsupported bounded route-recording geometry")
    _record[(1,)](ids, enabled, counter, lengths, records, ids.shape[0], ids.shape[1],
                  records.shape[1], records.shape[0], 256, num_warps=1)


@record.register_fake
def _fake(ids, enabled, counter, lengths, records):
    return None
