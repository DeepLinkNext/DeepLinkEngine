"""Compile a source-bounded official kernel family before scored requests.

The official 0.23 GDN kernel specializes NT * H as a constexpr. Closed-loop
prefill/decode mixtures can reach counts absent from a few HTTP cohorts.
Compile all counts permitted by the declared scheduler; launch no kernel and
leave the original Torch/Triton measurement guard unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import time

POLICY = 'official-023-gdn-chunk-compilation-v1'
MANIFEST = 'sha256:4ee78def8f33d59d48f116d1dfa793332c23c99ecab4f0d7dd5cd62d0fb4e6c1'
SOURCES = {
    'ops/triton/fla/chunk_scaled_dot_kkt.py': 'bcd1e9d8d4ec4f44ba24955ac9b85c6ce26e019802e12c49337f11a31bf64a98',
    'device/device_op.py': 'de6a8547df3ff92bf8666e475be56840005cccb3272452c2d6cac4c3f1ddff47',
}


def chunk_bound(tokens, sequences, chunk=64):
    if any(type(v) is not int or v <= 0 for v in (tokens, sequences, chunk)):
        raise ValueError('Positive integer scheduler geometry required')
    # For positive sequence lengths l_i, sum ceil(l_i/C) <=
    # floor((sum l_i + N*(C-1))/C). N is also bounded by the token budget.
    return (tokens + min(tokens, sequences) * (chunk - 1)) // chunk


def compile_family(kernel, torch, *, tokens, sequences, heads, key_heads, cores):
    """Use Triton's public compile-only API with aligned type-only pointers."""
    variants = []
    for chunks in range(1, chunk_bound(tokens, sequences) + 1):
        binary = kernel.warmup(
            torch.bfloat16, torch.bfloat16, torch.float32, torch.float32,
            torch.int32, torch.int32, T=tokens, B=1, H=heads, Hg=key_heads,
            K=128, BT=64, BK=128, IS_VARLEN=True, USE_G=True,
            bh_step=heads, task_num=chunks * heads, num_core=cores,
            num_warps=8, num_stages=3, multibuffer=True, grid=(cores,))
        variants.append(dict(chunks=chunks, task_num=chunks * heads,
                             binary_hash=binary.hash))
    return variants


def prepare(worker):
    from importlib import metadata
    versions = {name: metadata.version(name) for name in ('vllm', 'vllm-ascend', 'triton-ascend')}
    if versions['vllm-ascend'] != '0.23.0':
        return dict(policy=POLICY, applicable=False, versions=versions,
                    reason='This source-bound preparation applies only to the locked official 0.23 image')
    if hasattr(worker, '_deeplink_official_kernel_preparation'):
        return worker._deeplink_official_kernel_preparation
    if hasattr(worker, '_hust_compile_boundary'):
        raise ValueError('First-time kernel preparation must precede the original measurement guard')
    if versions['vllm'] not in ('0.23.0', '0.23.0+empty') or versions['triton-ascend'] != '3.2.2':
        raise ValueError('Unreviewed official GDN compiler/runtime tuple')
    sdk = Path(os.environ['VLLMHUST_OFFICIAL_SDK']).resolve()
    if json.loads((sdk / 'installation.json').read_bytes())['image_manifest'] != MANIFEST:
        raise ValueError('Kernel preparation requires the reviewed immutable image')
    import vllm_ascend
    package = Path(vllm_ascend.__file__).resolve().parent
    if not package.is_relative_to(sdk):
        raise ValueError('Official GDN package loaded outside admitted SDK')
    for name, expected in SOURCES.items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Official GDN implementation changed: ' + name)
    config = worker.vllm_config
    text = config.model_config.hf_text_config
    scheduler, parallel = config.scheduler_config, config.parallel_config
    if (parallel.tensor_parallel_size != 2 or scheduler.max_num_batched_tokens != 4096
            or scheduler.max_num_seqs != 24 or text.linear_key_head_dim != 128
            or text.linear_num_value_heads != 32 or text.linear_num_key_heads != 16):
        raise ValueError('Kernel preparation is outside its reviewed TP2 scheduler geometry')
    import torch
    from triton.runtime.jit import JITFunction
    from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_kernel
    from vllm_ascend.ops.triton.triton_utils import get_aicore_num
    # The pinned outer decorator supplies these two booleans. The inner JIT's
    # warmup method compiles with the same values without executing the kernel.
    kernel = chunk_scaled_dot_kkt_fwd_kernel.fn
    if not isinstance(kernel, JITFunction) or kernel.fn.__name__ != 'chunk_scaled_dot_kkt_fwd_kernel':
        raise ValueError('Unreviewed official GDN kernel wrapper')
    cores = get_aicore_num()
    if cores != 24:
        raise ValueError('Reviewed Ascend 910B2 target required')
    started = time.time()
    variants = compile_family(kernel, torch, tokens=4096, sequences=24, heads=16, key_heads=8, cores=cores)
    result = dict(schema='deeplinkengine-official-kernel-preparation/v1', policy=POLICY,
                  applicable=True, completed=True, rank=int(worker.rank),
                  started=started, ended=time.time(), versions=versions,
                  official_image_manifest=MANIFEST, source_sha256=SOURCES,
                  scheduler=dict(maximum_tokens=4096, maximum_sequences=24, chunk_size=64),
                  maximum_chunks=chunk_bound(4096, 24), variants=variants,
                  compilation_only=True, device_kernels_launched=False,
                  model_tensors_read=False, upstream_sources_modified=False,
                  measurement_guard_relaxed=False)
    path = Path(os.environ['VLLM_HUST_TEST_OUTPUT']) / ('rank-' + str(worker.rank) + '-official-kernel-preparation.json')
    raw = (json.dumps(result, indent=2) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
    receipt = dict(rank=int(worker.rank), policy=POLICY, applicable=True, completed=True,
                   maximum_chunks=result['maximum_chunks'], path=str(path),
                   sha256=hashlib.sha256(raw).hexdigest())
    worker._deeplink_official_kernel_preparation = receipt
    return receipt
