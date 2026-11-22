"""Opt-in CoreProc support for an independent, untimed official companion."""
import hashlib
import inspect
import os
from pathlib import Path


def identity(core):
    if os.environ.get('DEEPLINKENGINE_OFFICIAL_FT_COMPANION') != '1':
        raise ValueError('Official companion is not enabled')
    scheduler = core.scheduler
    config = scheduler.kv_cache_config
    coordinator = scheduler.kv_cache_manager.coordinator
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
    groups = []
    sources = {}
    for item in (type(core), type(scheduler), type(coordinator),
                 *map(type, coordinator.single_type_managers)):
        path = Path(inspect.getfile(item)).resolve()
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for group in config.kv_cache_groups:
        spec = group.kv_cache_spec
        path = Path(inspect.getfile(type(spec))).resolve()
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        groups.append(dict(type=type(spec).__module__ + '.' + type(spec).__name__,
                           block_size=spec.block_size, page_size_bytes=spec.page_size_bytes,
                           layers=list(group.layer_names), is_eagle_group=group.is_eagle_group,
                           mamba_cache_mode=getattr(spec, 'mamba_cache_mode', None),
                           num_speculative_blocks=getattr(spec, 'num_speculative_blocks', None),
                           host_resident=getattr(group, 'host_resident', None)))
    return dict(schema='deeplinkengine-official-resolved-core/v1', timing_eligible=False,
        num_blocks=config.num_blocks, groups=groups,
        manager_types=[type(m).__module__ + '.' + type(m).__name__ for m in coordinator.single_type_managers],
        in_flight_batches=core.batch_queue_size,
        scheduler=dict(async_scheduling=scheduler.scheduler_config.async_scheduling,
                       max_num_seqs=scheduler.scheduler_config.max_num_seqs,
                       max_num_batched_tokens=scheduler.scheduler_config.max_num_batched_tokens),
        source_sha256=sources, prefix_caching=scheduler.kv_cache_manager.enable_caching,
        capacity_at_max_length=get_max_concurrency_for_kv_cache_config(core.vllm_config, config),
        max_model_len=core.vllm_config.model_config.max_model_len,
        declared_kv_cache_memory_bytes=core.vllm_config.cache_config.kv_cache_memory_bytes)


def install():
    if os.environ.get('DEEPLINKENGINE_OFFICIAL_FT_COMPANION') != '1':
        return
    from m01_proc_plugin import install as capture_install
    capture_install()
    from vllm.v1.engine.core import EngineCoreProc
    previous = getattr(EngineCoreProc, 'hust_ft_core_identity', None)
    if previous is not None and previous is not identity:
        raise ValueError('Refusing to replace another official core identity utility')
    EngineCoreProc.hust_ft_core_identity = identity
