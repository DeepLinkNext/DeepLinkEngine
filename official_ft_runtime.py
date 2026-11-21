"""Read resolved official configuration outside native timing intervals."""


def gdn_backend():
    from importlib import metadata
    import vllm_ascend.ops.gdn as module
    cls = module.AscendGatedDeltaNetAttention
    if metadata.version('vllm-ascend') == '0.23.0':
        return dict(selected='triton-chunk64', probe_executed_by_observer=False)
    # Read the natural warmup's cached result. Never run the fused smoke
    # kernel or disable an available backend merely to qualify arithmetic.
    fla_branch = hasattr(module, '_get_fla_gdn_prefill_op')
    supported, fla = None, None
    if fla_branch:
        from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
        supported = get_current_hardware_profile().supports(HardwareCapability.FLA_GDN_PREFILL)
        fla = module._get_fla_gdn_prefill_op() if supported else None
    probe = cls._fused_chunk_available
    selected = 'fla-npu' if fla is not None else 'torch-npu-fused-chunk' if probe is True else 'triton-chunk64' if probe is False else None
    result = dict(selected=selected, fla_hardware_supported=supported,
        fla_branch_present=fla_branch,
        fla_operator_available=fla is not None, cached_fused_probe=probe,
        probe_executed_by_observer=False)
    if selected == 'fla-npu':
        from vllm_ascend.device.device_op import DeviceOperator
        from official_ft_fla import observe
        try:
            result['operator_identity'] = observe(fla, DeviceOperator.fla_gdn_prefill)
        except Exception as error:
            result['operator_identity'] = dict(admitted=False, error=type(error).__name__ + ': ' + str(error))
    elif selected == 'torch-npu-fused-chunk' and metadata.version('vllm-ascend').split('+')[0] == '0.28.0rc1':
        from official_ft_cann import observe
        try:
            result['operator_identity'] = observe(cls._chunk_gated_delta_rule_fused)
        except Exception as error:
            result['operator_identity'] = dict(error=type(error).__name__ + ': ' + str(error))
    return result


def configuration(worker):
    config, runner = worker.vllm_config, worker.model_runner
    parallel, scheduler, cache = config.parallel_config, config.scheduler_config, config.cache_config
    speculative = config.speculative_config
    groups = []
    for group in runner.kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        members = getattr(spec, 'kv_cache_specs', None)
        specs = list(members.values()) if members is not None else [spec]
        descriptions = [dict(type=type(s).__module__ + '.' + type(s).__name__,
            block_size=s.block_size, page_size_bytes=s.page_size_bytes,
            mamba_cache_mode=getattr(s, 'mamba_cache_mode', None),
            num_speculative_blocks=getattr(s, 'num_speculative_blocks', None),
            state_dtypes=[str(dtype) for dtype in getattr(s, 'dtypes', ())],
            state_shapes=[list(shape) for shape in getattr(s, 'shapes', ())]) for s in specs]
        groups.append(dict(layers=list(group.layer_names), specs=descriptions,
                           is_eagle_group=group.is_eagle_group))
    result = dict(schema='deeplinkengine-official-loaded-runtime/v1',
        rank=int(worker.rank), tensor_parallel_size=parallel.tensor_parallel_size,
        pipeline_parallel_size=parallel.pipeline_parallel_size,
        data_parallel_size=parallel.data_parallel_size,
        prefill_context_parallel_size=parallel.prefill_context_parallel_size,
        decode_context_parallel_size=parallel.decode_context_parallel_size,
        enable_expert_parallel=parallel.enable_expert_parallel,
        async_scheduling=scheduler.async_scheduling,
        maximum_sequences=scheduler.max_num_seqs, maximum_batched_tokens=scheduler.max_num_batched_tokens,
        maximum_model_length=config.model_config.max_model_len,
        kv_cache_memory_bytes=cache.kv_cache_memory_bytes, prefix_caching=cache.enable_prefix_caching,
        speculative_method=speculative.method if speculative else None,
        speculative_tokens=speculative.num_speculative_tokens if speculative else 0,
        parallel_drafting=speculative.parallel_drafting if speculative else False,
        speculative_tokens_per_batch_size=getattr(speculative, 'num_speculative_tokens_per_batch_size', None),
        constant_draft_positions=getattr(runner.drafter, 'constant_draft_positions', None),
        dynamic_spec_active=getattr(runner.drafter, 'dynamic_spec', None) is not None,
        gdn_backend=gdn_backend(),
        num_blocks=runner.kv_cache_config.num_blocks, groups=groups,
        tensor_values_observed=False, per_step_observer_installed=False)
    from importlib import metadata
    if metadata.version('vllm-ascend').split('+')[0] == '0.28.0rc1':
        from official_ft_factory import capture
        result['moe_factory_dispatch'] = capture(runner)
        from vllm_ascend.ascend_config import get_ascend_config
        from vllm_ascend.spec_decode.llm_base_proposer import lmhead_tp_enable
        result['enable_reduce_sample'] = get_ascend_config().enable_reduce_sample
        result['lmhead_tp_enabled'] = lmhead_tp_enable()
        result['target_xdrope_dimensions'] = runner.drafter.uses_xdrope_dim
        result['draft_xdrope_dimensions'] = runner.drafter.draft_uses_xdrope_dim
    return result
