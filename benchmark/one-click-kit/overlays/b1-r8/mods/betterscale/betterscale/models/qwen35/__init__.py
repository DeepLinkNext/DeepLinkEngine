"""Admission for the owned port of the qualified Qwen35 baseline.

Model forward, MTP and the async EngineCore remain the baseline implementation.
State ownership attaches below those owners, not through another serving loop.
"""

CAPTURE_SIZES = (
    3,
    6,
    12,
    16,
    24,
    32,
    40,
    48,
    64,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    4096,
)
UNIFIED_CAPTURE_SIZES = (3, 6, 12, 24, 48)


def capture_sizes(requests=16, mtp_tokens=2):
    from .count_policy import spec_capacities

    return tuple(
        sorted(
            set(CAPTURE_SIZES)
            | set(spec_capacities(requests, width=mtp_tokens + 1))
        )
    )


SCHEDULER = "betterscale.models.qwen35.apc_boundary.BoundaryScheduler"
STATE_SCHEDULER = "betterscale.models.qwen35.seat_scheduler.LiveStateScheduler"


def mtp_capture_sizes(mtp_tokens=2, *, unified=False):
    """Keep decode capacity keys disjoint from mixed-prefill keys."""
    if type(mtp_tokens) is not int or mtp_tokens not in (1, 2, 3, 4):
        raise ValueError("Qwen35 supports MTP1 through MTP4")
    if unified:
        if mtp_tokens != 2:
            raise ValueError("Unified donor capture policy currently requires MTP2")
        return UNIFIED_CAPTURE_SIZES
    return tuple(sorted(set(CAPTURE_SIZES) | ({80} if mtp_tokens >= 3 else set())))


def using_live_state(config):
    return (
        getattr(config, "additional_config", {}).get("using_live_runtime", False)
        is True
    )


def validate(config):
    import os

    p, s, m = config.parallel_config, config.scheduler_config, config.model_config
    hf = m.hf_text_config
    spec = config.speculative_config
    unified = os.environ.get("BETTERSCALE_QWEN35_CONTRACT") == "unified"
    if using_live_state(config):
        from .capacity import seat_counts

        seat_counts(config)
    expected_captures = (
        mtp_capture_sizes(spec.num_speculative_tokens, unified=True)
        if unified
        else capture_sizes(s.max_num_seqs, spec.num_speculative_tokens)
        if using_live_state(config)
        else mtp_capture_sizes(spec.num_speculative_tokens)
    )
    checks = {
        "35B-A3B BF16 text geometry": (
            hf.model_type == "qwen3_5_moe_text"
            and (
                hf.num_hidden_layers,
                hf.hidden_size,
                hf.head_dim,
                hf.num_attention_heads,
                hf.num_key_value_heads,
            )
            == (40, 2048, 256, 16, 2)
            and (
                hf.linear_num_key_heads,
                hf.linear_num_value_heads,
                hf.linear_key_head_dim,
                hf.linear_value_head_dim,
                hf.linear_conv_kernel_dim,
            )
            == (16, 32, 128, 128, 4)
            and str(m.dtype) == "torch.bfloat16"
            and m.quantization is None
            and config.load_config.load_format == "auto"
        ),
        "TP2/DP1/PP1 without EP or context parallelism": (
            p.tensor_parallel_size,
            p.data_parallel_size,
            p.pipeline_parallel_size,
            p.enable_expert_parallel,
            p.decode_context_parallel_size,
            p.prefill_context_parallel_size,
        )
        == (2, 1, 1, False, 1, 1),
        "execution<=36/query4096/context<=262144": (
            (
                1 <= s.max_num_seqs <= 36
                if using_live_state(config)
                else s.max_num_seqs == 16
            )
            and s.max_num_batched_tokens == 4096
            and 0 < m.max_model_len <= 262144
        ),
        "baseline asynchronous boundary scheduler": (
            s.async_scheduling
            and s.scheduler_cls
            == (STATE_SCHEDULER if using_live_state(config) else SCHEDULER)
        ),
        "native MTP1–4": spec is not None
        and spec.method == "mtp"
        and type(spec.num_speculative_tokens) is int
        and spec.num_speculative_tokens in (1, 2, 3, 4)
        and not spec.enforce_eager,
        "qualified graph contract": (
            str(config.compilation_config.cudagraph_mode)
            == (
                "FULL_AND_PIECEWISE"
                if unified
                else "FULL"
            )
            and set(config.compilation_config.cudagraph_capture_sizes)
            == set(expected_captures)
        ),
        "APC align without connectors or LoRA": (
            config.cache_config.enable_prefix_caching
            and config.cache_config.mamba_cache_mode == "align"
            and config.kv_transfer_config is None
            and config.lora_config is None
        ),
        "text or Qwen35 State image input (no video)": m.multimodal_config is None
        or (
            m.multimodal_config.get_limit_per_prompt("video") == 0
            and (
                m.multimodal_config.get_limit_per_prompt("image") == 0
                or (
                    using_live_state(config)
                    and spec.num_speculative_tokens == 2
                    and os.environ.get("BETTERSCALE_QWEN35_CONTRACT") != "unified"
                    and hf.model_type == "qwen3_5_moe_text"
                    and 1 <= m.multimodal_config.get_limit_per_prompt("image") <= 4
                )
            )
        ),
    }
    if not all(checks.values()):
        raise ValueError(
            "Outside Qwen35 baseline qualification: "
            + "; ".join(name for name, passed in checks.items() if not passed)
        )
    return "qwen35-live-state" if using_live_state(config) else "qwen35-baseline"


def check(config):
    import os

    required = {
        "MTP_TOKENS": str(config.speculative_config.num_speculative_tokens),
        "MTP_GDN_LAYOUT_FUSION": "1",
        "BETTERSCALE_MTP_GREEDY": "1",
        "BETTERSCALE_GDN_SMALL_COPIES": "1",
    }
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise ValueError(
            "Qwen35 baseline requires its qualified MTP/layout/small-fish flags"
        )
    if config.speculative_config.num_speculative_tokens > 2:
        if os.environ.get("BETTERSCALE_CONTEXT_PARALLEL") == "1":
            raise ValueError("MTP3/4 requires balanced attention disabled (Q1..3 contract)")
        os.environ["BETTERSCALE_CONTEXT_PARALLEL"] = "0"
    from ..qwen import check_runtime_profiles
    from ...patches import qwen_gdn, qwen_fia

    from .host_policy import metadata_default
    metadata_default(os.environ, unified=os.environ.get("BETTERSCALE_QWEN35_CONTRACT") == "unified")
    from ...patches.qwen_fia import owned
    if owned.enabled():
        if os.environ.get("BETTERSCALE_QWEN35_CONTRACT") == "school":
            check_runtime_profiles("qwen35_school_pins.json")
        else:
            check_runtime_profiles("qwen35_cann91_pins.json")
    else:
        check_runtime_profiles("qwen35_pins.json", "qwen35_unified_pins.json")
    qwen_gdn.check_library()
    if owned.enabled():
        owned.check_library()
    else:
        qwen_fia.check_library()
    from ...patches.qwen_fia.context_parallel import configure

    configure(os.environ)


def before_init(worker, config):
    from ...compilation_scope import CompilationScope
    from .integration import before_init

    worker._compilation_scope = CompilationScope()
    before_init(worker, config)
    if using_live_state(config):
        from .state_backend import install

        install()


def after_init(worker):
    from .integration import after_init

    after_init(worker)
    if using_live_state(worker.vllm_config):
        from .state_address import install

        install()


def determine_available_memory(worker, native):
    if not using_live_state(worker.vllm_config):
        return native()
    from .state_backend import determine_available_memory

    return determine_available_memory(worker, native)


def model_loaded(worker):
    from .integration import model_loaded

    model_loaded(worker)


def warmed(worker):
    """Finish continuation-only kernel preparation before service readiness."""
    from .input_reuse import warmup

    runner = worker.model_runner
    runner._bs_input_reuse_warmup_variants = warmup(runner)
    from .serving_warmup import warmup as warm_serving

    runner._bs_serving_warmup_variants = warm_serving(runner)
    worker._compilation_scope.seal()
    import logging

    logging.getLogger("vllm.betterscale").info(
        "Qwen35 Triton serving scope sealed; warmup calls=%s", runner._bs_serving_warmup_variants
    )
