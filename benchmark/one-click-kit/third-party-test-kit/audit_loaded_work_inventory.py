"""Validate loaded semantic dimensions/dispatch without reading tensor values."""


def validate(inventory, text):
    counts = {}
    for role in ("target", "draft"):
        entries = inventory[role]["modules"]
        modules = {row["name"]: row for row in entries}
        if len(modules) != len(entries) or inventory[role]["tensor_values_observed"] is not False:
            raise ValueError("Duplicate loaded modules or tensor-value observation")
        checked = set()

        def require(name, *, weights=None, scalars=None, object_scalars=None, quant=None):
            if name not in modules:
                raise ValueError("Missing loaded semantic module: " + name)
            row = modules[name]
            for field, shape in (weights or {}).items():
                value = row["parameters"].get(field)
                if value != dict(shape=shape, dtype="torch.bfloat16", device_type="npu"):
                    raise ValueError("Loaded semantic parameter differs: " + name + ":" + field)
            for field, expected in (scalars or {}).items():
                if type(row["scalars"].get(field)) is not type(expected) or row["scalars"][field] != expected:
                    raise ValueError("Loaded semantic scalar differs: " + name + ":" + field)
            for obj, fields in (object_scalars or {}).items():
                actual = row["objects"][obj]["scalars"]
                if any(type(actual.get(k)) is not type(v) or actual[k] != v for k, v in fields.items()):
                    raise ValueError("Loaded semantic object differs: " + name + ":" + obj)
            if quant and row["objects"]["quant_method"]["type"] != quant:
                raise ValueError("Loaded quantized/unreviewed dispatch: " + name)
            checked.add(name)

        linear = "vllm_ascend.ops.linear.AscendUnquantizedLinearMethod"
        prefix = "language_model.model" if role == "target" else "model"
        kinds = text["layer_types"] if role == "target" else ["full_attention"]
        shared_rotary_seen = False
        for i, kind in enumerate(kinds):
            base = f"{prefix}.layers.{i}"
            require(base, scalars=dict(layer_scale=False, layer_type=kind))
            for norm in ("input_layernorm", "post_attention_layernorm"):
                require(base + "." + norm, weights=dict(weight=[2048]), scalars=dict(variance_epsilon=1e-6))
            if kind == "linear_attention":
                attn = base + ".linear_attn"
                require(attn, scalars=dict(hidden_size=2048, tp_size=2, key_dim=2048, value_dim=4096,
                    head_k_dim=128, head_v_dim=128, num_k_heads=16, num_v_heads=32, activation="silu"))
                for leaf, shape in (("in_proj_qkvz", [6144, 2048]), ("in_proj_ba", [32, 2048]),
                                    ("out_proj", [2048, 2048]), ("conv1d", [4096, 1, 4])):
                    require(attn + "." + leaf, weights=dict(weight=shape), quant=linear)
                require(attn + ".norm", weights=dict(weight=[128]),
                    scalars=dict(norm_before_gate=True, eps=1e-6, activation="silu"))
            elif kind == "full_attention":
                attn = base + ".self_attn"
                require(attn, scalars=dict(hidden_size=2048, num_heads=8, num_kv_heads=1,
                    head_dim=256, attn_output_gate=True))
                for leaf, shape in (("qkv_proj", [4608, 2048]), ("o_proj", [2048, 2048])):
                    require(attn + "." + leaf, weights=dict(weight=shape), quant=linear)
                for leaf in ("q_norm", "k_norm"):
                    require(attn + "." + leaf, weights=dict(weight=[256]), scalars=dict(variance_epsilon=1e-6))
                rotary = attn + ".rotary_emb"
                if rotary in modules:
                    require(rotary, scalars=dict(rotary_dim=64, mrope_interleaved=True, mrope_section=[11, 11, 10]))
                    shared_rotary_seen = True
                elif not shared_rotary_seen:
                    raise ValueError("Missing first shared loaded rotary module")
                # The source-pinned get_rope memoizes this identical immutable
                # configuration. named_modules removes repeated shared objects;
                # later aliases therefore have no separate inventory row.
            else:
                raise ValueError("Unmapped loaded attention family")
            mlp = base + ".mlp"
            require(mlp, scalars=dict(n_routed_experts=256, tp_size=2, is_sequence_parallel=False,
                is_fused_shared_expert_enabled=False, replicate_shared_expert=False,
                enable_eplb=False, n_redundant_experts=0))
            for leaf, shape in (("gate", [256, 2048]), ("shared_expert_gate", [1, 2048]),
                                ("shared_expert.gate_up_proj", [512, 2048]),
                                ("shared_expert.down_proj", [2048, 256])):
                require(mlp + "." + leaf, weights=dict(weight=shape), quant=linear)
            require(mlp + ".experts.routed_experts",
                weights=dict(w13_weight=[256, 2048, 512], w2_weight=[256, 256, 2048]),
                scalars=dict(top_k=8, hidden_size=2048, intermediate_size_per_partition=256,
                    renormalize=True, apply_router_weight_on_input=False, routed_scaling_factor=1.0,
                    scoring_func="softmax", activation=dict(enum_type="vllm.model_executor.layers.fused_moe.activation.MoEActivation", name="SILU")),
                object_scalars=dict(moe_config=dict(num_experts=256, intermediate_size=512,
                    intermediate_size_per_partition=256, tp_size=2, ep_size=1, is_sequence_parallel=False)),
                quant="vllm_ascend.ops.fused_moe.routed_experts.AscendUnquantizedFusedMoEMethod")
        require(prefix + ".norm", weights=dict(weight=[2048]), scalars=dict(variance_epsilon=1e-6))
        require("language_model.lm_head" if role == "target" else "lm_head", weights=dict(weight=[124160, 2048]))
        if role == "draft":
            require("model.fc", weights=dict(weight=[1024, 4096]), quant=linear)
            for name in ("pre_fc_norm_hidden", "pre_fc_norm_embedding"):
                require("model." + name, weights=dict(weight=[2048]), scalars=dict(variance_epsilon=1e-6))
        counts[role] = len(checked)
    return dict(loaded_semantic_dimensions_and_dispatch_passed=True, checked_modules=counts,
        tensor_values_observed=False, necessary_work_certified=False)
