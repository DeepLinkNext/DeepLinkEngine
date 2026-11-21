"""Audit official loaded Qwen3.5 TP2 arithmetic, without tensor values.

The two upstream MoE module layouts are checked explicitly. Absent legacy
flags are not supplied with invented defaults. Source admission is separate.
"""


def validate(inventory, text):
    if inventory.get('tensor_values_observed') is not False:
        raise ValueError('Loaded inventory must not read parameter values')
    if (text['hidden_size'] != 2048 or text['num_experts'] != 256
            or text['num_experts_per_tok'] != 8 or text['moe_intermediate_size'] != 512
            or text['shared_expert_intermediate_size'] != 512
            or text['num_hidden_layers'] != 40 or text['mtp_num_hidden_layers'] != 1):
        raise ValueError('Unmapped necessary-arithmetic model family')
    counts, layouts = {}, {}
    for role in ('target', 'draft'):
        value = inventory[role]
        if value['tensor_values_observed'] is not False:
            raise ValueError('Loaded parameters were observed')
        modules = {r['name']: r for r in value['modules']}
        if len(modules) != len(value['modules']):
            raise ValueError('Duplicate loaded semantic module')
        checked = set()

        def require(name, *, weights=None, scalars=None, quant=None):
            if name not in modules:
                raise ValueError('Missing loaded semantic module: ' + name)
            row = modules[name]
            for field, shape in (weights or {}).items():
                if row['parameters'].get(field) != dict(shape=shape, dtype='torch.bfloat16', device_type='npu'):
                    raise ValueError('Loaded semantic parameter differs: ' + name + ':' + field)
            for field, expected in (scalars or {}).items():
                actual = row['scalars'].get(field)
                if type(actual) is not type(expected) or actual != expected:
                    raise ValueError('Loaded semantic scalar differs: ' + name + ':' + field)
            if quant and row['objects']['quant_method']['type'] != quant:
                raise ValueError('Unreviewed loaded dispatch: ' + name)
            checked.add(name)
            return row

        linear = 'vllm_ascend.ops.linear.AscendUnquantizedLinearMethod'
        prefix = 'language_model.model' if role == 'target' else 'model'
        rotary_seen = False
        kinds = text['layer_types'] if role == 'target' else ['full_attention']
        observed_layouts = set()
        for index, kind in enumerate(kinds):
            base = f'{prefix}.layers.{index}'
            require(base, scalars=dict(layer_scale=False, layer_type=kind))
            for norm in ('input_layernorm', 'post_attention_layernorm'):
                require(base + '.' + norm, weights=dict(weight=[2048]), scalars=dict(variance_epsilon=1e-6))
            if kind == 'linear_attention':
                attn = base + '.linear_attn'
                require(attn, scalars=dict(hidden_size=2048, tp_size=2, key_dim=2048, value_dim=4096,
                    head_k_dim=128, head_v_dim=128, num_k_heads=16, num_v_heads=32, activation='silu'))
                for leaf, shape in (('in_proj_qkvz', [6144, 2048]), ('in_proj_ba', [32, 2048]),
                                    ('out_proj', [2048, 2048]), ('conv1d', [4096, 1, 4])):
                    require(attn + '.' + leaf, weights=dict(weight=shape), quant=linear)
                require(attn + '.norm', weights=dict(weight=[128]),
                        scalars=dict(norm_before_gate=True, eps=1e-6, activation='silu'))
            elif kind == 'full_attention':
                attn = base + '.self_attn'
                require(attn, scalars=dict(hidden_size=2048, num_heads=8, num_kv_heads=1,
                                         head_dim=256, attn_output_gate=True))
                for leaf, shape in (('qkv_proj', [4608, 2048]), ('o_proj', [2048, 2048])):
                    require(attn + '.' + leaf, weights=dict(weight=shape), quant=linear)
                for leaf in ('q_norm', 'k_norm'):
                    require(attn + '.' + leaf, weights=dict(weight=[256]), scalars=dict(variance_epsilon=1e-6))
                if attn + '.rotary_emb' in modules:
                    require(attn + '.rotary_emb', scalars=dict(rotary_dim=64, mrope_interleaved=True,
                                                             mrope_section=[11, 11, 10]))
                    rotary_seen = True
                elif not rotary_seen:
                    raise ValueError('First loaded rotary module is absent')
            else:
                raise ValueError('Unmapped loaded attention family')
            mlp = base + '.mlp'
            require(mlp, scalars=dict(n_routed_experts=256, tp_size=2, is_sequence_parallel=False,
                                     enable_eplb=False, n_redundant_experts=0))
            for leaf, shape in (('gate', [256, 2048]), ('shared_expert_gate', [1, 2048]),
                                ('shared_expert.gate_up_proj', [512, 2048]),
                                ('shared_expert.down_proj', [2048, 256])):
                require(mlp + '.' + leaf, weights=dict(weight=shape), quant=linear)
            routed = mlp + '.experts.routed_experts'
            if routed in modules:
                observed_layouts.add('routed-experts')
                require(mlp, scalars=dict(is_fused_shared_expert_enabled=False, replicate_shared_expert=False))
                dispatch = 'vllm_ascend.ops.fused_moe.routed_experts.AscendUnquantizedFusedMoEMethod'
            else:
                observed_layouts.add('legacy-fused-moe')
                routed = mlp + '.experts'
                dispatch = 'vllm_ascend.ops.fused_moe.fused_moe.AscendUnquantizedFusedMoEMethod'
            expert = require(routed, weights=dict(w13_weight=[256, 2048, 512], w2_weight=[256, 256, 2048]),
                scalars=dict(top_k=8, hidden_size=2048, intermediate_size_per_partition=256,
                    renormalize=True, apply_router_weight_on_input=False, routed_scaling_factor=1.0,
                    scoring_func='softmax', activation=dict(
                        enum_type='vllm.model_executor.layers.fused_moe.activation.MoEActivation', name='SILU')),
                quant=dispatch)
            config = expert['objects']['moe_config']['scalars']
            for key, expected in dict(num_experts=256, intermediate_size_per_partition=256,
                                      tp_size=2, ep_size=1, is_sequence_parallel=False).items():
                if type(config.get(key)) is not type(expected) or config[key] != expected:
                    raise ValueError('Loaded MoE geometry differs: ' + routed + ':' + key)
        require(prefix + '.norm', weights=dict(weight=[2048]), scalars=dict(variance_epsilon=1e-6))
        require('language_model.lm_head' if role == 'target' else 'lm_head', weights=dict(weight=[124160, 2048]))
        if role == 'draft':
            require('model.fc', weights=dict(weight=[1024, 4096]), quant=linear)
            for leaf in ('pre_fc_norm_hidden', 'pre_fc_norm_embedding'):
                require('model.' + leaf, weights=dict(weight=[2048]), scalars=dict(variance_epsilon=1e-6))
        counts[role], layouts[role] = len(checked), sorted(observed_layouts)
    return dict(loaded_semantic_dimensions_and_dispatch_passed=True, checked_modules=counts,
                loaded_moe_layouts=layouts, tensor_values_observed=False)
