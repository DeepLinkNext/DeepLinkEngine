"""Source-specific FLA bounds; the frozen canonical/reference files stay intact."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def adjust_fla(base, model, geometry, workers):
    """Apply only after native_implementation validates actual FLA identities.

    Each genuine WY row saves k-1 multiplies: beta*exp(g) is computed once,
    then broadcast to k. Unknown one-token recurrent prompt rows save zero.
    Subtracting the maximum possible WY saving from the canonical lower bound
    is conservative for every admitted partition. The old upper bound remains
    valid. Neither corrected endpoint asserts an observed or tight partition.
    """
    from official_ft_fla import PROFILE_SHA256
    if len(workers) != 2 or {row['rank'] for row in workers} != {0, 1}:
        raise ValueError('Both native rank backend observations required')
    backends = {row['rank']: row['resolved_configuration']['gdn_backend'] for row in workers}
    if any(row.get('selected') not in ('fla-npu', 'triton-chunk64') for row in backends.values()):
        raise ValueError('Unmapped source-specific GDN arithmetic backend')
    if not any(row['selected'] == 'fla-npu' for row in backends.values()):
        return base
    for row in backends.values():
        if row['selected'] == 'fla-npu':
            identity = row.get('operator_identity', {})
            if identity.get('admitted') is not True or identity.get('profile_sha256') != PROFILE_SHA256:
                raise ValueError('Actual native FLA profile admission required')
    rows = geometry['requests']
    if (base['geometry_sha256'] != geometry['geometry_sha256'] or base['requests'] != len(rows)
            or not rows or any(type(row['input_tokens']) is not int or row['input_tokens'] < 1 for row in rows)
            or model.linear_key_head_dim != 128 or model.linear_value_head_dim != 128
            or type(model.linear_num_value_heads) is not int or model.linear_num_value_heads < 1):
        raise ValueError('Source-adjusted arithmetic geometry/model binding differs')
    layers = model.layer_types.count('linear_attention')
    if layers < 1:
        raise ValueError('Source-adjusted target GDN layer population is empty')
    canonical = deepcopy(base)
    result = deepcopy(base)
    deduction = sum(row['input_tokens'] for row in rows) * (model.linear_key_head_dim - 1) * layers * model.linear_num_value_heads
    lower = result['endpoints']['lower']
    if (base['F_interval'] != [base['endpoints'][side]['total_basic'] for side in ('lower', 'upper')]
            or lower['components']['target_gdn_prefill'] < deduction
            or any(sum(base['endpoints'][side]['components'].values()) != base['endpoints'][side]['total_basic']
                   for side in ('lower', 'upper'))):
        raise ValueError('Canonical arithmetic components do not conserve the interval')
    lower['components']['target_gdn_prefill'] -= deduction
    lower['total_basic'] -= deduction
    result['F_interval'][0] -= deduction
    result['schema'] = 'official-source-adjusted-necessary-arithmetic-interval/v1'
    result.pop('prefill_partition_envelope', None)
    result['base_canonical_ledger'] = canonical
    result['source_bound_correction'] = dict(
        family='fla-a2-26.9.1-wy64-factored-beta-exp/v1', profile_sha256=PROFILE_SHA256,
        geometry_sha256=geometry['geometry_sha256'], native_backend_by_rank={str(rank): row['selected'] for rank, row in backends.items()},
        scope='target GDN prompt prefill only; draft full-attention and recurrent prompt rows have zero correction',
        prompt_rows_by_request=[dict(request_ordinal=row['request_ordinal'], input_tokens=row['input_tokens']) for row in rows],
        key_dim=model.linear_key_head_dim, global_value_heads=model.linear_num_value_heads,
        target_linear_layers=layers, extra_tensor_parallel_multiplier=1,
        canonical_WY_basic_per_head_layer='2*n*k', source_WY_basic_per_head_layer='n+n*k',
        maximum_possible_basic_deduction=deduction, lower_adjustment=-deduction, upper_adjustment=0,
        recurrent_prompt_allowed=True, recurrent_prompt_deduction=0,
        source='internal/arch22/operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_vector.h:395-415',
        tight=False, corrected_endpoint_witness_claimed=False, base_witnesses_apply_only_to_canonical=True,
        mixed_rank_policy='If any rank selects FLA, all global heads bound the maximum possible saving; this remains conservative.',
        original_reference_denominator_modified=False, elapsed_time_used=False,
        cpu_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        canonical_ledger_sha256=hashlib.sha256((json.dumps(canonical, sort_keys=True, separators=(',', ':'))+'\n').encode()).hexdigest())
    return result
