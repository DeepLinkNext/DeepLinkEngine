"""Conservative page admission for the source-qualified official shared pool.

Use actual groups and pool blocks. No old log-to-page conversion, StateRoot
seats, host offloading, or inferred group count enters this calculation.
"""
import math


def envelope(core, prompt_lengths, *, maximum_output=256, maximum_sequences=24, speculative_tokens=2):
    if (not prompt_lengths or any(type(n) is not int or n < 1 for n in prompt_lengths)
            or type(core['num_blocks']) is not int or core['num_blocks'] < 1
            or type(core['in_flight_batches']) is not int or not 1 <= core['in_flight_batches'] <= 2
            or core['prefix_caching'] is not False or not core['groups']
            or core['scheduler']['max_num_seqs'] != maximum_sequences
            or core['scheduler']['max_num_batched_tokens'] != 4096):
        raise ValueError('Actual official pool, fresh prefixes and scheduler geometry required')
    # Charge every live seat, one complete admission in progress, both pending
    # grants, and the source's MTP lookahead. Extra per-group pages also cover
    # align-mode retention; none/all modes explicitly add speculative blocks.
    prefix = max(prompt_lengths) + maximum_output + core['in_flight_batches'] * (speculative_tokens + 1) + speculative_tokens
    groups = []
    for index, group in enumerate(core['groups']):
        block = group['block_size']
        if type(block) is not int or block < 1 or group['host_resident'] not in (None, False):
            raise ValueError('Unmapped block or host-resident KV group')
        kind = group['type'].rsplit('.', 1)[-1]
        if kind == 'FullAttentionSpec':
            extra = 0
        elif kind == 'MambaSpec':
            extra = group['num_speculative_blocks']
            if type(extra) is not int or extra < 0 or group['mamba_cache_mode'] not in ('none', 'all', 'align'):
                raise ValueError('Unmapped Mamba allocation geometry')
            extra += 2
        else:
            raise ValueError('Unmapped official KV group: ' + group['type'])
        per_request = (prefix + block - 1) // block + extra
        groups.append(dict(group=index, type=kind, block_size=block,
                           maximum_pages_per_request=per_request,
                           charged_pages=(maximum_sequences + 1) * per_request))
    required = sum(g['charged_pages'] for g in groups) + 1
    return dict(schema='deeplinkengine-official-shared-pool-envelope/v1',
        maximum_prefix_tokens=prefix, maximum_output_tokens=maximum_output,
        execution_seats=maximum_sequences, in_flight_batches=core['in_flight_batches'],
        speculative_tokens=speculative_tokens, groups=groups,
        actual_pool_blocks=core['num_blocks'], required_blocks=required,
        spare_blocks=core['num_blocks'] - required,
        sufficient_page_capacity=core['num_blocks'] >= required,
        old_capacity_conversion_used=False, elapsed_time_used=False)
