"""Conservative original018 hybrid-pool envelope without inferring group count.

The source log is floor(pool_blocks/groups)*minimum_block_size.  Bound every
group by the minimum block size and charge the null block once per group.
Enumerating all source-permitted minimum sizes avoids guessing a runtime
Mamba/attention grouping or reusing the newer runtime's capacity conversion.
Source ownership and completed-run applicability are separate qualifications.
"""


def envelope(prompt_lengths, *, reported_tokens, maximum_output=8192,
             execution_seats=24, maximum_block_size=4096,
             in_flight_batches=2, speculative_tokens=2, lookahead_tokens=2):
    values = dict(reported_tokens=reported_tokens, maximum_output=maximum_output,
        execution_seats=execution_seats, maximum_block_size=maximum_block_size,
        in_flight_batches=in_flight_batches, speculative_tokens=speculative_tokens,
        lookahead_tokens=lookahead_tokens)
    if (not prompt_lengths or any(type(n) is not int or n < 1 for n in prompt_lengths)
            or any(type(n) is not int or n < (0 if k in ("speculative_tokens", "lookahead_tokens") else 1)
                   for k, n in values.items())):
        raise ValueError("Positive current population and source capacity bounds required")
    maximum_prefix = max(prompt_lengths) + maximum_output + in_flight_batches * (speculative_tokens + 1) + lookahead_tokens
    sizes = [b for b in range(1, maximum_block_size + 1) if reported_tokens % b == 0]
    rows = []
    for block in sizes:
        full_pages = (maximum_prefix + block - 1) // block
        # align-mode Mamba retains at most two state pages plus the fixed
        # speculative pages; the next allocation is included in this maximum.
        pages_per_group = max(full_pages, 2 + speculative_tokens)
        # Charge one additional complete request before admission bookkeeping.
        # Finished request blocks have already been freed by the scheduler;
        # cached zero-refcount pages remain available on the eviction queue.
        required = (execution_seats + 1) * pages_per_group + 1
        rows.append(dict(minimum_block_size=block, per_group_capacity=reported_tokens // block,
            maximum_pages_per_request_group=pages_per_group, required_pages_per_group=required,
            spare_pages_per_group=reported_tokens // block - required))
    return dict(schema="original018-hybrid-page-capacity-envelope/v1", **values,
        maximum_prompt_tokens=max(prompt_lengths), maximum_prefix_tokens=maximum_prefix,
        admissible_minimum_block_sizes=rows,
        sufficient_page_capacity=bool(rows) and all(r["spare_pages_per_group"] >= 0 for r in rows),
        group_count_assumed=False, elapsed_time_used=False, historical_capacity_reused=False,
        source_ownership_qualified=False, no_preemption_certified=False,
        full_model_certificate=False)
