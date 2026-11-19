"""Finite page-demand envelope; source ownership obligations remain explicit."""


def envelope(prompt_lengths, *, maximum_output, resident_seats, execution_seats,
             block_size, pool_blocks, in_flight_batches, speculative_tokens,
             lookahead_tokens, transfer_directions=2):
    values = dict(maximum_output=maximum_output, resident_seats=resident_seats,
        execution_seats=execution_seats, block_size=block_size, pool_blocks=pool_blocks,
        in_flight_batches=in_flight_batches, speculative_tokens=speculative_tokens,
        lookahead_tokens=lookahead_tokens, transfer_directions=transfer_directions)
    if (not prompt_lengths or any(type(n) is not int or n < 1 for n in prompt_lengths)
            or any(type(n) is not int or n < (0 if k in ("speculative_tokens", "lookahead_tokens", "transfer_directions") else 1)
                   for k, n in values.items()) or resident_seats < execution_seats):
        raise ValueError("Positive source-bound capacities and current prompt lengths required")
    # The whole output budget is used, not observed average completion lengths.
    # Cover every queued verification grant plus allocation lookahead even if
    # EOS/length feedback has not reached the scheduler yet.
    queued_tokens = in_flight_batches * (speculative_tokens + 1)
    maximum_tokens = max(prompt_lengths) + maximum_output + queued_tokens + lookahead_tokens
    pages = (maximum_tokens + block_size - 1) // block_size
    # Every resident, each independently in-flight transfer direction, and one
    # successful native allocation before its seat claim are bounded separately.
    # Shared/refcounted pages may be charged more than once; never less.
    resident_pages = resident_seats * pages
    transfer_pages = transfer_directions * pages
    unclaimed_allocation_pages = pages
    null_pages = 1
    required = resident_pages + transfer_pages + unclaimed_allocation_pages + null_pages
    return dict(schema="native-page-capacity-envelope/v1", **values,
        maximum_prompt_tokens=max(prompt_lengths), queued_verification_tokens=queued_tokens,
        maximum_tokens_per_owned_prefix=maximum_tokens, maximum_pages_per_owned_prefix=pages,
        resident_page_bound=resident_pages, transfer_page_bound=transfer_pages,
        unclaimed_allocation_page_bound=unclaimed_allocation_pages,
        null_page_allowance=null_pages, required_pages=required,
        sufficient_page_capacity=required <= pool_blocks,
        spare_pages=pool_blocks - required, elapsed_time_used=False,
        source_ownership_qualified=False, no_preemption_certified=False,
        required_source_obligations=[
            "One uniform full-attention pool with the declared block size; no connector/sparse/Mamba cache groups.",
            "Every active, retained-hot and queued-writer prefix has one of the bounded resident seats.",
            "Deferred-free/pending-hot references share the same page IDs rather than allocating a duplicate prefix.",
            "At most one store and one load direction can pin detached/new pages; sources remain charged until TP quorum.",
            "A scheduler allocation claims its offered resident before another request allocation can run.",
            "The queue depth, fixed MTP width and lookahead bound every ahead-of-feedback reservation.",
            "Fresh unique text cache salts and a drained cold reset exclude external/previous-run prefix owners.",
            "Watermark/reserved-block constraints and active partial-reclaim demand do not exceed the same envelope.",
            "Only capacity refusal can preempt an active request on this source path; waiting seat refusal merely delays admission."],
        full_model_certificate=False)
