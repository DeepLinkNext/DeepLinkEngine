"""Derive time-independent public output facts for native necessary-work review.

These facts describe this run's own token population and ordered public DELTA
emissions. Source review must bind emissions to MTP acceptance before using the
latter interpretation. No scheduler/worker internals or elapsed time are inputs.
"""

import hashlib
import json

from native_public_client import validate_frames


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def geometry(raw_outputs, prompt_token_ids, maximum=8192, speculative_tokens=2):
    if (type(maximum) is not int or maximum < 1 or type(speculative_tokens) is not int
            or speculative_tokens < 0 or not prompt_token_ids
            or any(not tokens or any(type(t) is not int or t < 0 for t in tokens)
                   for tokens in prompt_token_ids)):
        raise ValueError("Positive native token population and source speculation width required")
    # This allowlist is the complete input to the public-output parser. Timing
    # fields, even if present in its original sealed record, cannot affect work.
    raw = {key: raw_outputs[key] for key in ("request_ids", "public_output_frames")}
    lengths = list(map(len, prompt_token_ids))
    responses = validate_frames(raw, lengths, maximum)
    ids = raw["request_ids"]
    emissions = {identity: [] for identity in ids}
    for frame_index, frame in enumerate(raw["public_output_frames"]):
        for identity, finished, cached, choices in frame:
            count = len(choices[0][1])
            emissions[identity].append(dict(public_step=frame_index, emitted_tokens=count,
                                            finished=finished))
    rows = []
    for ordinal, (response, prompt) in enumerate(zip(responses, prompt_token_ids, strict=True)):
        generated = response["output_ids"]
        ordered = emissions[response["request_id"]]
        nonempty = [event["emitted_tokens"] for event in ordered if event["emitted_tokens"]]
        if sum(nonempty) != len(generated):
            raise ValueError("Public emission population differs from this native completion")
        rows.append(dict(request_ordinal=ordinal, input_tokens=len(prompt), output_tokens=len(generated),
            input_ids_sha256=digest(prompt), output_ids_sha256=digest(generated),
            necessary_target_prefix_rows=len(prompt) + len(generated) - 1,
            ordered_public_emissions=ordered, positive_emission_lengths=nonempty,
            mtp_width_compatible=all(n <= speculative_tokens + 1 for n in nonempty),
            first_emission_is_one=nonempty[0] == 1,
            finish_reason=response["finish_reason"], stop_reason=response["stop_reason"]))
    signature = dict(requests=rows, maximum_output_tokens=maximum,
                     speculative_tokens=speculative_tokens,
                     public_steps=len(raw["public_output_frames"]))
    return dict(schema="native-public-output-geometry/v1", **signature,
        geometry_sha256=digest(signature), input_tokens=sum(lengths),
        output_tokens=sum(row["output_tokens"] for row in rows),
        necessary_target_prefix_rows=sum(row["necessary_target_prefix_rows"] for row in rows),
        all_emissions_mtp_width_compatible=all(row["mtp_width_compatible"] for row in rows),
        ordered_acceptance_source_binding_certified=False,
        necessary_work_certified=False, efficiency_percent=None,
        elapsed_time_used=False, private_engine_fields_read=False,
        pending="Bind public output batching to current source acceptance and target/draft dependencies; these token facts alone are not complete arithmetic.")
