"""Compare public-emission reconstruction with separately observed producers.

This checks a companion observation's own output population. Its times or
token counts never replace the clean native run's arithmetic or clock.
"""

from collections import defaultdict

from draft_producer_geometry import bind_producers
from draft_state_closure import backward, versions_from_calls
from public_request_work import bounds

FIELDS = ("necessary_kv_prefix_rows", "necessary_full_rows",
          "necessary_vocabulary_rows", "necessary_full_causal_pairs")


def audit(requests, scheduler, rank0_drafts, rank1_drafts):
    identities = [r["request"] for r in requests]
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("Unique companion request population required")
    emissions = defaultdict(list)
    for expected, wave in enumerate(scheduler):
        if wave["wave"] != expected:
            raise ValueError("Companion scheduler wave sequence is incomplete")
        for r in wave["retained"]:
            if r["request"] not in identities or r["stale"] and r["kept"]:
                raise ValueError("Unbound or stale companion public population")
            if r["kept"] or r["stopped"]:
                emissions[r["request"]].append(dict(public_step=expected,
                    emitted_tokens=r["kept"], finished=r["stopped"]))
    bindings = bind_producers(rank0_drafts, rank1_drafts, scheduler)
    versions, indices = versions_from_calls(rank0_drafts)
    roots = defaultdict(list)
    for binding in bindings["bindings"]:
        for step in range(binding["retained_drafts"]):
            roots[binding["request"]].append(indices[binding["producer_call"], binding["request"], step])
    results = []
    for ordinal, item in enumerate(requests):
        identity = item["request"]
        row = dict(request_ordinal=ordinal, input_tokens=item["input_tokens"],
            output_tokens=item["output_tokens"], input_ids_sha256=identity, output_ids_sha256=identity,
            ordered_public_emissions=emissions[identity], finish_reason="stop")
        interval = bounds(row)
        observed = backward(versions, roots[identity])
        for key in FIELDS:
            low, high = (interval[b]["closure"][key] for b in ("lower", "upper"))
            if not low <= observed[key] <= high:
                raise ValueError("Public reconstruction misses observed producer closure: " + identity + ":" + key)
        results.append(dict(request_ordinal=ordinal, request=identity,
            public_lower={k: interval["lower"]["closure"][k] for k in FIELDS},
            public_upper={k: interval["upper"]["closure"][k] for k in FIELDS},
            actual_observed={k: observed[k] for k in FIELDS}))
    return dict(schema="independent-public-to-observed-producer-crosscheck/v1",
        requests=results, all_request_closures_contained=True,
        companion_output_only=True, clean_native_counts_or_times_replaced=False,
        necessary_work_certified=False, full_model_certificate=False)
