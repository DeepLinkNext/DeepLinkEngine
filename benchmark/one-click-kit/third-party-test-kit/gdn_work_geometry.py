"""Join real host GDN metadata to request positions, without inventing splits."""

from collections import Counter, defaultdict
import re

from gdn_prefill_reference import segment_arithmetic
from gdn_reference import operation_counts


def exact_integer(value, minimum, label):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid " + label)
    return value


def join_wave(rows, captures, *, expected_layers):
    """Return source-bound per-layer-group chunks and recurrent row extents.

    FULL graph capacity and padding are deliberately not inferred from token
    totals. Every live request/layer must occur exactly once across metadata
    microbatches; an unsupported metadata layout fails before producing work.
    """
    scheduled = {}
    for row in rows:
        identity = row["request"]
        if (not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{64}", identity)
                or identity in scheduled):
            raise ValueError("Request identity is missing or repeated")
        exact_integer(row["start"], 0, "request start")
        exact_integer(row["count"], 1, "scheduled extent")
        scheduled[identity] = row
    expected_layers = set(expected_layers)
    if not scheduled or not expected_layers:
        raise ValueError("Require scheduled work and the fixed GDN layer population")
    seen, result = defaultdict(set), []
    for capture in captures:
        order = capture["requests"]
        if len(order) != len(set(order)) or not set(order) <= scheduled.keys():
            raise ValueError("Foreign or repeated metadata request")
        for group in capture["groups"]:
            layers = group["layers"]
            if (not layers or len(layers) != len(set(layers))
                    or not set(layers) <= expected_layers):
                raise ValueError("Foreign or repeated GDN layer")
            live = exact_integer(group["live"], 1, "live prefix")
            if live > len(order) or type(group["decode"]) is not bool:
                raise ValueError("GDN live prefix or execution mode is invalid")
            if group["decode"]:
                lengths = [scheduled[identity]["count"] for identity in order[:live]]
                verification = set(range(live))
                initial = None
            else:
                cu = group["cumulative_lengths"]
                initial = group["initial_state"]
                verify = group["verify_indices"]
                if (len(cu) != live + 1 or cu[0] != 0
                        or any(type(n) is not int or n < 0 for n in cu)
                        or any(b <= a for a, b in zip(cu, cu[1:]))
                        or len(initial) != live or any(type(n) is not bool for n in initial)
                        or len(verify) != len(set(verify))
                        or len(verify) != group["verify_count"]
                        or any(type(n) is not int or not 0 <= n < live for n in verify)):
                    raise ValueError("Incomplete host GDN cumulative/state/verification metadata")
                lengths = [b - a for a, b in zip(cu, cu[1:])]
                verification = set(verify)
            for i, identity in enumerate(order[:live]):
                if seen[identity].intersection(layers):
                    raise ValueError("Duplicate request/layer across GDN metadata captures")
                seen[identity].update(layers)
                row = scheduled[identity]
                if lengths[i] != row["count"]:
                    raise ValueError("Host GDN length differs from the native scheduled row")
                mode = "recurrent" if i in verification else "chunked_prefill"
                item = dict(request=identity, start=row["start"], count=lengths[i],
                            layers=list(layers), mode=mode)
                if mode == "chunked_prefill":
                    item["initial_state"] = initial[i]
                    item["arithmetic_per_head_layer"] = segment_arithmetic(lengths[i], initial[i])
                result.append(item)
    if any(seen[identity] != expected_layers for identity in scheduled):
        raise ValueError("GDN metadata does not cover every scheduled request/layer")
    return result


def target_prefill_work(waves, requests, *, expected_layers, global_heads):
    """Count observed target chunks only after checking request prefix coverage.

    This closes the GDN-prefill family, not the draft/scalar/reference families.
    Needed suffix trimming follows causality within a real published segment;
    it never moves a segment boundary or borrows another run's chunk histogram.
    """
    exact_integer(global_heads, 1, "global value heads")
    required = {}
    for row in requests:
        key = row["request"]
        if key in required:
            raise ValueError("Duplicate returned request")
        prompt = exact_integer(row["input_tokens"], 1, "prompt length")
        output = exact_integer(row["output_tokens"], 1, "output length")
        required[key] = prompt + output - 1
    if not required:
        raise ValueError("No current request population")
    covered = defaultdict(list)
    parts, special, comparison = Counter(), Counter(), Counter()
    segments = []
    for wave in waves:
        geometry = join_wave(wave["rows"], wave["gdn"], expected_layers=expected_layers)
        for row in geometry:
            key = row["request"]
            if key not in required:
                raise ValueError("Unbound request in GDN work")
            for layer in row["layers"]:
                covered[key, layer].append((row["start"], row["start"] + row["count"]))
            if row["mode"] != "chunked_prefill":
                continue
            needed = min(row["count"], max(0, required[key] - row["start"]))
            if not needed:
                continue
            if row["initial_state"] != (row["start"] > 0):
                raise ValueError("Prefill state origin is inconsistent with a fresh request prefix")
            for layer in row["layers"]:
                # The last entry is this segment, appended above. Any earlier
                # overlapping producer needs an explicit cache/producer join;
                # counting both from token lengths would inflate useful work.
                if any(max(start, row["start"]) < min(stop, row["start"] + needed)
                       for start, stop in covered[key, layer][:-1]):
                    raise ValueError("Overlapping prefill needs an explicit producer dependency closure")
            arithmetic = segment_arithmetic(needed, row["initial_state"])
            factor = len(row["layers"]) * global_heads
            for counter, name in ((parts, "basic"), (special, "special_functions"),
                                  (comparison, "comparisons")):
                for component, n in arithmetic[name].items():
                    counter[component] += factor * n
            segments.append(dict(request=key, start=row["start"], observed_count=row["count"],
                                 necessary_count=needed, initial_state=row["initial_state"],
                                 layers=row["layers"], wave=wave["wave"]))
    for key, length in required.items():
        for layer in expected_layers:
            end = 0
            for start, stop in sorted(covered[key, layer]):
                if start > end:
                    break
                end = max(end, stop)
            if end < length:
                raise ValueError("A necessary target GDN prefix has a gap")
    prefill_token_layers = sum(row["necessary_count"] * len(row["layers"]) for row in segments)
    necessary_token_layers = sum(required.values()) * len(expected_layers)
    recurrence_rows = necessary_token_layers - prefill_token_layers
    if recurrence_rows < 0:
        raise ValueError("Prefill exceeds the necessary target GDN population")
    initial_layers = defaultdict(set)
    for row in segments:
        if row["start"] == 0 and not row["initial_state"]:
            initial_layers[row["request"]].update(row["layers"])
    if any(initial_layers[request] != set(expected_layers) for request in required):
        raise ValueError("A fresh recurrent-only request needs a qualified zero-state arithmetic rule")
    recurrence = operation_counts(128, 128)
    recurrence = dict(basic={name: value * recurrence_rows * global_heads for name, value in recurrence["basic"].items()},
                      total_basic=recurrence["total_basic"] * recurrence_rows * global_heads,
                      special_functions={"exp": recurrence_rows * global_heads},
                      necessary_token_layers=recurrence_rows,
                      source_runtime_applicability_certified=False)
    return dict(scope="current_observed_target_gdn_prefill", basic=dict(parts),
                total_basic=sum(parts.values()), special_functions=dict(special),
                comparisons=dict(comparison), segments=segments,
                necessary_prefill_token_layers=prefill_token_layers,
                recurrence_candidate=recurrence,
                request_layer_coverage_verified=True, full_model_certificate=False)
