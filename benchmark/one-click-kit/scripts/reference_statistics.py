"""Cache verified reference denominators; recompute efficiency from matched numerators."""

from decimal import Decimal, localcontext
import hashlib
import json


def canonical_sha(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def statistics(evidence):
    source = evidence["m01"]
    if not source["evidence_qualified"] or source["chips"] != 2:
        raise ValueError("Reference evidence is not qualified")
    result = {
        "schema": "qualified-reference-denominator/v1",
        "benchmark_commit": evidence["benchmark_commit"],
        "manifest_sha256": evidence["manifest_sha256"],
        "unit": "TFLOP/s/card",
        "interpretation": "Workload-specific independent reference; upper denominator from a certified reference-time lower bound",
        "scope": {
            "profile": "mmlu-fixed200-v1",
            "requests": 200,
            "input_tokens": 282265,
            "output_tokens": 51200,
            "max_completion_tokens": 256,
            "dtype": "BF16",
            "tp": 2,
            "accelerator": "Ascend 910B2",
            "frontend": "native_inproc",
            "timing": "dual_rank_synchronized_after_warmup",
        },
        "calculation": {
            "reference_upper_formula": "C_ref_upper = sum(F_upper) / (k * sum(T_ref_lower) * 1e12)",
            "current_lower_formula": "C_current_lower = sum(F_current_lower) / (k * sum(T_current) * 1e12)",
            "efficiency_lower_formula": "eta_lower = 100 * C_current_lower / C_ref_upper",
            "shared_exact_work_formula": "Only when work is identical and cancellation is justified: eta_lower = 100 * sum(T_ref_lower) / sum(T_current)",
            "reference_time_construction": "finish(v) = duration(v) + max(finish(parent)); T_ref_lower = max(finish(v)) over the certified retained dependency/resource graph",
            "aggregation": "sum work and time across all repetitions before division; never average per-repeat rates or select the fastest repetition",
            "chips": 2,
            "flops_counting": "Necessary-work dependency closure; matrix multiply-add=2 operations; scalar catalog fixed; padding, redundant work and unused drafts excluded",
            "reference_bound": "Certified retained graph and nonnegative omitted work give a time lower bound; combining F_upper and T_ref_lower gives a reference-compute upper bound",
            "independence": "Target endpoint elapsed times and target efficiency are not inputs to reference calibration or DAG durations",
            "applicability": "Workload, arithmetic/reference domains, dtype, card count and timing boundary must match; out256/native and out8192/HTTP are separate",
        },
        "arms": {},
    }
    with localcontext() as ctx:
        ctx.prec = 40
        for arm in source["arms"]:
            repeats = arm["repetitions"]
            if len(repeats) != 3 or any(
                r["requests"] != 200
                or r["output_tokens"] != 51200
                or r["synchronized_ranks"] != 2
                or not r["uninstrumented"]
                for r in repeats
            ):
                raise ValueError("Reference domain differs")
            flops = [sum(r["F_interval"][i] for r in repeats) for i in (0, 1)]
            reference = sum(Decimal(r["Tref_lower_seconds"]) for r in repeats)
            # This is the reusable denominator, not an old efficiency score.
            result["arms"][arm["role"]] = {
                "sum_reference_seconds_lower": str(reference),
                "effective_flops_interval": flops,
                "reference_tflops_per_card_upper": str(
                    Decimal(flops[1]) / (2 * reference * 10**12)
                ),
                "reference_repetitions": [
                    {
                        k: r[k]
                        for k in (
                            "name",
                            "F_interval",
                            "Tref_lower_seconds",
                            "requests",
                            "input_tokens",
                            "output_tokens",
                            "source_sha256",
                        )
                    }
                    for r in repeats
                ],
            }
    result["domain_sha256"] = canonical_sha(
        {"scope": result["scope"], "arms": result["arms"]}
    )
    return result


def efficiency(cache, numerator):
    """Evaluate fresh, sealed same-domain arithmetic and timing; no old percent input."""
    if (
        numerator["reference_domain_sha256"] != cache["domain_sha256"]
        or numerator["scope"] != cache["scope"]
    ):
        raise ValueError(
            "Current numerator and reference denominator have different workload domains"
        )
    role = numerator["role"]
    reference = cache["arms"][role]
    rows = numerator["repetitions"]
    if len(rows) != len(reference["reference_repetitions"]) or any(
        not r["synchronized_ranks"] == 2
        or not r["uninstrumented"]
        or Decimal(r["T_seconds"]) <= 0
        for r in rows
    ):
        raise ValueError("Missing current synchronized timing")
    if numerator["effective_flops_interval"] != reference["effective_flops_interval"]:
        raise ValueError(
            "Changed necessary-work interval requires applicability revalidation"
        )
    with localcontext() as ctx:
        ctx.prec = 40
        elapsed = sum(Decimal(r["T_seconds"]) for r in rows)
        lower = Decimal(numerator["effective_flops_interval"][0]) / (
            2 * elapsed * 10**12
        )
        eta = 100 * lower / Decimal(reference["reference_tflops_per_card_upper"])
    return {
        "effective_tflops_per_card_lower": str(lower),
        "reference_tflops_per_card_upper": reference["reference_tflops_per_card_upper"],
        "efficiency_percent_lower": str(eta),
        "milestone_met": eta >= 50,
        "final_met": eta >= 90,
    }
