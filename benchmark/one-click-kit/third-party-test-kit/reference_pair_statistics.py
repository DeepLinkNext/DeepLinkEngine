"""Matched-F reference bounds; never divide unrelated interval endpoints.

For each arm the same unknown actual F occurs in achieved and reference
compute, so it cancels in eta. Achieved compute keeps its F interval; the
reference compute has an upper bound from F_upper and T_reference_lower.
These are bounds under the calibrated model, not confidence intervals.
"""

from decimal import Decimal, InvalidOperation, localcontext


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError("Invalid " + label)
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("Invalid " + label) from error
    if not number.is_finite() or number <= 0:
        raise ValueError("Nonpositive or nonfinite " + label)
    return number


def arm(work_interval, native_seconds, reference_lower_seconds, *, cards=2):
    if (not isinstance(work_interval, (list, tuple)) or len(work_interval) != 2
            or any(type(n) is not int or n <= 0 for n in work_interval)
            or work_interval[0] > work_interval[1] or type(cards) is not int or cards < 1):
        raise ValueError("Ordered positive integer work interval and card count required")
    low, high = map(Decimal, work_interval)
    native = positive(native_seconds, "matched native time")
    reference = positive(reference_lower_seconds, "qualified reference lower time")
    with localcontext() as context:
        context.prec = 40
        achieved = [n / (cards * native * 10**12) for n in (low, high)]
        upper = high / (cards * reference * 10**12)
        efficiency = 100 * reference / native
    return dict(F_interval=list(work_interval), T_seconds=str(native),
        reference_seconds_lower=str(reference), reference_seconds_exact=None,
        achieved_tflops_per_card_interval=[str(n) for n in achieved],
        reference_tflops_per_card_upper=str(upper), efficiency_lower_percent=str(efficiency),
        efficiency_exact_percent=None, hardware_peak_utilization_percent=None,
        statistic="Same actual necessary F cancels: eta >=100*T_reference_lower/T_native; no fastest repetition or independent-F endpoint division.",
        source_and_timing_qualification_required=True, full_model_certificate=False)


def pair(baseline, current):
    b = list(map(Decimal, baseline["achieved_tflops_per_card_interval"]))
    c = list(map(Decimal, current["achieved_tflops_per_card_interval"]))
    with localcontext() as context:
        context.prec = 40
        gain = [100 * (c[0] / b[1] - 1), 100 * (c[1] / b[0] - 1)]
    efficiency = positive(current["efficiency_lower_percent"], "current efficiency lower bound")
    return dict(arms=dict(B0=baseline, B1=current),
        effective_compute_gain_percent_interval=[str(n) for n in gain],
        efficiency_lower_observed_percent=float(efficiency),
        efficiency_lower_observed_percent_decimal=str(efficiency),
        milestone_met=efficiency >= 50, final_numeric_met=efficiency >= 90,
        milestone_decision="达成目标" if efficiency >= 50 else "未达成目标",
        final_decision="达成目标" if efficiency >= 90 else "未达成目标",
        below_threshold_interpretation="A lower bound below a threshold does not prove the unknown exact efficiency is below it.",
        necessary_work_interval_preserved=True, historical_counts_or_times_reused=False,
        full_model_certificate=False)


def aggregate(rows, *, expected_repetitions):
    """Preserve every matched repetition and divide sums, never average rates."""
    if (type(expected_repetitions) is not int or expected_repetitions < 1
            or len(rows) != expected_repetitions
            or any(type(r["repetition"]) is not int for r in rows)
            or [r["repetition"] for r in rows] != list(range(1, expected_repetitions + 1))):
        raise ValueError("Missing, duplicate or out-of-order formal repetitions")
    for row in rows:
        reproduced = arm(row["F_interval"], row["T_seconds"], row["reference_seconds_lower"])
        if any(row.get(key) != value for key, value in reproduced.items()):
            raise ValueError("Per-repetition reference statistics do not reproduce")
    with localcontext() as context:
        context.prec = 40
        work = [sum(row["F_interval"][index] for row in rows) for index in (0, 1)]
        elapsed = sum(positive(row["T_seconds"], "native repetition") for row in rows)
        reference = sum(positive(row["reference_seconds_lower"], "reference repetition") for row in rows)
    result = arm(work, elapsed, reference)
    result.update(repetitions=rows, formal_repetitions=expected_repetitions,
        aggregation="sum matched necessary work, native time and reference lower time before division")
    return result
