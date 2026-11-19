"""Recompute physical-card utilization from frozen raw sampler clocks."""

import bisect
import math
import re


def parse_cards(raw, expected=("5", "6")):
    values, card = {}, None
    for line in raw.splitlines():
        cols = [x.strip() for x in line.strip("|").split("|")]
        if len(cols) < 3:
            continue
        match = re.fullmatch(r"([0-9]+)\s+910B2", cols[0])
        if match:
            card = match[1] if match[1] in expected else None
        elif card and cols[0] == "0" and re.fullmatch(r"[0-9a-fA-F:.]+", cols[1]):
            value = float(cols[2].split()[0])
            if not 0 <= value <= 100 or card in values:
                raise ValueError("Invalid or duplicate physical-card sample")
            values[card], card = value, None
    if set(values) != set(expected):
        raise ValueError("Both physical910B2 cards are required")
    return values


def utilization(rows, start, end, expected=("5", "6")):
    if not (math.isfinite(start) and math.isfinite(end) and start < end):
        raise ValueError("Invalid request interval")
    samples = []
    for row in rows:
        if row.get("returncode", 0):
            raise ValueError("Failed device sampler")
        left, right = row["monotonic_start"], row["monotonic_end"]
        if not (math.isfinite(left) and math.isfinite(right) and left <= right):
            raise ValueError("Invalid sample clock interval")
        samples.append(((left + right) / 2, parse_cards(row["raw"], expected)))
    samples.sort(key=lambda x: x[0])
    times = [x[0] for x in samples]
    if len(set(times)) != len(times):
        raise ValueError("Duplicate sample clocks")
    cuts = [start] + [t for t in times if start < t < end] + [end]
    totals, covered, missing, max_hold = {card: 0.0 for card in expected}, 0.0, 0.0, 0.0
    for left, right in zip(cuts, cuts[1:]):
        index = bisect.bisect_right(times, left) - 1
        duration = right - left
        if index < 0:
            missing += duration
            continue
        covered += duration
        max_hold = max(max_hold, right - times[index])
        for card in totals:
            totals[card] += samples[index][1][card] * duration
    if covered <= 0:
        raise ValueError("No observed utilization interval")
    means = {k: v / covered for k, v in totals.items()}
    return {
        "per_card_percent": means,
        "mean_percent": sum(means.values()) / 2,
        "covered_seconds": covered,
        "missing_seconds": missing,
        "max_hold_age_seconds": max_hold,
        "method": "Sampler clock midpoint; request interval clipped; left-hold, time weighted. A sample estimate, not continuous hardware truth.",
    }
