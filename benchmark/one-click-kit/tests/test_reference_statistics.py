"""Synthetic arithmetic tests; no benchmark measurement is generated here."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reference_statistics import efficiency, statistics  # noqa:E402


def data():
    repeat = dict(
        name="fixture",
        F_interval=[10**12, 10**12],
        Tref_lower_seconds="10",
        requests=200,
        input_tokens=282265,
        output_tokens=51200,
        synchronized_ranks=2,
        uninstrumented=True,
        source_sha256="a" * 64,
    )
    cache = statistics(
        dict(
            benchmark_commit="b" * 40,
            manifest_sha256="c" * 64,
            m01=dict(
                evidence_qualified=True,
                chips=2,
                arms=[
                    dict(role="B1", repetitions=[deepcopy(repeat) for _ in range(3)])
                ],
            ),
        )
    )
    numerator = dict(
        reference_domain_sha256=cache["domain_sha256"],
        scope=deepcopy(cache["scope"]),
        role="B1",
        effective_flops_interval=[3 * 10**12, 3 * 10**12],
        repetitions=[
            dict(T_seconds="20", synchronized_ranks=2, uninstrumented=True)
            for _ in range(3)
        ],
    )
    return cache, numerator


def test_new_timing_changes_efficiency_using_same_cached_denominator():
    cache, n = data()
    assert float(efficiency(cache, n)["efficiency_percent_lower"]) == 50
    for row in n["repetitions"]:
        row["T_seconds"] = "40"
    assert float(efficiency(cache, n)["efficiency_percent_lower"]) == 25
    assert "efficiency_percent" not in str(cache)


def test_different_workload_cannot_borrow_cached_reference():
    cache, n = data()
    n["scope"]["max_completion_tokens"] = 8192
    with pytest.raises(ValueError, match="different workload"):
        efficiency(cache, n)


def test_changed_work_or_repeat_count_requires_revalidation():
    cache, n = data()
    n["effective_flops_interval"][0] -= 1
    with pytest.raises(ValueError, match="necessary-work"):
        efficiency(cache, n)
    cache, n = data()
    n["repetitions"].pop()
    with pytest.raises(ValueError, match="timing"):
        efficiency(cache, n)
