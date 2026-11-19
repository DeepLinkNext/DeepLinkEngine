"""Necessary expert rows follow actual ordering and latest cache producers."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from route_work_geometry import target_selection
from target_state_closure import close


def wave(index, rows):
    cu = [0]
    for row in rows: cu.append(cu[-1] + row["count"])
    return dict(wave=index, rows=list(reversed(rows)), gdn=[dict(
        requests=[row["request"] for row in rows], target_query_cumulative_lengths=cu)])


def test_rejected_suffix_is_replaced_by_latest_real_position_producer():
    requests = [dict(request="a", input_tokens=3, output_tokens=3),
                dict(request="b", input_tokens=2, output_tokens=1)]
    target = close(requests, dict(kv_intervals=[]))
    waves = [wave(0, [dict(request="a", start=0, count=3), dict(request="b", start=0, count=2)]),
             wave(1, [dict(request="a", start=3, count=3)]),
             wave(2, [dict(request="a", start=4, count=3)])]
    selected = target_selection(waves, requests, target)
    assert selected[0] == dict(prefix=[0, 1, 2, 3, 4], final=[2, 4], actual_rows=5)
    assert selected[1] == dict(prefix=[0], final=[0], actual_rows=3)
    assert selected[2] == dict(prefix=[0], final=[0], actual_rows=3)
    assert sum(len(row["prefix"]) for row in selected.values()) == 7


def test_query_order_cannot_be_guessed_from_scheduler_insertion_order():
    requests = [dict(request="a", input_tokens=2, output_tokens=1)]
    target = close(requests, dict(kv_intervals=[]))
    data = wave(0, [dict(request="a", start=0, count=2)])
    data["gdn"][0]["target_query_cumulative_lengths"] = [0, 3]
    with pytest.raises(ValueError, match="actual scheduled"):
        target_selection([data], requests, target)


def test_missing_route_prefix_blocks_complete_shape_catalog():
    requests = [dict(request="a", input_tokens=3, output_tokens=1)]
    target = close(requests, dict(kv_intervals=[]))
    with pytest.raises(ValueError, match="producer is absent"):
        target_selection([wave(0, [dict(request="a", start=1, count=2)])], requests, target)
