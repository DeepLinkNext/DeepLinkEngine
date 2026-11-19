"""Independent scalar cache replay checks public-emission closure endpoints."""

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from public_request_work import bounds, derive


def row(prompt, counts):
    return dict(request_ordinal=0, input_tokens=prompt, output_tokens=sum(counts),
        input_ids_sha256="a" * 64, output_ids_sha256="b" * 64, finish_reason="stop",
        ordered_public_emissions=[dict(public_step=i * 2, emitted_tokens=n,
            finished=i == len(counts) - 1) for i, n in enumerate(counts)])


def scalar_replay(prompt, counts, upper):
    latest, nodes, roots = {}, {}, []
    start, selected = 0, prompt - 1
    for stage, count in enumerate(counts[1:]):
        stop = prompt if stage == 0 else start + 3
        for position in range(start, stop):
            key = stage, 0, position
            latest[position] = key
            nodes[key] = dict(parent=None, reads=None)
        first = stage, 0, selected
        nodes[first]["reads"] = [latest[p] for p in range(selected + 1)]
        second = stage, 1, selected + 1
        latest[selected + 1] = second
        nodes[second] = dict(parent=first,
                             reads=[latest[p] for p in range(selected + 2)])
        retained = min(2, count) if upper and stage == len(counts) - 2 else count - 1
        roots.extend([first, second][:retained])
        start, selected = selected + 1, selected + count
    full, kv, todo = set(), set(), [("full", root) for root in roots]
    while todo:
        kind, key = todo.pop()
        if kind == "full":
            if key in full:
                continue
            full.add(key)
            todo.extend(("kv", dep) for dep in nodes[key]["reads"])
        else:
            if key in kv:
                continue
            kv.add(key)
            if nodes[key]["parent"] is not None:
                todo.append(("full", nodes[key]["parent"]))
    return kv, full


@pytest.mark.parametrize("prompt", [1, 2, 7])
def test_all_small_emission_sequences_match_independent_cache_replay(prompt):
    for steps in range(5):
        for counts in itertools.product((1, 2, 3), repeat=steps):
            counts = [1, *counts]
            for upper in (False, True):
                result = derive(row(prompt, counts), terminal="upper" if upper else "lower")
                kv, full = scalar_replay(prompt, counts, upper)
                got_kv = {(g["local_stage"], g["draft_step"], p)
                          for g in result["draft"]
                          for a, b in g["kv_position_intervals"] for p in range(a, b)}
                got_full = {(g["local_stage"], g["draft_step"], q["position"])
                            for g in result["draft"] for q in g["full_queries"]}
                assert got_kv == kv
                assert got_full == full
                assert result["closure"]["necessary_full_causal_pairs"] == sum(p + 1 for _, _, p in full)
                assert result["necessary_target_rows"] == prompt + sum(counts) - 1
                assert not result["source_runtime_applicability_certified"]


def test_terminal_ambiguity_keeps_both_branches_and_time_outside_geometry():
    source = row(7, [1, 3, 1, 2])
    result = bounds(source)
    assert result["lower"]["retained_output_roots"] == 3
    assert result["upper"]["retained_output_roots"] == 4
    source["seconds"] = -999
    assert bounds(source) == result
    assert result["efficiency_percent"] is None


@pytest.mark.parametrize("change", [
    lambda r: r["ordered_public_emissions"][-1].update(finished=False),
    lambda r: r["ordered_public_emissions"][0].update(finished=True),
    lambda r: r["ordered_public_emissions"][-1].update(emitted_tokens=0),
    lambda r: r["ordered_public_emissions"][-1].update(emitted_tokens=4),
    lambda r: r["ordered_public_emissions"][-1].update(public_step=0),
    lambda r: r.update(output_tokens=99),
    lambda r: r.update(finish_reason="abort"),
])
def test_ambiguous_or_incomplete_publication_is_rejected(change):
    source = row(2, [1, 2])
    change(source)
    with pytest.raises(ValueError):
        bounds(source)
