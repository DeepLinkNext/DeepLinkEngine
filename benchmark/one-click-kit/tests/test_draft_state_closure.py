"""Literal cache histories independently verify compressed backward KV unions."""

import importlib.util
from pathlib import Path
import random

import pytest

path = Path(__file__).resolve().parents[1] / "third-party-test-kit/draft_state_closure.py"
spec = importlib.util.spec_from_file_location("draft_state_closure", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def literal(versions, roots):
    caches, prefix, full, vocabulary = [], set(), set(), set()
    for i, version in enumerate(versions):
        parent = version["parent"]
        cache = list(caches[parent]) if parent is not None else []
        assert len(cache) >= version["start"]
        cache = cache[:version["start"]] + [(i, p) for p in range(version["start"], version["stop"] + 1)]
        caches.append(cache)
    pending = [("vocabulary", root) for root in roots]
    while pending:
        kind, index = pending.pop()
        if kind == "vocabulary":
            if index not in vocabulary:
                vocabulary.add(index)
                pending.append(("full", index))
        elif kind == "full":
            if index not in full:
                full.add(index)
                for owner, pos in caches[index][:versions[index]["stop"] + 1]:
                    pending.append(("kv", (owner, pos)))
        else:
            owner, pos = index
            if (owner, pos) not in prefix:
                prefix.add((owner, pos))
                v = versions[owner]
                if v["input_hidden"] is not None:
                    pending.append(("full", v["input_hidden"]))
                if v["input_token"] is not None:
                    pending.append(("vocabulary", v["input_token"]))
    return prefix, full, vocabulary


@pytest.mark.parametrize("seed", list(range(12)))
def test_compressed_prefix_versions_equal_literal_last_writer_history(seed):
    rng = random.Random(seed)
    versions, latest = [], None
    for call in range(10):
        start = 0 if latest is None else rng.randint(max(0, versions[latest]["stop"] - 2), versions[latest]["stop"] + 1)
        stop = start + rng.randint(0, 4)
        first = len(versions)
        versions.append(dict(call=call, request="same-request", step=0, row=0,
            start=start, stop=stop, parent=latest, input_hidden=None, input_token=None))
        versions.append(dict(call=call, request="same-request", step=1, row=0,
            start=stop + 1, stop=stop + 1, parent=first, input_hidden=first, input_token=first))
        latest = first + 1
    roots = rng.sample(list(range(len(versions))), 5)
    expected, full, vocabulary = literal(versions, roots)
    result = module.backward(versions, roots)
    actual = {(row["version"], p) for row in result["kv_intervals"] for p in range(row["start"], row["stop"] + 1)}
    assert actual == expected
    assert result["necessary_kv_prefix_rows"] == len(expected)
    assert result["full_versions"] == sorted(full)
    assert result["vocabulary_versions"] == sorted(vocabulary)
    assert result["effective_flops"] is None


def test_missing_state_is_rejected_instead_of_counted_as_zero():
    bad = dict(call=0, request="r", step=0, row=0, start=5, stop=7,
               parent=None, input_hidden=None, input_token=None)
    with pytest.raises(ValueError, match="no matching"):
        module.backward([bad], [0])


def call(ordinal, positions):
    end = positions[-1]
    planes = [dict(step=step, query_end_envelope=[len(positions)],
                   actual_kv_lengths=[end + 1 + step]) for step in (0, 1)]
    return dict(call=ordinal, requests=["r"], capacity=len(positions),
        sample_indices=[len(positions) - 1], first_pass_positions=positions,
        draft_width=2, draft_token_ids=[[11, 12]],
        length_planes_before=planes, length_planes_after=planes)


def test_accepted_second_step_keeps_prefix_kv_but_excludes_overwritten_draft_suffix():
    calls = [call(0, [0, 1]), call(1, [2, 3])]
    roots = dict(bindings=[dict(producer_call=1, request="r", retained_drafts=2)])
    result = module.close(calls, roots)
    assert result["necessary_kv_prefix_rows"] == 5
    assert result["necessary_full_rows"] == 2
    assert result["necessary_vocabulary_rows"] == 2
    # The old second-step write at position2 is overwritten by the next
    # first pass. Its full layer and vocabulary never become ancestors.
    assert result["full_versions"] == [2, 3]
    assert [r["version"] for r in result["kv_intervals"]] == [0, 2, 3]


def test_length_plane_lifetime_cannot_be_selected_after_seeing_a_result():
    calls = [call(0, [0, 1])]
    calls[0]["length_planes_after"] = []
    with pytest.raises(ValueError, match="read lifetime"):
        module.close(calls, dict(bindings=[]))


def test_selected_root_uses_its_causal_prefix_inside_full_padded_query_segment():
    item = call(0, [0, 1, 2])
    item['first_pass_kv_extent_contract'] = 'full_query_envelope'
    item['sample_indices'] = [0]
    for planes in (item['length_planes_before'], item['length_planes_after']):
        planes[1]['actual_kv_lengths'] = [2]
    closure = module.close([item], dict(bindings=[dict(producer_call=0, request='r', retained_drafts=2)]))
    assert closure['necessary_kv_prefix_rows'] == 2
    assert closure['necessary_full_rows'] == 2
    assert closure['necessary_full_causal_pairs'] == 3
    # The physical first pass also writes positions1/2. Only position0 is
    # needed by its selected logit; step1 replaces position1 using that root.
    item['length_planes_before'][0]['actual_kv_lengths'] = [1]
    with pytest.raises(ValueError, match='device-authoritative'):
        module.close([item], dict(bindings=[]))
