"""Retained prefixes, EOS trimming and per-request producer identity are distinct."""

import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "third-party-test-kit/draft_producer_geometry.py"
spec = importlib.util.spec_from_file_location("draft_producer_geometry", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(kept, offered=3, stale=False):
    draft = dict(call=0, wave=0, requests=["a" * 64], capacity=4,
        actual_input_tokens=2, first_pass_positions=[0, 1, 0, 0],
        first_pass_input_ids=[7, 8, 0, 0], sample_indices=[1],
        draft_token_ids=[[11, 12]], length_planes_before=[], length_planes_after=[])
    schedule = [dict(wave=0, rows=[], retained=[]), dict(wave=1,
        rows=[dict(request="a" * 64, draft=2, draft_token_ids=[11, 12])],
        retained=[dict(request="a" * 64, kept=kept, offered=offered,
                       kept_token_ids=[11, 12, 13][:kept], stale=stale)])]
    return draft, schedule


@pytest.mark.parametrize("kept,offered,stale,roots", [(1, 3, False, 1), (2, 3, False, 2),
    (3, 3, False, 2), (1, 1, False, 0), (0, 0, False, 0), (1, 3, True, 0)])
def test_only_native_retained_draft_prefix_creates_roots(kept, offered, stale, roots):
    draft, schedule = fixture(kept, offered, stale)
    result = module.bind_producers([draft], [copy.deepcopy(draft)], schedule)
    assert result["bindings"][0]["retained_drafts"] == roots
    assert result["effective_flops"] is None and not result["backward_state_closure_certified"]


@pytest.mark.parametrize("mutation", ["foreign_rank", "wrong_ids", "no_producer", "reused", "wrong_retained"])
def test_foreign_or_reused_producer_cannot_close(mutation):
    draft, schedule = fixture(2)
    other = copy.deepcopy(draft)
    if mutation == "foreign_rank":
        other["draft_token_ids"][0][0] = 42
    elif mutation == "wrong_ids":
        schedule[1]["rows"][0]["draft_token_ids"] = [42, 12]
    elif mutation == "no_producer":
        draft["wave"] = other["wave"] = 1
    elif mutation == "reused":
        schedule.append(dict(schedule[1], wave=2))
    else:
        schedule[1]["retained"][0]["kept_token_ids"] = [42, 12]
    with pytest.raises(ValueError):
        module.bind_producers([draft], [other], schedule)


def test_rank_local_route_filenames_are_not_a_producer_geometry_difference():
    draft, schedule = fixture(2)
    draft["routes"] = dict(file="rank-0-draft-0-routes.npz", sha256="a" * 64, layers=[40])
    other = copy.deepcopy(draft)
    other["routes"]["file"] = "rank-1-draft-0-routes.npz"
    assert module.bind_producers([draft], [other], schedule)["retained_second_drafts"] == 1
    other["routes"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="ranks disagree"):
        module.bind_producers([draft], [other], schedule)


def test_missing_retained_row_requires_an_already_observed_terminal_result():
    draft, schedule = fixture(2)
    schedule[1]['retained'][0]['stopped'] = True
    next_draft = copy.deepcopy(draft)
    next_draft.update(wave=1, call=1)
    schedule.append(dict(wave=2, rows=copy.deepcopy(schedule[1]['rows']), retained=[]))
    result = module.bind_producers([draft, next_draft], [copy.deepcopy(draft), copy.deepcopy(next_draft)], schedule)
    assert result['postcompletion_unretained_rows'] == 1
    assert result['retained_second_drafts'] == 1
    schedule[1]['retained'][0]['stopped'] = False
    with pytest.raises(ValueError, match='lacks a native retained'):
        module.bind_producers([draft, next_draft], [copy.deepcopy(draft), copy.deepcopy(next_draft)], schedule)
