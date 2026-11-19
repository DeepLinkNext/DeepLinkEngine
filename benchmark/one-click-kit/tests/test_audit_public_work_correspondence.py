import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from audit_public_work_correspondence import audit


def draft(call, wave, positions, sample, tokens):
    end = positions[sample]
    planes = [dict(step=s, query_end_envelope=[len(positions)], actual_kv_lengths=[end + 1 + s]) for s in (0, 1)]
    return dict(call=call, wave=wave, requests=["r"], capacity=len(positions), actual_input_tokens=len(positions),
        sample_indices=[sample], first_pass_positions=positions, draft_width=2, draft_token_ids=[tokens],
        length_planes_before=planes, length_planes_after=planes)


def wave(index, start, count, drafted, kept, *, offered=None, stopped=False):
    return dict(wave=index, rows=[dict(request="r", start=start, count=count,
        draft=len(drafted), draft_token_ids=drafted)], retained=[dict(request="r", kept=len(kept),
        offered=len(kept) if offered is None else offered, stopped=stopped, stale=False, kept_token_ids=kept)])


def test_rejected_suffix_is_overwritten_before_a_later_retained_second_draft():
    # Actual call0 produces11/12, both rejected. Call1 rewrites position2,
    # selecting that row out of a padded three-position first pass.
    drafts = [draft(0, 0, [0, 1], 1, [11, 12]), draft(1, 1, [2, 3, 4], 0, [14, 15])]
    waves = [wave(0, 0, 2, [], [10]), wave(1, 2, 3, [11, 12], [13]),
             wave(2, 3, 3, [14, 15], [14, 15, 16], stopped=True)]
    result = audit([dict(request="r", input_tokens=2, output_tokens=5)], waves, drafts, copy.deepcopy(drafts))
    actual = result["requests"][0]["actual_observed"]
    assert actual == dict(necessary_kv_prefix_rows=4, necessary_full_rows=2,
                          necessary_vocabulary_rows=2, necessary_full_causal_pairs=7)
    assert result["all_request_closures_contained"]
    assert not result["clean_native_counts_or_times_replaced"]


def test_terminal_eos_trim_is_a_real_interval_not_forced_to_lower_endpoint():
    drafts = [draft(0, 0, [0, 1], 1, [11, 12])]
    waves = [wave(0, 0, 2, [], [10]), wave(1, 2, 3, [11, 12], [11], offered=2, stopped=True)]
    result = audit([dict(request="r", input_tokens=2, output_tokens=2)], waves, drafts, copy.deepcopy(drafts))
    row = result["requests"][0]
    assert row["public_lower"]["necessary_vocabulary_rows"] == 0
    assert row["public_upper"]["necessary_vocabulary_rows"] == row["actual_observed"]["necessary_vocabulary_rows"] == 1


def test_observed_producer_identity_corruption_cannot_be_hidden_by_matching_counts():
    drafts = [draft(0, 0, [0, 1], 1, [11, 12])]
    waves = [wave(0, 0, 2, [], [10]), wave(1, 2, 3, [12, 11], [12, 11, 13], stopped=True)]
    with pytest.raises(ValueError, match="mismatched"):
        audit([dict(request="r", input_tokens=2, output_tokens=4)], waves, drafts, copy.deepcopy(drafts))
