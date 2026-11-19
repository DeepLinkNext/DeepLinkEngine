import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from canonical_workgroups import build
from public_request_work import derive


def request(identity, prompt, counts):
    return derive(dict(request_ordinal=identity, input_tokens=prompt,
        output_tokens=sum(counts), input_ids_sha256="a" * 64,
        output_ids_sha256="b" * 64, finish_reason="stop",
        ordered_public_emissions=[dict(public_step=i, emitted_tokens=n,
            finished=i == len(counts) - 1) for i, n in enumerate(counts)]))


def test_partitioned_prefill_and_finite_seats_preserve_all_dependencies():
    requests = {i: request(i, 7, [1, 3, 2]) for i in range(5)}
    result = build(requests, list(requests), {i: [4, 3] for i in requests},
                   budget_limit=8, max_active=2)
    assert result["totals"]["target_rows"] == 5 * 12
    assert result["totals"]["prefill_segments"] == 10
    assert result["totals"]["vocabulary_output_rows"] == 30
    assert all(w["active_requests"] <= 2 and w["scheduled_target_rows"] <= 8
               for w in result["waves"])
    assert len(result["admissions"]) == 5
    assert [a["previous_seat_owner"] for a in result["admissions"][:2]] == [None, None]
    for admission in result["admissions"][2:]:
        previous = admission["previous_seat_owner"]
        assert previous["terminal_wave"] < admission["wave"]
        assert requests[previous["request_id"]]
    for r in requests.values():
        entries = [e for w in result["waves"] for e in w["entries"]
                   if e["request_id"] == r["request_id"]]
        assert [e["canonical_target_interval"] for e in entries] == [[0, 4], [4, 7], [7, 10], [10, 12]]
    assert not result["source_runtime_applicability_certified"]
    assert not result["elapsed_time_used"]


def test_single_output_needs_no_draft_or_decode_wave():
    requests = {0: request(0, 2, [1])}
    result = build(requests, [0], {0: [2]}, budget_limit=2, max_active=1)
    assert len(result["waves"]) == 1
    assert result["totals"]["target_rows"] == 2
    assert result["totals"]["draft_kv_rows"] == 0


@pytest.mark.parametrize("parts", [[7, 1], [0, 7], [9, -2], [True, 6], []])
def test_incomplete_or_invalid_partition_is_rejected(parts):
    with pytest.raises(ValueError):
        build({0: request(0, 7, [1, 2])}, [0], {0: parts}, budget_limit=8, max_active=1)


def test_regrouping_cannot_silently_drop_a_draft_dependency():
    source = request(0, 7, [1, 3, 2])
    corrupt = copy.deepcopy(source)
    corrupt["draft"] = corrupt["draft"][1:]
    with pytest.raises(ValueError, match="lost or duplicated"):
        build({0: corrupt}, [0], {0: [7]}, budget_limit=8, max_active=1)
