"""Selection uses original history and declared lengths, never padding or outcomes."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from acceptance_inputs import select
from agent_trace import InvalidPublishedRequest


def test_first_common_band_preserves_bodies_and_ignores_source_scores():
    source = [
        dict(
            source_index=i,
            body=dict(messages=[dict(role="user", content=str(n))]),
            reward=i % 2,
        )
        for i, n in enumerate((3000, 5000, 6000, 20000, 33000, 40000))
    ]
    original = copy.deepcopy(source)
    selected, observed = select(
        source, lambda body: [1] * int(body["messages"][0]["content"])
    )
    assert [selected[k]["source_index"] for k in ("4096", "16384", "32768")] == [
        1,
        3,
        4,
    ]
    assert len(observed) == 5 and source == original
    assert selected["4096"]["source_body"] == source[1]["body"]
    source[1]["body"]["messages"][0]["content"] = "changed"
    assert selected["4096"]["source_body"]["messages"][0]["content"] == "5000"


def test_missing_or_invalid_history_is_not_repaired_into_a_context_case():
    rows = [
        dict(source_index=0, invalid_reason="missing_published_query"),
        dict(source_index=1, body=dict(messages=[dict(role="user", content="bad")])),
    ]

    def encode(_):
        raise InvalidPublishedRequest("invalid_published_tool_arguments")

    selected, observations = select(rows, encode)
    assert selected == {} and len(observations) == 2
    assert [r["excluded"] for r in observations] == [
        "missing_published_query",
        "invalid_published_tool_arguments",
    ]
