"""Async placeholders require independent device inputs, never wildcard IDs."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from resolved_http_inputs import resolve_records
from reference_evidence import post_capture_fix_only
from dataset_contract import sha


def records():
    request = "a" * 64
    hosts, drafts = [], []
    for wave, start, ids, offered in [
        (0, 0, [7, 8], 1),
        (1, 2, [11, 12, 13], 3),
        (2, 5, [21, 22, 23], None),
    ]:
        count = len(ids)
        kept = (
            []
            if offered is None
            else [
                dict(
                    request=request,
                    offered=offered,
                    kept=offered,
                    kept_token_ids=[8] if wave == 0 else ids,
                    stale=False,
                    stopped=wave == 1,
                )
            ]
        )
        row = dict(
            request=request,
            start=start + (wave > 0),
            count=count,
            draft=0 if wave == 0 else 2,
            draft_token_ids=[] if wave == 0 else [-1, -1],
        )
        hosts.append(dict(window="w", wave=wave, rows=[row], retained=kept))
        drafts.append(
            dict(
                window="w",
                wave=wave,
                call=wave,
                capacity=4,
                actual_input_tokens=count,
                requests=[request],
                sample_indices=[count - 1],
                first_pass_positions=list(range(start, start + count)),
                first_pass_input_ids=ids,
                draft_token_ids=[[11, 12] if wave == 0 else [21, 22]],
                length_planes_before=[dict(step=0, query_end_envelope=[count])],
                length_planes_after=[dict(step=0, query_end_envelope=[count])],
            )
        )
    for call in drafts:
        end = call["first_pass_positions"][-1] + 1
        call["length_planes_before"][0]["actual_kv_lengths"] = [end]
        call["length_planes_before"].append(
            dict(step=1, query_end_envelope=[1], actual_kv_lengths=[end + 1])
        )
        call["length_planes_after"] = copy.deepcopy(call["length_planes_before"])
    return (
        hosts,
        copy.deepcopy(hosts),
        copy.deepcopy(hosts),
        drafts,
        copy.deepcopy(drafts),
    )


def test_real_device_positions_and_prefixes_resolve_without_mutating_any_input():
    streams = records()
    original = copy.deepcopy(streams)
    resolved, changes = resolve_records(*streams)
    assert streams == original
    assert len(changes) == 2
    assert all(s[1]["rows"][0]["start"] == 2 for s in resolved[:3])
    assert all(s[1]["rows"][0]["draft_token_ids"] == [11, 12] for s in resolved[:3])
    assert changes[-1]["retained_output_absent_after_completed"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "mixed",
        "concrete",
        "peer",
        "position",
        "input",
        "retained",
        "count",
        "ownership",
        "offered",
        "unstopped",
        "planes",
        "truncated",
        "producer_gap",
    ],
)
def test_unobserved_or_inconsistent_producer_cannot_resolve(mutation):
    streams = records()
    h, left, r, d, p = streams
    if mutation in ("mixed", "concrete"):
        for s in (h, left, r):
            s[1]["rows"][0]["draft_token_ids"] = (
                [-1, 12] if mutation == "mixed" else [99, 12]
            )
    elif mutation == "peer":
        p[0]["draft_token_ids"][0][0] = 99
    elif mutation == "position":
        for s in (d, p):
            s[1]["first_pass_positions"] = [3, 4, 5]
    elif mutation == "input":
        for s in (d, p):
            s[1]["first_pass_input_ids"][0] = 99
    elif mutation == "retained":
        h[1]["retained"][0]["kept_token_ids"][0] = 99
    elif mutation == "count":
        for s in (h, left, r):
            s[1]["rows"][0]["count"] = 2
    elif mutation == "ownership":
        for s in (d, p):
            s[1]["requests"][0] = "b" * 64
    elif mutation == "offered":
        h[1]["retained"][0]["offered"] = 2
    elif mutation == "unstopped":
        h[1]["retained"][0]["stopped"] = False
    elif mutation == "planes":
        for s in (d, p):
            s[1]["length_planes_after"][0]["query_end_envelope"] = [2]
    elif mutation == "truncated":
        p.pop()
    else:
        for s in (d, p):
            s[1]["call"] = 0
    with pytest.raises(ValueError):
        resolve_records(*streams)


def test_only_cpu_finish_tail_may_change_for_closed_capture_reuse(tmp_path):
    old = tmp_path / "old.py"
    new = tmp_path / "new.py"
    before = 'import sys\ndef boundary(x): return x\ndef finish(output, model):\n    stopped = boundary("end")\n    return 1\n'
    old.write_text(before)
    new.write_text(before.replace("return 1", "return 2"))
    assert post_capture_fix_only(old, new, sha(old))
    for bad in [
        before.replace('boundary("end")', 'boundary("begin")'),
        before.replace("import sys", "import os"),
        before.replace("return x", "return None"),
    ]:
        new.write_text(bad)
        assert not post_capture_fix_only(old, new, sha(old))


def test_only_source_ordered_second_step_actual_length_is_published():
    from resolved_http_inputs import resolve_length_publication

    call = records()[3][1]
    call["sample_indices"] = [0]
    call["length_planes_after"][1]["actual_kv_lengths"] = [4]
    original = copy.deepcopy(call)
    resolved = resolve_length_publication(call)
    assert call == original
    assert resolved["length_planes_before"] == resolved["length_planes_after"]
    assert resolved["length_planes_before"][0] == original["length_planes_before"][0]
    assert resolved["length_planes_before"][1]["actual_kv_lengths"] == [4]
    call["length_planes_after"][1]["actual_kv_lengths"] = [5]
    with pytest.raises(ValueError, match="frozen causal publication"):
        resolve_length_publication(call)


def test_actual_async_source_contract_itself_is_hash_pinned():
    from resolved_http_inputs import PROTOCOL, PROTOCOL_SHA256

    assert sha(PROTOCOL) == PROTOCOL_SHA256


def test_existing_failed_derivations_are_never_linked_or_overwritten(
    tmp_path, monkeypatch
):
    import json
    import resolved_http_inputs as inputs

    raw = tmp_path / "raw"
    raw.mkdir()
    streams = records()
    names = [
        "scheduler-waves.jsonl",
        "rank-0-waves.jsonl",
        "rank-1-waves.jsonl",
        "rank-0-drafts.jsonl",
        "rank-1-drafts.jsonl",
    ]
    for name, rows in zip(names, streams, strict=True):
        (raw / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (raw / "request-wave-order.json").write_text("original failed attempt")
    monkeypatch.setattr(inputs, "qualify", lambda *args: {"reviewed": True})
    view = tmp_path / "new-view"
    proof = inputs.prepare(raw, tmp_path / "sdk", view)
    assert proof["completed"] is True
    assert not (view / "request-wave-order.json").exists()
    assert all(not (view / name).is_symlink() for name in names)
    assert (raw / "request-wave-order.json").read_text() == "original failed attempt"
