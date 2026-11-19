import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from qualify_native_capacity import PROTOCOL, PROTOCOL_SHA256, sha, validate_artifacts


def inputs():
    protocol = json.loads(PROTOCOL.read_bytes())
    args = dict(protocol["arguments"], additional_config=copy.deepcopy(protocol["additional"]))
    reset = dict(reset=True, last_snapshot=dict(pending=[], host=[], seats=[dict(owner=None)] * 54))
    # The fixed total is checked independently of the source ownership proof.
    work = dict(request_token_ids=[[1] * 1411 for _ in range(199)] + [[1] * 1476])
    log = "GPU KV cache size: 1,447,936 tokens, Maximum concurrency for 65,536 tokens per request: 22.09x"
    return args, reset, work, log, protocol


def test_reviewed_proof_and_full_output_envelope_are_bound_without_self_certification():
    assert sha(PROTOCOL) == PROTOCOL_SHA256
    values = inputs()
    result = validate_artifacts(*values)
    assert result["sufficient_page_capacity"]
    assert result["maximum_output"] == 8192
    assert not result["no_preemption_certified"]


@pytest.mark.parametrize("change", ["pending_dma", "owner", "paused", "old_budget", "other_pool", "conflicting_pool", "other_source_domain"])
def test_incompatible_or_incomplete_capacity_artifacts_are_rejected(change):
    args, reset, work, log, protocol = copy.deepcopy(inputs())
    if change == "pending_dma":
        reset["last_snapshot"]["pending"] = [1]
    elif change == "owner":
        reset["last_snapshot"]["seats"][0] = dict(owner="live")
    elif change == "paused":
        reset["last_snapshot"]["partial_reclaim"] = dict(paused=["live"])
    elif change == "old_budget":
        work["request_token_ids"].pop()
    elif change == "other_pool":
        log = "GPU KV cache size: 128 tokens, Maximum concurrency for 65,536 tokens per request: 0.00x"
    elif change == "conflicting_pool":
        log += "\nGPU KV cache size: 1,000,000 tokens, Maximum concurrency for 65,536 tokens per request: 1x"
    else:
        args["additional_config"]["state_resident_seats"] = 55
    with pytest.raises(ValueError):
        validate_artifacts(args, reset, work, log, protocol)
