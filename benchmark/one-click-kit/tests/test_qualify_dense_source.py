import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from qualify_dense_source import PROTOCOL, PROTOCOL_SHA256, digest, sha, validate_domain
from test_public_request_work import row


def inputs():
    protocol = json.loads(PROTOCOL.read_bytes())
    text = copy.deepcopy(protocol["model_text_fields"])
    args = dict(protocol["arguments"], max_num_seqs=36)
    rows = [row(1411, [1, 3, 2]) for _ in range(199)] + [row(1476, [1, 3, 2])]
    # Stagger the common first-token step so a wave never exceeds E36.
    for i, item in enumerate(rows):
        item["request_ordinal"] = i
        for j, event in enumerate(item["ordered_public_emissions"]):
            event["public_step"] = i if j == 0 else 200 + i * 2 + j
    signature = dict(requests=rows, maximum_output_tokens=8192, speculative_tokens=2, public_steps=602)
    geometry = dict(schema="native-public-output-geometry/v1", **signature, geometry_sha256=digest(signature))
    capacity = dict(source_ownership_qualified=True, no_preemption_certified=True,
                    no_external_prefix_reuse_certified=True)
    return text, args, geometry, capacity, protocol


def test_reviewed_domain_retains_all_tail_deliveries_at_exact_shapes():
    assert sha(PROTOCOL) == PROTOCOL_SHA256
    tail, cases = validate_domain(*inputs())
    assert tail["target_rows"] == 200 * 5
    assert {c["m"] for c in cases} == {2, 3}
    assert {c["n"] for c in cases} == {32, 2048, 4608, 6144}
    # T may vary without changing prospective source geometry or case choice.
    values = inputs()
    values[2]["T_seconds"] = .00001
    assert validate_domain(*values) == (tail, cases)


@pytest.mark.parametrize("change", ["width", "quant", "sequence_parallel", "capacity", "geometry_hash", "population"])
def test_unreviewed_domain_cannot_be_promoted_by_shape_similarity(change):
    text, args, geometry, capacity, protocol = inputs()
    if change == "width":
        text["hidden_size"] = 4096
    elif change == "quant":
        args["quantization"] = "fp8"
    elif change == "sequence_parallel":
        args["enable_expert_parallel"] = True
    elif change == "capacity":
        capacity["no_preemption_certified"] = False
    elif change == "geometry_hash":
        geometry["requests"][0]["output_tokens"] += 1
    else:
        geometry["requests"].pop()
    with pytest.raises(ValueError):
        validate_domain(text, args, geometry, capacity, protocol)
