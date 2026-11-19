import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from public_work_ledger import ledger
from test_public_request_work import row
from test_semantic_work import model


def architecture():
    result = model()
    result.vocab_size = 248320
    return result


def test_disjoint_ledger_has_every_family_but_does_not_self_certify():
    geometry = dict(requests=[row(7, [1, 3, 1, 2])], geometry_sha256="c" * 64)
    result = ledger(architecture(), geometry)
    low, high = result["F_interval"]
    assert 0 < low < high
    assert len(result["formula_families"]) == 8
    for branch in ("lower", "upper"):
        endpoint = result["endpoints"][branch]
        assert endpoint["total_basic"] == sum(endpoint["components"].values())
        assert endpoint["components"]["target_vocabulary"] == 2 * 2048 * 248320 * 7
        assert endpoint["components"]["target_gdn_recurrence"] == 6 * 30 * 32 * (7 * 128 * 128 + 128)
    assert not result["necessary_work_certified"]
    assert not result["full_model_certificate"]
    assert result["efficiency_percent"] is None


def test_request_local_identity_is_preserved_across_aggregation():
    first, second = row(3, [1, 3, 3]), row(5, [1, 1, 2])
    second["request_ordinal"] = 1
    geometry = dict(requests=[first, second], geometry_sha256="c" * 64)
    before = copy.deepcopy(geometry)
    result = ledger(architecture(), geometry)
    assert geometry == before
    assert result["requests"] == 2
    assert result["endpoints"]["lower"]["target_population"]["necessary_prefix_rows"] == 17
    geometry["T_seconds"] = .00001
    assert ledger(architecture(), geometry) == result


def test_first_token_only_has_no_speculative_or_recurrent_arithmetic():
    result = ledger(architecture(), dict(requests=[row(1, [1])], geometry_sha256="c" * 64))
    assert result["F_interval"][0] == result["F_interval"][1]
    parts = result["endpoints"]["lower"]["components"]
    assert parts["target_gdn_recurrence"] == 0
    assert all(value == 0 for key, value in parts.items() if key.startswith("draft"))
