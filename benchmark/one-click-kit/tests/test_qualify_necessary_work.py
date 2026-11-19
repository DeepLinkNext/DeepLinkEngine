import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit"))
from qualify_necessary_work import PROTOCOL, PROTOCOL_SHA256, sealed_read, sha


def test_all_reviewed_formula_sources_are_pinned_in_the_full_arithmetic_proof():
    assert sha(PROTOCOL) == PROTOCOL_SHA256
    protocol = json.loads(PROTOCOL.read_bytes())
    root = PROTOCOL.parent.parent / "third-party-test-kit"
    for name, pin in protocol["formula_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == pin
    assert len(protocol["claims"]) >= 8
    assert protocol["necessary_work_interval_allowed"]


def test_unsealed_derived_arithmetic_and_external_links_are_rejected(tmp_path):
    target = tmp_path / "facts.json"
    target.write_text('{"count": 2}')
    (tmp_path / "checksums.json").write_text(json.dumps({"facts.json": sha(target)}))
    assert sealed_read(tmp_path, "facts.json") == dict(count=2)
    target.write_text('{"count": 3}')
    with pytest.raises(ValueError, match="unsealed"):
        sealed_read(tmp_path, "facts.json")
    external = tmp_path.parent / (tmp_path.name + "-external.json")
    external.write_text('{"count": 2}')
    target.unlink()
    target.symlink_to(external)
    with pytest.raises(ValueError, match="unsealed"):
        sealed_read(tmp_path, "facts.json")
