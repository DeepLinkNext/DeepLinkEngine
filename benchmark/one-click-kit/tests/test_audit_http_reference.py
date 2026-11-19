import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_http_reference import run
from dataset_contract import sha


@pytest.mark.parametrize("failure,quiescent", [("Client failed after195 responses", True),
    ("ValueError: The two TP ranks disagree on real draft ownership/geometry", False)])
def test_offline_recovery_never_promotes_incomplete_or_live_observations(tmp_path, failure, quiescent):
    source = tmp_path / "source"
    source.mkdir()
    receipt = source / "receipt.json"
    receipt.write_text(json.dumps(dict(role="B1", completed=False, quiescent=quiescent, failure=failure)))
    (source / "checksums.json").write_text(json.dumps({"receipt.json": sha(receipt)}))
    original = receipt.read_bytes()
    with pytest.raises(ValueError, match="recoverable quiescent"):
        run(source, tmp_path / "sdk", tmp_path / "new-derivation")
    assert receipt.read_bytes() == original
    assert not (tmp_path / "new-derivation/receipt.json").exists()


def test_raw_recovery_rejects_a_changed_original_receipt_before_any_derived_work(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    receipt = source / "receipt.json"
    receipt.write_text("{}")
    (source / "checksums.json").write_text(json.dumps({"receipt.json": sha(receipt)}))
    receipt.write_text('{"completed": true}')
    with pytest.raises(ValueError, match="changed or is unsealed"):
        run(source, tmp_path / "sdk", tmp_path / "new-derivation")
