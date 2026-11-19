"""CANN diagnostics may have no newline; admission must use an exclusive JSON file."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import engine_entry


def test_non_newline_cann_stdout_does_not_corrupt_admission(tmp_path, monkeypatch):
    record = dict(devices=[dict(device=i, bytes=32 * 2**20, pinned=True, h2d_d2h_equal=True)
                           for i in (0, 1)], model_loaded=False, memlock=[65536, 65536])
    def child(command, **kwargs):
        assert Path(command[-1]) == tmp_path / "pinned-memory-preflight.json"
        Path(command[-1]).write_text(json.dumps(record))
        return SimpleNamespace(returncode=0, stdout="[LOG_WARNING] no newline.HOST_PIN_ADMISSION_FILE result", stderr="")
    monkeypatch.setattr(engine_entry.subprocess, "run", child)
    engine_entry.check_pinned_host_memory(tmp_path, {})
    assert json.loads((tmp_path / "pinned-memory-preflight.json").read_bytes()) == record
    with pytest.raises(FileExistsError, match="Preserve"):
        engine_entry.check_pinned_host_memory(tmp_path, {})


def test_stdout_success_cannot_replace_a_missing_device_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_entry.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="HOST_PIN_ADMISSION {}", stderr=""))
    with pytest.raises(ValueError, match="file missing"):
        engine_entry.check_pinned_host_memory(tmp_path, {})
