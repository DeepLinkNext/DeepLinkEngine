"""CANN stdout warnings must not corrupt the structured input-builder result."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/continuous_inputs.py"


def command(tmp_path, recipe):
    deployment = tmp_path / "sdk"
    entry = deployment / "runtime/entrypoints/prepare_continuous.py"
    entry.parent.mkdir(parents=True)
    entry.write_text(recipe)
    result = tmp_path / "result.json"
    out = tmp_path / "inputs"
    return [sys.executable, "-B", str(SCRIPT), "--deployment", str(deployment), "--out", str(out), "--result", str(result)], result, out


def test_warning_without_newline_and_trailing_log(tmp_path):
    cmd, result, out = command(tmp_path, "from pathlib import Path\nprint('[LOG_WARNING] unable to write log', end='')\nPath(__file__).with_name('continuous-inputs-r1.json').write_text('{\"real\":1}')\nprint('trailing runtime diagnostic')\nfor a,b in []:\n    raise RuntimeError('historical launch must not execute')\n")
    done = subprocess.run(cmd, text=True, capture_output=True, check=True)
    assert done.stdout.startswith("[LOG_WARNING]")
    assert "trailing runtime diagnostic" in done.stdout
    data = json.loads(result.read_bytes())
    assert data["path"] == str(out / "continuous-inputs-r1.json")
    assert data["sha256"] == hashlib.sha256(b'{"real":1}').hexdigest()
    # A reused result cannot silently replace an earlier attempt or run recipe.
    prior = result.read_bytes()
    repeat = subprocess.run(cmd, text=True, capture_output=True)
    assert repeat.returncode != 0 and repeat.stdout == ""
    assert result.read_bytes() == prior


def test_builder_failure_keeps_nonzero_and_diagnostics(tmp_path):
    cmd, result, _ = command(tmp_path, "print('native failure marker')\nraise RuntimeError('failed before producing inputs')\n")
    done = subprocess.run(cmd, text=True, capture_output=True)
    assert done.returncode != 0
    assert "native failure marker" in done.stdout
    assert "failed before producing inputs" in done.stderr
    assert result.read_bytes() == b""
