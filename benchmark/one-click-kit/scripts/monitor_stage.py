"""Find the active leaf arm without changing any campaign evidence."""

import json
from pathlib import Path
import re


def read(path):
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return {}


def locate(run, stage, campaign=None):
    run = Path(run)
    if not stage:
        failed = [s["name"] for s in (campaign or {}).get("stages", []) if s.get("status") == "failed"]
        stage = failed[-1] if failed else None
    if stage is None:
        return None, None
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", stage):
        raise ValueError("Unsafe monitor stage")
    parent = run / stage
    if stage != "mmlu-pair":
        return parent, stage
    arms = [(parent / arm, read(parent / arm / "receipt.json")) for arm in ("B0", "B1")]
    active = [(p, r) for p, r in arms if r and r.get("status") not in ("completed", "failed", "cancelled")]
    if active:
        selected = active[-1][0]
    else:
        attempted = [(p, r) for p, r in arms if r]
        selected = attempted[-1][0] if attempted else parent
    # POSIX form keeps nested stage checks such as "mmlu-pair/" valid on Windows.
    return selected, selected.relative_to(run).as_posix()
