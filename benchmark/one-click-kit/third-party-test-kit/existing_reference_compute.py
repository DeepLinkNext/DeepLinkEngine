"""Re-run benchmark's pinned M01 verifier on the original work and path archive."""

from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REVISION = "a3d96977005f7257502ded2938a051c70de60b02"
REPORT = Path("reports/m01-qualified-pair-20261103")
VERIFIER = Path("scripts/recompute_m01_qualified_pair.py")
VERIFIER_SHA = "97c5ab301a12ac10cc5d3b97a1c38d155e9a0862cbcc52650f0084ff3884220c"
MANIFEST_SHA = "5d1f4e6c50aa144bdc37ed293db9dd73fec5ab1c28788f91785a758146654cee"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def existing_rows(root, out, threshold):
    source = root.resolve()
    verifier = source / VERIFIER
    manifest = source / REPORT / "evidence/evidence-manifest.json"
    if sha(verifier) != VERIFIER_SHA or sha(manifest) != MANIFEST_SHA:
        raise ValueError("Existing M01 verifier or evidence manifest differs from the pinned benchmark source")
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONOPTIMIZE", "PYTHONPATH")}
    result = subprocess.run([sys.executable, "-I", "-B", str(verifier), "--check"],
                            cwd=source, env=env, capture_output=True, text=True)
    if result.returncode:
        raise ValueError("Original M01 recomputation failed: " + result.stderr.strip())
    computed = json.loads(result.stdout)
    case = "existing-mmlu200-out256"
    row = {"case": case, "kind": "compute", "repeat": 1, "arms": {},
           "scope": "Historical MMLU-Pro fixed200, out256, three repeats per arm",
           "source_url": f"https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/{REVISION}/{REPORT.as_posix()}"}
    for data in computed["arms"]:
        role = data["role"]
        elapsed = Decimal(data["sum_T_seconds"])
        reference = Decimal(data["sum_Tref_lower_seconds"])
        flops = data["sum_F_interval"]
        row["arms"][role] = {
            "effective_flops_interval": flops, "synchronized_s": float(elapsed),
            "reference_s_lower": float(reference),
            "tflops_per_card_interval": [float(value) for value in data["C_interval"]],
            "reference_tflops_per_card_upper": float(Decimal(flops[1]) / (2 * reference * 10**12)),
            "efficiency_pct_lower": float(100 * reference / elapsed),
            "reference_qualified": True, "repetitions": 3,
        }
    met = row["arms"]["B1"]["efficiency_pct_lower"] >= threshold
    row["checks"] = {"effective_compute_efficiency_lower": met}
    row["status"] = "criteria_met" if met else "criteria_not_met"
    out.mkdir(parents=True, exist_ok=False)
    (out / "benchmark-recompute.json").write_text(result.stdout, encoding="utf-8")
    (out / "source-evidence-manifest.json").write_bytes(manifest.read_bytes())
    return row, {"benchmark_commit": REVISION, "evidence_manifest_sha256": MANIFEST_SHA,
                 "verifier_sha256": VERIFIER_SHA, "source_url": row["source_url"]}

