"""Synthetic dataset streams; these are not model or benchmark results."""

import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import report_dataset


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, last=3, end=8):
    save(root / "deployment.json", {"release": "full-answer-b1-20261030-r4"})
    contract = {"solver": {"request": {"max_tokens": 2}, "model": "synthetic"}}
    save(root / "contract.json", contract)
    task = {"case_id": "synthetic", "problem": "Synthetic fixture"}
    spec = {
        "dataset": "frontier",
        "stage": "solver",
        "service_role": "B0",
        "solver_role": "B0",
        "diagnostic": True,
        "plan": str(root),
        "contract_sha256": report_dataset.sha(root / "contract.json"),
        "case_ids": [task["case_id"]],
    }
    save(root / "dataset-spec.json", spec)
    save(root / "receipt.json", {"quiescent": True, "completed": True})
    wire = root / "service/stream.sse.jsonl.gz"
    wire.parent.mkdir(parents=True)
    usage = {"prompt_tokens": 12, "completion_tokens": 2}
    events = [
        (1, {"choices": [{"delta": {"content": "first"}}]}),
        (3, {"choices": [{"delta": {"content": "last"}}]}),
        (7, {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage}),
    ]
    with gzip.open(wire, "wt") as stream:
        for clock, event in events:
            stream.write(
                json.dumps(
                    {"clock": {"monotonic": clock}, "raw": "data: " + json.dumps(event)}
                )
                + "\n"
            )
        stream.write(
            json.dumps({"clock": {"monotonic": 7}, "raw": "data: [DONE]"}) + "\n"
        )
    request = dict(
        contract["solver"]["request"],
        model="synthetic",
        messages=[{"role": "user", "content": task["problem"]}],
        stream=True,
        stream_options={"include_usage": True},
        cache_salt="synthetic",
    )
    record = {
        "case_id": task["case_id"],
        "status": "ok",
        "content": "firstlast",
        "usage": usage,
        "finish_reason": "stop",
        "start": {"monotonic": 0},
        "first_token": {"monotonic": 1},
        "last_token": {"monotonic": last},
        "end": {"monotonic": end},
        "stream_capture": str(wire),
        "stream_sha256": report_dataset.sha(wire),
        "request": request,
        "request_sha256": digest(request),
    }
    record["record_sha256"] = digest(record)
    save(root / "service/evaluation/requests/synthetic.json", record)
    save(
        root / "checksums.json",
        {
            str(p.relative_to(root)): report_dataset.sha(p)
            for p in root.rglob("*")
            if p.is_file()
        },
    )
    return SimpleNamespace(digest=digest, load_plan=lambda _: (contract, [task]))


class DatasetReportTests(unittest.TestCase):
    def test_outline_tpot_uses_completion_and_content_span_is_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            module = fixture(root)
            with patch.object(report_dataset, "client", return_value=module):
                result = report_dataset.inspect(root)
            self.assertEqual(result["mean_tpot_ms"], 7000)
            self.assertEqual(result["mean_content_span_per_token_ms"], 2000)
            self.assertIn("V1.5 3.4.2", result["tpot_definition"])
            self.assertEqual(result["mean_ttft_ms"], 1000)
            self.assertEqual(result["duration_seconds"], 8)
            self.assertEqual(result["output_tokens_per_second"], 0.25)

    def test_resealed_cached_last_token_cannot_override_raw_stream(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            module = fixture(root, last=7)
            with patch.object(report_dataset, "client", return_value=module):
                with self.assertRaisesRegex(ValueError, "timing differs"):
                    report_dataset.inspect(root)

    def test_response_end_before_last_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            module = fixture(root, end=2)
            with patch.object(report_dataset, "client", return_value=module):
                with self.assertRaisesRegex(ValueError, "out of order"):
                    report_dataset.inspect(root)


if __name__ == "__main__":
    unittest.main()
