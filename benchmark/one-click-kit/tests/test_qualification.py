"""Synthetic audit fixtures; these are never inference or benchmark evidence."""

import argparse
import ast
import contextlib
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qualification import qualify, sealed_files
from run_b1_r4 import check_memory, run


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def seal(root):
    save(
        root,
        "checksums.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file() and p != root / "checksums.json"
        },
    )


def fixture(root, deployment):
    bindings = {"mods/packed-host/packed_host_state.py": "fixture-adapter"}
    for folder, name in (
        (root, "installed-files.json"),
        (deployment, "provenance/installed-files.json"),
    ):
        save(folder, name, bindings)
    for folder in (root, deployment):
        save(folder, "deployment.json", {"release": "synthetic-fixture-not-an-engine"})
    save(
        root,
        "receipt.json",
        dict(status="completed", completed=True, quiescent=True, kind="continuous"),
    )
    save(root, "cleanup.json", dict(quiescent=True))
    save(root, "resource-admission.json", dict(physical_cards=[5, 6], admitted=True))
    save(root, "model-weights.json", {"fixture": "not-real-weights"})
    save(
        root,
        "service/runtime-environment.json",
        {"BETTERSCALE_STATE_CACHE_HOST_BYTES": str(4 * 2**30)},
    )
    inputs = {
        name: [
            dict(input_tokens=tokens, token_ids_sha256=f"fixture-{i}") for i in range(8)
        ]
        for name, tokens in (("4096", 4096), ("16384", 16384))
    }
    save(root, "service/request-plan.json", inputs)
    save(
        root,
        "service/plan.json",
        dict(
            host_budget_per_rank_bytes=4 * 2**30,
            physical_cards=[5, 6],
            input_sha256=hashlib.sha256(
                (root / "service/request-plan.json").read_bytes()
            ).hexdigest(),
        ),
    )
    save(
        root,
        "service/tokenize-preflight.json",
        [c for cases in inputs.values() for c in cases],
    )
    phases, policy, engine = [], [], []
    start = 1000
    for name, cases in inputs.items():
        for suffix, duration in (("warmup", 60), ("formal", 300)):
            phase = name + "-" + suffix
            phases.append(
                dict(
                    name=phase,
                    seconds=duration,
                    start={"monotonic": start},
                    end={"monotonic": start + duration + 2},
                    submission_deadline_monotonic=start + duration,
                    failed=0,
                    requests=8,
                )
            )
            for lane, case in enumerate(cases):
                request_id = f"fixture-{phase}-lane{lane}-00000"
                usage = dict(prompt_tokens=case["input_tokens"], completion_tokens=256)
                row = dict(
                    id=request_id,
                    phase=phase,
                    input_tokens=case["input_tokens"],
                    success=True,
                    http_status=200,
                    usage=usage,
                    finish_reason="length",
                    start={"monotonic": start + 1},
                    first_token={"monotonic": start + 2},
                    last_token={"monotonic": start + 3},
                    end={"monotonic": start + 4},
                )
                save(root, "service/" + request_id + ".json", row)
                lines = [
                    dict(
                        clock={"monotonic": start + stamp},
                        raw="data: " + json.dumps(event),
                    )
                    for stamp, event in (
                        (2, {"choices": [{"delta": {"content": "fixture"}}]}),
                        (
                            3,
                            {
                                "choices": [
                                    {
                                        "delta": {"content": "fixture"},
                                        "finish_reason": "length",
                                    }
                                ],
                                "usage": usage,
                            },
                        ),
                    )
                ]
                lines.append(dict(clock={"monotonic": start + 4}, raw="data: [DONE]"))
                (root / "service" / (request_id + ".sse.jsonl.gz")).write_bytes(
                    gzip.compress(
                        ("\n".join(json.dumps(line) for line in lines) + "\n").encode()
                    )
                )
                policy.append(
                    dict(
                        runtime_effective_decision="recompute",
                        runtime_control_path="request_scoped_prefix_cache_bypass",
                        reused_tokens=0,
                        runtime_decision_supported=True,
                        runtime_cache_salt=request_id,
                        prompt_tokens=case["input_tokens"],
                    )
                )
                engine.append(
                    dict(
                        event="school028_request_salt_admitted",
                        prompt_tokens=case["input_tokens"],
                        cache_salt_sha256=hashlib.sha256(
                            request_id.encode()
                        ).hexdigest(),
                    )
                )
            start += duration + 10
    save(root, "service/continuous.json", dict(status="completed", phases=phases))
    for name, rows in (("kvmat-policy.jsonl", policy), ("kvmat-engine.jsonl", engine)):
        (root / "service" / name).write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
    for rank in (0, 1):
        rows = [dict(event="model_loaded", rank=rank, active=True)]
        rows += [
            dict(event="partition_equivalence", tokens=t, exact=[True] * 5, rank=rank)
            for t in (1024, 2048, 4096)
        ]
        rows += [
            dict(event="prefix", split=True),
            dict(
                event="graph_replay", active=True, target=True, speculative_enabled=True
            ),
        ]
        (root / "service" / f"microflow-{rank}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
        save(
            root,
            f"service/packed-host-install-{rank}.json",
            dict(adapter_sha256="fixture-adapter"),
        )
    save(
        root,
        "service/runtime-identity.json",
        dict(worker="corrected_serving_runtime.CorrectedWorker"),
    )
    seal(root)


class QualificationTests(unittest.TestCase):
    def test_unsigned_writeback_underflow_is_retained_and_resampled(self):
        bad = {
            "working_estimate": 127,
            "limit": 128,
            "stat": {"file": 110, "inactive_file": 48, "file_writeback": 2**64 - 4096},
        }
        valid = {
            "working_estimate": 79,
            "limit": 128,
            "stat": {"file": 110, "inactive_file": 48, "file_writeback": 0},
        }
        guard = Mock()
        guard.sample.side_effect = [copy.deepcopy(bad), copy.deepcopy(valid)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            value = check_memory(guard, {}, root)
            rows = [
                json.loads(s)
                for s in (root / "memory-samples.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rows[0]["stat"]["file_writeback"], 2**64 - 4096)
            self.assertFalse(rows[0]["counter_validation"]["valid"])
            self.assertTrue(value["counter_validation"]["valid"])
            self.assertEqual(value["working_estimate"], valid["working_estimate"])
            guard.check.assert_called_once_with(value, {})

    def test_persistent_invalid_counters_check_raw_current_without_cache_credit(self):
        bad = {
            "working_estimate": 79,
            "current": 100,
            "limit": 128,
            "stat": {"file": 110, "file_writeback": 2**64 - 4096},
        }
        guard = Mock()
        guard.sample.side_effect = [copy.deepcopy(bad) for _ in range(3)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            value = check_memory(guard, {}, root)
            guard.check.assert_called_once_with(value, {})
            self.assertFalse(value["counter_validation"]["valid"])
            self.assertEqual(value["working_estimate"], bad["current"])
            self.assertEqual(value["sampled_working_estimate"], 79)
            self.assertEqual(
                value["counter_validation"]["fallback"],
                "raw_current_no_cache_credit",
            )
            self.assertEqual(
                len((root / "memory-samples.jsonl").read_text().splitlines()), 4
            )

    def test_invalid_counters_do_not_bypass_original_reserve_or_oom_failure(self):
        bad = {
            "working_estimate": 80,
            "current": 127,
            "limit": 128,
            "stat": {"file": 110, "file_writeback": 2**64 - 4096},
        }
        for reason in ("reserve reached", "new oom event"):
            guard = Mock()
            guard.sample.side_effect = [copy.deepcopy(bad) for _ in range(3)]
            guard.check.side_effect = RuntimeError(reason)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with self.assertRaisesRegex(RuntimeError, reason):
                    check_memory(guard, {}, root)
                value = json.loads((root / "memory-latest.json").read_bytes())
                self.assertEqual(value["working_estimate"], bad["current"])
                guard.check.assert_called_once_with(value, {})

    def test_memory_guard_trigger_is_saved_before_raising(self):
        class Guard:
            @staticmethod
            def sample():
                return {
                    "working_estimate": 133,
                    "limit": 137,
                    "events": {"oom": 0},
                    "stat": {"file": 100},
                }

            @staticmethod
            def check(value, initial):
                raise RuntimeError("fixture reserve reached")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "fixture reserve reached"):
                check_memory(Guard, {}, root)
            expected = Guard.sample()
            expected["counter_validation"] = {
                "valid": True,
                "attempt": 1,
                "invalid_fields": [],
            }
            self.assertEqual(
                json.loads((root / "memory-latest.json").read_bytes()), expected
            )
            self.assertEqual(
                json.loads((root / "memory-samples.jsonl").read_text().strip()),
                expected,
            )

    def test_original_audit_function_bodies_remain_unchanged(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        text = (scripts / "original_audit.py").read_text()
        functions = {
            n.name: ast.get_source_segment(text, n)
            for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef)
        }
        for source in json.loads(
            (scripts / "original-audit-sources.json").read_bytes()
        ):
            self.assertEqual(
                hashlib.sha256(functions[source["function"]].encode()).hexdigest(),
                source["function_sha256"],
            )

    def test_preparation_failure_is_sealed_before_any_model_start(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(temp) / "failed"
            self.assertEqual(
                run(
                    argparse.Namespace(
                        out=root, deployment=Path(temp) / "missing", kind="continuous"
                    )
                ),
                2,
            )
            self.assertFalse(
                json.loads((root / "receipt.json").read_bytes())[
                    "model_service_started"
                ]
            )
            self.assertTrue(sealed_files(root))
            with self.assertRaises(FileExistsError):
                run(argparse.Namespace(out=root))

    def test_full_raw_qualification_and_rejection_of_a_resealed_failed_rank(self):
        with tempfile.TemporaryDirectory() as temp:
            root, deployment = Path(temp) / "run", Path(temp) / "deployment"
            fixture(root, deployment)
            self.assertEqual(qualify(root, deployment)["requests"], 32)
            path = root / "service/microflow-1.jsonl"
            path.write_text(
                path.read_text().replace('"active": true', '"active": false')
            )
            seal(root)
            with self.assertRaisesRegex(ValueError, "Inactive Microflow rank"):
                qualify(root, deployment)

    def test_historical_completed_text_does_not_replace_actual_request_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root, deployment = Path(temp) / "run", Path(temp) / "deployment"
            fixture(root, deployment)
            path = next((root / "service").glob("fixture-*.json"))
            row = json.loads(path.read_bytes())
            row["success"] = False
            save(root, str(path.relative_to(root)), row)
            seal(root)
            with self.assertRaisesRegex(ValueError, "raw SSE/scalar mismatch"):
                qualify(root, deployment)


if __name__ == "__main__":
    unittest.main()
