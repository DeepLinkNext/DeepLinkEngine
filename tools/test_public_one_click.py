"""CPU regressions for the public entry and reference acceptance boundaries."""

from decimal import Decimal
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_entry", ROOT / "one_click.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
sys.path.insert(0, str(entry.KIT / "scripts"))
from full_reference_reuse import arm_statistics
from compute_reference import target


class PublicEntryTests(unittest.TestCase):
    def test_real_public_supplement_route_reaches_frozen_parser_without_profile_injection(self):
        observed = []
        def execute(executable, command, environment):
            self.assertEqual(Path(command[1]), entry.KIT / "scripts/acceptance_suite.py")
            self.assertNotIn("--engine-profile", command)
            spec = importlib.util.spec_from_file_location("public_supplement_contract", command[1])
            suite = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(suite)
            with patch.object(sys, "argv", command[1:]), patch.object(suite, "run", return_value=0) as run:
                self.assertEqual(suite.main(), 0)
                observed.append(run.call_args.args[0])
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(entry, "reference_evidence", return_value=Path(temp)), \
                patch.object(entry.os, "execve", side_effect=execute), \
                patch.object(sys, "argv", list(sys.argv)):
            with self.assertRaises(SystemExit) as exited:
                entry.main(["--context-tenant-only", "--no-follow", "--deployment", "/sdk",
                    "--baseline-deployment", "/b0", "--b1-python", sys.executable,
                    "--model", "/model", "--cache", temp, "--out", str(Path(temp) / "new-run"),
                    "--physical-devices", "1,6", "--devices", "0,1"])
            self.assertIn(exited.exception.code, (None, 0))
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].physical_devices, "1,6")
        self.assertEqual(observed[0].devices, "0,1")

    def test_published_sources_match_every_inventory_pin(self):
        self.assertEqual(entry.verify_kit()["tools"]["files"], 510)

    def test_changed_source_and_unlisted_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "one_click.py").write_text("original")
            manifest = dict(commit="a" * 40, files={"one_click.py": entry.digest(root / "one_click.py")})
            (root / "DEPLOYMENT-TOOLS.json").write_text(json.dumps(manifest))
            lock = root.parent / (root.name + "-lock.json")
            lock.write_text(json.dumps(dict(tools=dict(upstream_commit="a" * 40,
                manifest_sha256=entry.digest(root / "DEPLOYMENT-TOOLS.json")))))
            try:
                with patch.object(entry, "KIT", root), patch.object(entry, "LOCK", lock):
                    entry.verify_kit()
                    (root / "one_click.py").write_text("changed")
                    with self.assertRaisesRegex(ValueError, "source changed"):
                        entry.verify_kit()
                    (root / "one_click.py").write_text("original")
                    (root / "unexpected.py").write_text("added")
                    with self.assertRaisesRegex(ValueError, "inventory differs"):
                        entry.verify_kit()
            finally:
                lock.unlink()

    def test_mismatched_reference_download_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = entry.verify_kit()
            with patch.dict(os.environ, {"XDG_CACHE_HOME": temp}, clear=True), \
                    patch.object(entry.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong bytes")):
                with self.assertRaisesRegex(ValueError, "size/SHA-256 mismatch"):
                    entry.reference_evidence(lock)
                self.assertFalse((Path(temp) / "vllm-hust-engine/project-evidence-53541118/benchmark-evidence").exists())

    def test_unsafe_reference_archive_never_escapes_cache(self):
        # A compromised mirror must not gain extraction privileges even if a
        # future operator updates the outer archive pin.
        body = io.BytesIO()
        with tarfile.open(fileobj=body, mode="w:gz") as archive:
            info = tarfile.TarInfo("../escape")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        raw = body.getvalue()
        lock = entry.verify_kit()
        lock["reference_evidence"].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        module = type("Contract", (), dict(ASSET_BYTES=len(raw), ASSET_SHA256=hashlib.sha256(raw).hexdigest()))
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(os.environ, {"XDG_CACHE_HOME": temp}, clear=True), \
                patch.object(entry, "evidence_module", return_value=module), \
                patch.object(entry.urllib.request, "urlopen", return_value=io.BytesIO(raw)):
            with self.assertRaisesRegex(ValueError, "Unsafe reference archive"):
                entry.reference_evidence(lock)

    def test_anonymous_endpoint_failure_uses_verified_public_fallback(self):
        body = io.BytesIO()
        with tarfile.open(fileobj=body, mode="w:gz") as archive:
            info = tarfile.TarInfo("benchmark-evidence/sentinel")
            info.size = 6
            archive.addfile(info, io.BytesIO(b"golden"))
        raw = body.getvalue()
        lock = entry.verify_kit()
        pin = hashlib.sha256(raw).hexdigest()
        lock["reference_evidence"].update(bytes=len(raw), sha256=pin,
            api_url="https://api.github.com/public-fixture", url="https://github.com/public-fixture")
        def verify(path):
            self.assertEqual((path / "sentinel").read_bytes(), b"golden")
        module = SimpleNamespace(ASSET_BYTES=len(raw), ASSET_SHA256=pin, verify=verify)
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(os.environ, {"XDG_CACHE_HOME": temp}, clear=True), \
                patch.object(entry, "evidence_module", return_value=module), \
                patch.object(entry.urllib.request, "urlopen", side_effect=[
                    entry.urllib.error.URLError("transient fixture"), io.BytesIO(raw)]) as opened:
            result = entry.reference_evidence(lock)
            self.assertEqual((result / "sentinel").read_bytes(), b"golden")
            self.assertEqual(opened.call_count, 2)
            for call in opened.call_args_list:
                self.assertFalse(call.args[0].has_header("Authorization"))

    def test_reference_different_work_does_not_cancel(self):
        result = arm_statistics([dict(repetition=1, F_interval=[100, 110], T_seconds="2")],
            dict(effective_flops_interval=[150, 200], sum_reference_seconds_lower="2",
                 reference_tflops_per_card_upper="0.00000000005", reference_repetitions=[1]))
        self.assertEqual(Decimal(result["efficiency_lower_percent"]), Decimal("50"))
        self.assertFalse(result["shared_work_cancellation_used"])

    def test_all_predeclared_repetitions_contribute_to_efficiency(self):
        rows = [dict(repetition=index, F_interval=[100, 100], T_seconds=str(seconds))
                for index, seconds in enumerate([1, 1, 4], 1)]
        result = arm_statistics(rows, dict(effective_flops_interval=[300, 300],
            sum_reference_seconds_lower="3", reference_tflops_per_card_upper="0.00000000005",
            reference_repetitions=[1, 2, 3]))
        self.assertEqual(result["T_seconds"], "6")
        self.assertEqual(Decimal(result["efficiency_lower_percent"]), Decimal("50"))

    def test_retired_partial_certificate_cannot_revive_old_pass(self):
        for kind in ("matched_complete_necessary_work_and_full_reference_lower_bound", "partial_dense_projection_diagnostic"):
            result = target(dict(status="measured", milestone_met=True,
                certificate_kind=kind, efficiency_lower_observed_percent=99))
            self.assertFalse(result["accepted"])
            self.assertIsNone(result["measured_value"])

    def test_unqualified_new_certificate_cannot_pass(self):
        result = target(dict(status="measured", milestone_met=True,
            certificate_kind="qualified_full_reference_denominator_reuse/v1"))
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["measured_value"])


if __name__ == "__main__":
    unittest.main()
