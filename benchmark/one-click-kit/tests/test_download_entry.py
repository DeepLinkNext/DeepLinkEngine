"""Standalone downloader identity/authentication checks; never start inference."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "engine_download_entry", ROOT / "download_and_run.py"
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


class DownloadEntryTests(unittest.TestCase):
    def test_verified_bundle_launches_the_log_enabled_root_entry(self):
        root = Path("/synthetic/verified-engine")
        args = ["--model", "/synthetic/model", "--out", "/synthetic/run"]
        with (
            patch.object(entry, "ensure", return_value=root),
            patch.object(entry.os, "execve") as execute,
        ):
            entry.main(args)
        self.assertEqual(
            execute.call_args.args[1],
            [sys.executable, str(root / "one_click.py"), *args],
        )

    def test_dotenv_is_not_executed_and_unrelated_keys_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "not-created"
            env = root / ".env"
            env.write_text(
                f'UNRELATED=$(touch {marker})\nGITHUB_TOKEN="synthetic-key" # fixture\n'
            )
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(entry.token(env), "synthetic-key")
            self.assertFalse(marker.exists())

    def test_source_and_manifest_cannot_approve_themselves_after_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "one_click.py"
            source.write_text("synthetic original")
            manifest = root / "DEPLOYMENT-TOOLS.json"
            value = {
                "repository": entry.REPOSITORY,
                "commit": entry.COMMIT,
                "files": {"one_click.py": entry.sha(source)},
            }
            manifest.write_text(json.dumps(value))
            with patch.object(entry, "MANIFEST_SHA256", entry.sha(manifest)):
                self.assertEqual(entry.verify_tools(root), root)
                source.write_text("changed")
                value["files"]["one_click.py"] = entry.sha(source)
                manifest.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "externally fixed"):
                    entry.verify_tools(root)

    def test_redirect_does_not_forward_private_token_to_asset_host(self):
        req = urllib.request.Request(
            "https://api.github.com/example",
            headers={"Authorization": "Bearer synthetic-key"},
        )
        result = entry.SafeRedirect().redirect_request(
            req,
            None,
            302,
            "Found",
            {},
            "https://release-assets.githubusercontent.com/example",
        )
        self.assertIsNone(result.get_header("Authorization"))


class R8ConfigurationDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "r8_config_bootstrap", ROOT / "runners/r8/download_and_run.py"
        )
        cls.entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.entry)

    def test_r8_config_bootstrap_forwards_full_scope_to_verified_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evaluation.json"
            path.write_text(
                json.dumps(
                    dict(
                        engine_profile="r8",
                        model="weights",
                        deployment="sdk",
                        physical_devices=[1, 6],
                        out="results",
                    )
                )
            )
            with (
                patch.object(self.entry, "ensure", return_value=ROOT),
                patch.object(self.entry.os, "execve") as execute,
            ):
                self.entry.main(["run", "--config", str(path)])
            args = execute.call_args.args[1][2:]
            self.assertNotIn("--config", args)
            self.assertEqual(
                args[args.index("--model") + 1], str(Path(folder) / "weights")
            )
            self.assertEqual(
                args[args.index("--datasets") + 1], "mmlu,hle,frontier,swe,terminal"
            )
            self.assertEqual(args[args.index("--engine-profile") + 1], "r8")
            self.assertNotIn("--performance-only", args)

    def test_r8_runner_rejects_config_for_another_release_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evaluation.json"
            path.write_text(
                json.dumps(
                    dict(
                        engine_profile="r6",
                        model="weights",
                        deployment="sdk",
                        physical_devices=[1, 6],
                        out="results",
                    )
                )
            )
            with (
                patch.object(self.entry, "ensure", return_value=ROOT),
                patch.object(self.entry.os, "execve") as execute,
                patch("builtins.print"),
            ):
                self.assertEqual(self.entry.main(["run", "--config", str(path)]), 2)
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
