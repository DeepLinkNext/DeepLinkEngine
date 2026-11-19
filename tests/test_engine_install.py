"""Verify offline engine delivery, integrity rejection and pip source isolation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmark/one-click-kit/scripts")]
import engine_install


class EngineInstallTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.wheels = self.root / "local wheels"
        self.wheels.mkdir()
        self.records = []
        resolver = patch.object(engine_install, "resolve_runtime_dependencies", return_value={"verified": True})
        self.resolver = resolver.start()
        self.addCleanup(resolver.stop)
        for index in range(4):
            data = ("fixed engine artifact " + str(index)).encode()
            name = "fixture_" + str(index) + ".whl"
            (self.wheels / name).write_bytes(data)
            self.records.append(dict(name=name, filename=name, bytes=len(data),
                                     sha256=hashlib.sha256(data).hexdigest(),
                                     url="https://example.invalid/" + name))

    def test_frozen_index_contains_exactly_the_four_engine_artifacts(self):
        records = engine_install.sources()
        self.assertEqual(len(records), 4)
        self.assertTrue(all("api.github.com/repos/DeepLinkNext/DeepLinkEngine/releases/assets/" in r["api_url"]
                            for r in records))

    def test_offline_install_pins_local_files_and_hashes_without_scanning_web_pages(self):
        with patch.object(engine_install, "sources", return_value=self.records), \
             patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")), \
             patch("child_runner.run_stage", return_value=types.SimpleNamespace(returncode=0)) as run:
            result = engine_install.install(self.root / "cache", wheelhouse=self.wheels)
        self.assertEqual(result["status"], "installed")
        self.resolver.assert_called_once()
        args = run.call_args.args[0]
        self.assertIn("--no-deps", args)
        self.assertIn("--require-hashes", args)
        requirements = Path(args[args.index("-r") + 1]).read_text().splitlines()
        self.assertEqual(len(requirements), 4)
        self.assertTrue(all(line.startswith("file://") and "%20" in line and " --hash=sha256:" in line
                            for line in requirements))
        self.assertFalse(any("expanded_assets" in item or "https:" in item for item in args + requirements))

    def test_download_only_never_runs_pip(self):
        with patch.object(engine_install, "sources", return_value=self.records), \
             patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")), \
             patch("child_runner.run_stage") as run:
            result = engine_install.install(self.root / "cache", wheelhouse=self.wheels, download_only=True)
        self.assertEqual(result["status"], "downloaded")
        self.resolver.assert_not_called()
        run.assert_not_called()

    def test_corrupt_artifact_cannot_reach_pip(self):
        (self.wheels / self.records[0]["name"]).write_bytes(b"corrupt")
        with patch.object(engine_install, "sources", return_value=self.records), \
             patch("urllib.request.urlopen", side_effect=OSError("network unavailable")), \
             patch("child_runner.run_stage") as run:
            with self.assertRaisesRegex(OSError, "Dependency download failed"):
                engine_install.install(self.root / "cache", wheelhouse=self.wheels)
        run.assert_not_called()

    def test_pip_failure_is_preserved_in_receipt(self):
        with patch.object(engine_install, "sources", return_value=self.records), \
             patch("child_runner.run_stage", return_value=types.SimpleNamespace(returncode=17)):
            with self.assertRaisesRegex(RuntimeError, "installation failed"):
                engine_install.install(self.root / "cache", wheelhouse=self.wheels)
        receipt = next((self.root / "cache/engine-install").glob("*.json"))
        self.assertEqual(json.loads(receipt.read_text())["returncode"], 17)


class RuntimeDependencyTests(unittest.TestCase):
    def test_source_only_manager_dependencies_are_required_even_without_wheel_metadata(self):
        for installed in (None, "3.9", "4.5"):
            with self.subTest(installed=installed):
                version = {"return_value": installed} if installed else {"side_effect": engine_install.metadata.PackageNotFoundError}
                with patch.object(engine_install, "RUNTIME_ROOTS", ()), \
                     patch.object(engine_install, "ADAPTER_REQUIREMENTS", ("platformdirs>=4",)), \
                     patch.object(engine_install.metadata, "version", **version), \
                     patch.object(engine_install.metadata, "distribution", return_value=types.SimpleNamespace(version=installed, requires=[])):
                    result = engine_install.dependency_audit()
                self.assertEqual(result["verified"], installed == "4.5")
                if installed != "4.5":
                    self.assertEqual(result["issues"][0]["package"], "runtime-adapter")
                    self.assertEqual(result["issues"][0]["requires"], "platformdirs>=4")

    def test_extras_and_transitive_requirements_are_checked_even_when_core_versions_match(self):
        distributions = {
            "vllm": types.SimpleNamespace(version="0.28", requires=["fastapi[standard]>=0.133"]),
            "fastapi": types.SimpleNamespace(version="0.136", requires=["starlette>=1.0.1", "uvicorn>=0.40; extra == 'standard'"]),
            "starlette": types.SimpleNamespace(version="1.0.1", requires=[]),
            "uvicorn": types.SimpleNamespace(version="0.35", requires=[]),
        }
        with patch.object(engine_install, "RUNTIME_ROOTS", ("vllm",)), \
             patch.object(engine_install, "ADAPTER_REQUIREMENTS", ()), \
             patch.object(engine_install.metadata, "distribution", side_effect=lambda name: distributions[name]), \
             patch.object(engine_install.metadata, "version", side_effect=lambda name: distributions[name].version):
            before = engine_install.dependency_audit()
            self.assertFalse(before["verified"])
            self.assertEqual(before["issues"][0]["found"], "0.35")
            distributions["uvicorn"].version = "0.40"
            self.assertTrue(engine_install.dependency_audit()["verified"])

    def test_missing_transitive_package_is_not_hidden_by_an_installed_parent(self):
        dist = types.SimpleNamespace(version="0.28", requires=["missing-api-dependency>=1"])
        with patch.object(engine_install, "RUNTIME_ROOTS", ("vllm",)), \
             patch.object(engine_install, "ADAPTER_REQUIREMENTS", ()), \
             patch.object(engine_install.metadata, "distribution", return_value=dist), \
             patch.object(engine_install.metadata, "version", side_effect=engine_install.metadata.PackageNotFoundError):
            result = engine_install.dependency_audit()
        self.assertFalse(result["verified"])
        self.assertIsNone(result["issues"][0]["found"])


if __name__ == "__main__":
    unittest.main()
