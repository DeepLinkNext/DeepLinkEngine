"""CPU regressions for automatic environment setup; no model packages are imported."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark/one-click-kit/scripts"))
import environment_setup as setup
import run_all as workflow


def info(version="2", python=None):
    return dict(python=python or sys.executable, prefix="/fixture", version=[3, 12],
                implementation="CPython", machine="aarch64", system="Linux", packages={"vendor": version})


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "engine").mkdir()
        (self.root / "engine/r8-lock.json").write_text(json.dumps({"required_python_packages": {"vendor": "2"}}))
        self.data = {"cache": str(self.root / "cache"), "out": str(self.root / "new-run")}
        self.payload = b"fixed wheel fixture"
        self.source = dict(filename="fixture.whl", url="https://fixture.invalid/wheel",
                           sha256=hashlib.sha256(self.payload).hexdigest(), bytes=len(self.payload))

    def prepare(self, before=None, probes=None, command=None):
        observations = iter(probes or [info("1"), info()])
        def observed(python, *args):
            value = dict(next(observations), python=str(python), prefix=str(Path(python).parent.parent))
            return value
        with patch.object(setup, "ROOT", self.root), \
             patch.object(setup.platform, "system", return_value="Linux"), \
             patch.object(setup.platform, "machine", return_value="aarch64"), \
             patch.object(setup, "cann_environment", return_value=({"PATH": os.environ.get("PATH", "")}, {"version": "9.1.0"})), \
             patch.object(setup, "select_python", return_value=(sys.executable, before or info())), \
             patch.object(setup, "probe", side_effect=observed), \
             patch.object(setup, "command", side_effect=command) as execute, \
             patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            result = setup.ensure_environment(self.data)
        return result, execute

    def test_qualified_base_is_readonly_and_a_private_environment_is_created(self):
        result, execute = self.prepare()
        self.assertEqual(len(execute.call_args_list), 3)
        self.assertNotEqual(result["python"], sys.executable)
        report = json.loads(Path(result["report"]).read_text())
        self.assertEqual(report["status"], "ready")
        self.assertFalse(report["shared_environment_modified"])
        self.assertFalse(Path(self.data["out"]).exists())
        self.assertEqual(result["env"]["DEEPLINKENGINE_ENVIRONMENT_REPORT"], result["report"])

    def test_mismatch_repairs_only_the_private_venv(self):
        result, execute = self.prepare(before=info("1"))
        commands = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertIn("venv", commands[0])
        self.assertIn("--system-site-packages", commands[0])
        for args in commands[1:]:
            self.assertTrue(args[0].startswith(self.data["cache"]))
            self.assertNotEqual(args[0], sys.executable)
        self.assertIn("--ignore-installed", commands[1])
        self.assertNotIn("--ignore-installed", commands[2])
        report = json.loads(Path(result["report"]).read_text())
        self.assertEqual(report["after"]["packages"], {"vendor": "2"})

    def test_cached_matching_venv_skips_all_package_installs(self):
        first, _ = self.prepare(before=info("1"))
        python = Path(first["python"])
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        _, execute = self.prepare(before=info("1"), probes=[info(), info()])
        execute.assert_not_called()

    def test_failed_cached_repair_is_not_trusted_just_because_versions_match(self):
        first, _ = self.prepare(before=info("1"))
        python = Path(first["python"])
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        report_path = Path(first["report"])
        report = json.loads(report_path.read_text())
        report["status"] = "failed"
        report_path.write_text(json.dumps(report))
        _, execute = self.prepare(before=info("1"), probes=[info(), info()])
        self.assertEqual(len(execute.call_args_list), 2)
        self.assertEqual(json.loads(report_path.read_text())["status"], "ready")

    def test_failed_repair_records_failure_without_creating_result_directory(self):
        def fail(*args):
            raise ValueError("fixture download failed")
        with self.assertRaisesRegex(ValueError, "fixture download failed"):
            self.prepare(before=info("1"), command=fail)
        reports = list(Path(self.data["cache"]).glob("environments/*/environment.json"))
        self.assertEqual(json.loads(reports[0].read_text())["status"], "failed")
        self.assertFalse(Path(self.data["out"]).exists())

    def test_post_repair_mismatch_is_not_marked_ready(self):
        with self.assertRaisesRegex(ValueError, "remain inconsistent"):
            self.prepare(before=info("1"), probes=[info("1"), info("wrong")])

    def test_wrong_architecture_never_starts_pip(self):
        with patch.object(setup.platform, "machine", return_value="x86_64"), \
             patch.object(setup, "command") as execute:
            with self.assertRaisesRegex(ValueError, "architecture"):
                setup.ensure_environment(self.data)
            execute.assert_not_called()

    def test_selects_available_python312(self):
        old = dict(info(), version=[3, 11])
        with patch.object(setup.sys, "executable", "/python311"), \
             patch.object(setup.shutil, "which", return_value="/python312"), \
             patch.object(setup, "probe", side_effect=[old, info(python="/python312")]):
            python, _ = setup.select_python({}, {"vendor": "2"})
        self.assertEqual(python, "/python312")

    def test_real_metadata_probe_does_not_import_target_package(self):
        observed = setup.probe(sys.executable, ["definitely-not-an-installed-package"])
        self.assertIsNone(observed["packages"]["definitely-not-an-installed-package"])

    def test_hash_matching_local_wheel_needs_no_download(self):
        local = self.root / "local"
        local.mkdir()
        (local / "fixture.whl").write_bytes(self.payload)
        with patch.object(setup.urllib.request, "urlopen") as network:
            result = setup.cached_wheel(self.source, [str(local)], self.root / "wheels")
        self.assertEqual(result, local / "fixture.whl")
        network.assert_not_called()

    def test_anonymous_download_is_size_and_hash_checked(self):
        with patch.object(setup.urllib.request, "urlopen", return_value=io.BytesIO(self.payload)) as network:
            result = setup.cached_wheel(self.source, [], self.root / "wheels")
        self.assertEqual(result.read_bytes(), self.payload)
        self.assertNotIn("Authorization", network.call_args.args[0].headers)

    def test_wrong_download_hash_is_rejected(self):
        with patch.object(setup.urllib.request, "urlopen", return_value=io.BytesIO(b"x" * len(self.payload))):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                setup.cached_wheel(self.source, [], self.root / "wheels")
        self.assertFalse((self.root / "wheels/fixture.whl").exists())

    def test_early_eof_is_retried_and_only_complete_verified_bytes_are_installed(self):
        with patch.object(setup.urllib.request, "urlopen", side_effect=[
                io.BytesIO(self.payload[:5]), io.BytesIO(self.payload)]) as network:
            result = setup.cached_wheel(self.source, [], self.root / "wheels")
        self.assertEqual(network.call_count, 2)
        self.assertEqual(result.read_bytes(), self.payload)
        self.assertFalse(list((self.root / "wheels").glob("wheel-*")))

    def test_persistent_truncation_retains_failure_and_never_creates_valid_cache_entry(self):
        with patch.object(setup.urllib.request, "urlopen", side_effect=lambda *a, **k: io.BytesIO(b"short")):
            with self.assertRaisesRegex(OSError, "received 5, expected"):
                setup.cached_wheel(self.source, [], self.root / "wheels")
        self.assertFalse((self.root / "wheels/fixture.whl").exists())

    def test_corrupted_cache_is_quarantined_and_replaced(self):
        cache = self.root / "wheels"
        cache.mkdir()
        (cache / "fixture.whl").write_bytes(b"corrupted")
        with patch.object(setup.urllib.request, "urlopen", return_value=io.BytesIO(self.payload)):
            result = setup.cached_wheel(self.source, [], cache)
        self.assertEqual(result.read_bytes(), self.payload)
        self.assertEqual(len(list(cache.glob("*.invalid-*"))), 1)

    def test_cann_setup_uses_positional_path_and_keeps_values_out_of_report(self):
        cann = self.root / "cann-9.1.0"
        (cann / "lib64").mkdir(parents=True)
        (cann / "lib64/libascendcl.so").touch()
        (cann / "set_env.sh").touch()
        fake = type("Result", (), {"stdout": b"PATH=/cann/bin\0SECRET=fixture-secret\0"})()
        with patch.object(setup.subprocess, "run", return_value=fake) as run:
            env, report = setup.cann_environment({"cann_home": str(cann)}, {})
        self.assertEqual(env["PATH"], "/cann/bin")
        self.assertNotIn("fixture-secret", json.dumps(report))
        self.assertEqual(run.call_args.args[0][-1], str(cann / "set_env.sh"))

    def test_environment_options_are_not_forwarded_to_strict_evaluator_config(self):
        model = self.root / "model"
        model.mkdir()
        config = self.root / "evaluation.json"
        config.write_text(json.dumps(dict(self.data, model="model", physical_devices=[0, 7],
                              environment={"wheelhouse": "wheels", "python": "python3.12"})))
        data = workflow.configuration(config)
        self.assertEqual(data["environment"]["wheelhouse"], str(self.root / "wheels"))
        self.assertEqual(data["environment"]["python"], "python3.12")

    def test_repairs_use_the_verified_owned_child_runner(self):
        report = {"steps": []}
        result = type("Result", (), {"returncode": 0})()
        with patch("one_click.verify_kit") as verify, patch("child_runner.run_stage", return_value=result) as owned:
            setup.command(["fixture-python", "-m", "pip"], {}, self.root, "fixture", report, self.root / "report.json")
        verify.assert_called_once()
        owned.assert_called_once()
        self.assertEqual(report["steps"][0]["status"], "completed")

    def test_child_uses_managed_python_even_when_config_names_the_original_python(self):
        self.data["environment"] = {"python": "/original/python"}
        prepared, _ = self.prepare()
        managed = prepared["python"]
        observed = dict(info(python=managed), prefix=str(Path(managed).parent.parent))
        with patch.object(setup, "ROOT", self.root), \
             patch.object(setup.sys, "executable", managed), \
             patch.object(setup.sys, "prefix", observed["prefix"]), \
             patch.object(setup.platform, "system", return_value="Linux"), \
             patch.object(setup.platform, "machine", return_value="aarch64"), \
             patch.object(setup, "cann_environment", return_value=({"PATH": "fixture"}, {"version": "9.1.0"})), \
             patch.object(setup, "select_python", return_value=(managed, observed)) as select, \
             patch.dict(os.environ, {"DEEPLINKENGINE_ENVIRONMENT_REPORT": prepared["report"],
                                    "DEEPLINKENGINE_MANAGED_PYTHON": managed}, clear=True):
            self.assertIsNone(setup.ensure_environment(self.data))
        self.assertEqual(select.call_args.args[0]["python"], managed)

    def test_environment_options_are_removed_before_the_real_v3_config_parser(self):
        spec = importlib.util.spec_from_file_location("workflow_fixtures", ROOT / "tests/test_run_all.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fixture = module.WorkflowTests()
        fixture.setUp()
        try:
            fixture.data["environment"] = {"wheelhouse": str(self.root / "wheels")}
            result, _ = fixture.invoke()
            self.assertEqual(result, 0)
            config = fixture.root / "result/evaluation.json"
            self.assertNotIn("environment", json.loads(config.read_text()))
            spec = importlib.util.spec_from_file_location("strict_v3_config", ROOT / "benchmark/one-click-kit/one_click.py")
            entry = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(entry)
            args = entry.configuration("prepare", ["--config", str(config), "--out", str(self.root / "new-prepare")])
            self.assertEqual(args[args.index("--datasets") + 1], "mmlu,hle,frontier,swe,terminal")
        finally:
            fixture.doCleanups()

    def test_prepare_environment_flag_never_starts_the_workload(self):
        with patch.object(sys, "argv", ["run_all.py", "--config", "fixture.json", "--prepare-environment"]), \
             patch.object(workflow, "configuration", return_value=self.data), \
             patch.object(setup, "ensure_environment", return_value=None), \
             patch.object(workflow, "preflight"), patch.object(workflow, "run") as execute, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(workflow.main(), 0)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
