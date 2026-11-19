"""Entrypoint routing checks without SDK loading, NPU work or task generation."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("engine_one_click", ROOT / "one_click.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def write_config(directory, **values):
    data = {
        "engine_profile": "r6",
        "model": "models/Qwen3.5-35B-A3B",
        "b1_python": "sdk/bin/python",
        "physical_devices": [0, 1],
        "cache": "cache",
        "out": "runs/three-datasets-r1",
    }
    data.update(values)
    data = {key: value for key, value in data.items() if value is not None}
    path = Path(directory) / "evaluation.json"
    path.write_text(json.dumps(data))
    return path, data


def printed(display):
    """Join the message argument of each mocked print call.

    Reading call.args[0] avoids the repr escaping that breaks Windows paths.
    """
    return "\n".join(str(call.args[0]) for call in display.call_args_list if call.args)


def status_run(directory, campaign=None, stages=None, exit_code=None,
               report=False, receipt=None, cases=0):
    run = Path(directory) / "run"
    run.mkdir()
    if campaign is not None:
        (run / "campaign.json").write_text(json.dumps(campaign))
    (run / "execution-checklist.json").write_text(
        json.dumps(
            {
                "status": (campaign or {}).get("status", "running"),
                "stages": stages or [],
            }
        )
    )
    if receipt:
        leaf = run / receipt[0]
        (leaf / "service").mkdir(parents=True)
        (leaf / "receipt.json").write_text(json.dumps({"status": receipt[1]}))
        for index in range(cases):
            (leaf / "service" / ("case-%03d.json" % index)).write_text("{}")
    if exit_code is not None:
        logs = Path(directory) / "run-logs"
        logs.mkdir()
        (logs / "exit.json").write_text(json.dumps({"returncode": exit_code}))
    if report:
        report_dir = Path(directory) / "run-report"
        report_dir.mkdir()
        (report_dir / "report.html").write_text("<html></html>")
    return run


class OneClickTests(unittest.TestCase):
    def test_engine_entry_preserves_arguments_and_defaults_to_live_logs(self):
        args = [
            "--model",
            "/synthetic/model",
            "--out",
            "/synthetic/result",
            "--physical-devices",
            "5,6",
        ]
        with (
            patch.object(entry.os, "execve") as execute,
            patch("builtins.print") as display,
        ):
            entry.main(args)
        self.assertTrue(
            any(entry.NOTICE_ZH in call.args for call in display.call_args_list)
        )
        executable, command, env = execute.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertEqual(
            command,
            [
                sys.executable,
                str(ROOT / "scripts/live_console.py"),
                "--run",
                "/synthetic/result",
                "--",
                sys.executable,
                str(ROOT / "scripts/third_party.py"),
                *args,
            ],
        )
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(env["PYTHONPATH"], "")
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")

    def test_existing_prepare_and_cache_arguments_reach_the_same_controller(self):
        args = [
            "--prepare-only",
            "--no-follow",
            "--model=/synthetic/model",
            "--out=/synthetic/result",
            "--cache",
            "/synthetic/cache",
        ]
        with patch.object(entry.os, "execve") as execute, patch("builtins.print"):
            entry.main(args)
        self.assertEqual(
            execute.call_args.args[1][2:], [x for x in args if x != "--no-follow"]
        )

    def test_help_does_not_create_a_console_session(self):
        with patch.object(entry.os, "execve") as execute, patch("builtins.print"):
            entry.main(["--help"])
        self.assertEqual(
            execute.call_args.args[1][1], str(ROOT / "scripts/third_party.py")
        )

    def test_prepare_subcommand_defaults_to_full_five_with_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            path, data = write_config(folder)
            with patch.object(entry.os, "execve") as execute, patch("builtins.print"):
                entry.main(["prepare", "--config", str(path)])
            executable, command, env = execute.call_args.args
            self.assertEqual(executable, sys.executable)
            self.assertEqual(
                command[:7],
                [
                    sys.executable,
                    str(ROOT / "scripts/live_console.py"),
                    "--run",
                    str(Path(folder) / "runs/three-datasets-r1-prepare"),
                    "--",
                    sys.executable,
                    str(ROOT / "scripts/third_party.py"),
                ],
            )
            self.assertEqual(
                command[7:],
                [
                    "--engine-profile",
                    "r6",
                    "--model",
                    str(Path(folder) / "models/Qwen3.5-35B-A3B"),
                    "--out",
                    str(Path(folder) / "runs/three-datasets-r1-prepare"),
                    "--physical-devices",
                    "0,1",
                    "--datasets",
                    "mmlu,hle,frontier,swe,terminal",
                    "--reference-profile",
                    "submission-v2",
                    "--cache",
                    str(Path(folder) / "cache"),
                    "--b1-python",
                    str(Path(folder) / "sdk/bin/python"),
                    "--prepare-only",
                ],
            )
            self.assertEqual(env["PYTHONPATH"], "")
            self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")

    def test_run_subcommand_uses_the_configured_out_and_skips_prepare(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = write_config(folder)
            with patch.object(entry.os, "execve") as execute, patch("builtins.print"):
                entry.main(["run", "--config", str(path)])
            command = execute.call_args.args[1]
            self.assertEqual(
                command[2:4],
                [
                    "--run",
                    str(Path(folder) / "runs/three-datasets-r1"),
                ],
            )
            forwarded = command[7:]
            self.assertNotIn("--prepare-only", forwarded)
            self.assertEqual(
                forwarded[forwarded.index("--datasets") + 1],
                "mmlu,hle,frontier,swe,terminal",
            )
            self.assertNotIn("--performance-only", forwarded)

    def test_quick_config_only_selects_fixed_mmlu_and_keeps_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = write_config(folder, engine_profile="r8", test_mode="quick")
            with patch.object(entry.os, "execve") as execute, patch("builtins.print"):
                entry.main(["run", "--config", str(path)])
            args = execute.call_args.args[1][7:]
            self.assertEqual(args[args.index("--datasets") + 1], "mmlu")
            self.assertEqual(args[args.index("--engine-profile") + 1], "r8")
            self.assertNotIn("--performance-only", args)

    def test_invalid_test_mode_cannot_silently_run_another_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            for value in ("three", "FULL", 5, [], {}):
                path, _ = write_config(folder, test_mode=value)
                with (
                    patch.object(entry.os, "execve") as execute,
                    patch("builtins.print") as display,
                ):
                    self.assertEqual(entry.main(["run", "--config", str(path)]), 2)
                execute.assert_not_called()
                self.assertIn("test_mode", printed(display))

    def test_prepare_out_override_is_used_without_the_suffix(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = write_config(folder)
            override = Path(folder) / "runs/manual-prepare"
            with patch.object(entry.os, "execve") as execute, patch("builtins.print"):
                entry.main(["prepare", "--config", str(path), "--out", str(override)])
            forwarded = execute.call_args.args[1][7:]
            self.assertEqual(forwarded[forwarded.index("--out") + 1], str(override))

    def test_existing_deployment_and_devices_are_forwarded(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = write_config(
                folder,
                b1_python=None,
                deployment="existing/b1-r6",
                devices=[0, 1],
                baseline_deployment="b0/original018",
                dataset_assets="data/assets",
                env_file=".env",
            )
            with patch.object(entry.os, "execve") as execute, patch("builtins.print"):
                entry.main(["run", "--config", str(path)])
            forwarded = execute.call_args.args[1][7:]
            self.assertIn("--deployment", forwarded)
            self.assertNotIn("--b1-python", forwarded)
            self.assertEqual(
                forwarded[forwarded.index("--deployment") + 1],
                str(Path(folder) / "existing/b1-r6"),
            )
            self.assertEqual(forwarded[forwarded.index("--devices") + 1], "0,1")
            self.assertEqual(
                forwarded[forwarded.index("--baseline-deployment") + 1],
                str(Path(folder) / "b0/original018"),
            )
            self.assertEqual(
                forwarded[forwarded.index("--dataset-assets") + 1],
                str(Path(folder) / "data/assets"),
            )
            self.assertEqual(
                forwarded[forwarded.index("--env-file") + 1],
                str(Path(folder) / ".env"),
            )

    def test_invalid_physical_devices_name_the_field_before_startup(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = write_config(folder, physical_devices=[0, 0])
            with (
                patch.object(entry.os, "execve") as execute,
                patch("builtins.print") as display,
            ):
                result = entry.main(["run", "--config", str(path)])
            self.assertEqual(result, 2)
            execute.assert_not_called()
            self.assertIn("physical_devices", printed(display))

    def test_unknown_configuration_field_is_rejected_by_name(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = write_config(folder, datasets="mmlu")
            with (
                patch.object(entry.os, "execve") as execute,
                patch("builtins.print") as display,
            ):
                result = entry.main(["run", "--config", str(path)])
            self.assertEqual(result, 2)
            execute.assert_not_called()
            self.assertIn("datasets", printed(display))

    def test_missing_b1_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = write_config(folder, b1_python=None)
            with (
                patch.object(entry.os, "execve") as execute,
                patch("builtins.print") as display,
            ):
                result = entry.main(["run", "--config", str(path)])
            self.assertEqual(result, 2)
            execute.assert_not_called()
            self.assertIn("b1_python", printed(display))

    def test_existing_output_directory_is_refused_with_a_hint(self):
        with tempfile.TemporaryDirectory() as folder:
            path, data = write_config(folder)
            (Path(folder) / data["out"]).mkdir(parents=True)
            with (
                patch.object(entry.os, "execve") as execute,
                patch("builtins.print") as display,
            ):
                result = entry.main(["run", "--config", str(path)])
            self.assertEqual(result, 2)
            execute.assert_not_called()
            message = printed(display)
            self.assertIn("already occupied", message)
            self.assertIn("--out", message)

    def test_explicit_empty_out_is_a_configuration_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = write_config(folder)
            with (
                patch.object(entry.os, "execve") as execute,
                patch("builtins.print") as display,
            ):
                result = entry.main(["run", "--config", str(path), "--out="])
            self.assertEqual(result, 2)
            execute.assert_not_called()
            message = printed(display)
            self.assertIn("--out", message)
            self.assertIn("non-empty", message)

    def test_existing_log_or_report_directory_is_refused(self):
        for suffix in ("-logs", "-report"):
            with tempfile.TemporaryDirectory() as folder:
                path, data = write_config(folder)
                conflict = Path(folder) / (data["out"] + suffix)
                conflict.mkdir(parents=True)
                with (
                    patch.object(entry.os, "execve") as execute,
                    patch("builtins.print") as display,
                ):
                    result = entry.main(["run", "--config", str(path)])
                self.assertEqual(result, 2)
                execute.assert_not_called()
                message = printed(display)
                self.assertIn(str(conflict), message)
                self.assertIn("--out", message)

    def test_dangling_output_symlink_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path, data = write_config(folder)
            link = Path(folder) / (data["out"] + "-report")
            link.parent.mkdir(parents=True)
            try:
                os.symlink(Path(folder) / "moved-away", link, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable on this platform")
            self.assertFalse(link.exists())
            self.assertTrue(os.path.lexists(link))
            with (
                patch.object(entry.os, "execve") as execute,
                patch("builtins.print") as display,
            ):
                result = entry.main(["run", "--config", str(path)])
            self.assertEqual(result, 2)
            execute.assert_not_called()
            message = printed(display)
            self.assertIn(str(link), message)
            self.assertIn("--out", message)

    def test_devices_must_be_two_distinct_devices(self):
        for devices in ([0], [0, 0], [-1, 0]):
            with tempfile.TemporaryDirectory() as folder:
                path, _ = write_config(folder, devices=devices)
                with (
                    patch.object(entry.os, "execve") as execute,
                    patch("builtins.print") as display,
                ):
                    result = entry.main(["run", "--config", str(path)])
                self.assertEqual(result, 2)
                execute.assert_not_called()
                message = printed(display)
                self.assertIn("devices", message)

    def test_example_configuration_and_readme_match_the_entry_contract(self):
        example = json.loads((ROOT / "examples/evaluation.json").read_bytes())
        self.assertLessEqual(set(example), set(entry.CONFIG_FIELDS))
        for field in ("model", "physical_devices", "out"):
            self.assertIn(field, example)
        published = json.loads((ROOT / "publication-files.json").read_bytes())
        self.assertIn("examples/evaluation.json", published)
        self.assertEqual(published, sorted(published))
        readme = "".join((ROOT / "README.md").read_text().split())
        self.assertIn("one_click.pyprepare--configevaluation.json", readme)
        self.assertIn("one_click.pyrun--configevaluation.json", readme)
        self.assertIn("one_click.pystatus--run", readme)
        self.assertIn("<out>-report/report.html", readme)
        self.assertIn("<out>-prepare/frozen-execution-plan.json", readme)

    def test_subcommand_help_lists_fields_without_starting_a_console(self):
        with (
            patch.object(entry.os, "execve") as execute,
            patch("builtins.print") as display,
        ):
            result = entry.main(["prepare", "--help"])
        self.assertEqual(result, 0)
        execute.assert_not_called()
        text = printed(display)
        for phrase in (
            "physical_devices",
            "engine_profile",
            "examples/evaluation.json",
        ):
            self.assertIn(phrase, text)

    def test_status_reports_a_prepared_campaign(self):
        with tempfile.TemporaryDirectory() as folder:
            run = status_run(
                folder,
                campaign={"status": "prepared_only"},
                stages=[{"name": "continuous-B0", "status": "planned"}],
            )
            with patch.object(entry.os, "execve") as execute, patch(
                "builtins.print"
            ) as display:
                result = entry.main(["status", "--run", str(run), "--json"])
            self.assertEqual(result, 0)
            execute.assert_not_called()
            value = json.loads(printed(display))
            self.assertEqual(value["state"], "prepared")
            self.assertEqual(value["campaign_status"], "prepared_only")
            self.assertIsNone(value["returncode"])
            self.assertIsNone(value["report_path"])
            self.assertEqual(
                (value["stages_done"], value["stages_total"], value["stage"]),
                (0, 1, "prepare"),
            )

    def test_status_shows_completed_campaign_with_failed_command(self):
        with tempfile.TemporaryDirectory() as folder:
            run = status_run(
                folder,
                campaign={"status": "completed"},
                stages=[{"name": "continuous-B1", "status": "completed"}],
                exit_code=7,
                report=True,
            )
            with patch.object(entry.os, "execve") as execute, patch(
                "builtins.print"
            ) as display:
                result = entry.main(["status", "--run", str(run)])
            self.assertEqual(result, 0)
            execute.assert_not_called()
            text = printed(display)
            self.assertIn("评测阶段已完成", text)
            self.assertIn("命令执行失败", text)
            self.assertIn("退出码: 7", text)
            self.assertIn(str(Path(folder) / "run-report/report.html"), text)
            with patch("builtins.print") as display:
                json_result = entry.main(["status", "--run", str(run), "--json"])
            value = json.loads(printed(display))
            self.assertEqual(json_result, 0)
            self.assertEqual(value["state"], "completed_command_failed")
            self.assertEqual(value["returncode"], 7)
            self.assertEqual(
                value["report_path"], str(Path(folder) / "run-report/report.html")
            )

    def test_status_reports_failure_reason_and_unknown_stage_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            run = status_run(
                folder,
                campaign={"status": "failed", "failure": "RuntimeError: boom"},
                stages=[{"name": "hle-solver-B1", "status": "failed"}],
                exit_code=2,
            )
            with patch.object(entry.os, "execve"), patch("builtins.print") as display:
                result = entry.main(["status", "--run", str(run), "--json"])
            self.assertEqual(result, 0)
            value = json.loads(printed(display))
            self.assertEqual(value["state"], "failed")
            self.assertEqual(value["failure"], "RuntimeError: boom")
            self.assertEqual(value["returncode"], 2)

    def test_status_without_terminal_record_keeps_last_stage_and_unknown_process(self):
        with tempfile.TemporaryDirectory() as folder:
            run = status_run(
                folder,
                stages=[{"name": "mmlu-pair", "status": "running"}],
                receipt=("mmlu-pair/B1", "running-client"),
                cases=1,
            )
            with patch.object(entry.os, "execve"), patch("builtins.print") as display:
                result = entry.main(["status", "--run", str(run), "--json"])
            self.assertEqual(result, 0)
            value = json.loads(printed(display))
            self.assertEqual(value["state"], "unknown")
            self.assertIsNone(value["campaign_status"])
            self.assertIsNone(value["returncode"])
            self.assertEqual(
                (value["stage"], value["phase"], value["responses"], value["planned"]),
                ("mmlu-pair/B1", "running-client", 1, 200),
            )

    def test_status_is_queryable_before_the_first_checklist(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            run.mkdir()
            (run / "output-volume-admission.json").write_text(
                json.dumps({"prepare_only": False})
            )
            logs = Path(folder) / "run-logs"
            logs.mkdir()
            (logs / "controller.log").write_text("restoring SDK\n")
            with patch.object(entry.os, "execve") as execute, patch(
                "builtins.print"
            ) as display:
                result = entry.main(["status", "--run", str(run), "--json"])
            self.assertEqual(result, 0)
            execute.assert_not_called()
            value = json.loads(printed(display))
            self.assertEqual(value["state"], "unknown")
            self.assertEqual(value["stage"], "prepare")
            self.assertIsNone(value["campaign_status"])
            self.assertIsNone(value["returncode"])
            self.assertIsNone(value["report_path"])

    def test_status_follows_a_symlinked_run_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            real = status_run(
                folder,
                campaign={"status": "completed"},
                stages=[{"name": "continuous-B1", "status": "completed"}],
                exit_code=2,
                report=True,
            )
            link = Path(folder) / "link"
            try:
                os.symlink(real, link, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable on this platform")
            with patch("builtins.print") as display:
                result = entry.main(["status", "--run", str(link), "--json"])
            self.assertEqual(result, 0)
            value = json.loads(printed(display))
            self.assertEqual(value["state"], "completed_command_failed")
            self.assertEqual(value["returncode"], 2)
            self.assertEqual(
                value["report_path"], str(Path(folder) / "run-report/report.html")
            )

    def test_status_rejects_missing_directory_or_record(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("builtins.print") as display:
                missing = entry.main(
                    ["status", "--run", str(Path(folder) / "absent")]
                )
            self.assertEqual(missing, 2)
            self.assertIn("run directory not found", printed(display))
            empty = Path(folder) / "empty"
            empty.mkdir()
            with patch("builtins.print") as display:
                record = entry.main(["status", "--run", str(empty)])
            self.assertEqual(record, 2)
            self.assertIn("no evaluation record", printed(display))

    def test_status_help_and_missing_run_argument(self):
        with patch.object(entry.os, "execve"), patch("builtins.print") as display:
            self.assertEqual(entry.main(["status", "--help"]), 0)
        text = printed(display)
        for phrase in ("--run", "--json", "unknown"):
            self.assertIn(phrase, text)
        with patch("builtins.print") as display:
            self.assertEqual(entry.main(["status"]), 2)
        self.assertIn("--run is required", printed(display))

    def test_real_help_explains_shared_weights_and_extra_scoring_time(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "one_click.py"), "--help"],
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        text = "".join(result.stdout.split())
        for phrase in [
            "无需另备模型或外部裁判服务",
            "评分增加整轮耗时",
            "不是独立模型验证",
            "不等同引擎请求失败",
        ]:
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()


def test_status_does_not_revive_withdrawn_dense_certificate(tmp_path):
    from one_click import status_value
    import json

    (tmp_path / "campaign.json").write_text(json.dumps(dict(status="completed")))
    reference = tmp_path / "compute-reference"
    reference.mkdir()
    (reference / "status.json").write_text(json.dumps(dict(status="measured",
        full_model_certificate=True,
        certificate_kind="matched_complete_necessary_work_and_full_reference_lower_bound")))
    value = status_value(tmp_path)
    assert value["state"] == "reference_incomplete"
    assert "撤回" in value["failure"]
