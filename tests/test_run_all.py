"""Regression checks for the installed entry points, completion records and failures."""
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmark/one-click-kit/scripts")]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark import run_progress
from test_public_run_all import reference
spec = importlib.util.spec_from_file_location("workflow", ROOT / "run_all.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "model").mkdir()
        self.data = dict(model=str(self.root / "model"), physical_devices=[0, 7],
                         cache=str(self.root / "cache"), out=str(self.root / "result"), rerun_b0=True)
        self.calls = []
        self.incomplete = False

    def write(self, relative, value):
        path = self.root / "result" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def execute(self, args, **kwargs):
        self.calls.append(args)
        if "install" in args or (len(args)>1 and Path(args[1]).name=="runtime_storage.py"):
            self.write("installation/installation.json", {"release": "full-answer-b1-20261117-r8"})
        if "run" in args:
            self.write("fixed-five/frozen-execution-plan.json", dict(
                optimized_deployment="/actual/b1", baseline_deployment="/actual/b0", logical_devices="0,1",
                datasets=["mmlu", "hle", "frontier", "swe", "terminal"]))
            self.write("fixed-five/campaign.json", dict(status="completed",
                       selected_execution_complete=not self.incomplete,
                       compute_reference=reference()))
            self.write("fixed-five-report/report.html", "fixture report")
        if "--context-tenant-only" in args:
            self.assertEqual(args[1], str(ROOT / "benchmark/one_click.py"))
            self.assertNotIn("--engine-profile", args)
            self.assertEqual(args[args.index("--physical-devices") + 1], "0,7")
            self.assertEqual(args[args.index("--devices") + 1], "0,1")
            self.assertEqual(args[args.index("--baseline-deployment") + 1], "/actual/b0")
            self.write("context-tenant/execution.json", {"status": "completed"})
            self.write("context-tenant/paired-summary.json", dict(execution_complete=True,
                       comparison=[{"phase": phase} for phase in
                       ["4096-formal", "16384-formal", "32768-formal", "multi-tenant-formal"]]))
            self.write("context-tenant/paired-report/report.html", "fixture report")
        if len(args)>2 and "report_python" in args[2]:
            targets = {k: dict(milestone=dict(met=True, status="达成目标"),
                               final=dict(met=False, status="未达成目标"))
                       for k in ("4.1", "4.2", "4.3", "4.4")}
            for goal in targets["4.4"].values():
                goal["evidence_status"] = "published_rental"
            self.write("full-report/summary.json", dict(schema="deeplinkengine-full-acceptance-report/v1",
                                                        execution_complete=True, targets=targets))
            for name in ("report.html","fixed-five.svg","fixed-five.png"):
                self.write("full-report/"+name,"fixture only")
        return types.SimpleNamespace(returncode=0)

    def invoke(self, execute=None):
        with patch("child_runner.run_stage", side_effect=execute or self.execute), \
             patch("benchmark.one_click.verify_kit"), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = workflow.run(self.data)
        receipt = json.loads((self.root / "result/results.json").read_text())
        return result, receipt

    def test_full_flow_routes_supplement_without_package_profile_injection(self):
        from benchmark.b0_run_policy import DISCOVERY_ROOT, DISCOVERY_SOURCE

        environments = []

        def execute(args, **kwargs):
            if "run" in args or "--context-tenant-only" in args:
                environments.append(kwargs["env"])
            return self.execute(args, **kwargs)

        inherited = {DISCOVERY_ROOT: "/another-run", DISCOVERY_SOURCE: "/another-source"}
        with patch.dict(workflow.os.environ, inherited):
            result, receipt = self.invoke(execute)
            self.assertEqual({key: workflow.os.environ[key] for key in inherited}, inherited)
        self.assertEqual(len(environments), 2)
        for env in environments:
            self.assertEqual(env[DISCOVERY_ROOT], str(self.root.resolve()))
            self.assertNotIn(DISCOVERY_SOURCE, env)
        self.assertEqual(result, 0)
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["automated_acceptance_complete"], True)
        self.assertEqual(len(receipt["evidence_sha256"]), 11)
        names = [row["name"] for row in receipt["stages"]]
        self.assertLess(names.index("verify-installed-tools"), names.index("run-context-tenant"))

    def test_plot_only_failure_continues_to_cpu_report_without_inference_retry(self):
        def execute(args, **kwargs):
            result = self.execute(args, **kwargs)
            if '--context-tenant-only' in args:
                self.write('context-tenant/execution.json', dict(status='failed', failure='alternate rendering error',
                           stages=[dict(role=r, status='completed', returncode=0) for r in ('B0', 'B1')]))
                (self.root/'result/context-tenant/paired-report/report.html').unlink()
                return types.SimpleNamespace(returncode=7)
            return result

        with patch('benchmark.full_report.context_sources') as admit:
            code, receipt = self.invoke(execute)
        admit.assert_called_once()
        self.assertEqual(code, 0)
        self.assertEqual(receipt['status'], 'completed')
        rows = [row for row in receipt['stages'] if row['name'] == 'run-context-tenant']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['returncode'], 7)
        self.assertEqual(rows[0]['status'], 'sealed_with_report_failure')
        self.assertEqual(receipt['reports']['context_tenant'], 'full-report/report.html#context-results')

    def test_completed_measurements_keep_report_when_actual_cost_is_missing(self):
        def execute(args, **kwargs):
            result = self.execute(args, **kwargs)
            if len(args) > 2 and 'report_python' in args[2]:
                path = self.root / 'result/full-report/summary.json'
                data = json.loads(path.read_bytes())
                data['targets']['4.4']['milestone'] = dict(met=False, status='未达成目标', evidence_status='missing')
                self.write('full-report/summary.json', data)
            return result

        code, receipt = self.invoke(execute)
        self.assertEqual(code, 0)
        self.assertEqual(receipt['status'], 'completed')
        self.assertFalse(receipt['automated_acceptance_complete'])
        self.assertEqual(receipt['targets']['4.4']['milestone']['evidence_status'], 'missing')

    def test_report_only_loads_fee_config_without_model_or_device_fields(self):
        config = self.root / 'evaluation.json'
        config.write_text(json.dumps(dict(out='result', cache='cache', actual_cost='fees/configured.json',
                                          physical_devices='old-unavailable-device-schema',
                                          environment={'future_vendor_option': None})))
        fees = self.root / 'fees'
        fees.mkdir()
        configured, override = fees / 'configured.json', fees / 'override with spaces.json'
        for path in (configured, override):
            path.write_text('{}')
        command = ['run_all.py', '--config', str(config), '--report-only']
        for selected, suffix in ((configured, []), (override, ['--actual-cost', str(override)])):
            with self.subTest(selected=selected), patch.object(sys, 'argv', command + suffix), \
                 patch.dict(workflow.os.environ, {}), \
                 patch.object(workflow.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)) as runner:
                self.assertEqual(workflow.main(), 0)
            runner.assert_called_once()
            args = runner.call_args.args[0]
            self.assertEqual(args[-2:], ['--actual-cost', str(selected)])
            self.assertIn('full-report-', args[-3])

    def test_default_configuration_starts_the_full_workflow_with_35b(self):
        example = json.loads((ROOT / "benchmark/run-all.example.json").read_bytes())
        self.assertEqual(workflow.DEFAULT_MODEL, "/models/Qwen3.5-35B-A3B")
        self.assertEqual(example["model"], workflow.DEFAULT_MODEL)
        config = self.root / "evaluation.json"
        config.write_text(json.dumps({k: v for k, v in self.data.items() if k != "model"}))
        with contextlib.chdir(self.root), patch.object(sys, "argv", ["run_all.py"]), \
             patch.dict(workflow.os.environ, {}), patch.object(workflow, "select_run_engine"), \
             patch("engine.release_selection.ensure_metadata_client"), \
             patch("engine.environment_setup.ensure_environment", return_value=None), \
             patch.object(workflow, "DEFAULT_MODEL", self.data["model"]), \
             patch.object(workflow, "preflight"), patch.object(workflow, "run", return_value=0) as run:
            self.assertEqual(workflow.main(), 0)
        self.assertEqual(run.call_args.args[0], self.data)

    def test_full_progress_survives_python_handoff_and_reaches_report(self):
        config = self.root / "evaluation.json"
        config.write_text(json.dumps(self.data))
        out = Path(self.data["out"])
        environment_receipt = self.root / "environment.json"
        argv = ["run_all.py", "--config", str(config)]

        class ReplacedPython(BaseException):
            pass

        def prepare(data, *, progress):
            progress.detail("prepare-python", receipt=environment_receipt)
            environment_receipt.write_text(json.dumps({
                "progress_id": progress.run_id,
                "status": "checking", "steps": [{"name": "resolve-python-dependencies", "status": "running"}],
            }))
            pending = self.read_progress(config)
            self.assertEqual(pending["stage"], "prepare-environment")
            self.assertEqual(pending["detail"]["step"], "resolve-python-dependencies")
            self.assertFalse(out.exists())
            return {"python": sys.executable, "env": dict(workflow.os.environ)}

        during = []

        def execute(args, **kwargs):
            status = self.read_progress(config)
            during.append(status["stage"])
            self.assertEqual(status["run_id"], handed_off["run_id"])
            if status["stage"] == "run-fixed-five":
                self.write("fixed-five/frozen-stages.json", [{"name": "continuous-B0"}, {"name": "continuous-B1"}])
                self.write("fixed-five/execution-checklist.json", {"status": "running", "stages": [
                    {"name": "continuous-B0", "status": "reused_verified"},
                    {"name": "continuous-B1", "status": "running"},
                ]})
                detail = self.read_progress(config)["detail"]
                self.assertEqual((detail["stage"], detail["stages_done"]), ("continuous-B1", 1))
                self.assertEqual(detail["stages"][0]["status"], "reused_verified")
            if status["stage"] == "run-context-tenant":
                arm = out / "context-tenant/context-B0"
                self.write("context-tenant/execution.json", {
                    "schema": "vllm-hust-acceptance-supplement-execution/v1", "status": "running",
                    "planned_roles": ["B0", "B1"], "stages": [{"role": "B0", "status": "running", "run": str(arm)}],
                })
                self.write("context-tenant/context-B0/receipt.json", {"status": "running-client"})
                detail = self.read_progress(config)["detail"]
                self.assertEqual((detail["stage"], detail["phase"]), ("B0", "running-client"))
            return self.execute(args, **kwargs)

        with patch.dict(workflow.os.environ), patch.object(workflow, "select_run_engine"),              patch("engine.release_selection.ensure_metadata_client"), patch.object(workflow, "preflight"),              patch("benchmark.one_click.verify_kit"), patch("child_runner.run_stage", side_effect=execute),              contextlib.redirect_stdout(io.StringIO()) as console:
            with patch("engine.environment_setup.ensure_environment", side_effect=prepare),                  patch.object(sys, "argv", argv), patch.object(workflow.os, "execve", side_effect=ReplacedPython) as switch:
                with self.assertRaises(ReplacedPython):
                    workflow.main()

            handed_off = self.read_progress(config)
            self.assertEqual(handed_off["detail"]["step"], "python-handoff")
            self.assertEqual(switch.call_args.args[2][run_progress.RUN_ID_ENV], handed_off["run_id"])
            with patch.dict(workflow.os.environ, switch.call_args.args[2]), patch.object(sys, "argv", argv),                  patch("engine.environment_setup.ensure_environment", return_value=None):
                self.assertEqual(workflow.main(), 0)

        final = self.read_progress(config)
        self.assertEqual(final["state"], "completed")
        self.assertEqual(final["run_id"], handed_off["run_id"])
        self.assertEqual(final["stage"], "verify-report")
        self.assertEqual(Path(final["report_path"]), out / "full-report/report.html")
        self.assertEqual([row["name"] for row in final["stages"]].count("prepare-environment"), 1)
        self.assertEqual(during, ["prepare-datasets", "install-package", "verify-engine", "install-engine",
                                  "prepare-fixed-five", "run-fixed-five", "verify-installed-tools",
                                  "run-context-tenant", "render-full-report"])
        self.assertIn("[PROGRESS]", console.getvalue())
        self.assertEqual(console.getvalue().count("读取配置"), 1)

    def read_progress(self, config):
        with patch.object(sys, "argv", ["run_all.py", "--status", "--json", "--config", str(config)]),              contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(workflow.main(), 0)
        return json.loads(output.getvalue())

    def test_status_summarizes_current_workload_and_preserves_json(self):
        config = self.root / "evaluation.json"
        config.write_text(json.dumps(self.data))
        self.write("results.json", {
            "status": "running", "started_at": "2026-11-23T03:20:19+00:00",
            "baseline_policy": {"rerun_b0": True},
            "stages": [{"name": "run-fixed-five", "status": "running",
                        "started_at": "2026-11-23T03:56:49+00:00"}],
        })
        self.write("fixed-five/execution-checklist.json", {"status": "running", "stages": [
            {"name": "terminal-trace-B0", "status": "completed"},
            {"name": "terminal-trace-B1", "status": "running"},
        ]})
        self.write("fixed-five/terminal-trace-B1/receipt.json", {"status": "starting-model"})
        self.write("fixed-five/terminal-trace-B1/dataset-spec.json", {"case_ids": list(range(48))})
        args = ["run_all.py", "--status", "--config", str(config)]
        with patch.object(run_progress, "datetime", wraps=datetime) as clock:
            clock.now.return_value = datetime(2026, 11, 23, 14, 54, 43, tzinfo=timezone.utc)
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(workflow.main(), 0)
            human = output.getvalue()
            self.assertIn("当前阶段  固定五项评测", human)
            self.assertIn("执行进度  已完成 1/2 项", human)
            self.assertIn("评测项目  第 2 项 · Terminal-Bench · B1 轨迹回放", human)
            self.assertIn("当前动作  启动模型", human)
            self.assertIn("请求进度  已记录 0/48 个请求", human)
            self.assertIn("运行耗时  总计 11 小时 34 分 24 秒", human)
            self.assertIn("阶段耗时  10 小时 57 分 54 秒", human)
            self.assertNotIn("terminal-trace-B0", human)

            with patch.object(sys, "argv", [*args, "--watch"]), \
                 patch.object(run_progress.time, "sleep", side_effect=KeyboardInterrupt), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(workflow.main(), 130)
            self.assertEqual(len(output.getvalue().splitlines()), 1)
            self.assertIn("Terminal-Bench · B1 轨迹回放", output.getvalue())
            self.assertIn("0/48 个请求", output.getvalue())

            machine = self.read_progress(config)
        self.assertEqual(machine["elapsed_seconds"], 39474)
        self.assertEqual(machine["detail"]["stage"], "terminal-trace-B1")
        self.assertEqual(machine["detail"]["phase"], "starting-model")

    def test_progress_records_failure_before_output_exists_and_watch_is_read_only(self):
        config = self.root / "evaluation.json"
        config.write_text(json.dumps(self.data))
        out = Path(self.data["out"])
        self.assertEqual(self.read_progress(config)["state"], "not_started")

        with patch.object(sys, "argv", ["run_all.py", "--config", str(config)]),              patch("engine.release_selection.ensure_metadata_client"), patch.object(workflow, "select_run_engine"),              patch("engine.environment_setup.ensure_environment", side_effect=ValueError("missing fixture CANN")),              contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            workflow.main()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(out.exists())
        failure = self.read_progress(config)
        self.assertEqual((failure["state"], failure["stage"], failure["failure"]),
                         ("failed", "prepare-environment", "missing fixture CANN"))

        path = run_progress.progress_path(out)
        contents = path.read_bytes()
        with patch.object(sys, "argv", ["run_all.py", "--status", "--watch", "--json", "--config", str(config)]),              patch.object(run_progress.time, "sleep", side_effect=AssertionError("watch must stop at failure")),              patch.object(workflow, "select_run_engine", side_effect=AssertionError("status must not run")),              contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(workflow.main(), 0)
        self.assertEqual(json.loads(output.getvalue())["state"], "failed")
        self.assertEqual(path.read_bytes(), contents)

    def test_progress_distinguishes_recovery_from_old_results_and_controller_exit(self):
        config = self.root / "evaluation.json"
        config.write_text(json.dumps(self.data))
        out = Path(self.data["out"])
        self.write("results.json", {"status": "failed", "stages": [{"name": "run-fixed-five", "status": "failed"}]})
        progress = run_progress.BootstrapProgress(out, run_progress.timestamp(), False)
        progress.advance("verify-recovery")
        with patch.object(run_progress, "owner_alive", return_value=False):
            interrupted = self.read_progress(config)
        self.assertEqual((interrupted["state"], interrupted["stage"]), ("interrupted", "verify-recovery"))
        self.assertIsNone(interrupted["results_path"])
        self.assertFalse(interrupted["rerun_b0"])
        path = run_progress.progress_path(out)
        before = path.read_bytes()
        with patch.object(sys, "argv", ["run_all.py", "--status", "--watch", "--config", str(config)]),              patch.object(run_progress, "owner_alive", return_value=True),              patch.object(run_progress.time, "sleep", side_effect=KeyboardInterrupt),              contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(workflow.main(), 130)
        self.assertEqual(path.read_bytes(), before)

    def test_config_failure_is_visible_and_recovery_cannot_replace_an_active_run(self):
        config = self.root / "evaluation.json"
        config.write_text(json.dumps(dict(self.data, physical_devices=[0, 0])))
        out = Path(self.data["out"])
        with patch.object(sys, "argv", ["run_all.py", "--config", str(config)]),              contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            workflow.main()
        failed = self.read_progress(config)
        self.assertEqual((failed["state"], failed["stage"]), ("failed", "read-config"))
        self.assertFalse(out.exists())

        original = run_progress.BootstrapProgress(out, run_progress.timestamp(), True)
        original.finish("handed_off")
        self.write("results.json", {"status": "running", "controller_owner": original.data["owner"],
                                   "progress_id": original.run_id,
                                   "stages": [{"name": "run-fixed-five", "status": "running"}]})
        with patch.object(run_progress, "owner_alive", return_value=True):
            with self.assertRaisesRegex(ValueError, "already executing"):
                run_progress.BootstrapProgress(out, run_progress.timestamp(), False)
        self.assertEqual(self.read_progress(config)["run_id"], original.run_id)

    def test_progress_output_survives_windows_console_encoding(self):
        config = self.root / "evaluation.json"
        config.write_text(json.dumps(self.data))
        failure = "\u73af\u5883\u51c6\u5907\u5931\u8d25 \U0001f680"
        out = Path(self.data["out"])
        progress = run_progress.BootstrapProgress(out, run_progress.timestamp(), True)
        progress.finish("failed", failure)
        status_args = ["run_all.py", "--status", "--config", str(config)]

        with io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict") as console:
            with contextlib.redirect_stdout(console):
                run_progress.announce("prepare-environment", "check-cann")
                run_progress.announce_state("completed")
                with patch.object(sys, "argv", status_args):
                    self.assertEqual(workflow.main(), 0)
                with patch.object(sys, "argv", [*status_args, "--json"]):
                    self.assertEqual(workflow.main(), 0)
            console.flush()
            lines = console.buffer.getvalue().decode("cp1252").splitlines()

        self.assertEqual(lines[0], "[PROGRESS] prepare-environment | check-cann")
        self.assertEqual(lines[1], "[PROGRESS] completed")
        self.assertIn("[PROGRESS] failed | read-config", lines[2])
        status = json.loads(lines[3])
        self.assertEqual((status["state"], status["failure"]), ("failed", failure))

    def test_terminal_progress_is_available_with_a_partial_child_receipt(self):
        config = self.root / "evaluation.json"
        config.write_text(json.dumps(self.data))
        self.write("results.json", {"status": "failed", "error": "controller stopped",
                                   "stages": [{"name": "run-fixed-five", "status": "failed"}]})
        child = Path(self.data["out"]) / "fixed-five"
        child.mkdir()
        (child / "campaign.json").write_text("{")
        status = self.read_progress(config)
        self.assertEqual((status["state"], status["stage"], status["failure"]),
                         ("failed", "run-fixed-five", "controller stopped"))
        self.assertIsNone(status["detail"])

    def test_advanced_flags_are_listed_in_full_help(self):
        for flag in ("--help", "--help-all"):
            with self.subTest(flag=flag), patch.object(sys, "argv", ["run_all.py", flag]), \
                 contextlib.redirect_stdout(io.StringIO()) as stdout, self.assertRaises(SystemExit) as exit_code:
                workflow.main()
            self.assertEqual(exit_code.exception.code, 0)
            self.assertIn("--config", stdout.getvalue())
            self.assertEqual("--resume-bootstrap" in stdout.getvalue(), flag == "--help-all")

    def test_full_entry_routes_the_fee_config_to_report_not_measurement(self):
        config = self.root / 'evaluation.json'
        config.write_text(json.dumps(dict(self.data, actual_cost='fees.json')))
        expense = self.root / 'fees.json'
        command = ['run_all.py', '--config', str(config)]
        expense.write_text('{}')
        with patch.object(sys, 'argv', command), patch.object(workflow, 'select_run_engine') as select, \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_code:
            workflow.main()
        self.assertEqual(exit_code.exception.code, 2)
        select.assert_not_called()

        bill = self.root / 'fixture-receipt.txt'
        bill.write_bytes(b'synthetic actual cost for preflight')
        scenario = {role: dict(measurement_seal_sha256='a'*64, monthly_cost_per_card_cny='1200',
                               evidence=[dict(file=bill.name, sha256=hashlib.sha256(bill.read_bytes()).hexdigest())])
                    for role in ('B0', 'B1')}
        expense.write_text(json.dumps(dict(schema='deeplinkengine-v2-actual-card-cost/v1',
                                          scenarios={'mmlu': scenario})))
        with patch.object(sys, 'argv', command), patch.object(workflow, 'select_run_engine'), \
             patch('engine.environment_setup.ensure_environment', return_value=None), patch.object(workflow, 'preflight'), \
             patch.dict(workflow.os.environ, {}), patch.object(workflow, 'run', return_value=0) as run:
            self.assertEqual(workflow.main(), 0)
        self.assertEqual(run.call_args.kwargs['actual_cost'], expense)
        self.assertNotIn('actual_cost', run.call_args.args[0])

    def test_recorded_installation_version_drives_recovery_without_selection_file(self):
        out = self.root / 'old'
        (out/'installation').mkdir(parents=True)
        installed = out/'installation/installation.json'
        installed.write_text(json.dumps(dict(package_version='9.42', release='recorded-B1')))
        data = dict(self.data, out=str(out))
        with patch('engine.release_selection.select', return_value={'version':'9.42','release':'recorded-B1'}) as select:
            self.assertEqual(workflow.select_run_engine(data, recovery=True)['version'], '9.42')
        select.assert_called_once_with(data['cache'], version='9.42', wheelhouse=None)
        installed.write_text('{}')
        with patch('engine.release_selection.select') as select, self.assertRaisesRegex(ValueError, 'missing from preserved'):
            workflow.select_run_engine(data, recovery=True)
        select.assert_not_called()

    def test_changed_engine_can_choose_b0_from_a_mixed_sealed_campaign_with_new_cache(self):
        source = self.root/'previous'
        source.mkdir()
        before = dict(self.data, out=str(source), cache=str(self.root/'historical-cache'))
        (source/'results.json').write_text(json.dumps(dict(configuration=before,
                engine_selection=dict(version='1'))))
        names = ('continuous-B0', 'continuous-B1', 'mmlu-pair', 'hle-judge-B0')
        campaign = dict(status='completed', stages=[dict(name=n, status='completed') for n in names])
        with patch('benchmark.recovery_source.stopped'), patch('reuse_campaign.source',
                return_value=(None, {'engine_release':'B1-old'}, campaign, {})), \
             patch('reuse_campaign.audit_stage') as audit, \
             patch('engine.release_selection.active', return_value={'release':'B1-new','version':'2','client':None}):
            result = workflow.completed_run_source(self.data, source)
        self.assertEqual(result['completed_stages'], ['continuous-B0', 'mmlu-pair'])
        audit.assert_called_once_with(source/'fixed-five', 'continuous-B0')
        self.assertNotIn('compute_resume_source', result)

    def test_continuation_verifies_historical_interpreter_against_source_cache(self):
        source = self.root/'previous'
        install = source/'installation'
        install.mkdir(parents=True)
        cache = self.root/'historical-cache'
        python = cache/'environment/bin/python'
        python.parent.mkdir(parents=True)
        (python.parent.parent/'pyvenv.cfg').touch()
        (install/'installation.json').write_text(json.dumps(dict(python=str(python))))
        with patch('engine.python_storage.admitted', return_value=True) as admitted, \
             patch('engine.python_storage.ensure'), patch('engine.environment_setup.cann_environment', return_value=({}, None)), \
             patch('engine.environment_setup.probe', return_value={}), patch('engine.environment_setup.compatible_python', return_value=True), \
             patch('engine.environment_setup.differences', return_value=[]), patch('engine.environment_setup.vendor_integrity', return_value={'verified':True}), \
             patch('engine.environment_setup.runtime_imports'), patch.object(sys, 'executable', str(python)):
            self.assertIsNone(workflow.recovery_environment(self.data, source=source, source_cache=str(cache)))
        admitted.assert_called_once_with(python, str(cache))
        self.assertFalse((source/'environment-recovery.json').exists())

    def test_prepare_protocol_stops_before_every_model_and_formal_client(self):
        with patch('child_runner.run_stage', side_effect=self.execute), patch('benchmark.one_click.verify_kit'), \
             contextlib.redirect_stdout(io.StringIO()):
            result = workflow.run(self.data, prepare_protocol_only=True)
        receipt = json.loads((self.root / 'result/results.json').read_text())
        self.assertEqual(result, 0)
        self.assertEqual([r['name'] for r in receipt['stages']],
            ['prepare-datasets', 'install-package', 'verify-engine', 'install-engine', 'prepare-fixed-five'])
        self.assertEqual(receipt['status'], 'prepared')
        self.assertFalse(receipt['model_service_started'])
        self.assertFalse(receipt['formal_client_started'])
        self.assertFalse(receipt['automated_acceptance_complete'])
        self.assertFalse(any('run' in command for command in self.calls))

    def test_failed_protocol_preparation_cannot_publish_prepared(self):
        def execute(args, **kwargs):
            if 'prepare' in args:
                return types.SimpleNamespace(returncode=9)
            return self.execute(args, **kwargs)
        with patch('child_runner.run_stage', side_effect=execute), patch('benchmark.one_click.verify_kit'), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = workflow.run(self.data, prepare_protocol_only=True)
        receipt = json.loads((self.root / 'result/results.json').read_text())
        self.assertEqual(result, 1)
        self.assertEqual(receipt['status'], 'failed')
        self.assertNotIn('protocol_preparation_complete', receipt)

    def test_sealed_stage_reuse_is_forwarded_and_failure_originals_are_untouched(self):
        self.data["rerun_b0"] = False
        source=self.root/'previous';source.mkdir();original=b'original failed ledger\n'
        (source/'results.json').write_bytes(original)
        (source/'installation').mkdir();(source/'installation/installation.json').write_text(json.dumps(dict(package_version='previous')))
        def execute(args,**kwargs):
            if '--reuse-completed-campaign' in args:
                self.assertEqual(args[args.index('--reuse-completed-campaign')+1],str(source/'fixed-five'))
                self.assertIn('--datasets',args)
                self.assertEqual(args[args.index('--datasets')+1],'mmlu,hle,frontier,swe,terminal')
                return self.execute([*args,'run'],**kwargs)
            return self.execute(args,**kwargs)
        with patch.object(workflow,'completed_run_source',return_value=dict(root=str(source),completed_stages=['continuous-B0'])), \
             patch('engine.release_selection.active',return_value=dict(version='new',client=None,engine_profile='r9')),patch('engine.release_selection.save'), \
             patch('benchmark.one_click.configuration_arguments',return_value=['--datasets','mmlu,hle,frontier,swe,terminal']), \
             patch('benchmark.one_click.verify_kit'),patch('child_runner.run_stage',side_effect=execute), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(workflow.run(self.data,reuse_completed_run=source),0)
        self.assertEqual((source/'results.json').read_bytes(),original)
        receipt=json.loads((self.root/'result/results.json').read_text())
        self.assertEqual(receipt['reused_run']['completed_stages'],['continuous-B0'])

    def test_reuse_rejects_changed_configuration_before_auditing_or_execution(self):
        source=self.root/'previous';source.mkdir()
        (source/'results.json').write_text(json.dumps(dict(status='failed',configuration=dict(self.data,physical_devices=[2,3]))))
        with patch('reuse_campaign.source') as audit:
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                workflow.completed_run_source(self.data,source)
            audit.assert_not_called()

    def test_reuse_source_cannot_overlap_new_output(self):
        with self.assertRaisesRegex(ValueError,'independent'):
            workflow.completed_run_source(self.data,Path(self.data['out'])/'nested')

    def test_claimed_completed_stage_must_pass_independent_raw_audit(self):
        source=self.root/'previous';source.mkdir()
        (source/'results.json').write_text(json.dumps(dict(status='failed',configuration=self.data)))
        with patch('reuse_campaign.source',return_value=(source,{},dict(status='failed',stages=[dict(name='continuous-B0',status='completed')]),{})), \
             patch('reuse_campaign.audit_stage',side_effect=ValueError('modified raw stream')):
            with self.assertRaisesRegex(ValueError,'modified raw stream'):
                workflow.completed_run_source(self.data,source)

    def test_bootstrap_retry_preserves_and_reaudits_prior_completed_stage_selection(self):
        self.failed_preflight()
        ledger=self.root/'result/results.json';prior=json.loads(ledger.read_text())
        reused=dict(root=str(self.root/'previous'),completed_stages=['continuous-B0'])
        prior['reused_run']=reused;ledger.write_text(json.dumps(prior))
        def execute(args,**kwargs):
            return self.execute([*args,'run'] if '--reuse-completed-campaign' in args else args,**kwargs)
        with patch.object(workflow,'completed_run_source',return_value=reused) as audit, \
             patch('benchmark.one_click.configuration_arguments',return_value=['--datasets','mmlu,hle,frontier,swe,terminal']), \
             patch('benchmark.one_click.verify_kit'),patch('child_runner.run_stage',side_effect=execute), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(workflow.run(self.data,resume_bootstrap=True),0)
            audit.assert_called_once_with(self.data,reused['root'])
        self.assertTrue(any('--reuse-completed-campaign' in a for a in self.calls))

    def test_zero_exit_without_all_four_binary_decisions_cannot_complete(self):
        def execute(args, **kwargs):
            result=self.execute(args, **kwargs)
            if len(args)>2 and "report_python" in args[2]:
                path=self.root/"result/full-report/summary.json"
                data=json.loads(path.read_text());data['targets']['4.3']['final']['status']='复核中'
                path.write_text(json.dumps(data))
            return result
        result,receipt=self.invoke(execute)
        self.assertEqual(result,1)
        self.assertFalse(receipt['automated_acceptance_complete'])
        self.assertIn('four completed acceptance decisions',receipt['error'])

    def test_reports_and_zero_exit_do_not_hide_incomplete_execution(self):
        self.incomplete = True
        result, receipt = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("not show all", receipt["error"])

    def test_selected_three_dataset_success_cannot_pass_as_the_full_suite(self):
        def execute(args, **kwargs):
            result = self.execute(args, **kwargs)
            if "run" in args:
                path = self.root / "result/fixed-five/frozen-execution-plan.json"
                plan = json.loads(path.read_text())
                plan["datasets"] = ["mmlu", "hle", "frontier"]
                path.write_text(json.dumps(plan))
            return result
        result, receipt = self.invoke(execute)
        self.assertEqual(result, 1)
        self.assertIn("all five fixed datasets", receipt["error"])
        self.assertFalse(any("--context-tenant-only" in args for args in self.calls))

    def test_failed_install_never_starts_tests_or_retries(self):
        def failed(args, **kwargs):
            self.calls.append(args)
            return types.SimpleNamespace(returncode=0 if "prepare_datasets.py" in str(args[1]) else 9)
        result, receipt = self.invoke(failed)
        self.assertEqual(result, 1)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(receipt["stages"][-1]["name"], "install-package")
        self.assertEqual(receipt["stages"][-1]["returncode"], 9)

    def test_launch_failure_does_not_leave_running_stage(self):
        def failed(*args, **kwargs):
            raise OSError("cannot launch")
        result, receipt = self.invoke(failed)
        self.assertEqual(result, 1)
        self.assertEqual(receipt["stages"][0]["status"], "launch_failed")

    def test_interruption_retains_interrupted_stage(self):
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt()
        result, receipt = self.invoke(interrupted)
        self.assertEqual(result, 1)
        self.assertEqual(receipt["stages"][0]["status"], "interrupted")

    def test_existing_output_and_dangling_link_are_rejected(self):
        config = self.root / "evaluation.json"
        config.write_text(json.dumps(self.data))
        out = Path(self.data["out"])
        out.mkdir()
        with self.assertRaisesRegex(ValueError, "new output"):
            workflow.configuration(config)
        out.rmdir()
        try:
            out.symlink_to(self.root / "missing", target_is_directory=True)
        except OSError:
            return
        with self.assertRaisesRegex(ValueError, "new output"):
            workflow.configuration(config)

    def test_relative_paths_are_bound_to_configuration_file(self):
        config = self.root / "evaluation.json"
        config.write_text(json.dumps(dict(self.data, model="model", out="output", cache="cache")))
        data = workflow.configuration(config)
        self.assertEqual(data["model"], str(self.root / "model"))
        self.assertEqual(data["out"], str(self.root / "output"))

    def test_wrong_dependencies_are_rejected_without_native_imports(self):
        with patch.object(workflow.sys, "version_info", (3, 12)), \
             patch.object(workflow.platform, "python_implementation", return_value="CPython"), \
             patch.object(workflow.platform, "system", return_value="Linux"), \
             patch.object(workflow.platform, "machine", return_value="aarch64"), \
             patch.object(workflow.metadata, "version", return_value="wrong"), \
             patch("child_runner.run_stage") as execute:
            with self.assertRaisesRegex(ValueError, "expected 2.13.0"):
                workflow.preflight()
            execute.assert_not_called()

    def failed_preflight(self):
        def execute(args, **kwargs):
            if "prepare" in args:
                self.calls.append(args)
                return types.SimpleNamespace(returncode=9)
            return self.execute(args, **kwargs)
        result, _ = self.invoke(execute)
        self.assertEqual(result, 1)

    def test_bootstrap_recovery_verifies_existing_installation_and_retains_original_failure(self):
        self.data["environment"] = {"wheelhouse": str(self.root / "wheelhouse")}
        self.failed_preflight()
        before = (self.root / "result/results.json").read_bytes()
        self.calls.clear()
        with patch("benchmark.one_click.verify_kit"), patch("child_runner.run_stage", side_effect=self.execute), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = workflow.run(self.data, resume_bootstrap=True)
        self.assertEqual(result, 0)
        self.assertFalse(any("install" in args for args in self.calls))
        repair = next(args for args in self.calls if "--repair-dependencies" in args)
        self.assertEqual(repair[repair.index("--wheelhouse") + 1], self.data["environment"]["wheelhouse"])
        receipt = json.loads((self.root / "result/results.json").read_text())
        self.assertEqual(receipt["stages"][1]["mode"], "verify_existing")
        self.assertEqual(receipt["stages"][3]["mode"], "verify_existing")
        old = receipt["previous_attempts"][0]
        self.assertEqual(old["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual((self.root / "result" / old["path"]).read_bytes(), before)
        prepare = next(args for args in self.calls if "prepare" in args)
        self.assertIn("--out", prepare)
        self.assertNotEqual(prepare[prepare.index("--out") + 1], str(self.root / "result/fixed-five-prepare"))

    def test_installation_verification_timeout_is_recoverable_before_any_campaign(self):
        def failed(args, **kwargs):
            result=self.execute(args, **kwargs)
            if len(args)>1 and Path(args[1]).name=="runtime_storage.py":
                return types.SimpleNamespace(returncode=124)
            return result
        result,_=self.invoke(failed)
        self.assertEqual(result,1)
        prior=workflow.bootstrap_resume(self.data)
        self.assertEqual(prior["stages"][-1]["name"],"install-engine")
        with patch("benchmark.one_click.verify_kit"),patch("child_runner.run_stage",side_effect=self.execute), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(workflow.run(self.data,resume_bootstrap=True),0)
        self.assertTrue(any(Path(a[1]).name=="runtime_storage.py" and "--reuse" in a for a in self.calls))

    def test_bootstrap_recovery_rejects_changed_physical_devices_before_work(self):
        self.failed_preflight()
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            workflow.bootstrap_resume(dict(self.data, physical_devices=[2, 3]))

    def test_bootstrap_recovery_refuses_an_existing_inference_campaign(self):
        self.failed_preflight()
        (self.root / "result/fixed-five").mkdir()
        with self.assertRaisesRegex(ValueError, "inference campaign already exists"):
            workflow.bootstrap_resume(self.data)

    def before_model_failure(self):
        def fail(args, **kwargs):
            result = self.execute(args, **kwargs)
            if "run" in args:
                self.write("fixed-five/campaign.json", dict(status="failed", selected_execution_complete=False,
                    stages=[dict(name="continuous-B0", status="failed"), dict(name="continuous-B1", status="planned")]))
                self.write("fixed-five/continuous-B0/receipt.json", dict(completed=False,
                    model_service_started=False, client_started=False, quiescent=True))
                return types.SimpleNamespace(returncode=2)
            return result
        self.invoke(fail)

    def test_pre_model_failure_can_be_archived_and_recovered_without_losing_evidence(self):
        self.data["environment"] = {"wheelhouse": str(self.root / "wheelhouse")}
        self.before_model_failure()
        original = (self.root / "result/fixed-five/campaign.json").read_bytes()
        with patch("benchmark.one_click.verify_kit"), patch("child_runner.run_stage", side_effect=self.execute), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(workflow.run(self.data, resume_before_model=True), 0)
        receipt = json.loads((self.root / "result/results.json").read_text())
        previous = self.root / "result" / receipt["previous_attempts"][-1]["campaign_directory"]
        self.assertEqual((previous / "fixed-five/campaign.json").read_bytes(), original)
        repair = [args for args in self.calls if "--repair-dependencies" in args][-1]
        self.assertEqual(repair[repair.index("--wheelhouse") + 1], self.data["environment"]["wheelhouse"])
        self.assertEqual(receipt["stages"][1]["mode"], "verify_existing")

    def startup_failure(self):
        self.before_model_failure()
        self.write("fixed-five/continuous-B0/receipt.json",dict(completed=False,role="B0",kind="continuous",model_service_started=True,client_started=False,quiescent=True))
        self.write("fixed-five/continuous-B0/service-process.json",dict(pid=123))

    def test_failed_model_startup_can_resume_without_repeating_any_formal_request(self):
        self.startup_failure()
        with patch("qualification.sealed_files") as sealed,patch("third_party_report.verify_campaign_bindings") as bindings, \
             patch("benchmark.one_click.verify_kit"),patch("child_runner.run_stage",side_effect=self.execute), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(workflow.run(self.data,resume_before_inference=True),0)
        sealed.assert_called_once();bindings.assert_called_once()
        receipt=json.loads((self.root/'result/results.json').read_text())
        self.assertTrue(receipt['previous_attempts'][-1]['campaign_directory'])

    def test_startup_recovery_rejects_a_started_formal_client(self):
        self.startup_failure()
        path=self.root/'result/fixed-five/continuous-B0/receipt.json'
        data=json.loads(path.read_text());data['client_started']=True;path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'refuses to repeat'):workflow.before_model_resume(self.data,allow_model_startup=True)

    def test_startup_recovery_rejects_client_evidence_even_if_flag_is_false(self):
        self.startup_failure();self.write('fixed-five/continuous-B0/client.log','request sent')
        with self.assertRaisesRegex(ValueError,'possible formal client'):workflow.before_model_resume(self.data,allow_model_startup=True)

    def test_startup_recovery_requires_intact_sealed_originals(self):
        self.startup_failure()
        with patch('qualification.sealed_files',side_effect=ValueError('seal changed')):
            with self.assertRaisesRegex(ValueError,'seal changed'):workflow.before_model_resume(self.data,allow_model_startup=True)

    def test_pre_model_recovery_rejects_any_started_service(self):
        self.before_model_failure()
        self.write("fixed-five/continuous-B0/service-process.json", dict(pid=123))
        with self.assertRaisesRegex(ValueError, "refuses to repeat"):
            workflow.before_model_resume(self.data)

    def test_pre_model_recovery_requires_explicit_false_model_receipt(self):
        self.before_model_failure()
        self.write("fixed-five/continuous-B0/receipt.json", dict(completed=False, quiescent=True))
        with self.assertRaisesRegex(ValueError, "refuses to repeat"):
            workflow.before_model_resume(self.data)

    def test_pre_model_recovery_rejects_changed_devices(self):
        self.before_model_failure()
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            workflow.before_model_resume(dict(self.data, physical_devices=[2, 3]))

    def test_bootstrap_recovery_never_repeats_a_completed_workflow(self):
        self.invoke()
        with self.assertRaisesRegex(ValueError, "failed fixed-five preflight"):
            workflow.bootstrap_resume(self.data)


    def test_cli_report_environment_uses_configured_cache_and_preserves_overrides(self):
        import report_environment
        for explicit in (False,True):
            with self.subTest(explicit=explicit),patch.dict(workflow.os.environ,{},clear=True):
                expected=self.root/('explicit-cache' if explicit else 'cache/xdg')
                if explicit:workflow.os.environ['XDG_CACHE_HOME']=str(expected)
                selected=[]
                def prepare(data, *, progress=None):
                    with patch.object(report_environment.Path,'home',return_value=self.root/'unused-home'), \
                         patch.object(report_environment.subprocess,'run',return_value=types.SimpleNamespace(returncode=0)):
                        selected.append(report_environment.report_python()[0])
                    return None
                with patch.object(workflow,'configuration',return_value=self.data),patch.object(workflow,'preflight'), \
                     patch('engine.release_selection.select',return_value=None), \
                     patch('engine.environment_setup.ensure_environment',side_effect=prepare), \
                     patch.object(sys,'argv',['run_all.py','--config','unused.json','--prepare-environment']), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(workflow.main(),0)
                self.assertTrue(selected[0].is_relative_to(expected),selected[0])
                self.assertEqual(workflow.os.environ['MPLCONFIGDIR'],str(expected/'matplotlib'))



class TerminalResultsTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name).resolve()
        self.out = self.root / "recorded-run"
        self.config = self.root / "evaluation.json"
        self.config.write_text(json.dumps({"out": self.out.name}), encoding="utf-8")
        self.report = self.out / "full-report/summary.json"
        self.report.parent.mkdir(parents=True)
        self.summary = {
            "schema": "deeplinkengine-full-acceptance-report/v1", "engine_release": "fixture-r10",
            "execution_complete": True,
            "targets": {
                "4.1": {"milestone": {"met": True}, "final": {"met": False},
                        "efficiency_lower_percent": "82.345"},
                "4.4": {"milestone": {"met": False, "evidence_status": "missing"},
                        "final": {"met": False, "evidence_status": "missing_measurement"},
                        "cost_basis": "actual_expense"},
            },
            "workloads": {
                "mmlu": {
                    "complete": True, "quality": {"met": True},
                    "arms": {"B0": {"output_tokens_per_second": 125, "mean_ttft_ms": 100},
                             "B1": {"output_tokens_per_second": 250, "mean_ttft_ms": 75}},
                },
                "swe": {"complete": True, "arms": {}},
            },
            "context_results": [
                {"phase": "4096-formal",
                 "B0": {"metrics": {"ttft_ms": {"p95": 400}}},
                 "B1": {"metrics": {"ttft_ms": {"p95": 350}}}},
            ],
        }
        self.report.write_text(json.dumps(self.summary), encoding="utf-8")
        self.report.with_name("report.html").write_text("rendered fixture", encoding="utf-8")
        self.receipt = self.out / "results.json"

    def query(self, *args):
        with patch.object(sys, "argv", ["run_all.py", "--results", *map(str, args)]), \
             patch.object(workflow, "select_run_engine", side_effect=AssertionError("results must not select an engine")), \
             patch.object(workflow, "run", side_effect=AssertionError("results must not execute a run")), \
             contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            code = workflow.main()
        return code, out.getvalue(), err.getvalue()

    def write_receipt(self, **extra):
        value = {"status": "completed", "stages": [{"name": "render-full-report", "status": "completed"}],
                 "evidence_sha256": {"full-report/summary.json": hashlib.sha256(self.report.read_bytes()).hexdigest()}}
        value.update(extra)
        self.receipt.write_text(json.dumps(value), encoding="utf-8")

    def test_terminal_results_show_sealed_decisions_and_both_arms(self):
        self.write_receipt(automated_acceptance_complete=False)
        before = {p: p.read_bytes() for p in self.out.rglob("*") if p.is_file()}
        code, text, errors = self.query("--config", self.config)
        self.assertEqual((code, errors), (0, ""))
        for fragment in ("fixture-r10", "4.1 Effective compute", "MISSING COST EVIDENCE",
                         "MISSING MEASUREMENT", "125.00 -> 250.00", "400.00 -> 350.00", "trace only"):
            self.assertIn(fragment, text)
        code, text, errors = self.query(self.receipt, "--json")
        self.assertEqual((code, errors), (0, ""))
        result = json.loads(text)
        self.assertEqual(result["summary"], self.summary)
        self.assertIs(result["summary_digest_verified"], True)
        self.assertIs(result["automated_acceptance_complete"], False)
        self.assertEqual(before, {p: p.read_bytes() for p in self.out.rglob("*") if p.is_file()})

    def test_changed_report_is_rejected_even_via_summary_path(self):
        self.write_receipt()
        self.report.write_bytes(self.report.read_bytes() + b" ")
        code, output, errors = self.query(self.report, "--json")
        self.assertEqual((code, output), (2, ""))
        self.assertIn("differs from the result receipt", errors)

    def test_failed_run_uses_its_receipt_instead_of_publishing_an_old_report(self):
        self.write_receipt(status="failed", error="NPU fixture stopped",
                           stages=[{"name": "run-fixed-five", "status": "failed", "returncode": 2}])
        code, text, errors = self.query(self.out, "--json")
        self.assertEqual((code, errors), (0, ""))
        result = json.loads(text)
        self.assertEqual((result["state"], result["stage"], result["returncode"]), ("failed", "run-fixed-five", 2))
        self.assertEqual(result["failure"], "NPU fixture stopped")
        self.assertNotIn("summary", result)

    def test_copied_summary_is_read_without_a_run_environment(self):
        copied = self.root / "copied-summary.json"
        copied.write_bytes(self.report.read_bytes())
        code, text, errors = self.query(copied, "--json")
        self.assertEqual((code, errors), (0, ""))
        result = json.loads(text)
        self.assertEqual(result["state"], "summary_only")
        self.assertEqual(result["summary"], self.summary)
        self.assertIs(result["summary_digest_verified"], False)


if __name__ == "__main__":
    unittest.main()
