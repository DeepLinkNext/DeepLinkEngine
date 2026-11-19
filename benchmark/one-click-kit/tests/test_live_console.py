"""Real foreground log streaming and cancellation without devices or model imports."""

import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from live_console import Follow, progress  # noqa: E402 - standalone script import


class LiveConsoleTests(unittest.TestCase):
    def test_reference_only_counts_declared_stages_and_current_http_responses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "compute-reference"
            leaf = reference / "http-current-work"
            (leaf / "service").mkdir(parents=True)
            ids = ["qualified-" + str(i) for i in range(5)] + ["current-http-work-observation"] + ["final-" + str(i) for i in range(4)]
            (reference / "automatic-plan.json").write_text(json.dumps(dict(stages=[dict(id=i) for i in ids])))
            (reference / "status.json").write_text(json.dumps(dict(status="running", stages=[
                dict(id=i, status="completed" if index < 5 else "running") for index, i in enumerate(ids[:6])
            ])))
            (leaf / "receipt.json").write_text('{"status":"running-client"}')
            (leaf / "service/case-000.json").write_text("{}")
            state = progress(root)
            self.assertEqual((state["stages_done"], state["stages_total"], state["responses"], state["planned"]), (5, 10, 1, 200))
            self.assertEqual(state["stage"], "compute-reference/current-http-work-observation")
            self.assertEqual(state["phase"], "running-client")

    def test_reference_monitor_streams_current_service_without_scanning_compiler_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            leaf = root / "compute-reference/native-current-timing"
            leaf.mkdir(parents=True)
            (root / "compute-reference/status.json").write_text(json.dumps(dict(status="running", stages=[
                dict(id="current-native-synchronized-timing", status="running")
            ])))
            (leaf / "service.log").write_text("real native initialization\n")
            cache = leaf / "compiler-cache"
            cache.mkdir()
            (cache / "collector.log").write_text("transient cache must not be followed\n")
            logs = Path(temp) / "logs"
            logs.mkdir()
            output = io.StringIO()
            follow = Follow(root, logs, stream=output, persist=False)
            follow.tick()
            self.assertIn("real native initialization", output.getvalue())
            self.assertNotIn("transient cache", output.getvalue())
            self.assertIn("stages=0/1", output.getvalue())

    def test_nested_pair_counts_and_reused_stages_are_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "execution-checklist.json").write_text(
                json.dumps(
                    {
                        "stages": [
                            {"name": "continuous-B0", "status": "reused_verified"},
                            {"name": "mmlu-pair", "status": "running"},
                        ]
                    }
                )
            )
            leaf = root / "mmlu-pair/B1"
            (leaf / "service").mkdir(parents=True)
            (leaf / "receipt.json").write_text('{"status":"running-client"}')
            (leaf / "service/case-000.json").write_text("{}")
            d = progress(root)
            self.assertEqual(
                (d["stage"], d["responses"], d["planned"], d["stages_done"]),
                ("mmlu-pair/B1", 1, 200, 1),
            )
            # Nested stage checks match on the POSIX name on every platform.
            self.assertNotIn("\\", d["stage"])

    def test_progress_exposes_prepared_and_failed_stages_for_status_queries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "execution-checklist.json").write_text(
                json.dumps(
                    {
                        "status": "prepared_only",
                        "stages": [{"name": "continuous-B0", "status": "planned"}],
                    }
                )
            )
            prepared = progress(root)
            self.assertEqual(
                (prepared["stage"], prepared["phase"], prepared["stages_done"]),
                ("prepare", "prepared_only", 0),
            )
            (root / "execution-checklist.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "failure": "RuntimeError: boom",
                        "stages": [{"name": "hle-solver-B1", "status": "failed"}],
                    }
                )
            )
            leaf = root / "hle-solver-B1"
            leaf.mkdir()
            (leaf / "receipt.json").write_text(
                '{"status":"failed","failure":"RuntimeError: boom"}'
            )
            failed = progress(root)
            self.assertEqual(
                (
                    failed["stage"],
                    failed["phase"],
                    failed["failure"],
                    failed["stages_done"],
                ),
                ("hle-solver-B1", "failed", "RuntimeError: boom", 0),
            )

    def test_follow_handles_split_utf8_and_file_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            logs = root / "logs"
            logs.mkdir()
            stream = io.StringIO()
            follow = Follow(root / "run", logs, stream)
            path = logs / "controller.log"
            raw = "启动".encode()
            path.write_bytes(raw[:2])
            follow.tail(path)
            with path.open("ab") as f:
                f.write(raw[2:])
            follow.tail(path)
            (logs / "new.log").write_text("replacement")
            (logs / "new.log").replace(path)
            follow.tail(path)
            follow.console.close()
            self.assertIn("启动", stream.getvalue())
            self.assertIn("replacement", stream.getvalue())

    def test_logs_and_failure_exit_are_preserved_outside_campaign(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            child = root / "child.py"
            child.write_text(
                "from pathlib import Path\nimport json\n"
                f"r=Path({str(run)!r});r.mkdir()\n"
                '(r/"execution-checklist.json").write_text(json.dumps({"status":"failed",'
                '"failure":"synthetic failure","stages":[{"name":"hle-solver-B1","status":"failed"}]}))\n'
                'p=r/"hle-solver-B1";p.mkdir()\n'
                '(p/"service.log").write_text("MODEL LOAD LOG\\n")\n'
                '(p/"client.log").write_text("CLIENT ERROR\\n")\n'
                'print("CONTROLLER ERROR",flush=True)\nraise SystemExit(7)\n'
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "live_console.py"),
                    "--run",
                    str(run),
                    "--",
                    sys.executable,
                    str(child),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 7, result.stderr)
            for expected in [
                "CONTROLLER ERROR",
                "MODEL LOAD LOG",
                "CLIENT ERROR",
                "stages=0/1",
                "code=7",
            ]:
                self.assertIn(expected, result.stdout)
                self.assertIn(expected, (root / "run-logs/console.log").read_text())
            self.assertFalse((run / "console.log").exists())
            self.assertTrue(
                json.loads((root / "run-logs/exit.json").read_text())[
                    "controller_reaped"
                ]
            )

    def test_interrupt_and_terminate_wait_for_cleanup(self):
        for signum in [signal.SIGINT, signal.SIGTERM]:
            with self.subTest(signal=signum), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                run = root / "run"
                child = root / "child.py"
                child.write_text(
                    "import os,time\nfrom pathlib import Path\n"
                    f"r=Path({str(root)!r})\n"
                    'try:\n (r/"ready").write_text(str(os.getpid()))\n time.sleep(30)\n'
                    'finally:\n time.sleep(0.2)\n (r/"cleaned").write_text("yes")\n'
                )
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(SCRIPTS / "live_console.py"),
                        "--run",
                        str(run),
                        "--",
                        sys.executable,
                        str(child),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                try:
                    deadline = time.monotonic() + 10
                    while not (root / "ready").exists():
                        if process.poll() is not None or time.monotonic() > deadline:
                            self.fail("Synthetic controller did not start")
                        time.sleep(0.02)
                    pid = int((root / "ready").read_text())
                    process.send_signal(signum)
                    self.assertEqual(process.wait(timeout=10), 130)
                    self.assertTrue((root / "cleaned").exists())
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()

    def test_follow_only_cannot_launch_inference_and_waits_for_report_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            run.mkdir()
            (run / "campaign.json").write_text('{"status":"completed"}')
            logs = root / "run-logs"
            logs.mkdir()
            (logs / "controller.log").write_text("report rendering\n")
            p = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPTS / "live_console.py"),
                    "--run",
                    str(run),
                    "--follow-only",
                ],
                stdout=subprocess.DEVNULL,
            )
            try:
                time.sleep(0.3)
                self.assertIsNone(p.poll())
                (logs / "exit.json").write_text('{"returncode":7}')
                self.assertEqual(p.wait(timeout=10), 7)
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait()


if __name__ == "__main__":
    unittest.main()
