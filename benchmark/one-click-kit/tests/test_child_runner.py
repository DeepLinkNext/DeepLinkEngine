"""Exercise cancellation through real subprocesses without loading a model."""

import os
import contextlib
import fcntl
import json
from unittest.mock import patch
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


class ChildRunnerTests(unittest.TestCase):
    def test_parent_waits_for_child_cleanup_on_interrupt_and_terminate(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        for cancellation in (signal.SIGINT, signal.SIGTERM):
            with (
                self.subTest(signal=cancellation),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                child = root / "child.py"
                child.write_text(
                    "import os,time\nfrom pathlib import Path\n"
                    f"root=Path({str(root)!r})\n"
                    "try:\n (root/'ready').write_text(str(os.getpid()))\n time.sleep(30)\n"
                    "finally:\n time.sleep(0.3)\n (root/'cleaned').write_text('yes')\n"
                )
                parent = root / "parent.py"
                parent.write_text(
                    "import sys\nfrom pathlib import Path\n"
                    f"sys.path.insert(0,{str(scripts)!r})\n"
                    "from child_runner import cancellation_handlers,run_stage\n"
                    "with cancellation_handlers():\n"
                    " try:\n"
                    f"  run_stage([sys.executable,{str(child)!r}])\n"
                    " except KeyboardInterrupt:\n"
                    f"  assert Path({str(root / 'cleaned')!r}).is_file()\n"
                    "  raise SystemExit(23)\n"
                )
                process = subprocess.Popen([sys.executable, str(parent)])
                try:
                    deadline = time.monotonic() + 10
                    while not (root / "ready").is_file():
                        if process.poll() is not None or time.monotonic() > deadline:
                            self.fail("Synthetic child did not become ready")
                        time.sleep(0.02)
                    child_pid = int((root / "ready").read_text())
                    process.send_signal(cancellation)
                    self.assertEqual(process.wait(timeout=10), 23)
                    self.assertTrue((root / "cleaned").is_file())
                    with self.assertRaises(ProcessLookupError):
                        os.kill(child_pid, 0)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()


    @contextlib.contextmanager
    def lease_fixture(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        import resource_guard
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            root = Path(temp)
            paths = [str(root / name) for name in ("card-0.lock", "card-7.lock", "model.lock")]
            handles = [stack.enter_context(open(name, "a+b")) for name in paths]
            for handle in handles:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            entries = [dict(path=name, fd=handle.fileno()) for name, handle in zip(paths, handles)]
            with patch.dict(os.environ, {"VLLMHUST_ENGINE_LOCKS": json.dumps(entries),
                    "LEASE_TEST_PATHS": json.dumps(paths)}), patch.object(resource_guard, "paths", return_value=paths):
                yield root, entries, handles

    def test_lease_descriptors_survive_two_real_stage_boundaries(self):
        with self.lease_fixture() as (root, entries, handles):
            from child_runner import run_stage
            worker = root / "worker.py"
            marker = root / "inherited.json"
            scripts = Path(__file__).resolve().parents[1] / "scripts"
            worker.write_text("\n".join([
                "import fcntl,json,os,sys", "from pathlib import Path",
                f"sys.path.insert(0, {str(scripts)!r})", "import resource_guard",
                "resource_guard.paths=lambda:json.loads(os.environ['LEASE_TEST_PATHS'])",
                "from child_runner import run_stage",
                "if len(sys.argv)==1:",
                " result=run_stage([sys.executable,__file__,'leaf'],env=dict(os.environ))",
                " raise SystemExit(result.returncode)",
                "entries=resource_guard.inherited()", "assert len(entries)==3",
                "for entry in entries:", " other=os.open(entry['path'],os.O_RDWR)",
                " try:", "  try:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)",
                "  except BlockingIOError:pass", "  else:raise AssertionError('Lease lost exclusivity')",
                " finally:os.close(other)",
                f"Path({str(marker)!r}).write_text(json.dumps([e['path'] for e in entries]))", ""
            ]))
            with (root / "worker.log").open("w") as log:
                result = run_stage([sys.executable, str(worker)], stdout=log, stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode, 0, (root / "worker.log").read_text())
            self.assertEqual(json.loads(marker.read_text()), [e["path"] for e in entries])
            for entry in entries:
                self.assertEqual(os.fstat(entry["fd"]).st_ino, os.stat(entry["path"]).st_ino)

    def test_invalid_parent_leases_fail_before_launch(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        import child_runner
        for failure in ("closed", "different_environment"):
            with self.subTest(failure=failure), self.lease_fixture() as (root, entries, handles):
                env = None
                if failure == "closed":
                    handles[0].close()
                    expected = OSError
                else:
                    env = dict(os.environ, VLLMHUST_ENGINE_LOCKS="[]")
                    expected = ValueError
                with patch.object(child_runner.subprocess, "Popen") as launch:
                    with self.assertRaises(expected):
                        child_runner.run_stage([sys.executable, "-c", "pass"], env=env)
                    launch.assert_not_called()

    def test_explicit_environment_can_drop_parent_lease_authority(self):
        with self.lease_fixture() as (root, entries, handles):
            from child_runner import run_stage
            env = dict(os.environ)
            env.pop("VLLMHUST_ENGINE_LOCKS")
            code = "import os,sys\nfor raw in sys.argv[1:]:\n try:os.fstat(int(raw))\n except OSError:pass\n else:raise AssertionError('Unexpected descriptor inheritance')\n"
            self.assertEqual(run_stage([sys.executable, "-c", code, *[str(e["fd"]) for e in entries]], env=env).returncode, 0)



if __name__ == "__main__":
    unittest.main()
