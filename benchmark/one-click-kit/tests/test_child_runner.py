"""Exercise cancellation through real subprocesses without loading a model."""

import os
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


if __name__ == "__main__":
    unittest.main()
