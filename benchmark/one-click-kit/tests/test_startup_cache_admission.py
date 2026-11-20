"""Synthetic memory observations exercise guard order; no model/NPU work."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_b1_r4 import admit_startup_memory  # noqa: E402
from baseline_image_sdk import file_map  # noqa: E402


class Guard:
    def __init__(self, current, oom=False):
        self.current = current
        self.released = False
        self.oom = oom
        self.checked = False

    def sample(self):
        return {
            "limit": 64 * 2**30,
            "current": self.current * 2**30,
            "working_estimate": self.current * 2**30,
            "stat": {"file": 0},
            "events": {"oom": int(self.oom and self.released)},
        }

    def check(self, value, initial):
        self.checked = True
        if value["events"]["oom"] > initial["events"]["oom"]:
            raise RuntimeError("OOM counter increased")
        if value["limit"] - value["working_estimate"] < 4 * 2**30:
            raise RuntimeError("4GiB reserve reached")


class StartupCacheAdmissionTests(unittest.TestCase):
    def test_sdk_hash_cache_cannot_poison_checkpoint_only_startup_admission(self):
        # Reproduce the observed sequence without allocating GiB: SDK hashing
        # raises cgroup cache to63GiB; checkpoint hints cannot free SDK pages.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sdk = root / "sdk/rootfs/usr/local"
            sdk.mkdir(parents=True)
            binary = sdk / "original-library.so"
            binary.write_bytes(b"unchanged pinned SDK")
            weights = self.fixture(root)
            guard = Guard(24)
            original_digest = hashlib.file_digest

            def digest(stream, algorithm):
                guard.current = 63
                return original_digest(stream, algorithm)

            def hint(fd, _offset, _length, _advice):
                import os

                if Path(os.readlink(f"/proc/self/fd/{fd}")) == binary:
                    guard.current = 24

            with (
                patch("baseline_image_sdk.hashlib.file_digest", side_effect=digest),
                patch("baseline_image_sdk.os.posix_fadvise", side_effect=hint),
            ):
                admitted = file_map(root / "sdk")
                admit_startup_memory(guard, root, weights, root)
            self.assertTrue(guard.checked)
            self.assertEqual(
                admitted["rootfs/usr/local/original-library.so"],
                {
                    "bytes": binary.stat().st_size,
                    "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                },
            )

    def fixture(self, root):
        p = root / "fixture.safetensors"
        p.write_bytes(b"fixture")
        st = p.stat()
        return {
            p.name: {
                "bytes": st.st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "inode": st.st_ino,
                "mtime_ns": st.st_mtime_ns,
            }
        }

    def test_active_file_cache_is_actually_released_before_guard_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = self.fixture(root)
            guard = Guard(63)

            def hint(*args):
                self.assertFalse(guard.checked)
                guard.released = True
                guard.current = 2

            with patch("checkpoint_cache.os.posix_fadvise", side_effect=hint) as advise:
                admit_startup_memory(guard, root, weights, root)
            advise.assert_called_once()
            self.assertTrue(guard.checked)
            self.assertTrue((root / "memory-initial.json").exists())
            self.assertTrue((root / "checkpoint-cache-startup.json").exists())

    def test_hint_without_actual_headroom_cannot_bypass_reserve(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = self.fixture(root)
            guard = Guard(63)
            with patch("checkpoint_cache.os.posix_fadvise"):
                with self.assertRaisesRegex(
                    RuntimeError, "Insufficient raw cgroup headroom"
                ):
                    admit_startup_memory(guard, root, weights, root)
            self.assertTrue((root / "checkpoint-cache-startup.json").exists())

    def test_oom_during_reclamation_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = self.fixture(root)
            guard = Guard(63, oom=True)

            def hint(*args):
                guard.released = True
                guard.current = 2

            with patch("checkpoint_cache.os.posix_fadvise", side_effect=hint):
                with self.assertRaisesRegex(RuntimeError, "OOM counter increased"):
                    admit_startup_memory(guard, root, weights, root)


if __name__ == "__main__":
    unittest.main()
