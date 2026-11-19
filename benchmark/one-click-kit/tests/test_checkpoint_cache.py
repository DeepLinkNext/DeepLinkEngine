import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_cache import release_loaded_checkpoint_cache


def sample():
    return {"limit": 128 * 2**30, "current": 80 * 2**30}


class CheckpointCacheTests(unittest.TestCase):
    def test_read_only_cache_hint_preserves_checkpoint_bytes_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            path = model / "fixture.safetensors"
            raw = b"checkpoint-fixture" * 8192
            path.write_bytes(raw)
            original = path.stat()
            weights = {
                path.name: {
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "inode": original.st_ino,
                    "mtime_ns": original.st_mtime_ns,
                }
            }
            receipt = release_loaded_checkpoint_cache(
                model, weights, root / "receipt.json", sample
            )
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(path.stat().st_mtime_ns, original.st_mtime_ns)
            self.assertFalse(receipt["global_drop_caches"])
            self.assertFalse(receipt["file_content_written"])
            self.assertEqual(json.loads((root / "receipt.json").read_bytes()), receipt)

    def test_low_raw_headroom_stops_even_with_a_low_working_estimate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "fixture.safetensors"
            path.write_bytes(b"fixture")
            weights = {
                path.name: {
                    "bytes": 7,
                    "sha256": hashlib.sha256(b"fixture").hexdigest(),
                }
            }
            with self.assertRaisesRegex(
                RuntimeError, "Insufficient raw cgroup headroom"
            ):
                release_loaded_checkpoint_cache(
                    root,
                    weights,
                    root / "receipt.json",
                    lambda: {
                        "limit": 128 * 2**30,
                        "current": 127 * 2**30,
                        "working_estimate": 40 * 2**30,
                    },
                )
            self.assertTrue((root / "receipt.json").exists())

    def test_cache_hint_cannot_target_an_external_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            other = root / "other.safetensors"
            other.write_bytes(b"outside")
            (model / "alias.safetensors").symlink_to(other)
            with self.assertRaisesRegex(ValueError, "admitted checkpoint"):
                release_loaded_checkpoint_cache(
                    model,
                    {"alias.safetensors": {"bytes": 7, "sha256": "fixture"}},
                    root / "receipt.json",
                    sample,
                )
            self.assertEqual(other.read_bytes(), b"outside")


if __name__ == "__main__":
    unittest.main()
