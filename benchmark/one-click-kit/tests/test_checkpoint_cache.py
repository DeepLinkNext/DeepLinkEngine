import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_cache import release_loaded_checkpoint_cache, reclaim_plan, reclaim_before_service
from unittest.mock import patch
from types import SimpleNamespace


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


class BoundedPreServiceReclaimTests(unittest.TestCase):
    def state(self, current=63):
        gib = 2**30
        return dict(limit=64*gib, current=current*gib, working_estimate=23*gib,
            stat=dict(file=(current-8)*gib, inactive_file=(current-23)*gib,
                      shmem=0, file_dirty=0, file_writeback=0, file_mapped=0, unevictable=0), events=dict(oom=0, oom_kill=0))

    def test_plan_preserves_original_working_reserve_and_caps_temporary_memory(self):
        value = self.state()
        plan = reclaim_plan(value, 12*2**30)
        self.assertTrue(plan["eligible"])
        self.assertLessEqual(plan["temporary_bytes"], 16*2**30)
        self.assertLess(plan["non_file_estimate"] + plan["temporary_bytes"], value["limit"] - 4*2**30)

    def test_repeated_hashing_active_cache_is_reclaimed_before_model_creation(self):
        value=self.state();value["stat"]["inactive_file"]=6*2**30
        value["working_estimate"]=value["current"]-6*2**30
        plan=reclaim_plan(value,12*2**30)
        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["temporary_bytes"],16*2**30)
        self.assertEqual(plan["inactive_file_observed_bytes"],6*2**30)

    def test_mapped_and_locked_file_pages_are_never_credited(self):
        for name in ("file_mapped","unevictable"):
            value=self.state();value["stat"][name]=48*2**30
            self.assertFalse(reclaim_plan(value,12*2**30)["eligible"])

    def test_anon_dirty_and_missing_cache_are_not_credited_as_reclaimable(self):
        for mutate in (lambda s: s["stat"].pop("inactive_file"),
                       lambda s: s["stat"].update(file_writeback=2**64-1),
                       lambda s: s["stat"].update(file_dirty=40*2**30),
                       lambda s: s["stat"].update(shmem=40*2**30)):
            value=self.state();mutate(value)
            self.assertFalse(reclaim_plan(value, 12*2**30)["eligible"])

    def test_every_temporary_page_is_released_before_the_final_sample(self):
        allocated, closed = [], []
        def allocate(size):
            allocated.append(size)
            return SimpleNamespace(close=lambda: closed.append(size))
        def sample():
            return self.state(47 if allocated and len(closed)==len(allocated) else 63)
        with patch("checkpoint_cache.allocate_private_chunk", side_effect=allocate):
            result = reclaim_before_service(sample, 12*2**30)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(sum(allocated), 16*2**30)
        self.assertEqual(allocated, closed)
        self.assertGreaterEqual(result["after"]["limit"]-result["after"]["current"],12*2**30)
        self.assertFalse(result["model_service_started"])
        self.assertFalse(result["memory_limit_changed"])

    def test_oom_aborts_and_releases_partial_allocation(self):
        allocated, closed = [], []
        def allocate(size):
            allocated.append(size)
            return SimpleNamespace(close=lambda: closed.append(size))
        def sample():
            value = self.state()
            value["events"]["oom"] = int(len(allocated)>=2)
            return value
        with patch("checkpoint_cache.allocate_private_chunk", side_effect=allocate):
            result = reclaim_before_service(sample, 12*2**30)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(allocated),2)
        self.assertEqual(allocated, closed)

    def test_persistent_invalid_counters_never_trigger_allocation(self):
        value=self.state();value["stat"]["file_writeback"]=2**64-1
        with patch("checkpoint_cache.allocate_private_chunk") as allocate:
            result=reclaim_before_service(lambda: value,12*2**30)
        allocate.assert_not_called()
        self.assertEqual(result["status"],"declined")
        self.assertEqual(len(result["sampling_attempts"]),3)

    def test_one_invalid_counter_sample_is_retained_before_a_valid_observation(self):
        invalid=self.state();invalid["stat"]["file_writeback"]=2**64-1
        valid=self.state(40)
        with patch("checkpoint_cache.allocate_private_chunk") as allocate:
            samples=iter([invalid,valid])
            result=reclaim_before_service(lambda: next(samples),12*2**30)
        allocate.assert_not_called()
        self.assertEqual(result["sampling_attempts"],[invalid,valid])
        self.assertFalse(result["plan"]["eligible"])

    def test_after_load_entry_never_runs_temporary_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/"fixture.safetensors";p.write_bytes(b"fixture")
            with patch("checkpoint_cache.reclaim_before_service", side_effect=AssertionError("not before service")):
                with self.assertRaisesRegex(RuntimeError,"Insufficient raw cgroup headroom"):
                    release_loaded_checkpoint_cache(root,{p.name:dict(bytes=7,sha256="fixture")},root/"receipt.json",self.state)


if __name__ == "__main__":
    unittest.main()
