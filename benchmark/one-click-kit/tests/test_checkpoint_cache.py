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
                    before_service=True,
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


class StartupReserveTests(unittest.TestCase):
    def test_real_pages_are_held_until_healthy_idle_boundary_and_released_once(self):
        from checkpoint_cache import StartupMemoryReserve
        chunks = [SimpleNamespace(close=__import__('unittest.mock', fromlist=['Mock']).Mock()) for _ in range(64)]
        value = dict(limit=64*2**30, current=50*2**30)
        checks = []
        with tempfile.TemporaryDirectory() as temp, patch('checkpoint_cache.allocate_private_chunk', side_effect=chunks):
            reserve = StartupMemoryReserve(Path(temp)/'reserve.json', lambda: value, lambda: checks.append(True))
            self.assertFalse(any(c.close.called for c in chunks))
            reserve.before_warmup();reserve.close('cleanup')
            self.assertTrue(all(c.close.call_count == 1 for c in chunks))
            self.assertEqual(len(checks),65)
            self.assertEqual(reserve.record['release_reason'],'healthy-idle-service-before-first-warmup')

    def test_guard_failure_releases_every_allocated_page_before_model_start(self):
        from checkpoint_cache import StartupMemoryReserve
        from unittest.mock import Mock
        chunk=SimpleNamespace(close=Mock())
        with tempfile.TemporaryDirectory() as temp, patch('checkpoint_cache.allocate_private_chunk',return_value=chunk):
            with self.assertRaisesRegex(RuntimeError,'working reserve'):
                StartupMemoryReserve(Path(temp)/'reserve.json', lambda:dict(limit=64*2**30,current=50*2**30), lambda: (_ for _ in ()).throw(RuntimeError('working reserve')))
            chunk.close.assert_called_once()
            self.assertTrue(json.loads((Path(temp)/'reserve.json').read_text())['released'])

    def test_cache_refill_cannot_silently_enter_warmup_without_raw_headroom(self):
        from checkpoint_cache import StartupMemoryReserve
        with tempfile.TemporaryDirectory() as temp, patch('checkpoint_cache.allocate_private_chunk',return_value=SimpleNamespace(close=lambda:None)):
            value=dict(limit=64*2**30,current=50*2**30)
            reserve=StartupMemoryReserve(Path(temp)/'reserve.json',lambda:value,lambda:None)
            value['current']=63*2**30
            with self.assertRaisesRegex(RuntimeError,'raw memory before warmup'):
                reserve.before_warmup()
            self.assertTrue(reserve.record['released'])

    def test_input_cache_refill_cannot_fill_reserve_against_a_live_startup(self):
        from checkpoint_cache import StartupMemoryReserve
        with tempfile.TemporaryDirectory() as temp, patch('checkpoint_cache.allocate_private_chunk') as allocate:
            with self.assertRaisesRegex(RuntimeError,'12GiB raw memory immediately before'):
                StartupMemoryReserve(Path(temp)/'reserve.json',lambda:dict(limit=64*2**30,current=60*2**30),lambda:None)
            allocate.assert_not_called()
            record=json.loads((Path(temp)/'reserve.json').read_text())
            self.assertTrue(record['released'])
            self.assertFalse(record['model_started_at_allocation'])

    def test_final_cache_admission_runs_before_holding_reserve_or_starting_model(self):
        from run_b1_r4 import prepare_startup_reserve
        from unittest.mock import Mock
        value=dict(limit=64*2**30,current=50*2**30,working_estimate=40*2**30,stat=dict(file=20*2**30),events={})
        guard=SimpleNamespace(sample=lambda:value,check=Mock())
        sequence=[]
        def cleanup(*args,**kwargs):
            self.assertTrue(kwargs['before_service']);sequence.append('cleanup')
        def reserve(*args):
            self.assertEqual(sequence,['cleanup']);sequence.append('reserve');return 'held'
        with tempfile.TemporaryDirectory() as temp, patch('run_b1_r4.release_loaded_checkpoint_cache',side_effect=cleanup), patch('checkpoint_cache.StartupMemoryReserve',side_effect=reserve):
            self.assertEqual(prepare_startup_reserve(guard,{},Path(temp),{},Path(temp)),'held')
        guard.check.assert_called_once()
        self.assertEqual(sequence,['cleanup','reserve'])

    def test_buffer_lifetime_excludes_both_warmup_and_formal_clients(self):
        import ast
        module=ast.parse((Path(__file__).resolve().parents[1]/'scripts/run_b1_r4.py').read_text())
        function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='admitted_run')
        lines={}
        for n in ast.walk(function):
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('startup_reserve','service','warmup','client') for t in n.targets):
                if isinstance(n.value,ast.Call):
                    for t in n.targets:
                        if isinstance(t,ast.Name):lines[t.id]=n.lineno
        releases=[n.lineno for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='before_warmup']
        self.assertLess(lines['startup_reserve'],lines['service'])
        self.assertTrue(any(lines['service']<line<lines['warmup'] for line in releases))
        self.assertTrue(any(lines['warmup']<line<lines['client'] for line in releases))


class RuntimeMemoryReserveTests(unittest.TestCase):
    def state(self, working=44):
        return dict(limit=64*2**30,current=63*2**30,working_estimate=working*2**30,
                    stat=dict(file=40*2**30,inactive_file=20*2**30,shmem=0,file_dirty=0,file_writeback=0),events=dict(oom=0))

    def test_after_load_clean_cache_does_not_invoke_startup_pressure_or_raw_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'fixture.safetensors';p.write_bytes(b'fixture')
            weights={p.name:dict(bytes=7,sha256=hashlib.sha256(b'fixture').hexdigest())}
            with patch('checkpoint_cache.allocate_private_chunk',side_effect=AssertionError('No pressure allocation with a live model')):
                record=release_loaded_checkpoint_cache(root,weights,root/'cache.json',self.state)
            self.assertEqual(record['required_raw_headroom_bytes'],0)
            self.assertEqual(p.read_bytes(),b'fixture')

    def test_loaded_worker_uses_original_working_and_oom_guard_even_when_raw_cache_is_full(self):
        from run_b1_r4 import monitor_loaded_memory
        checks=[]
        guard=SimpleNamespace(sample=self.state,check=lambda value,initial:checks.append(value))
        with tempfile.TemporaryDirectory() as temp:
            self.assertTrue(monitor_loaded_memory(guard,{},Path(temp),Path(temp),{},True))
        self.assertEqual(len(checks),1)
        self.assertLess(checks[0]['limit']-checks[0]['current'],4*2**30)
        self.assertGreater(checks[0]['limit']-checks[0]['working_estimate'],4*2**30)

    def test_real_working_pressure_and_new_oom_still_abort_before_service_progress(self):
        from run_b1_r4 import monitor_loaded_memory
        for reason in ('4GiB working-set reserve reached','New cgroup OOM event'):
            guard=SimpleNamespace(sample=self.state,check=lambda *args:(_ for _ in ()).throw(RuntimeError(reason)))
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(RuntimeError,reason):monitor_loaded_memory(guard,{},Path(temp),Path(temp),{},True)

    def test_invalid_cache_counters_still_fall_back_to_raw_reserve(self):
        from run_b1_r4 import monitor_loaded_memory
        def state():
            value=self.state();value['stat']['file_writeback']=2**64-1;return value
        def check(value,initial):
            self.assertEqual(value['working_estimate'],value['current'])
            raise RuntimeError('raw fallback reserve reached')
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError,'raw fallback'):
                monitor_loaded_memory(SimpleNamespace(sample=state,check=check),{},Path(temp),Path(temp),{},True)

    def test_underflow_longer_than_three_reads_recovers_without_zeroing_counters(self):
        from run_b1_r4 import check_memory
        bad=self.state();bad['stat']['file_writeback']=2**64-28672
        good=self.state();checks=[]
        with tempfile.TemporaryDirectory() as temp, patch('run_b1_r4.time.sleep'):
            guard=SimpleNamespace(sample=__import__('unittest.mock',fromlist=['Mock']).Mock(side_effect=[bad,bad,bad,good]),check=lambda value,initial:checks.append(value))
            value=check_memory(guard,{},Path(temp))
            rows=[json.loads(x) for x in (Path(temp)/'memory-samples.jsonl').read_text().splitlines()]
        self.assertEqual(value['counter_validation']['attempt'],4)
        self.assertEqual(len(checks),1)
        self.assertEqual(rows[0]['stat']['file_writeback'],2**64-28672)
        self.assertEqual(checks[0]['working_estimate'],good['working_estimate'])

    def test_oom_during_invalid_sample_aborts_without_waiting_for_valid_counters(self):
        from run_b1_r4 import check_memory
        from unittest.mock import Mock
        for name in ('oom','oom_kill','oom_group_kill'):
            value=self.state();value['stat']['file_writeback']=2**64-1;value['events'][name]=1
            guard=SimpleNamespace(sample=Mock(return_value=value),check=Mock())
            with tempfile.TemporaryDirectory() as temp, patch('run_b1_r4.time.sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError,'New cgroup '+name):check_memory(guard,{'events':{name:0}},Path(temp))
                self.assertEqual(guard.sample.call_count,1)
                sleep.assert_not_called();guard.check.assert_not_called()

    def test_persistent_invalid_counters_have_bounded_wait_and_keep_raw_guard(self):
        from run_b1_r4 import check_memory
        from unittest.mock import Mock
        value=self.state();value['stat']['file_writeback']=2**64-1
        guard=SimpleNamespace(sample=Mock(side_effect=lambda:json.loads(json.dumps(value))),check=Mock())
        with tempfile.TemporaryDirectory() as temp, patch('run_b1_r4.time.sleep') as sleep:
            result=check_memory(guard,{},Path(temp))
        self.assertEqual(guard.sample.call_count,20)
        self.assertEqual(sleep.call_count,19)
        self.assertEqual(result['working_estimate'],result['current'])
        self.assertEqual(result['counter_validation']['fallback'],'raw_current_no_cache_credit')
        guard.check.assert_called_once()


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
                result=release_loaded_checkpoint_cache(root,{p.name:dict(bytes=7,sha256="fixture")},root/"receipt.json",self.state)
                self.assertEqual(result["required_raw_headroom_bytes"],0)


if __name__ == "__main__":
    unittest.main()
