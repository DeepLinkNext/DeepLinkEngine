"""CPU checks for separate official identities and safe OCI restoration."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official import enable_module_imports, worker_pythonpath

enable_module_imports()
from official_baselines import configuration
from official_image import destination, extract, read_lock, environment, matches_version
from official_storage import mounts, candidates


class OfficialBaselines(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    @unittest.skipUnless(sys.platform == 'linux', 'Sticky device leases are a Linux host interface')
    def test_official_leases_keep_the_existing_sticky_lock_and_check_idle(self):
        import errno
        from official_resource_compat import acquire

        scripts = Path(__file__).resolve().parents[1] / 'benchmark/one-click-kit/scripts'
        with patch.object(sys, 'path', [str(scripts), *sys.path]):
            import resource_guard as guard
        sticky = self.root / 'sticky'
        sticky.mkdir()
        sticky.chmod(0o1777)
        lock = sticky / 'device.lock'
        lock.write_bytes(b'original')
        identity = lock.stat()
        original_acquire = guard.acquire

        def acquire_original(output, timeout):
            if 'VLLMHUST_ENGINE_LOCKS' not in os.environ:
                raise PermissionError(errno.EACCES, 'protected_regular', str(lock))
            return original_acquire(output, timeout)

        sample = ('| 0 910B2 | OK\n| 1 910B2 | OK\nNo running processes found in NPU 0\n'
                  'No running processes found in NPU 1\n')
        with patch.dict(os.environ, {'VLLM_HUST_PHYSICAL_DEVICES': '0,1'}, clear=True), \
                patch.object(guard, 'paths', return_value=[str(lock)]), \
                patch.object(guard, 'acquire', side_effect=acquire_original), \
                patch('official_resource_compat.subprocess.run',
                      return_value=subprocess.CompletedProcess(['npu-smi'], 0, stdout=sample)):
            entries = acquire(self.root, 0)
            try:
                self.assertEqual(os.fstat(entries[0]['fd']).st_ino, identity.st_ino)
                self.assertTrue(json.loads((self.root/'resource-compatibility.json').read_bytes())['admitted'])
            finally:
                guard.release(entries)
        self.assertEqual(lock.read_bytes(), b'original')
        self.assertEqual(lock.stat().st_ino, identity.st_ino)

    def test_cgroup_v1_guard_admits_real_counters_and_checks_ancestors_and_oom(self):
        from types import SimpleNamespace
        from official_memory import CgroupV1Memory

        proc = self.root / 'proc'
        (proc / 'self').mkdir(parents=True)
        mount = self.root / 'cgroup-memory'
        leaf = mount / 'jobs' / 'solver'
        leaf.mkdir(parents=True)
        (proc / 'self/cgroup').write_text('11:memory:/jobs/solver\n')
        (proc / 'self/mountinfo').write_text(f'45 31 0:42 / {mount} ro - cgroup cgroup rw,memory\n')
        gib = 2**30
        (proc / 'meminfo').write_text(f'MemTotal: {128 * gib // 1024} kB\nMemFree: {80 * gib // 1024} kB\n')
        (proc / 'vmstat').write_text('oom_kill 1\n')

        for scope, current, limit in ((leaf, 8, 128), (leaf.parent, 30, 64), (mount, 36, 128)):
            (scope / 'memory.use_hierarchy').write_text('1\n')
            (scope / 'memory.stat').write_text(f'cache {4*gib}\ninactive_file {2*gib}\n'
                f'shmem 0\ndirty 0\nwriteback 0\nhierarchical_memory_limit {64*gib if scope != mount else 128*gib}\n')
            (scope / 'memory.usage_in_bytes').write_text(str(current*gib))
            (scope / 'memory.limit_in_bytes').write_text(str(limit*gib))
            (scope / 'memory.oom_control').write_text('under_oom 0\noom_kill 0\n')
            (scope / 'memory.failcnt').write_text('0\n')

        def check(value, initial):
            if value['events']['oom_kill'] > initial['events']['oom_kill']:
                raise RuntimeError('New cgroup oom_kill event')

        original = SimpleNamespace(estimate=lambda current, stat: current-stat['inactive_file'], check=check)
        guard = CgroupV1Memory(original, proc)
        initial = guard.sample()
        self.assertEqual(initial['limit'] - initial['current'], 34*gib)
        self.assertEqual(initial['working_estimate'], 28*gib)
        self.assertEqual(len(initial['scopes']), 3)
        self.assertFalse(initial['host_raw_headroom_selected'])
        guard.check(initial, initial)

        (leaf.parent / 'memory.usage_in_bytes').write_text(str(62*gib))
        with self.assertRaisesRegex(RuntimeError, '4GiB'):
            guard.check(guard.sample(), initial)
        (leaf.parent / 'memory.oom_control').write_text('under_oom 0\noom_kill 1\n')
        with self.assertRaisesRegex(RuntimeError, 'oom_kill'):
            guard.check(guard.sample(), initial)

    def test_protocol_clients_keep_driver_paths_and_use_their_own_torch(self):
        from official_suite import clean_client_environment

        libraries = ['/image/site-packages/torch/lib', '/usr/local/Ascend/driver/lib64',
                     '/image/site-packages/torch_npu/lib/', '/lib/aarch64-linux-gnu']
        with patch.dict(os.environ, {'LD_LIBRARY_PATH': os.pathsep.join(libraries),
                                     'PYTHONPATH': '/image/site-packages'}, clear=True):
            env = clean_client_environment()
        self.assertEqual(env['LD_LIBRARY_PATH'].split(os.pathsep), [libraries[1], libraries[3]])
        self.assertNotIn('PYTHONPATH', env)

    def config(self, **changes):
        model = self.root / 'model'
        model.mkdir(exist_ok=True)
        data = dict(model='model', cache='cache', out='new', physical_devices=[0, 7], versions=['0.23.0'])
        data.update(changes)
        path = self.root / 'evaluation.json'
        path.write_text(json.dumps(data))
        return path

    def test_sdk_child_discovers_original_worker_and_plugin_names(self):
        repo = Path(__file__).resolve().parents[1]
        scripts = repo / 'benchmark/one-click-kit/scripts'
        env = dict(os.environ, PYTHONPATH=worker_pythonpath(repo, scripts, ''))
        code = '''
from importlib.metadata import entry_points
from pathlib import Path
import sys
from official_worker import OfficialWorker

entries = entry_points(group='vllm.general_plugins', name='deeplinkengine_official_ft')
plugin, = entries
assert plugin.value == 'official.official_ft_plugin:install'
assert plugin.load().__module__ == 'official.official_ft_plugin'
assert OfficialWorker.__module__ == 'official_worker'
official = Path(sys.argv[1])
for name in ('official_worker', 'official.official_ft_plugin'):
    assert Path(sys.modules[name].__file__).resolve().parent == official
'''
        subprocess.run([sys.executable, '-S', '-B', '-c', code, str(repo / 'official')],
                       cwd=self.root, env=env, check=True, timeout=20)

    def test_explicit_version_and_paths(self):
        value = configuration(self.config())
        self.assertEqual(value['versions'], ['0.23.0'])
        self.assertEqual(value['out'], str(self.root / 'new'))

    def test_no_historical_relabelling_or_unknown_versions(self):
        for value in [['0.18.0'], ['0.23.0', '0.23.0'], ['latest'], ['0.23']]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                configuration(self.config(versions=value))

    def test_exact_official_empty_build_marker_without_fork_relabelling(self):
        self.assertTrue(matches_version('vllm', '0.23.0+empty', '0.23.0'))
        self.assertTrue(matches_version('vllm-ascend', '0.23.0', '0.23.0'))
        for distribution, version in [('vllm', '0.23.0+custom'), ('vllm', '0.23.0.dev1+empty'),
                                      ('vllm', '0.30.0+empty'), ('vllm-ascend', '0.23.0+empty')]:
            with self.subTest(version=version):
                self.assertFalse(matches_version(distribution, version, '0.23.0'))

    def test_reject_existing_or_dangling_output(self):
        (self.root / 'new').symlink_to(self.root / 'missing')
        with self.assertRaises(ValueError):
            configuration(self.config())

    def test_reject_duplicate_cards_and_unknown_options(self):
        for changes in [dict(physical_devices=[0, 0]), dict(physical_devices=[True, 7]), dict(enforce_eager=True)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                configuration(self.config(**changes))

    def test_public_dispatch_never_enters_original_pair(self):
        import run_all
        with patch.object(sys, 'argv', ['run_all.py', '--official-baselines', '--config', 'evaluation.json']), patch('official_baselines.main', return_value=7) as official:
            self.assertEqual(run_all.main(), 7)
            official.assert_called_once_with(['--config', 'evaluation.json'])

    def layer(self, rows):
        path = self.root / 'layer.tar'
        with tarfile.open(path, 'w') as archive:
            for name, raw, target in rows:
                item = tarfile.TarInfo(name)
                item.mode = 0o755
                if target is not None:
                    item.type = tarfile.SYMTYPE
                    item.linkname = target
                    archive.addfile(item)
                else:
                    item.size = len(raw)
                    archive.addfile(item, io.BytesIO(raw))
        return path

    def test_whiteout_and_exact_file_modes(self):
        rootfs = self.root / 'rootfs'
        (rootfs / 'usr/local').mkdir(parents=True)
        (rootfs / 'usr/local/old').write_bytes(b'previous')
        path = self.layer([('usr/local/.wh.old', b'', None), ('usr/local/new', b'exact', None)])
        extract(path, rootfs)
        self.assertFalse((rootfs / 'usr/local/old').exists())
        self.assertEqual((rootfs / 'usr/local/new').read_bytes(), b'exact')
        self.assertEqual((rootfs / 'usr/local/new').stat().st_mode & 0o777, 0o755)

    def test_symlink_escape_rejected(self):
        rootfs = self.root / 'rootfs'
        rootfs.mkdir()
        with self.assertRaises(ValueError):
            destination(rootfs, '../outside')
        (rootfs / 'escape').symlink_to(self.root)
        with self.assertRaises(ValueError):
            destination(rootfs, 'escape/outside')
        path = self.layer([('usr/local/escape', b'', '../../../../outside')])
        with self.assertRaises(ValueError):
            extract(path, rootfs)

    def test_runtime_environment_is_private_and_removes_optimized_flags(self):
        image = self.root / 'image'
        home = image / 'rootfs/usr/local/python3.12.13'
        (home / 'bin').mkdir(parents=True)
        (home / 'bin/python3').write_bytes(b'fixture')
        (image / 'image-config.json').write_text(json.dumps(dict(config=dict(Env=[
            'PATH=/usr/local/python3.12.13/bin:/usr/bin',
            'LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/cann-9.1.0/lib64',
            'ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0']))))
        with patch.dict(os.environ, dict(BETTERSCALE_EXPERIMENT='1', HUST_TEST='1', GITHUB_TOKEN='fixture-secret', VLLM_PLUGINS='hust_joint_packed')):
            python, env = environment(image, self.root / 'cache', devices='0,1')
        self.assertEqual(python, home / 'bin/python3')
        self.assertNotIn('GITHUB_TOKEN', env)
        self.assertNotIn('BETTERSCALE_EXPERIMENT', env)
        self.assertNotIn('HUST_TEST', env)
        self.assertNotIn('VLLM_PLUGINS', env)
        self.assertIn('/usr/local/Ascend/driver/lib64', env['LD_LIBRARY_PATH'])
        self.assertEqual(env['ASCEND_HOME_PATH'], str(image / 'rootfs/usr/local/Ascend/cann-9.1.0'))
        self.assertEqual(env['TMPDIR'], str(self.root / 'cache/tmp'))
        self.assertEqual(env['TORCH_EXTENSIONS_DIR'], str(self.root / 'cache/torch-extensions'))

    def test_storage_alias_requires_same_export_namespace_and_server(self):
        records = mounts('server:/data /shared nfs rw,vers=3 0 0\nserver:/data/workspace-a /fast nfs4 rw,vers=4.2 0 0\nother:/data/workspace-a /wrong-server nfs4 rw,vers=4.2 0 0\nserver:/data/workspace-b /wrong-workspace nfs4 rw,vers=4.2 0 0\n')
        found = candidates(Path('/shared/workspace-a/task/cache'), records)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['preferred'], '/fast/task/cache')
        self.assertEqual(found[0]['exact_remote_path'], '/data/workspace-a/task/cache')
        self.assertEqual(candidates(Path('/fast/task/cache'), records), [])

    def test_failed_results_remain_failed_and_sealed_without_efficiency(self):
        from official_report import report
        phase = self.root / 'mmlu'
        (phase / 'service').mkdir(parents=True)
        (phase / 'cache').mkdir()
        raw = phase / 'service/case-000.json'
        raw.write_text(json.dumps(dict(success=False, error='preserved timeout')))
        (phase / 'cache/jit-binary').write_bytes(b'unscored cache')
        (phase / 'receipt.json').write_text(json.dumps(dict(completed=False)))
        (self.root / 'campaign.json').write_text(json.dumps(dict(
            baseline_image=dict(id='official-vllm-0.23.0'), status='failed',
            fixed_five_denominators=dict(mmlu=200, hle=200, frontier=160, swe=64, terminal=48),
            stages=[dict(name='mmlu', status='failed')])) )
        result = report(self.root)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['phases']['mmlu']['status'], 'failed')
        self.assertIsNone(result['qualified_efficiency_percent'])
        checksums = json.loads((self.root / 'checksums.json').read_bytes())
        self.assertIn('mmlu/service/case-000.json', checksums)
        self.assertNotIn('mmlu/cache/jit-binary', checksums)
        self.assertEqual(json.loads(raw.read_bytes())['error'], 'preserved timeout')


if __name__ == '__main__':
    unittest.main()
