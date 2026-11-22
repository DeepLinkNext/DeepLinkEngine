import io
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
import sys
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] if Path(__file__).parent.name == 'tests' else Path(__file__).resolve().parent))
from official import enable_module_imports

enable_module_imports()
import official_source_candidate as candidate
import official_baselines as public_entry
import official_storage


class CandidateTests(unittest.TestCase):
    def test_isolated_stage_registry_keeps_core_and_ascend_torch_separate(self):
        rows=candidate.build_dependency_artifacts()
        core=rows['sha256:'+candidate.TORCH_BUILD_WHEEL['sha256']]
        self.assertEqual((core['stage'],core['requirement']),('01-core','torch==2.13.0'))
        for name,row in candidate.ASCEND_BUILD_WHEELS.items():
            selected=rows['sha256:'+row['sha256']]
            self.assertEqual(selected['stage'],'03-ascend')
            self.assertEqual(selected['package'],name)
        nested=rows['sha256:'+candidate.NESTED_BUILD_ARTIFACTS['arctic-build-torch']['sha256']]
        self.assertEqual((nested['stage'],nested['requirement']),('03-ascend/arctic-inference-build','torch==2.7.0'))
        self.assertNotEqual(nested['sha256'],candidate.ASCEND_BUILD_WHEELS['torch']['sha256'])
        self.assertEqual(len(rows),5)

    def test_ascend_build_requires_both_original_torch_and_npu_pins(self):
        from engine import environment_setup
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'pyproject.toml').write_text('[build-system]\nrequires=["torch==2.10.0","torch-npu==2.13.0rc1"]\n')
            with patch.object(environment_setup,'cached_wheel') as fetch:
                with self.assertRaisesRegex(ValueError,'build requirement changed'):
                    candidate.primary_build_constraint(Path(sys.executable),root,root,root,{},sdk=root,stage='03-ascend')
                fetch.assert_not_called()

    def test_build_torch_cache_binds_only_exact_requirement_and_fresh_bytes(self):
        import official_build_assets
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);core=root/'core';core.mkdir();build=root/'build';build.mkdir();(root/'engine').mkdir()
            pyproject=core/'pyproject.toml';pyproject.write_text('[build-system]\nrequires=["torch == 2.13.0"]\n')
            wheel=root/'cached.whl';wheel.write_bytes(b'cpu-wheel-fixture')
            item=dict(candidate.TORCH_BUILD_WHEEL,bytes=wheel.stat().st_size,sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
            (root/'engine/r8-lock.json').write_text(json.dumps(dict(environment_sources=dict(torch=item))))
            constraint=build/'01-core-metadata-root-constraints.txt'
            with patch.object(candidate,'ROOT',root),patch.object(candidate,'TORCH_BUILD_WHEEL',item), \
                 patch.object(official_build_assets,'fetch',return_value=(wheel,dict(cpu_fixture=True))) as fetch:
                path,proof=candidate.primary_build_constraint(Path(sys.executable),core,root/'cache',build,{},sdk=root/'sdk')
                self.assertEqual(path,constraint);self.assertIn('torch @ '+wheel.as_uri(),path.read_text())
                self.assertFalse(proof['runtime_constraint_added'])
                self.assertEqual(proof['scope'],'metadata-root-only');self.assertFalse(proof['global_build_constraint_added'])
                wheel.write_bytes(b'bad-bytes')
                with self.assertRaisesRegex(ValueError,'fresh whole SHA'):candidate.primary_build_constraint(Path(sys.executable),core,root/'cache',build,{},sdk=root/'sdk')
                pyproject.write_text('[build-system]\nrequires=["torch == 2.10.0"]\n');fetch.reset_mock()
                with self.assertRaisesRegex(ValueError,'build requirement changed'):candidate.primary_build_constraint(Path(sys.executable),core,root/'cache',build,{},sdk=root/'sdk')
                fetch.assert_not_called()

    def test_build_cmake_launchers_keep_scripts_and_bind_private_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"private python's tools";root.mkdir()
            python=root/'python3';python.write_bytes(b'private interpreter fixture')
            originals={}
            for name in ('cmake','ctest','cpack'):
                body=b'#!/usr/local/python3.12.13/bin/python3\nprint("copied package")\n'
                (root/name).write_bytes(body);originals[name]=body
            out=Path(directory)/'build-launchers';records=candidate.build_console_launchers(python,out)
            self.assertEqual(set(records),{'cmake','ctest','cpack'})
            import shlex
            for name,row in records.items():
                self.assertEqual((root/name).read_bytes(),originals[name])
                self.assertEqual(row['original_script_sha256'],hashlib.sha256(originals[name]).hexdigest())
                self.assertEqual(shlex.split((out/name).read_text().splitlines()[1])[:5],['exec',str(python.resolve()),'-B','-S',str(root/name)])
                self.assertTrue(row['build_only']);self.assertFalse(row['original_script_modified'])

    def test_unknown_cmake_shebang_refused_without_sdk_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);python=root/'python3';python.write_bytes(b'fixture')
            original=b'#!/different/python\nprint(1)\n'
            (root/'cmake').write_bytes(original)
            with self.assertRaisesRegex(ValueError,'Unexpected upstream CMake console'):candidate.build_console_launchers(python,root/'launchers')
            self.assertEqual((root/'cmake').read_bytes(),original)

    def setUp(self):
        delivery = patch('official_source_delivery.prepare', return_value=dict(path='cpu-fixture-delivery', sha256='a'*64))
        self.delivery = delivery.start()
        self.addCleanup(delivery.stop)
        storage = patch.object(official_storage, 'prefer', side_effect=lambda p: (Path(p), None))
        storage.start()
        self.addCleanup(storage.stop)
        # The local overlay need not copy the complete repository just for
        # controller fixtures. In CI this is the ordinary checkout root.
        source = patch.object(public_entry, 'ROOT', Path(official_storage.__file__).resolve().parents[1])
        source.start()
        self.addCleanup(source.stop)
        context = patch('official_execution.maybe_enter_official', return_value=None)
        context.start()
        self.addCleanup(context.stop)
        if sys.platform != 'linux':
            # The platform-mocked CLI fixture has no Linux /proc owner.
            owner = patch('official_sdk_state.owner_info', return_value={'cpu_fixture': True})
            owner.start()
            self.addCleanup(owner.stop)

    def archive(self, directory, names):
        path = Path(directory) / 'package.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            for name, kind, body in names:
                member = tarfile.TarInfo(name)
                if kind == 'file':
                    member.size = len(body)
                    archive.addfile(member, io.BytesIO(body))
                else:
                    member.type = tarfile.SYMTYPE
                    member.linkname = body
                    archive.addfile(member)
        return path

    def test_complete_safe_sdist(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = self.archive(temp, [('p/setup.py', 'file', b'pass\n'), ('p/PKG-INFO', 'file', b'Version: 0.28.0\n')])
            out = Path(temp) / 'out'
            candidate.safe_extract(archive, out)
            self.assertEqual((out / 'setup.py').read_bytes(), b'pass\n')

    def test_escape_or_link_or_duplicate_rejected(self):
        for members in ([('../outside', 'file', b'x')], [('p/x', 'link', '../../outside')],
                        [('p/x', 'file', b'a'), ('p/x', 'file', b'b')],
                        [('p/x', 'file', b'a'), ('q/y', 'file', b'b')]):
            with self.subTest(members=members), tempfile.TemporaryDirectory() as temp:
                archive = self.archive(temp, members)
                with self.assertRaises(ValueError):
                    candidate.safe_extract(archive, Path(temp) / 'out')
                self.assertFalse((Path(temp) / 'outside').exists())

    def test_incomplete_sdist_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = self.archive(temp, [('p/setup.py', 'file', b'pass')])
            with self.assertRaises(ValueError):
                candidate.safe_extract(archive, Path(temp) / 'out')

    def test_explicit_source_identity_not_official_image(self):
        identity = candidate.identity()
        self.assertTrue(identity['candidate'])
        self.assertFalse(identity['official_release_image'])
        self.assertEqual(identity['ascend_version'], '0.28.0rc1')
        self.assertNotEqual(identity['version'], '0.28.0')
        self.assertFalse(identity['sdist_git_file_equivalence_certified'])
        self.assertEqual(len(candidate.key()), 64)

    def test_build_only_batch_override_is_absent_from_serving_environment(self):
        from official_image import environment
        import os
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sdk=root/'sdk';home=sdk/'rootfs/usr/local/python3.12.13'
            (home/'bin').mkdir(parents=True);(home/'bin/python3').write_text('CPU fixture')
            (sdk/'image-config.json').write_text(json.dumps({'config':{'Env':[]}}))
            (sdk/'image-lock.json').write_text(json.dumps({'upstream_version':'0.28.0'}))
            with patch.dict(os.environ,{'VLLM_BATCH_INVARIANT':'1'}):
                _,env=environment(sdk,root/'cache')
            self.assertNotIn('VLLM_BATCH_INVARIANT',env)

    def test_remove_only_copied_editable_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            site = root / 'site'
            for package in ('vllm', 'vllm_ascend', 'torch'):
                d = site / (package + '-1.dist-info')
                d.mkdir(parents=True)
                (d / 'METADATA').write_text('Name: ' + package + '\n')
            (site / '__editable___vllm_023_finder.py').write_text('pass')
            removed = candidate.remove_old_editables(root, site)
            self.assertEqual(len(removed), 3)
            self.assertTrue((site / 'torch-1.dist-info').is_dir())
            self.assertFalse((site / 'vllm-1.dist-info').exists())

    def test_cached_sdk_rejects_changed_build_receipt_before_metadata_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk=Path(directory);(sdk/'source-build').mkdir()
            identity={'id':'cpu-source-candidate'}
            lock=dict(identity,manifest_digest='sha256:'+'a'*64,manifest_kind='source-build-derivation')
            receipt=sdk/'source-build/receipt.json';receipt.write_text('{"status":"prepared"}')
            (sdk/'installed-files.json').write_text('{}')
            installed=dict(schema='deeplinkengine-source-candidate-sdk/v1',source_candidate=identity,
                image_manifest=lock['manifest_digest'],installed_files_sha256=hashlib.sha256(b'{}').hexdigest(),
                build_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
            (sdk/'installation.json').write_text(json.dumps(installed))
            (sdk/'source-candidate-lock.json').write_text(json.dumps(identity))
            (sdk/'image-lock.json').write_text(json.dumps(lock))
            receipt.write_text('{"status":"prepared","post_install_change":true}')
            with patch.object(candidate,'identity',return_value=identity),patch.object(candidate,'key',return_value='a'*64), \
                 patch('official_image.probe') as probe:
                with self.assertRaisesRegex(ValueError,'identity or installed files changed'):candidate.reuse(sdk)
            probe.assert_not_called()

    def test_successful_prepare_receipt_is_still_sealed_after_finally_and_reuse(self):
        import official_image
        import official_execution
        import os
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache';base=cache/'official-sdks/0.23.0-4ee78def8f33d59d'
            base.mkdir(parents=True);output=cache/'source-candidate-sdks/cpu-fixture'
            lock=Path(public_entry.ROOT)/'engine/official-baselines/0.23.0.json'
            (base/'image-lock.json').write_bytes(lock.read_bytes());(base/'installed-files.json').write_text('{}')
            (base/'installation.json').write_text(json.dumps(dict(image_manifest=candidate.BASE_MANIFEST,
                installed_files_sha256=hashlib.sha256(b'{}').hexdigest())))
            (base/'image-manifest.json').write_text('{}');(base/'image-config.json').write_text('{}')
            def extract(archive,path):
                path.mkdir(parents=True);(path/'setup.py').write_text('CPU fixture')
                version='0.28.0rc1' if path.name=='vllm-ascend' else '0.28.0'
                (path/'PKG-INFO').write_text('Metadata-Version: 2.1\nVersion: '+version+'\n')
            env=dict(os.environ,PYTHONHOME=str(output/'cpu-python'),CARGO_HOME=str(output/'cargo'),
                     RUSTUP_HOME=str(output/'rustup'),PATH=os.environ['PATH'],
                     LD_LIBRARY_PATH='/private/python/lib:/private/site/torch/lib:/private/site/torch_npu/lib:/private/cann/lib64')
            with patch.object(official_image,'file_map',return_value={}), \
                 patch.object(official_image,'probe',return_value={'cpu_fixture':True}), \
                 patch.object(official_image,'environment',return_value=(Path(sys.executable),env)), \
                 patch.object(official_execution,'context_identity',return_value={'source_build_system_root':True}), \
                 patch.object(official_execution,'sdk_run',return_value=subprocess.CompletedProcess([],0,stdout='',stderr='')), \
                 patch.object(candidate,'source_archive',return_value=root/'unused-archive'), \
                 patch.object(candidate,'safe_extract',side_effect=extract),patch.object(candidate,'command') as commands, \
                 patch.object(candidate,'toolchain_evidence',return_value={'cpu_fixture':True}), \
                 patch.object(candidate,'build_console_launchers',return_value={'cpu_fixture':True}), \
                 patch.object(candidate,'torch_build_constraint',side_effect=lambda *a,**k:(root/(k.get('stage','01-core')+'-isolated-build-only.txt'),{'cpu_fixture':True})), \
                 patch.object(candidate.shutil,'which',return_value='/cpu-rustup'), \
                 patch.object(candidate.shutil,'disk_usage',return_value=SimpleNamespace(free=200*1024**3)):
                installed=candidate.prepare(lock,cache,output)
                raw=(output/'source-build/receipt.json').read_bytes()
                self.assertEqual(installed['build_receipt_sha256'],hashlib.sha256(raw).hexdigest())
                self.assertEqual(json.loads(raw)['status'],'prepared')
                reused=candidate.reuse(output)
                self.assertEqual(reused['build_receipt_sha256'],installed['build_receipt_sha256'])
                self.assertTrue(Path(env['PIP_CACHE_DIR']).is_relative_to(cache.resolve()/'source-build-downloads'))
                self.assertFalse(Path(env['PIP_CACHE_DIR']).is_relative_to(output.resolve()))
                steps={call.args[4]:call.args[1] for call in commands.call_args_list}
                self.assertEqual(steps['01-core']['LD_LIBRARY_PATH'],'/private/python/lib:/private/cann/lib64')
                self.assertEqual(steps['01-core']['PIP_FIND_LINKS'],str(root/'01-core-isolated-build-only.txt'))
                self.assertEqual(steps['03-ascend']['PIP_FIND_LINKS'],str(root/'03-ascend-isolated-build-only.txt'))
                self.assertTrue(all('PIP_BUILD_CONSTRAINT' not in value for value in steps.values()))
                self.assertTrue(all('PIP_CONSTRAINT' not in value for value in steps.values()))
                self.assertEqual(steps['03-ascend']['LD_LIBRARY_PATH'],'/private/python/lib:/private/cann/lib64')
                self.assertEqual(steps['01-core']['VLLM_TARGET_DEVICE'],'empty')
                self.assertNotIn('VLLM_BATCH_INVARIANT',steps['01-core'])
                self.assertEqual(steps['03-ascend']['VLLM_BATCH_INVARIANT'],'1')
                self.assertNotIn('VLLM_TARGET_DEVICE',steps['03-ascend'])
                for name in ['02-remove-base-triton','06-rust-build-tools','09-upstream-rust']:
                    self.assertNotIn('VLLM_TARGET_DEVICE',steps[name])
                    self.assertNotIn('VLLM_BATCH_INVARIANT',steps[name])

    def partial(self,sdk,status='building'):
        sdk.mkdir();(sdk/'source-build').mkdir()
        marker=dict(schema='deeplinkengine-source-sdk-preparation/v1',candidate=candidate.identity(),
                    destination=str(sdk.resolve()),owner={'cpu_fixture':True},model_started=False)
        (sdk/'source-preparation.json').write_text(json.dumps(marker))
        (sdk/'source-build/receipt.json').write_text(json.dumps(dict(status=status,
                    candidate=candidate.identity(),model_started=False)))
        (sdk/'original-error.log').write_bytes(b'original failure')

    def test_stopped_partial_source_sdk_is_preserved_for_a_fresh_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sdk=root/'sdk';self.partial(sdk,'failed')
            cached=root/'source-build-downloads/pip/file';cached.parent.mkdir(parents=True);cached.write_bytes(b'CPU cache fixture')
            with patch('official_sdk_state.refuse_active_owner',return_value={'reason':'process_absent'}):
                record=candidate.quarantine_partial(sdk)
            self.assertFalse(sdk.exists())
            self.assertEqual((Path(record['preserved'])/'original-error.log').read_bytes(),b'original failure')
            self.assertEqual(cached.read_bytes(),b'CPU cache fixture')

    def test_live_or_completed_source_build_is_not_quarantined(self):
        for status in ['building','prepared']:
            with self.subTest(status=status),tempfile.TemporaryDirectory() as directory:
                sdk=Path(directory)/'sdk';self.partial(sdk,status)
                with patch('official_sdk_state.refuse_active_owner',side_effect=ValueError('active owner')):
                    with self.assertRaises(ValueError):candidate.quarantine_partial(sdk)
                self.assertTrue(sdk.exists())

    def test_unknown_incomplete_source_sdk_is_not_moved(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk=Path(directory)/'sdk';sdk.mkdir();(sdk/'unknown').write_bytes(b'preserve')
            with self.assertRaisesRegex(ValueError,'Unknown source SDK'):candidate.quarantine_partial(sdk)
            self.assertEqual((sdk/'unknown').read_bytes(),b'preserve')

    def test_outside_package_destination_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'private').mkdir()
            (root / 'site').mkdir()
            with self.assertRaises(ValueError):
                candidate.remove_old_editables(root / 'private', root / 'site')

    def config(self, root, version):
        model = root / 'model'
        model.mkdir(exist_ok=True)
        path = root / 'evaluation.json'
        path.write_text(json.dumps(dict(model=str(model), cache=str(root / 'cache'),
            out=str(root / 'result'), physical_devices=[0, 1], versions=[version])))
        return path

    def test_public_candidate_prepare_uses_explicit_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.config(root, candidate.VERSION)
            events=[]
            def deliver(cache,sdk,out):
                if sys.platform=='linux':
                    import fcntl
                    lock=sdk.parent/(sdk.name+'.prepare.lock')
                    with lock.open('a+b') as stream:
                        with self.assertRaises(BlockingIOError):fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                events.append('delivery');return dict(path='cpu-fixture-delivery',sha256='a'*64)
            self.delivery.side_effect=deliver
            def prepare(*args):events.append('original-prepare');return dict(schema='fixture-only')
            with patch.object(public_entry.platform, 'system', return_value='Linux'), \
                 patch.object(public_entry.platform, 'machine', return_value='aarch64'), \
                 patch.object(candidate, 'prepare', side_effect=prepare) as build:
                code = public_entry.main(['--config', str(path), '--prepare-environment'])
            self.assertEqual(code, 0)
            self.assertEqual(events,['delivery','original-prepare'])
            build.assert_called_once()
            self.delivery.assert_called_once()
            actual = json.loads((root / 'result/results.json').read_bytes())
            self.assertEqual(actual['stages'][0]['build_dependency_delivery']['path'], 'cpu-fixture-delivery')
            self.assertFalse(actual['model_execution_authorized'])
            self.assertEqual(actual['stages'][0]['baseline_id'], candidate.ID)
            self.assertFalse(actual['stages'][0]['official_release_image'])

    def test_public_completed_candidate_reuse_skips_delivery_and_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=self.config(root,candidate.VERSION)
            sdk=root/'cache/source-candidate-sdks'/(candidate.ID+'-'+candidate.key()[:16]);sdk.mkdir(parents=True)
            with patch.object(public_entry.platform,'system',return_value='Linux'), \
                 patch.object(public_entry.platform,'machine',return_value='aarch64'), \
                 patch.object(candidate,'quarantine_partial',return_value=None), \
                 patch.object(candidate,'reuse',return_value=dict(schema='cpu-strict-reuse-fixture')) as reuse, \
                 patch.object(candidate,'prepare') as build:
                self.assertEqual(public_entry.main(['--config',str(path),'--prepare-environment']),0)
            reuse.assert_called_once_with(sdk);build.assert_not_called();self.delivery.assert_not_called()

    def test_public_delivery_failure_stops_before_original_sdk_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=self.config(root,candidate.VERSION)
            self.delivery.side_effect=ValueError('verified delivery rejected bad cache')
            with patch.object(public_entry.platform,'system',return_value='Linux'), \
                 patch.object(public_entry.platform,'machine',return_value='aarch64'), \
                 patch.object(candidate,'prepare') as build:
                self.assertEqual(public_entry.main(['--config',str(path),'--prepare-environment']),1)
            build.assert_not_called()
            actual=json.loads((root/'result/results.json').read_bytes())
            self.assertIn('delivery rejected bad cache',actual['stages'][0]['error'])

    def test_public_candidate_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.config(root, candidate.VERSION)
            with patch.object(public_entry.platform, 'system', return_value='Linux'), \
                 patch.object(public_entry.platform, 'machine', return_value='aarch64'), \
                 patch.object(candidate, 'prepare', side_effect=RuntimeError('controlled source build failure')):
                code = public_entry.main(['--config', str(path), '--prepare-environment'])
            self.assertEqual(code, 1)
            actual = json.loads((root / 'result/results.json').read_bytes())
            self.assertEqual(actual['status'], 'incomplete')
            self.assertEqual(actual['stages'][0]['status'], 'failed')
            self.assertIn('controlled source build failure', actual['stages'][0]['error'])

    def test_stable_028_still_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.config(root, '0.28.0')
            with patch.object(public_entry.platform, 'system', return_value='Linux'), \
                 patch.object(public_entry.platform, 'machine', return_value='aarch64'), \
                 patch.object(candidate, 'prepare') as build:
                self.assertEqual(public_entry.main(['--config', str(path), '--prepare-environment']), 1)
            build.assert_not_called()
            actual = json.loads((root / 'result/results.json').read_bytes())
            self.assertEqual(actual['stages'][0]['status'], 'blocked')

    @unittest.skipUnless(sys.platform == 'linux', 'Actual Linux owned-build process evidence')
    def test_real_failed_build_command_sealed_and_quiescent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(RuntimeError):
                candidate.command([sys.executable, '-B', '-c', 'import sys;print("controlled");sys.exit(7)'],
                                  dict(__import__('os').environ), root, root, 'fixture-failure', timeout=10)
            receipt = json.loads((root / 'fixture-failure.json').read_bytes())
            self.assertEqual(receipt['returncode'], 7)
            self.assertTrue(receipt['quiescent'])
            self.assertIn('controlled', (root / 'fixture-failure.log').read_text())


if __name__ == '__main__':
    unittest.main()
