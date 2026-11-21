"""CPU-only arithmetic, custody and public-entry regression checks.

Fixtures below are synthetic; they are never published as model observations.
"""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import official_rebuild as replay
import official_baselines


class C8Replay(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = {key: [dict(input_tokens=int(key)) for _ in range(8)] for key in ('4096', '16384')}
        self.phases, self.records = [], []
        for i, name in enumerate(replay.PHASES):
            start = i * 400.0
            seconds = 60 if i % 2 == 0 else 300
            self.phases.append(dict(name=name, seconds=seconds, failed=0, requests=8,
                start=dict(monotonic=start), end=dict(monotonic=start+seconds+2),
                submission_deadline_monotonic=start+seconds))
            for lane in range(8):self.records.append(dict(id='service-'+name+f'-lane{lane}-00000', phase=name,
                input_tokens=int(name.split('-')[0]), usage=dict(completion_tokens=256),
                start=dict(monotonic=start+1), first_token=dict(monotonic=start+2),
                last_token=dict(monotonic=start+3), end=dict(monotonic=start+4)))

    def dump(self, path, value):
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def test_original_tpot_uses_request_end_not_last_token(self):
        rows = replay.aggregate(self.records, self.phases, self.inputs)
        formal = rows['4096-formal']
        self.assertEqual(formal['mean_tpot_ms'], 2000/255)
        self.assertEqual(formal['mean_ttft_ms'], 1000)
        self.assertEqual(formal['output_tokens_per_second'], 8*256/302)
        self.assertEqual(formal['requests'], 8)
        self.assertEqual(rows['4096-warmup']['original_submission_seconds'], 60)

    def test_tool_header_never_becomes_driver_evidence(self):
        for version in ('26.0.rc1','99.9'):
            hardware=replay.recorded_hardware(dict(npu_before='| npu-smi '+version+' |\n| 0 910B2 |',physical_cards=[0,7],admitted=True),'a'*64)
            self.assertEqual(hardware['npu_smi_version'],version)
            self.assertIsNone(hardware['driver_version'])
            self.assertEqual(hardware['driver_version_status'],'not-recorded')

    def test_fixed_geometry_and_prefix_policy_cannot_be_relabelled(self):
        lock={'id':'official-vllm-0.23.0'}
        values={'--tensor-parallel-size':'2','--dtype':'bfloat16','--max-model-len':'65536',
                '--max-num-batched-tokens':'4096','--max-num-seqs':'24','--kv-cache-memory-bytes':str(20*1024**3),
                '--speculative-config':json.dumps(dict(method='mtp',num_speculative_tokens=2)),
                '--compilation-config':json.dumps(dict(cudagraph_mode='FULL'))}
        arguments=[item for pair in values.items() for item in pair]+['--no-enable-prefix-caching']
        plan=dict(baseline_id=lock['id'],role='B0',kind='continuous',runtime_versions=replay.VERSIONS,
            original_baseline_018_relabelled=False,necessary_work_certified=False,
            model='/models/Qwen3.5-35B-A3B',serve_arguments=arguments)
        replay.admit_plan(plan,lock)
        for key in values:
            wrong=copy.deepcopy(plan);wrong['serve_arguments'][arguments.index(key)+1]='{}' if key.endswith('-config') else '1'
            with self.subTest(key=key),self.assertRaises(ValueError):replay.admit_plan(wrong,lock)
        for args in (arguments[:-1],arguments+['--enable-prefix-caching'],arguments+['--max-num-seqs=100']):
            with self.subTest(args=args),self.assertRaises(ValueError):replay.admit_plan(dict(plan,serve_arguments=args),lock)

    def test_json_transport_is_identical_on_windows_and_linux(self):
        path=self.root/'data.json';replay.write(path,dict(actual='原值',value=1.25))
        self.assertNotIn(b'\r',path.read_bytes())
        self.assertTrue(path.read_bytes().endswith(b'\n'))

    def test_duplicate_missing_or_reordered_populations_refused(self):
        for records, phases in ((self.records+self.records[:1], self.phases),
                                 (self.records[1:], self.phases),
                                 (self.records, self.phases[::-1])):
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                replay.aggregate(records, phases, self.inputs)

    def test_record_clock_input_lane_or_population_changes_refused(self):
        for key, value in [('id', 'service-4096-warmup-lane8-00000'),
                           ('input_tokens', 16), ('end', dict(monotonic=99)),
                           ('first_token', dict(monotonic=float('nan')))]:
            rows = copy.deepcopy(self.records); rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                replay.aggregate(rows, self.phases, self.inputs)
        for key, value in [('failed', 1), ('requests', 2), ('seconds', 300),
                           ('submission_deadline_monotonic', 61)]:
            phases = copy.deepcopy(self.phases); phases[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                replay.aggregate(self.records, phases, self.inputs)

    def evidence(self):
        path = self.root / 'source'; path.mkdir()
        (path/'continuous').mkdir();(path/'context-inputs').mkdir()
        (path/'continuous/raw').write_bytes(b'original')
        (path/'context-inputs/private.json').write_bytes(b'private-unrun-task')
        for name in ('campaign.json','model-weights.json','resource-admission.json'):
            self.dump(path/name, {'synthetic': True})
        self.dump(path/'continuous/checksums.json', {'raw':replay.digest(path/'continuous/raw')})
        self.dump(path/'checksums.json', {p.relative_to(path).as_posix():replay.digest(p) for p in path.rglob('*') if p.is_file()})
        return replay.Evidence(path)

    def test_changed_raw_cannot_pass_old_seal(self):
        evidence = self.evidence(); (evidence.root/'continuous/raw').write_bytes(b'changed')
        with self.assertRaises(ValueError):replay.Evidence(evidence.root)
        with self.assertRaises(ValueError):evidence.raw('continuous/raw')

    def test_portable_roundtrip_preserves_original_bytes(self):
        evidence = self.evidence(); source = self.root/'support.py'; source.write_bytes(b'original source')
        archive = self.root/'portable.tar.gz'
        replay.package(evidence, {'/old/host/support.py':source}, archive)
        output = self.root/'restored'; output.mkdir()
        restored = replay.unpack(archive, output, evidence.seal_sha256)
        self.assertEqual(restored.raw('continuous/raw'), b'original')
        self.assertEqual(restored.logical_root, evidence.logical_root)
        self.assertEqual(restored.auxiliary('/old/host/support.py', replay.digest(source)).read_bytes(), b'original source')
        self.assertEqual((restored.root/'checksums.json').read_bytes(),(evidence.root/'checksums.json').read_bytes())
        self.assertIn('context-inputs/private.json',restored.seal)
        self.assertNotIn('context-inputs/private.json',restored.available)
        self.assertFalse((restored.root/'context-inputs/private.json').exists())
        self.assertEqual(evidence.available,set(evidence.seal))

    def test_directory_still_requires_all_original_member_bytes(self):
        evidence=self.evidence();(evidence.root/'context-inputs/private.json').unlink()
        with self.assertRaises(OSError):replay.Evidence(evidence.root)

    def test_portable_requires_external_original_seal_and_exact_c8_closure(self):
        evidence=self.evidence();archive=self.root/'portable.tar.gz';replay.package(evidence,{},archive)
        with tarfile.open(archive) as stream:
            original={row.name:stream.extractfile(row).read() for row in stream if row.isfile()}
        for expected in (None,'0'*64):
            with tempfile.TemporaryDirectory(dir=self.root) as output,self.assertRaisesRegex(ValueError,'External original'):
                replay.unpack(archive,Path(output),expected)
        for mode in ('missing-c8','extra-private-payload','wrong-omitted-map','changed-outer-seal'):
            files=dict(original);manifest=json.loads(files['manifest.json'])
            if mode=='missing-c8':
                del files['evidence/continuous/raw'];del manifest['files']['evidence/continuous/raw']
            elif mode=='extra-private-payload':
                files['evidence/context-inputs/private.json']=b'private-unrun-task'
                manifest['files']['evidence/context-inputs/private.json']=replay.digest(evidence.root/'context-inputs/private.json')
            elif mode=='wrong-omitted-map':manifest['omitted_original_members']={}
            else:
                seal=json.loads(files['evidence/checksums.json']);seal['context-inputs/private.json']='0'*64
                files['evidence/checksums.json']=json.dumps(seal).encode()
                import hashlib
                manifest['files']['evidence/checksums.json']=hashlib.sha256(files['evidence/checksums.json']).hexdigest()
            files['manifest.json']=json.dumps(manifest).encode();changed=self.root/(mode+'.tar.gz')
            with tarfile.open(changed,'x:gz') as stream:
                for name,raw in files.items():
                    item=tarfile.TarInfo(name);item.size=len(raw);stream.addfile(item,io.BytesIO(raw))
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=self.root) as output,self.assertRaises(ValueError):
                replay.unpack(changed,Path(output),evidence.seal_sha256)

    def test_unsafe_or_duplicate_tar_members_refused(self):
        for name, kind in [('../escape', None), ('/absolute', None), ('C:/drive', None),
                           ('back\\slash', None), ('symlink', tarfile.SYMTYPE), ('duplicate', None)]:
            with self.subTest(name=name):
                archive = self.root/'unsafe.tar.gz'
                with tarfile.open(archive, 'w:gz') as stream:
                    item = tarfile.TarInfo(name); item.size = 1
                    if kind:item.type=kind;item.linkname='target'
                    stream.addfile(item, io.BytesIO(b'x'))
                    if name=='duplicate':stream.addfile(item, io.BytesIO(b'x'))
                with tempfile.TemporaryDirectory(dir=self.root) as output, self.assertRaises(ValueError):
                    replay.unpack(archive, Path(output))
        self.assertFalse((self.root/'escape').exists())

    def test_missing_archive_pin_and_existing_output_refused(self):
        source = self.root/'archive'; source.write_bytes(b'not an archive')
        config = self.dump(self.root/'config.json', dict(source=str(source), out=str(self.root/'new')))
        with self.assertRaisesRegex(ValueError, 'SHA256'):replay.run(config)
        out = self.root/'existing';out.mkdir();(out/'retained').write_bytes(b'retained')
        self.dump(config, dict(source=str(source), out=str(out)))
        with self.assertRaisesRegex(ValueError, 'new output'):replay.run(config)
        self.assertEqual((out/'retained').read_bytes(), b'retained')

    def test_directory_pin_is_checked_before_replay(self):
        evidence=self.evidence()
        config=self.dump(self.root/'config.json', dict(source=str(evidence.root), out=str(self.root/'new'), source_sha256='0'*64))
        with patch.object(replay, 'replay', side_effect=AssertionError('must not replay')):
            with self.assertRaisesRegex(ValueError, 'SHA256'):replay.run(config)

    def test_output_cannot_modify_original_directory(self):
        evidence=self.evidence()
        config=self.dump(self.root/'config.json', dict(source=str(evidence.root),out=str(evidence.root/'new'),source_sha256=evidence.seal_sha256))
        with self.assertRaisesRegex(ValueError, 'outside'):replay.run(config)
        self.assertFalse((evidence.root/'new').exists())

    def test_changed_outer_seal_cannot_be_exported(self):
        evidence=self.evidence();changed=dict(evidence.seal);changed['campaign.json']='0'*64;self.dump(evidence.root/'checksums.json',changed)
        with self.assertRaisesRegex(ValueError, 'changed while archiving'):
            replay.package(evidence,{},self.root/'new.tar.gz')

    def test_report_only_dispatch_bypasses_device_platform_and_normal_config(self):
        with patch.object(replay, 'run', return_value=0) as run, \
             patch.object(official_baselines, 'configuration', side_effect=AssertionError('normal campaign')):
            self.assertEqual(official_baselines.main(['--report-only', '--config', str(self.root/'config.json')]), 0)
            run.assert_called_once()

    def test_report_only_cannot_prepare_or_resume_model_campaign(self):
        for flags in (['--prepare-environment'], ['--reuse-completed-continuous', 'old']):
            with patch.object(replay, 'run', side_effect=AssertionError('must not replay')), self.assertRaises(SystemExit) as error:
                official_baselines.main(['--report-only','--config','missing']+flags)
            self.assertEqual(error.exception.code, 2)

    def test_actual_public_cli_rejects_before_model_import_or_output(self):
        source=self.root/'archive';source.write_bytes(b'wrong archive')
        config=self.dump(self.root/'config.json', dict(source=str(source),out=str(self.root/'new'),source_sha256='0'*64))
        result=subprocess.run([sys.executable,'-B',str(replay.ROOT/'run_all.py'),
            '--official-baselines','--report-only','--config',str(config)], capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,2)
        self.assertIn('SHA256',result.stderr)
        self.assertFalse((self.root/'new').exists())


if __name__ == '__main__':unittest.main()
