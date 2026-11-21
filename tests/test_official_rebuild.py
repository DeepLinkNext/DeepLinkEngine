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
        (path/'raw').write_bytes(b'original')
        self.dump(path/'checksums.json', {'raw':replay.digest(path/'raw')})
        return replay.Evidence(path)

    def test_changed_raw_cannot_pass_old_seal(self):
        evidence = self.evidence(); (evidence.root/'raw').write_bytes(b'changed')
        with self.assertRaises(ValueError):replay.Evidence(evidence.root)
        with self.assertRaises(ValueError):evidence.raw('raw')

    def test_portable_roundtrip_preserves_original_bytes(self):
        evidence = self.evidence(); source = self.root/'support.py'; source.write_bytes(b'original source')
        archive = self.root/'portable.tar.gz'
        replay.package(evidence, {'/old/host/support.py':source}, archive)
        output = self.root/'restored'; output.mkdir()
        restored = replay.unpack(archive, output)
        self.assertEqual(restored.raw('raw'), b'original')
        self.assertEqual(restored.logical_root, evidence.logical_root)
        self.assertEqual(restored.auxiliary('/old/host/support.py', replay.digest(source)).read_bytes(), b'original source')

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
        evidence=self.evidence();self.dump(evidence.root/'checksums.json', {'raw':'0'*64})
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
