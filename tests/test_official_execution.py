import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import official_execution as execution


class ExecutionContextTests(unittest.TestCase):
    def test_actual_libc_floor_not_python_binary_minimum(self):
        for version, expected in [('2.34', True), ('2.35', False), ('2.39', False), (None, False)]:
            with self.subTest(version=version), patch.object(execution, 'libc_version', return_value=version):
                self.assertEqual(execution.needed(), expected)

    def test_modern_host_subprocess_is_unchanged(self):
        with patch.object(execution, 'needed', return_value=False), \
             patch.object(execution.subprocess, 'run', return_value='completed') as run:
            result = execution.sdk_run(['private-python', '-S', '-c', 'pass'], sdk='sdk', env={'KEY': 'fixture'}, capture_output=True)
        self.assertEqual(result, 'completed')
        run.assert_called_once_with(['private-python', '-S', '-c', 'pass'], env={'KEY': 'fixture'}, capture_output=True)

    def test_older_host_routes_metadata_through_private_root(self):
        with patch.object(execution, 'needed', return_value=True), \
             patch.object(execution, 'ensure_for_sdk', return_value=Path('/fixture-root')), \
             patch.object(execution, 'run_in_context', return_value='completed') as run:
            self.assertEqual(execution.sdk_run(['sdk-python', '-S'], sdk='/fixture-sdk', env={'P': 'Q'}), 'completed')
        self.assertEqual(run.call_args.args[:2], (['sdk-python', '-S'], Path('/fixture-root')))
        self.assertEqual(run.call_args.kwargs['env'], {'P': 'Q'})

    def test_wrapper_does_not_start_host_python_with_sdk_pythonhome(self):
        request = dict(root='fixture', command=['sdk-python'], env={'PYTHONHOME': 'sdk-home', 'LD_LIBRARY_PATH': 'sdk-libraries'})
        with patch.object(execution, 'request_for', return_value=request), \
             patch.dict(os.environ, {'PYTHONHOME': 'invalid-host-bootstrap', 'LD_LIBRARY_PATH': 'invalid-sdk-loader'}), \
             patch.object(execution.subprocess, 'run', return_value='completed') as run:
            execution.run_in_context(['sdk-python'], Path('fixture'), env=request['env'], text=True, capture_output=True)
        call = run.call_args
        self.assertNotIn('PYTHONHOME', call.kwargs['env'])
        self.assertNotIn('LD_LIBRARY_PATH', call.kwargs['env'])
        self.assertEqual(json.loads(call.kwargs['input'])['env']['PYTHONHOME'], 'sdk-home')
        self.assertNotIn('sdk-home', call.args[0])
        self.assertEqual(call.args[0][0], sys.executable)

    def test_stale_inherited_context_cannot_fake_the_current_root(self):
        old = dict(schema='deeplinkengine-private-glibc-context/v1',
                   base_layer_sha256=execution.BASE_DIGEST, root_dev_inode=[-1, -1])
        with patch.dict(os.environ, {'DEEPLINK_EXECUTION_CONTEXT': json.dumps(old)}):
            with self.assertRaises(ValueError):
                execution.context_identity()

    def test_archive_parent_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('../escape', '/absolute', 'a/../../escape', 'a\\escape'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    execution.destination(root, name)

    def test_changed_static_execution_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'rootfs').mkdir()
            (root / 'rootfs/file').write_bytes(b'original')
            inventory = {'file': {'sha256': hashlib.sha256(b'original').hexdigest()}}
            (root / 'static-files.json').write_text(json.dumps(inventory))
            (root / 'receipt.json').write_text(json.dumps(dict(schema='deeplinkengine-private-glibc-base/v1',
                layer_sha256=execution.BASE_DIGEST, static_inventory_sha256=execution.sha(root / 'static-files.json'))))
            self.assertEqual(execution.verify_root(root), root.resolve() / 'rootfs')
            (root / 'rootfs/file').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                execution.verify_root(root)

    def test_pure_download_does_not_enter_context(self):
        import official_baselines
        import official_download
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'evaluation.json'
            path.write_text(json.dumps(dict(cache=str(root / 'cache'), out=str(root / 'out'), versions=['0.23.0'])))
            with patch.object(official_download, 'run', return_value=0), \
                 patch.object(execution, 'maybe_enter_official', side_effect=AssertionError('download entered context')):
                self.assertEqual(official_baselines.main(['--config', str(path), '--download-only']), 0)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux execution-root lock')
    def test_only_known_stopped_partial_root_is_preserved_and_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);sdk=cache/'official-sdks/sdk'
            output=cache/'execution-contexts'/('ubuntu-'+execution.BASE_DIGEST[:16])
            output.mkdir(parents=True);(output/'original').write_bytes(b'partial')
            marker=dict(schema='deeplinkengine-private-glibc-preparation/v1',
                        layer_sha256=execution.BASE_DIGEST,destination=str(output),owner={'fixture': True})
            (output/'preparation.json').write_text(json.dumps(marker))
            with patch('official_sdk_state.refuse_active_owner', return_value={'reason':'process_absent'}), \
                 patch('baseline_setup.fetch', return_value=cache/'layer'), \
                 patch.object(execution,'prepare_root',return_value=Path('/rebuilt')) as prepare:
                self.assertEqual(execution.ensure_for_sdk(sdk),Path('/rebuilt'))
            preserved=list(output.parent.glob(output.name+'.interrupted-*'))
            self.assertEqual(len(preserved),1)
            self.assertEqual((preserved[0]/'original').read_bytes(),b'partial')
            self.assertTrue((preserved[0]/'interrupted-recovery.json').exists())
            prepare.assert_called_once_with(cache/'layer',output)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux execution-root lock')
    def test_live_partial_root_is_not_moved(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);sdk=cache/'official-sdks/sdk'
            output=cache/'execution-contexts'/('ubuntu-'+execution.BASE_DIGEST[:16]);output.mkdir(parents=True)
            marker=dict(schema='deeplinkengine-private-glibc-preparation/v1',
                        layer_sha256=execution.BASE_DIGEST,destination=str(output),owner={})
            (output/'preparation.json').write_text(json.dumps(marker))
            with patch('official_sdk_state.refuse_active_owner',side_effect=ValueError('active')):
                with self.assertRaisesRegex(ValueError,'active'):
                    execution.ensure_for_sdk(sdk)
            self.assertTrue(output.exists());self.assertEqual(list(output.parent.glob('*.interrupted-*')),[])

    @unittest.skipUnless(sys.platform == 'linux', 'Linux execution-root lock')
    def test_completed_root_drift_does_not_quarantine_or_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);sdk=cache/'official-sdks/sdk'
            output=cache/'execution-contexts'/('ubuntu-'+execution.BASE_DIGEST[:16]);output.mkdir(parents=True)
            (output/'receipt.json').write_text('{}')
            with patch.object(execution,'verify_root',side_effect=ValueError('drift')), \
                 patch.object(execution,'prepare_root') as prepare:
                with self.assertRaisesRegex(ValueError,'drift'):
                    execution.ensure_for_sdk(sdk)
            self.assertTrue(output.exists());prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
