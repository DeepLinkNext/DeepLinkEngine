import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmark/one-click-kit/scripts')]
from official_image import configure_fla_opp, environment
from startup_failure import inspect


class OppEnvironmentTests(unittest.TestCase):
    def sdk(self, root, version='0.30.0', vendor=True):
        root.mkdir()
        (root / 'image-lock.json').write_text(json.dumps(dict(upstream_version=version)))
        (root / 'image-config.json').write_text(json.dumps(dict(config=dict(Env=[]))))
        home = root / 'rootfs/usr/local/python3.12.16'
        (home / 'bin').mkdir(parents=True)
        (home / 'bin/python3').write_text('CPU fixture only')
        site = home / 'lib/python3.12/site-packages'
        package = site / 'fla_npu'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text("raise AssertionError('environment detection must not import this module')")
        target = package / 'opp/vendors/fla_npu_transformer'
        if vendor:
            (target / 'op_api/lib').mkdir(parents=True)
            (target / 'op_api/lib/libcust_opapi.so').write_bytes(b'CPU fixture only')
            (target / 'op_impl').mkdir()
        else:
            target.parent.mkdir(parents=True)
        return site, target

    def test_actual_environment_binds_embedded_opp_before_python_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'sdk'
            _, vendor = self.sdk(root)
            _, env = environment(root, Path(directory) / 'cache')
            self.assertEqual(env['ASCEND_CUSTOM_OPP_PATH'].split(':'), [str(vendor), str(vendor / 'op_api/lib')])
            self.assertNotIn(str(vendor / 'op_api/lib'), env['LD_LIBRARY_PATH'].split(':'))
            self.assertNotIn('fla_npu', sys.modules)

    def test_preserves_image_vendor_entries_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'sdk'
            site, vendor = self.sdk(root)
            env = dict(ASCEND_CUSTOM_OPP_PATH=str(vendor) + ':/prior/image/vendor')
            configure_fla_opp(root, site, env)
            self.assertEqual(env['ASCEND_CUSTOM_OPP_PATH'].split(':'), [str(vendor), str(vendor / 'op_api/lib'), '/prior/image/vendor'])

    def test_023_environment_does_not_receive_030_operator_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'sdk'
            site, _ = self.sdk(root, version='0.23.0')
            env = dict(ASCEND_CUSTOM_OPP_PATH='/existing')
            configure_fla_opp(root, site, env)
            self.assertEqual(env, dict(ASCEND_CUSTOM_OPP_PATH='/existing'))

    def test_missing_or_external_operator_library_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'sdk'
            site, vendor = self.sdk(root, vendor=False)
            with self.assertRaisesRegex(ValueError, 'complete'):
                configure_fla_opp(root, site, {})
            external = Path(directory) / 'external'
            external.mkdir()
            vendor.symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'inside'):
                configure_fla_opp(root, site, {})

    def test_inherited_fla_library_overrides_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'sdk'
            self.sdk(root)
            with patch.dict(os.environ, dict(FLA_NPU_STABLE_LIB='/foreign/lib.so', FLA_NPU_OP_API_LIB='/foreign/opapi.so')):
                _, env = environment(root, Path(directory) / 'cache')
            self.assertNotIn('FLA_NPU_STABLE_LIB', env)
            self.assertNotIn('FLA_NPU_OP_API_LIB', env)

    def test_existing_fatal_parser_distinguishes_failure_from_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'service.log'
            log.write_text('RuntimeWarning: embedded OPP is not configured\n')
            self.assertIsNone(inspect(log))
            log.write_text('(Worker_TP0 pid=21665) ERROR 11-21 04:56:09 [multiproc_executor.py:1055] WorkerProc hit an exception.\n'
                           '(Worker_TP0 pid=21665) ERROR 11-21 04:56:09 [multiproc_executor.py:1055] RuntimeError: fla_npu(stable): aclnnCausalConv1dGetWorkspaceSize failed: 561002\n')
            found = inspect(log)
            self.assertEqual(found['pid'], 21665)
            self.assertIn('561002', found['root_error'])


if __name__ == '__main__':
    unittest.main()
