"""Adversarial sealed-arm admission and zero/one-start orchestration, CPU only."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import arm_reuse as reuse
import baseline_admission
import run_paired


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ArmAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'old'
        self.root.mkdir()
        root = self.root
        write(root / 'receipt.json', dict(role='B0', kind='joint', completed=True,
            quiescent=True, status='completed', started=10))
        write(root / 'device-mapping.json', dict(physical_cards=[1, 6], logical_cards=[0, 1]))
        write(root / 'model-weights.json', {'weight':dict(bytes=1, sha256='fixed')})
        task = root / 'original-task.json'
        write(task, {'requests':list(range(200))})
        write(root / 'service/request-plan.json', dict(source=str(task), request_count=200,
            request_rate='inf', warmups=0, requests=list(range(200)), scorer_sha256='score'))
        files = {'actual-baseline-file':'frozen'}
        self.pinned = hashlib.sha256(json.dumps(files, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        write(root / 'baseline-source-admission.json', dict(files=files,
            source_map_sha256=self.pinned, deployment='/actual/B0'))
        write(root / 'deployment.json', dict(release='earlier-B1-companion'))
        write(root / 'installed-files.json', {'earlier-companion':'old'})
        write(root / 'checksums.json', {'receipt.json':reuse.sha(root / 'receipt.json')})
        write(root / 'controller-source.json', {'old':'pinned'})
        self.conditions = dict(physical_cards=[1, 6], logical_cards=[0, 1],
            task_plan_sha256=reuse.sha(task), scorer_sha256='score',
            model_weights=reuse.weights(root), baseline_deployment='/actual/B0',
            optimized_deployment_sha256='new-B1-profile',
            optimized_installed_files_sha256='new-B1-files', optimized_release='new-B1')
        self.inspected = dict(role='B0', recorded_requests=200, successful_requests=200,
            failed_requests=0, quality=dict(unreturned=0, scorer_sha256='score', correct=143),
            serving_warmup=dict(completed=True))
        self.addCleanup(patch.stopall)
        patch.object(baseline_admission, 'PINNED_SOURCE_MAP_SHA256', self.pinned).start()

    def check(self, role='B0'):
        reuse.validate_conditions(self.root, role, self.conditions, self.inspected)

    def test_old_companion_B1_map_does_not_change_actual_baseline(self):
        self.check()
        self.assertEqual(reuse.read(self.root / 'deployment.json')['release'], 'earlier-B1-companion')

    def test_wrong_role_or_failed_arm_never_reaches_raw_audit(self):
        with self.assertRaisesRegex(ValueError, 'requested role'):
            self.check('B1')
        write(self.root / 'receipt.json', dict(completed=False, quiescent=True))
        with patch('report_execution.inspect_run') as inspect:
            with self.assertRaisesRegex(ValueError, 'Failed or running'):
                reuse.admit(self.root, 'B0', self.conditions)
            inspect.assert_not_called()

    def test_mapping_arrival_scorer_weights_and_baseline_mutations_rejected(self):
        changes = [
            ('device-mapping.json', 'physical_cards', [0, 1]),
            ('device-mapping.json', 'logical_cards', [1, 0]),
            ('service/request-plan.json', 'request_rate', '1'),
            ('service/request-plan.json', 'warmups', 1),
            ('service/request-plan.json', 'scorer_sha256', 'other'),
            ('service/request-plan.json', 'requests', [0]),
            ('model-weights.json', 'weight', dict(bytes=1, sha256='changed')),
            ('baseline-source-admission.json', 'deployment', '/another/B0'),
            ('baseline-source-admission.json', 'files', {'changed':'source'}),
        ]
        for name, key, value in changes:
            path = self.root / name
            original = path.read_bytes()
            changed = reuse.read(path)
            changed[key] = value
            write(path, changed)
            try:
                with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                    self.check()
            finally:
                path.write_bytes(original)

    def test_failed_requests_missing_responses_and_incomplete_warmup_rejected(self):
        for key, value in [('recorded_requests', 199), ('successful_requests', 199),
                           ('failed_requests', 1), ('serving_warmup', {'completed':False})]:
            original = copy.deepcopy(self.inspected)
            self.inspected[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check()
            self.inspected = original

    def test_old_optimized_arm_cannot_be_relabelled_as_current(self):
        receipt = reuse.read(self.root / 'receipt.json')
        receipt['role'] = 'B1'
        write(self.root / 'receipt.json', receipt)
        self.inspected['role'] = 'B1'
        with self.assertRaisesRegex(ValueError, 'exact current optimized'):
            self.check('B1')
        self.conditions.update(optimized_deployment_sha256=reuse.sha(self.root / 'deployment.json'),
            optimized_installed_files_sha256=reuse.sha(self.root / 'installed-files.json'),
            optimized_release='earlier-B1-companion')
        self.check('B1')

    def test_binding_preserves_original_date_and_rejects_later_tampering(self):
        with patch('report_execution.inspect_run', return_value=self.inspected):
            binding = reuse.admit(self.root, 'B0', self.conditions)
            self.assertEqual(binding['original_started'], 10)
            self.assertEqual(binding['original_release'], 'earlier-B1-companion')
            self.assertEqual(reuse.recheck(binding), self.root)
            write(self.root / 'controller-source.json', {'other':'changed'})
            with self.assertRaisesRegex(ValueError, 'changed after prospective admission'):
                reuse.recheck(binding)


class PairOrchestrationTests(unittest.TestCase):
    def run_pair(self, reuse_roles):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            deployment, root = base / 'sdk', base / 'new'
            root.mkdir()
            for name in ('inputs/mmlu-quality018-r2/quality-request-plan.json',
                         'scorers/mmlu_pro_pyramidkv.py', 'provenance/installed-files.json'):
                write(deployment / name, {})
            write(deployment / 'deployment.json', {'release':'full-answer-b1-20261116-r7'})
            sources = {role:base / ('old-' + role) for role in reuse_roles}
            args = SimpleNamespace(deployment=deployment, baseline_deployment=base / 'base',
                physical_devices='1,6', devices='0,1', qualified_continuous=base / 'c8',
                model=base / 'model', report_python=None,
                reuse_baseline_arm=sources.get('B0'), reuse_optimized_arm=sources.get('B1'))
            def baseline(*args):
                write(args[1], {})
                return dict(python='python', versions={}, source_map_sha256='base', adaptations={})
            def admission(source, role, conditions):
                return dict(run=str(source), role=role, seal_sha256=role + '-seal')
            def child(command):
                role = command[command.index('--role') + 1]
                write(root / role / 'checksums.json', {'fresh':True})
                return SimpleNamespace(returncode=0)
            with (
                patch.object(run_paired, 'verify_installed_sources'),
                patch.object(run_paired, 'verify_mmlu_scope', return_value={}),
                patch.object(run_paired, 'admit', side_effect=baseline),
                patch.object(run_paired, 'admit_dependencies', return_value={}),
                patch.object(run_paired, 'qualify', return_value=dict(weights={}, run='c8', seal_sha256='seal')),
                patch.object(reuse, 'expected', return_value={}),
                patch.object(reuse, 'admit', side_effect=admission),
                patch.object(reuse, 'recheck'),
                patch.object(run_paired, 'run_stage', side_effect=child) as start,
                patch('report_environment.report_python', return_value=(sys.executable, {})),
                patch.object(run_paired.subprocess, 'run', return_value=SimpleNamespace(returncode=0)),
            ):
                self.assertEqual(run_paired.execute_pair(args, root), 0)
            self.assertEqual(start.call_count, 2 - len(reuse_roles))
            receipt = reuse.read(root / 'receipt.json')
            self.assertEqual(receipt['new_inference_arms'], 2 - len(reuse_roles))
            self.assertEqual(receipt['reused_arms'], sorted(reuse_roles))
            for row in receipt['attempts']:
                if row['role'] in reuse_roles:
                    self.assertTrue(row['reused_verified'])
                    self.assertNotIn('returncode', row)
                    self.assertFalse((root / row['role']).exists())
            frozen = reuse.read(root / 'frozen-pair-plan.json')
            self.assertEqual(frozen.get('reused_arms', {}).keys(), sources.keys())
            seal = reuse.read(root / 'pair-checksums.json')
            for role in ('B0', 'B1'):
                self.assertEqual(role + '/checksums.json' in seal, role not in reuse_roles)

    def test_both_verified_arms_launch_no_model(self):
        self.run_pair(['B0', 'B1'])

    def test_reused_baseline_launches_only_new_B1(self):
        self.run_pair(['B0'])

    def test_original_fresh_pair_still_launches_both_arms(self):
        self.run_pair([])


if __name__ == '__main__':
    unittest.main()
