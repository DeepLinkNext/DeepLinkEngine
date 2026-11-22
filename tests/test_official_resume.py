"""CPU custody and refusal checks for the narrow six-solver continuation."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official import enable_module_imports
enable_module_imports()
from official_suite import public_arguments
from official.official_resume import admit, PHASES, RETAIN, _closed
from official.official_rebuild import digest


class Resume(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.old = self.root / 'old-code'
        self.now = self.root / 'new-code'
        self.source = self.root / 'closed'
        self.dest = self.root / 'new-output'
        self.source.mkdir()
        self.config = dict(model=str(self.root / 'model'),
                           protocol_run=str(self.root / 'protocol'),
                           protocol_deployment=str(self.root / 'protocol-sdk'))
        self.inputs = Path(self.config['protocol_run']) / 'continuous-B0/inputs/continuous-inputs-r1.json'
        self.inputs.parent.mkdir(parents=True)
        self.inputs.write_text('{"4096": [1,2], "16384": [3,4]}')
        self.protocol = Path(self.config['protocol_run']) / 'frozen-execution-plan.json'
        self.protocol.write_text('{"fixed": true}')
        self.lock = dict(id='upstream-vllm-0.28.0-ascend-0.28.0rc1-source',
                         version='0.28.0rc1-source', manifest_kind='source-build-derivation',
                         manifest_digest='sha256:' + 'a' * 64)
        self.runtime = dict(runtime_kind='source-build-candidate', build_receipt_sha256='b' * 64)
        self.weights = {'shard.safetensors': dict(bytes=42, sha256='c' * 64,
                         device=1, inode=2, mtime_ns=3, ctime_ns=4)}
        self.mapping = {'0': 0, '1': 1}
        self.sources, self.old_sources = {}, {}
        for name in ('official/official_suite.py', 'official/official_judge.py',
                     'official/official_report.py',
                     'official/official_worker.py', 'benchmark/one-click-kit/scripts/workload.py',
                     'engine/service.py'):
            for root in (self.old, self.now):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(('original-' + name).encode())
            self.old_sources[str(self.old / name)] = digest(self.old / name)
            self.sources[str(self.now / name)] = digest(self.now / name)
        self.external = self.root / 'protocol-sdk/client.py'
        self.external.parent.mkdir(parents=True)
        self.external.write_bytes(b'original external client')
        self.old_sources[str(self.external)] = self.sources[str(self.external)] = digest(self.external)
        self.campaign = dict(schema='deeplinkengine-official-full-suite/v1', status='failed', ended=123.0,
            baseline_image=self.lock, physical_mapping=self.mapping, runtime_identity=self.runtime,
            continuous_input_sha256=digest(self.inputs), sources=self.old_sources,
            stages=[dict(name=name, status='completed' if i < 6 else 'failed' if i == 6 else 'not_attempted')
                    for i, name in enumerate(PHASES)])
        self.write('model-weights.json', self.weights)
        self.write('resource-admission.json', dict(admitted=True, physical_cards=[0, 1], mapping=self.mapping))
        for name in RETAIN:
            kind = 'continuous' if name == 'continuous' else 'joint' if name == 'mmlu' else 'dataset'
            receipt = dict(schema='deeplinkengine-official-arm/v1', baseline_id=self.lock['id'],
                phase=name, role='B0', status='completed', completed=True, model_ready=True,
                client_started=True, quiescent=True, cleanup_actions=[],
                input_plan_sha256=digest(self.protocol))
            plan = dict(baseline_id=self.lock['id'], role='B0', runtime_identity=self.runtime,
                kind=kind, serve_arguments=public_arguments(kind),
                model=self.config['model'], protocol_deployment=self.config['protocol_deployment'],
                continuous_inputs=str(self.inputs), input_sha256=digest(self.inputs),
                client_tools=str(self.old / 'benchmark/one-click-kit'), client_sources=self.old_sources,
                worker_sources={str(self.old / 'official/official_worker.py'):
                                self.old_sources[str(self.old / 'official/official_worker.py')]})
            self.write(name + '/receipt.json', receipt)
            self.write(name + '/service/plan.json', plan)
            self.write(name + '/model-weights.json', self.weights)
            self.write(name + '/resource-admission.json', self.load('resource-admission.json'))
            self.write(name + '/deployment.json', dict(release=self.lock['id'], **self.runtime))
            if kind == 'dataset':
                suffix = 'trace' if name in ('swe', 'terminal') else 'solver'
                spec = Path(self.config['protocol_run']) / 'plans' / (name + '-' + suffix + '-B0.json')
                spec.parent.mkdir(parents=True, exist_ok=True)
                spec.write_text(json.dumps(dict(dataset=name, case_ids=[1, 2, 3]), sort_keys=True))
                self.write(name + '/dataset-spec.json', json.loads(spec.read_bytes()))
                plan['dataset_spec'] = str(spec)
                self.write(name + '/service/plan.json', plan)
            self.write(name + '/service/evaluation/phase.json', {'status': 'completed', 'requests': 3})
            self.campaign['stages'][PHASES.index(name)]['receipt'] = receipt
        self.write('frontier-judge/receipt.json', dict(schema='deeplinkengine-official-fixed-judge/v1',
            dataset='frontier', status='failed', completed=False, quiescent=True, cleanup_actions=[]))
        self.write('frontier-judge/service/plan.json', dict(kind='dataset'))
        self.write('frontier-judge/service/judge-spec.json', dict(case_ids=[1, 2, 3]))
        self.reseal()

    def write(self, name, value):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')

    def load(self, name):
        return json.loads((self.source / name).read_bytes())

    def change(self, name, **values):
        data = self.load(name)
        data.update(values)
        self.write(name, data)

    def reseal(self):
        self.write('campaign.json', self.campaign)
        for name in (*RETAIN, 'frontier-judge'):
            directory = self.source / name
            pins = {p.relative_to(directory).as_posix(): digest(p)
                    for p in directory.rglob('*') if p.is_file() and p != directory / 'checksums.json'}
            self.write(name + '/checksums.json', pins)
        pins = {p.relative_to(self.source).as_posix(): digest(p)
                for p in self.source.rglob('*') if p.is_file() and p != self.source / 'checksums.json'}
        self.write('checksums.json', pins)
        self.seal = digest(self.source / 'checksums.json')

    def call(self, **changes):
        args = dict(expected_seal=self.seal, config=self.config, image_lock=self.lock,
                    runtime=self.runtime, sources=self.sources, weights=self.weights,
                    mapping=self.mapping, inputs=self.inputs, current_root=self.now, phases=PHASES)
        args.update(changes)
        return admit(self.source, self.dest, **args)

    def test_preserves_six_seals_receipts_payloads_and_source(self):
        before = {p.relative_to(self.source).as_posix(): p.read_bytes()
                  for p in self.source.rglob('*') if p.is_file()}
        result = self.call()
        self.assertEqual(set(result), set(RETAIN))
        for name in RETAIN:
            row = result[name]
            self.assertEqual(row['original_checksums_sha256'], digest(self.source / name / 'checksums.json'))
            self.assertEqual(row['original_receipt_sha256'], digest(self.source / name / 'receipt.json'))
            self.assertEqual(row['original_campaign_checksums_sha256'], self.seal)
            self.assertFalse(row['inference_repeated'])
            for path in (self.source / name).rglob('*'):
                if path.is_file():
                    self.assertEqual(path.read_bytes(), (self.dest / name / path.relative_to(self.source / name)).read_bytes())
        self.assertFalse((self.dest / 'frontier-judge').exists())
        self.assertEqual(before, {p.relative_to(self.source).as_posix(): p.read_bytes()
                                for p in self.source.rglob('*') if p.is_file()})

    def test_only_three_controller_changes_are_recorded(self):
        for name in ('official_suite.py', 'official_judge.py', 'official_report.py'):
            path = self.now / 'official' / name
            path.write_bytes(b'new reviewed controller')
            self.sources[str(path)] = digest(path)
        result = self.call()
        self.assertEqual(set(result['hle']['reviewed_controller_changes']),
                         {'official/official_suite.py', 'official/official_judge.py', 'official/official_report.py'})

    def test_noncontroller_change_refuses_before_copy(self):
        path = self.now / 'engine/service.py'
        path.write_bytes(b'changed helper')
        self.sources[str(path)] = digest(path)
        with self.assertRaisesRegex(ValueError, 'Non-controller'):
            self.call()
        self.assertFalse(self.dest.exists())

    def test_original_source_must_exist_with_original_hash(self):
        path = self.old / 'official/official_suite.py'
        path.write_bytes(b'mutated old source')
        with self.assertRaisesRegex(ValueError, 'supporting source'):
            self.call()

    def test_external_source_and_current_admission_are_checked(self):
        self.sources[str(self.external)] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Current source admission'):
            self.call()

    def test_entire_outer_seal_is_verified_not_just_solvers(self):
        self.write('unrelated/report-fragment.json', {'original': True})
        self.reseal()
        self.write('unrelated/report-fragment.json', {'original': False})
        with self.assertRaisesRegex(ValueError, 'sealed evidence changed'):
            self.call()

    def test_external_seal_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'External whole'):
            self.call(expected_seal='0' * 64)

    def test_active_or_wrong_phase_boundary_is_refused(self):
        for field, value in [('status', 'running'), ('status', 'completed')]:
            self.campaign[field] = value
            self.reseal()
            with self.assertRaisesRegex(ValueError, 'terminal candidate'):
                self.call()
        self.campaign['status'] = 'failed'
        self.campaign['stages'][7]['status'] = 'completed'
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'terminal candidate'):
            self.call()

    def test_candidate_runtime_mapping_and_weight_readback_match(self):
        cases = [dict(image_lock={**self.lock, 'version': '0.30.0'}),
                 dict(runtime={**self.runtime, 'build_receipt_sha256': 'd' * 64}),
                 dict(mapping={'0': 2, '1': 3}),
                 dict(weights={'shard.safetensors': {**self.weights['shard.safetensors'], 'inode': 99}})]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.call(**change)

    def test_protocol_and_continuous_input_are_not_relabelled(self):
        self.inputs.write_text('different workload')
        with self.assertRaisesRegex(ValueError, 'terminal candidate'):
            self.call()

    def test_phase_receipt_and_plan_runtime_refuse(self):
        for name, change in [('mmlu/receipt.json', dict(quiescent=False)),
                             ('swe/service/plan.json', dict(protocol_deployment='other-sdk'))]:
            saved = (self.source / name).read_bytes()
            self.change(name, **change)
            self.reseal()
            with self.assertRaises(ValueError):
                self.call()
            (self.source / name).write_bytes(saved)
        self.reseal()

    def test_inner_outer_population_mismatch(self):
        pins = self.load('mmlu/checksums.json')
        pins.pop('service/evaluation/phase.json')
        self.write('mmlu/checksums.json', pins)
        outer = self.load('checksums.json')
        outer['mmlu/checksums.json'] = digest(self.source / 'mmlu/checksums.json')
        self.write('checksums.json', outer)
        self.seal = digest(self.source / 'checksums.json')
        with self.assertRaisesRegex(ValueError, 'populations differ'):
            self.call()

    def test_failed_judge_must_be_quiescent_and_pre_ready(self):
        for change in (dict(quiescent=False), dict(client_started=True), dict(model_ready=True)):
            original = (self.source / 'frontier-judge/receipt.json').read_bytes()
            self.change('frontier-judge/receipt.json', **change)
            self.reseal()
            with self.assertRaises(ValueError):
                self.call()
            (self.source / 'frontier-judge/receipt.json').write_bytes(original)

    def test_even_unsealed_judge_ready_or_request_output_refuses(self):
        for name in ('ready-models.json', 'service/client.log', 'service/evaluation/request.json'):
            self.write('frontier-judge/' + name, {})
            with self.assertRaisesRegex(ValueError, 'readiness or client'):
                self.call()
            (self.source / 'frontier-judge' / name).unlink()

    def test_worker_binding_must_equal_published_sources(self):
        self.change('terminal/service/plan.json', worker_sources={str(self.external): '0' * 64})
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'worker/tool'):
            self.call()

    def test_path_traversal_in_outer_seal_refuses(self):
        outer = self.load('checksums.json')
        outer['../escape'] = '0' * 64
        self.write('checksums.json', outer)
        self.seal = digest(self.source / 'checksums.json')
        with self.assertRaisesRegex(ValueError, 'Unsafe evidence'):
            self.call()

    def test_existing_destination_is_not_overwritten(self):
        path = self.dest / 'mmlu'
        path.mkdir(parents=True)
        (path / 'keep').write_text('unchanged')
        with self.assertRaisesRegex(ValueError, 'must be absent'):
            self.call()
        self.assertEqual((path / 'keep').read_text(), 'unchanged')

    def test_original_cleanup_pid_birth_cannot_still_be_alive(self):
        receipt = dict(quiescent=True, cleanup_actions=[dict(pid=999999, start_ticks='456')])
        body = '999999 (original with spaces) ' + ' '.join(['S'] + ['0'] * 18 + ['456'])
        with patch.object(Path, 'read_text', return_value=body):
            with self.assertRaisesRegex(ValueError, 'still exists'):
                _closed(receipt)
        with patch.object(Path, 'read_text', return_value=body.replace('456', '789')):
            _closed(receipt)
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError):
            _closed(receipt)

    def test_original_serving_arguments_must_equal_current_public_arguments(self):
        self.change('continuous/service/plan.json', serve_arguments=['--dtype', 'float32'])
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'runtime, model or protocol'):
            self.call()

    def test_fixed_dataset_material_and_resource_admission_are_checked(self):
        self.write('swe/dataset-spec.json', dict(case_ids=['different']))
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'fixed dataset workload'):
            self.call()

    def test_changed_current_protocol_plan_is_refused(self):
        self.protocol.write_text('{"fixed": false}')
        with self.assertRaisesRegex(ValueError, 'receipt differs'):
            self.call()

    @unittest.skipUnless(os.name != 'nt', 'Symlink custody case uses an unprivileged Unix symlink')
    def test_sealed_symlink_refuses(self):
        path = self.source / 'mmlu/model-weights.json'
        body = path.read_bytes()
        path.unlink()
        external = self.root / 'weights.json'
        external.write_bytes(body)
        path.symlink_to(external)
        with self.assertRaisesRegex(ValueError, 'Symlink evidence'):
            self.call()


if __name__ == '__main__':
    unittest.main()
