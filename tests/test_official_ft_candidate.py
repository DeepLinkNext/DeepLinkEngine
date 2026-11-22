"""CPU contracts for the source candidate; no candidate runtime is certified."""
import base64
from contextlib import nullcontext
import copy
from enum import Enum
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests'), str(ROOT / 'benchmark/one-click-kit/scripts'),
    str(ROOT / 'benchmark/one-click-kit/third-party-test-kit'),
    str(ROOT / 'benchmark/one-click-kit/third-party-test-kit/proc_capture')]
from official import enable_module_imports

enable_module_imports()
from official_ft_factory import capture, validate as factory_validate
from official_ft_inventory import validate
from official_ft_audit import runtime, candidate_backend_admission
from official_ft_selection import version, selection, admit_candidate, VERSION, BASELINE
import test_official_ft as original_ft
loaded = original_ft.loaded


def factory_binding():
    source = '/sdk/rootfs/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py'
    row = dict(runner=dict(type='vllm_ascend.ops.fused_moe.fused_moe.AscendMoERunner', source=source),
        routed=dict(type='vllm_ascend.ops.fused_moe.routed_experts.AscendRoutedExperts', source=source),
        shared=dict(type='vllm_ascend.ops.fused_moe.shared_experts.AscendSharedExperts', source=source),
        shared_layer_is_inventory_mlp=True, n_shared_experts=0,
        quant_type=dict(type='vllm_ascend.quantization.quant_type.QuantType', name='NONE'),
        shared_quant_type=dict(type='vllm_ascend.quantization.quant_type.QuantType', name='NONE'),
        weights_replicated=False, shared_sequence_parallel=False, multistream_overlap=True)
    return dict(schema='official028-loaded-moe-factory-dispatch/v1',
        roles={role: [dict(copy.deepcopy(row), block=str(i)) for i in range(count)]
               for role, count in (('target', 40), ('draft', 1))},
        source_sha256={source: 'a'*64}, tensor_values_observed=False,
        forward_invoked=False, observer_installed=False)


class CandidateFactoryTests(unittest.TestCase):
    def test_loaded_factory_layout_uses_real_routed_count_without_invented_flags(self):
        inventory, text = loaded()
        for role in ('target', 'draft'):
            for row in inventory[role]['modules']:
                if row['name'].endswith('.mlp.experts'):
                    row['name'] += '.routed_experts'
                    row['scalars']['n_shared_experts'] = 0
                    row['objects']['quant_method']['type'] = 'vllm_ascend.ops.fused_moe.routed_experts.AscendUnquantizedFusedMoEMethod'
        result = validate(inventory, text, source_version=VERSION)
        self.assertEqual(result['loaded_moe_layouts']['target'], ['028-factory-routed-experts'])
        expert = next(row for row in inventory['target']['modules'] if row['name'].endswith('.mlp.experts.routed_experts'))
        del expert['scalars']['n_shared_experts']
        with self.assertRaisesRegex(ValueError, 'scalar differs'):
            validate(inventory, text, source_version=VERSION)

    def test_legacy_inventory_cannot_be_relabelled_as_candidate(self):
        inventory, text = loaded()
        with self.assertRaisesRegex(ValueError, 'factory routed owner'):
            validate(inventory, text, source_version=VERSION)

    def test_actual_shared_alias_quant_and_layout_are_required(self):
        self.assertFalse(factory_validate(factory_binding())['missing030_flags_filled'])
        for field, wrong in [('shared_layer_is_inventory_mlp', False), ('n_shared_experts', False),
                ('weights_replicated', True), ('shared_sequence_parallel', True),
                ('quant_type', dict(type='other.QuantType', name='NONE')),
                ('multistream_overlap', None), ('shared', None)]:
            with self.subTest(field=field):
                value = factory_binding()
                value['roles']['target'][0][field] = wrong
                with self.assertRaises(ValueError):
                    factory_validate(value)

    def test_capture_reads_plain_shared_executor_without_forward_or_tensor_values(self):
        quant = Enum('QuantType', ['NONE'], module='vllm_ascend.quantization.quant_type')
        def obj(module, name, **fields):
            cls = type(name, (), {'__module__': module})
            value = cls()
            value.__dict__.update(fields)
            return value
        def model(count):
            rows = []
            for i in range(count):
                mlp = SimpleNamespace()
                routed = obj('vllm_ascend.ops.fused_moe.routed_experts', 'AscendRoutedExperts', n_shared_experts=0)
                shared = obj('vllm_ascend.ops.fused_moe.shared_experts', 'AscendSharedExperts',
                    layer=mlp, quant_type=quant.NONE, weights_replicated=False,
                    is_sequence_parallel=False, multistream_overlap=True)
                executor = obj('vllm_ascend.ops.fused_moe.fused_moe', 'AscendMoERunner',
                    routed_experts=routed, ascend_shared_experts=shared, quant_type=quant.NONE)
                rows.append((str(i), SimpleNamespace(n_routed_experts=256, shared_expert=mlp, experts=executor)))
            return SimpleNamespace(named_modules=lambda: iter(rows))
        runner = SimpleNamespace(model=model(40), drafter=SimpleNamespace(model=model(1)))
        with patch('official_ft_factory.inspect.getfile', return_value=__file__):
            observed = capture(runner)
        factory_validate(observed)
        self.assertEqual(len(observed['source_sha256']), 1)
        self.assertFalse(observed['forward_invoked'])

    def test_candidate_resolved_sampling_fields_are_observed_not_defaulted(self):
        value = original_ft.RuntimeTests().value()
        value.update(constant_draft_positions=False, moe_factory_dispatch=factory_binding(),
            enable_reduce_sample=True, lmhead_tp_enabled=False, target_xdrope_dimensions=3,
            draft_xdrope_dimensions=3)
        runtime(value, VERSION)
        del value['enable_reduce_sample']
        with self.assertRaisesRegex(ValueError, 'source configuration'):
            runtime(value, VERSION)


class CandidateSelectionTests(unittest.TestCase):
    def identity(self):
        return dict(id=BASELINE, version=VERSION, upstream_version='0.28.0', candidate=True,
                    official_release_image=False, manifest_kind='source-build-derivation', manifest_digest='sha256:'+'b'*64)

    def test_candidate_uses_derivation_version_not_stable_upstream_label(self):
        self.assertEqual(version(self.identity()), VERSION)
        value = self.identity()
        value['official_release_image'] = True
        with self.assertRaises(ValueError):
            version(value)

    def test_candidate_cache_is_separate_and_exact_public_derivation_is_required(self):
        value = self.identity()
        declared = {k: v for k, v in value.items() if k not in ('manifest_digest', 'manifest_kind')}
        helper = SimpleNamespace(identity=lambda: copy.deepcopy(declared), key=lambda: 'b'*64)
        with patch.dict(sys.modules, official_source_candidate=helper):
            selected, _, cache = selection(value, '/cache', ROOT)
            self.assertEqual(selected, VERSION)
            self.assertEqual(cache.name, BASELINE+'-'+'b'*16)
            self.assertEqual(cache.parent.name, 'source-candidate-sdks')
            value['manifest_digest'] = 'sha256:'+'c'*64
            with self.assertRaisesRegex(ValueError, 'public derivation'):
                selection(value, '/cache', ROOT)

    def test_candidate_continuation_reuses_strict_public_sdk_admission(self):
        value = self.identity()
        installed = dict(schema='deeplinkengine-source-candidate-sdk/v1', image_manifest=value['manifest_digest'],
            official_release_image=False, installed_files_sha256='d'*64, build_receipt_sha256='e'*64, metadata={'versions': {}})
        calls = []
        helper = SimpleNamespace(reuse=lambda image: calls.append(image) or installed)
        state = SimpleNamespace(sdk_preparation=lambda image: nullcontext())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'image-lock.json').write_text(json.dumps(value))
            with patch.dict(sys.modules, official_source_candidate=helper, official_sdk_state=state):
                result = admit_candidate(root, value, root)
            self.assertEqual(calls, [root])
            self.assertEqual(result['manifest_kind'], 'source-build-derivation')
            self.assertFalse(result['official_release_image'])
            self.assertEqual(result['build_receipt_sha256'], 'e'*64)

    def test_public_candidate_cpu_rejection_does_not_require_private_build_controller(self):
        import subprocess
        from official_baselines import save
        from official_report import seal
        from official_ft_audit import Evidence
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, native, companion, config = original_ft.CompletionTests().prefix_fixture(root)
            domain = json.loads((ROOT/'engine/official-ft-source-domains.json').read_bytes())['domains'][VERSION]
            identity = self.identity()
            identity.update(base_image_manifest=domain['base_image_manifest'], upstream_source_patched=False,
                sources={name.replace('_', '-'): pin for name, pin in domain['source_archives'].items()})
            campaign = json.loads((original/'campaign.json').read_bytes())
            campaign['baseline_image'] = identity
            save(original/'campaign.json', campaign)
            for path in (native, companion):
                plan = json.loads((path/'native-plan.json').read_bytes())
                plan.update(baseline_id=BASELINE, expected_versions=dict(vllm='0.28.0', ascend='0.28.0rc1'))
                save(path/'native-plan.json', plan)
                seal(path)
            seal(original)
            config['versions'] = [VERSION]
            save(root/'evaluation.json', config)
            result = subprocess.run([sys.executable, '-B', str(ROOT/'run_all.py'), '--official-baselines',
                '--config', str(root/'evaluation.json'), '--complete-ft', str(original),
                '--source-version-seal-sha256', hashlib.sha256((original/'checksums.json').read_bytes()).hexdigest(),
                '--reuse-ft-companion', str(companion), '--companion-seal-sha256',
                hashlib.sha256((companion/'checksums.json').read_bytes()).hexdigest()],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
            self.assertIn('expected=1383, observed=1385', result.stderr)
            record = Evidence(config['out']).read('continuation.json')
            report = Evidence(config['out']).read('necessary-work/audit/producer-prefix-rejection.json')
            self.assertFalse(record['sdk_and_model_admission_performed'])
            self.assertIsNone(report['F_interval'])
            self.assertEqual(report['source_family'], domain['id'])
            self.assertFalse(Path(config['cache']).exists())


class CandidateBackendTests(unittest.TestCase):
    def test_backend_collector_reads_natural_cached_choice_without_calling_probe(self):
        from types import ModuleType
        import official_ft_runtime
        from importlib import metadata
        package, ops, gdn = (ModuleType(name) for name in
            ('vllm_ascend', 'vllm_ascend.ops', 'vllm_ascend.ops.gdn'))
        package.ops, ops.gdn = ops, gdn
        def forbidden(*args, **kwargs):
            raise AssertionError('Observer must not invoke a probe or forward')
        cls = SimpleNamespace(_fused_chunk_available=True,
            _probe_fused_chunk=forbidden, _chunk_gated_delta_rule_fused=forbidden)
        gdn.AscendGatedDeltaNetAttention = cls
        with patch.dict(sys.modules, vllm_ascend=package,
                **{'vllm_ascend.ops': ops, 'vllm_ascend.ops.gdn': gdn}), \
                patch.object(metadata, 'version', return_value='0.28.0rc1'), \
                patch('official_ft_cann.observe', return_value={'loaded_libraries': {}}) as collect:
            result = official_ft_runtime.gdn_backend()
        self.assertEqual(result['selected'], 'torch-npu-fused-chunk')
        self.assertFalse(result['probe_executed_by_observer'])
        collect.assert_called_once_with(forbidden)

    class Source:
        def __init__(self, path, values):
            self.values, self.selected = values, {}
            self.seal_path = path
            path.write_text('{}')
        def read(self, name):
            self.selected[name] = hashlib.sha256(json.dumps(self.values[name], sort_keys=True).encode()).hexdigest()
            return copy.deepcopy(self.values[name])

    def sources(self, root):
        fixture = json.loads((ROOT/'tests/fixtures/official-fixed200-tokens.json').read_bytes())
        tokens = json.loads(gzip.decompress(base64.b64decode(fixture['gzip_base64'])))
        plan = dict(baseline_id=BASELINE, expected_versions={'vllm': '0.28.0'}, purpose='arithmetic-companion')
        def workers():
            return [dict(rank=rank, resolved_configuration=dict(gdn_backend=dict(
                selected='torch-npu-fused-chunk', cached_fused_probe=True,
                operator_identity=dict(loaded_libraries={'opapi.so': 'd'*64},
                    host_tiling_parameters_observed=False)))) for rank in (0, 1)]
        native = {'native-plan.json': plan, 'token-workload.json': dict(request_token_ids=tokens)}
        native.update({f'client/formal-{i}-pre-audit.json': workers() for i in range(1, 4)})
        companion = {'native-plan.json': plan, 'token-workload.json': dict(request_token_ids=tokens),
            'receipt.json': dict(completed=True, quiescent=True),
            'client/companion-receipt.json': dict(completed=True, observers_detached=True, native_counts_or_times_replaced=False),
            'client/loaded-inventory-bindings-after.json': workers()}
        return self.Source(root/'native-seal.json', native), self.Source(root/'companion-seal.json', companion)

    def test_natural_cann_gap_writes_specific_sealed_rejection_without_repeating_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native, companion = self.sources(root)
            domain = dict(id='candidate-028/v1', baseline_id=BASELINE,
                supported_prefill_families=['triton-chunk64'], unmapped_backend_reasons={
                    'torch-npu-fused-chunk': dict(code='cann-host-tiling-source-binding-missing', missing=['actual host tiling arithmetic binding'])})
            out = root/'audit'
            out.mkdir()
            with self.assertRaisesRegex(ValueError, 'cann-host-tiling-source-binding-missing'):
                candidate_backend_admission(native, companion, domain, out)
            report = json.loads((out/'source-admission-rejection.json').read_bytes())
            seal = json.loads((out/'checksums.json').read_bytes())
            self.assertEqual(report['status'], 'ineligible')
            self.assertEqual(len(report['actual_backend_observations']), 8)
            self.assertIsNone(report['F_interval'])
            self.assertFalse(report['native_timing_repeated'])
            self.assertEqual(seal['source-admission-rejection.json'], hashlib.sha256((out/'source-admission-rejection.json').read_bytes()).hexdigest())

    def test_different_population_does_not_become_a_candidate_counterexample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native, companion = self.sources(root)
            companion.values['token-workload.json']['request_token_ids'][0][0] += 1
            with self.assertRaisesRegex(ValueError, 'different token population'):
                candidate_backend_admission(native, companion, {}, root)
            self.assertFalse((root/'source-admission-rejection.json').exists())


if __name__ == '__main__':
    unittest.main()
