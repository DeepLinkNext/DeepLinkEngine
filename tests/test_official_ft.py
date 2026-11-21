import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmark/one-click-kit/scripts'),
               str(ROOT / 'benchmark/one-click-kit/third-party-test-kit'),
               str(ROOT / 'benchmark/one-click-kit/third-party-test-kit/proc_capture')]
from official_ft_inventory import validate
from official_ft_capacity import envelope
from official_ft_audit import Evidence, runtime
from official_ft_draft import length_planes
from official_ft_inputs import stable_lengths


def loaded():
    fixture = json.loads((ROOT / 'tests/fixtures/official-023-loaded-semantics.json').read_bytes())
    raw = gzip.decompress(base64.b64decode(fixture['gzip_base64']))
    assert hashlib.sha256(raw).hexdigest() == fixture['payload_sha256']
    return json.loads(raw), fixture['model_text']


class LoadedInventoryTests(unittest.TestCase):
    def test_real_official_inventory_uses_actual_legacy_moe_layout(self):
        inventory, text = loaded()
        result = validate(inventory, text)
        self.assertEqual(result['checked_modules'], dict(target=593, draft=20))
        self.assertEqual(result['loaded_moe_layouts'], dict(target=['legacy-fused-moe'], draft=['legacy-fused-moe']))
        mlp = next(r for r in inventory['target']['modules'] if r['name'] == 'language_model.model.layers.0.mlp')
        self.assertNotIn('is_fused_shared_expert_enabled', mlp['scalars'])

    def test_loaded_expert_partition_cannot_be_inferred_from_config(self):
        inventory, text = loaded()
        expert = next(r for r in inventory['target']['modules'] if r['name'].endswith('.layers.0.mlp.experts'))
        expert['parameters']['w13_weight']['shape'][0] = 257
        with self.assertRaisesRegex(ValueError, 'parameter differs'):
            validate(inventory, text)

    def test_changed_draft_dispatch_is_rejected(self):
        inventory, text = loaded()
        expert = next(r for r in inventory['draft']['modules'] if r['name'] == 'model.layers.0.mlp.experts')
        expert['objects']['quant_method']['type'] = 'other.Method'
        with self.assertRaisesRegex(ValueError, 'dispatch'):
            validate(inventory, text)

    def test_changed_norm_scalar_is_rejected(self):
        inventory, text = loaded()
        norm = next(r for r in inventory['target']['modules'] if r['name'].endswith('layers.0.input_layernorm'))
        norm['scalars']['variance_epsilon'] = 1e-5
        with self.assertRaisesRegex(ValueError, 'scalar differs'):
            validate(inventory, text)


class CapacityTests(unittest.TestCase):
    def core(self):
        return dict(num_blocks=1000, in_flight_batches=2, prefix_caching=False,
            scheduler=dict(max_num_seqs=24, max_num_batched_tokens=4096), groups=[
                dict(type='vllm.v1.kv_cache_interface.FullAttentionSpec', block_size=2048,
                     host_resident=None),
                dict(type='vllm.v1.kv_cache_interface.MambaSpec', block_size=2048,
                     host_resident=None, mamba_cache_mode='none', num_speculative_blocks=2)])

    def test_actual_shared_pool_and_none_mode_charge_extra_speculative_pages(self):
        proof = envelope(self.core(), [6000, 100])
        pages = (proof['maximum_prefix_tokens'] + 2047) // 2048
        self.assertEqual(proof['required_blocks'], 25 * (pages + pages + 2 + 2) + 1)
        self.assertTrue(proof['sufficient_page_capacity'])
        self.assertFalse(proof['old_capacity_conversion_used'])

    def test_insufficient_actual_pool_does_not_certify(self):
        core = self.core()
        core['num_blocks'] = 1
        self.assertFalse(envelope(core, [6000])['sufficient_page_capacity'])

    def test_host_pool_is_not_silently_charged_to_device_pool(self):
        core = self.core()
        core['groups'][0]['host_resident'] = True
        with self.assertRaisesRegex(ValueError, 'host-resident'):
            envelope(core, [6000])

    def test_unmapped_queue_depth_is_rejected(self):
        core = self.core()
        core['in_flight_batches'] = 3
        with self.assertRaises(ValueError):
            envelope(core, [6000])


class RuntimeTests(unittest.TestCase):
    def value(self):
        return dict(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1,
            prefill_context_parallel_size=1, decode_context_parallel_size=1, enable_expert_parallel=False,
            async_scheduling=True, maximum_sequences=24, maximum_batched_tokens=4096,
            maximum_model_length=65536, kv_cache_memory_bytes=20 * 1024**3, prefix_caching=False,
            speculative_method='mtp', speculative_tokens=2, parallel_drafting=False,
            speculative_tokens_per_batch_size=None, dynamic_spec_active=False,
            tensor_values_observed=False, per_step_observer_installed=False,
            constant_draft_positions=None)

    def test_original_023_unconditional_positions_are_not_filled_with_false(self):
        value = self.value()
        self.assertIs(runtime(value, '0.23.0')['constant_draft_positions'], None)

    def test_030_requires_observed_position_policy(self):
        value = self.value()
        with self.assertRaisesRegex(ValueError, 'constant-position'):
            runtime(value, '0.30.0')
        value['constant_draft_positions'] = False
        runtime(value, '0.30.0')

    def test_dynamic_k_does_not_inherit_the_configured_maximum(self):
        value = self.value()
        value['speculative_tokens_per_batch_size'] = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, 'speculative_tokens_per_batch_size'):
            runtime(value, '0.23.0')

    def test_unresolved_async_is_not_guessed(self):
        value = self.value()
        value['async_scheduling'] = None
        with self.assertRaisesRegex(ValueError, 'async_scheduling'):
            runtime(value, '0.23.0')


class Vector:
    ndim = 1

    def __init__(self, values):
        self.values = values

    def __getitem__(self, item):
        return Vector(self.values[item])

    def numel(self):
        return len(self.values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class LengthPlaneTests(unittest.TestCase):
    def test_actual_official_kernel_fields_are_read(self):
        from types import SimpleNamespace
        meta = SimpleNamespace(seq_lens=Vector([3]), seq_lens_list=[3], actual_seq_lengths_q=[2])
        result = length_planes([{'layer': meta}], 1)
        self.assertEqual(result[0]['actual_kv_lengths'], [3])
        self.assertFalse(hasattr(meta, '_mtp_device_seq_lens'))

    def test_disagreeing_kernel_arguments_are_rejected(self):
        from types import SimpleNamespace
        meta = SimpleNamespace(seq_lens=Vector([4]), seq_lens_list=[3], actual_seq_lengths_q=[2])
        with self.assertRaisesRegex(ValueError, 'disagree'):
            length_planes([{'layer': meta}], 1)

    def test_optimized_length_publication_rule_is_not_applied(self):
        row = dict(length_planes_before=[dict(step=0), dict(step=1)],
                   length_planes_after=[dict(step=0), dict(step=1, changed=True)])
        with self.assertRaisesRegex(ValueError, 'changed'):
            stable_lengths(row)


class CompletionTests(unittest.TestCase):
    def test_companion_plugin_is_discovered_but_inactive_in_ordinary_runs(self):
        import importlib.metadata
        import official_ft_plugin
        import os
        plugins = {entry.name for distribution in importlib.metadata.distributions(path=[str(ROOT)])
                   for entry in distribution.entry_points if entry.group == 'vllm.general_plugins'}
        self.assertIn('deeplinkengine_official_ft', plugins)
        modules = set(sys.modules)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(official_ft_plugin.install())
        self.assertEqual(set(sys.modules), modules)

    def test_sealed_completion_marker_is_last(self):
        from official_ft import finish
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'receipt.json').write_text('{"completed":false}')
            finish(root, 'receipt.json', dict(status='completed', completed=True))
            self.assertTrue(Evidence(root).read('receipt.json')['completed'])

    def test_final_seal_failure_cannot_leave_completed(self):
        import official_ft
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'receipt.json').write_text('{"completed":false}')
            actual = official_ft.save

            def fail_seal(path, value):
                if Path(path).name == 'checksums.json':
                    raise OSError('injected final seal failure')
                return actual(path, value)

            with patch.object(official_ft, 'save', side_effect=fail_seal):
                with self.assertRaises(OSError):
                    official_ft.finish(root, 'receipt.json', dict(status='completed', completed=True))
            self.assertFalse(json.loads((root / 'receipt.json').read_bytes())['completed'])


if __name__ == '__main__':
    unittest.main()
