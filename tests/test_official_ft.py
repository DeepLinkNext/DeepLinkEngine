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

    def clone(self):
        return Vector(copy.deepcopy(self.values))

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


class DeferredTensor:
    """CPU reads fail until the original proposer publishes graph updates."""
    def __init__(self, values, state):
        self.values, self.state = copy.deepcopy(values), state
        self.ndim = 2 if values and isinstance(values[0], list) else 1

    def __getitem__(self, index):
        return DeferredTensor(self.values[index], self.state)

    def numel(self):
        return sum(len(row) for row in self.values) if self.ndim == 2 else len(self.values)

    def detach(self):
        return self

    def clone(self):
        self.state['clones'].append(self.state['updated'])
        return DeferredTensor(self.values, self.state)

    def cpu(self):
        if not self.state['updated']:
            raise AssertionError('device synchronization before post-replay update')
        self.state['reads'] += 1
        return self

    def tolist(self):
        if not self.state['updated']:
            raise AssertionError('host read before post-replay update')
        return copy.deepcopy(self.values)


class SafeDraftProposer:
    method = 'mtp'
    num_speculative_tokens = 2
    needs_extra_input_slots = False
    pass_hidden_states_to_model = True
    pcp_size = dcp_size = 1
    parallel_drafting = False

    def __init__(self, state):
        self.state = state
        self.input_ids = DeferredTensor([11, 12], state)
        self.positions = DeferredTensor([0, 1], state)
        self.output = DeferredTensor([[21, 22]], state)
        self.fail = False

    def _get_positions(self, capacity):
        return self.positions[:capacity]

    def _runnable(self, **kwargs):
        # Real merged draft mutates first-pass buffers in place.
        self.input_ids.values[:] = [91, 92]
        self.positions.values[:] = [8, 9]
        if self.fail:
            raise RuntimeError('original draft failed')
        return self.output

    def _propose(self, sentinel, **kwargs):
        self.state['sentinel'] = sentinel
        result = self._runnable(**kwargs)
        self.state['updated'] = True  # Actual source post-replay update boundary.
        return result


class SafeDraftBoundaryTests(unittest.TestCase):
    def fixture(self, directory):
        from official_ft_draft import DraftObservation
        from types import SimpleNamespace
        state = dict(updated=False, reads=0, clones=[])
        proposer = SafeDraftProposer(state)
        runner = SimpleNamespace(drafter=proposer, input_batch=SimpleNamespace(
            req_ids=['request-1'], sampling_metadata=SimpleNamespace(all_greedy=True)))
        metadata = SimpleNamespace(seq_lens=DeferredTensor([3], state),
            seq_lens_list=[3], actual_seq_lengths_q=[2])
        kwargs = dict(batch_size=1, num_input_tokens=2, num_tokens=2,
            token_indices_to_sample=DeferredTensor([1], state),
            multi_steps_attn_metadata=[{'layer': metadata}])
        observer = DraftObservation(runner, directory, 0, 'a' * 32, lambda: 7)
        return proposer, observer, state, kwargs

    def test_reads_follow_original_post_replay_update_and_preserve_first_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            proposer, observer, state, kwargs = self.fixture(path)
            sentinel = object()
            self.assertIs(proposer._propose(sentinel, **kwargs), proposer.output)
            self.assertIs(state['sentinel'], sentinel)
            self.assertTrue(state['clones'])
            self.assertFalse(any(state['clones']))
            self.assertGreater(state['reads'], 0)
            self.assertEqual(observer.stop(), 1)
            record = json.loads((path / 'rank-0-drafts.jsonl').read_text())
            self.assertEqual(record['first_pass_input_ids'], [11, 12])
            self.assertEqual(record['first_pass_positions'], [0, 1])
            self.assertEqual(record['draft_token_ids'], [[21, 22]])
            self.assertEqual(record['wave'], 7)
            self.assertEqual(record['host_read_boundary'], 'after_original_propose_post_replay_update')
            self.assertNotIn('_runnable', vars(proposer))
            self.assertNotIn('_propose', vars(proposer))

    def test_original_failure_never_reads_device_or_leaks_method_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposer, observer, state, kwargs = self.fixture(Path(temporary))
            proposer.fail = True
            with self.assertRaisesRegex(RuntimeError, 'original draft failed'):
                proposer._propose(object(), **kwargs)
            self.assertEqual(state['reads'], 0)
            self.assertEqual(observer.stop(), 0)
            self.assertNotIn('_runnable', vars(proposer))
            self.assertNotIn('_propose', vars(proposer))

    def test_direct_runnable_without_outer_boundary_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposer, observer, state, kwargs = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, 'safe outer propose'):
                proposer._runnable(**kwargs)
            self.assertEqual(state['reads'], 0)
            self.assertEqual(proposer.input_ids.values, [11, 12])
            observer.stop()


def fixed_fla_entry(fused_fwd):
    return fused_fwd(output_final_state=True, chunk_size=64, layout='BNSD',
        use_exp2=False, use_qk_l2norm_in_kernel=False, allow_neg_eigval=False,
        disable_recompute=True, state_v_first=False)


def changed_fla_entry(fused_fwd):
    return fused_fwd(output_final_state=True, chunk_size=128, layout='BNSD',
        use_exp2=False, use_qk_l2norm_in_kernel=False, allow_neg_eigval=False,
        disable_recompute=True, state_v_first=False)


def loaded_fla_operator(*, output_final_state=False, chunk_size=64, layout='BNSD',
        use_exp2=False, use_qk_l2norm_in_kernel=False, allow_neg_eigval=False,
        disable_recompute=True, state_v_first=False, use_gate_in_kernel=False):
    raise AssertionError('CPU signature audit must never call the device operator')


class FlaDomainTests(unittest.TestCase):
    def test_actual_flags_combine_source_literals_and_loaded_defaults_without_call(self):
        from official_ft_fla import flags, profile
        expected = dict(profile()['call_flags'], use_gate_in_kernel=False)
        self.assertEqual(flags(loaded_fla_operator, fixed_fla_entry, expected), expected)

    def test_different_actual_chunk_family_is_rejected(self):
        from official_ft_fla import flags, profile
        with self.assertRaisesRegex(ValueError, 'flags differ'):
            flags(loaded_fla_operator, changed_fla_entry, profile()['call_flags'])

    def test_claimed_admission_without_actual_mapped_libraries_is_rejected(self):
        import official_ft_fla as module
        domain = module.profile()
        value = dict(profile_sha256=module.PROFILE_SHA256, observed_files=module.files(domain),
            admitted=True, actual_library_source_stamp=domain['source_stamp'],
            probe_kernel_executed=False, backend_modified=False, tensor_values_observed=False,
            loaded_libraries={})
        with self.assertRaisesRegex(ValueError, 'loaded library set'):
            module.validate(value, ROOT)

    def test_fla_closure_does_not_claim_whole_python_package_git_equivalence(self):
        from official_ft_fla import profile
        value = profile()
        self.assertFalse(value['whole_python_package_matches_public_git'])
        self.assertEqual(len(value['packaged_kernel_sources']), 88)

    def test_source_and_formula_policy_pins_reproduce(self):
        import official_ft_audit as module
        path = ROOT / 'engine/official-ft-source-domains.json'
        self.assertEqual(module.sha(path), module.SOURCE_DOMAINS_SHA256)
        for name, pin in json.loads(path.read_bytes())['formula_sha256'].items():
            self.assertEqual(module.sha(module.KIT / 'third-party-test-kit' / name), pin)


class CompletionTests(unittest.TestCase):
    def image_fixture(self, root):
        import official_ft
        sdk, out = root / 'sdk', root / 'out'
        sdk.mkdir()
        out.mkdir()
        (sdk / 'payload').write_bytes(b'original')
        manifest = {'payload': hashlib.sha256(b'original').hexdigest()}
        (sdk / 'installed-files.json').write_text(json.dumps(manifest))
        identity = dict(manifest_digest='sha256:fixed-image')
        (sdk / 'image-lock.json').write_text(json.dumps(identity))
        (sdk / 'installation.json').write_text(json.dumps(dict(image_manifest=identity['manifest_digest'],
            installed_files_sha256=official_ft.sha(sdk / 'installed-files.json'),
            metadata=dict(versions=dict(vllm='0.23.0')))))
        return sdk, out, identity

    def test_cached_sdk_payload_is_checked_before_a_live_probe(self):
        import official_ft
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temporary:
            sdk, out, identity = self.image_fixture(Path(temporary))
            (sdk / 'payload').write_bytes(b'modified')
            probe = Mock()
            module = SimpleNamespace(digest=official_ft.sha,
                file_map=lambda path: {'payload': official_ft.sha(path / 'payload')}, probe=probe)
            with patch.dict(sys.modules, {'official_image': module}):
                with self.assertRaisesRegex(ValueError, 'SDK files differ'):
                    official_ft.admit_image(sdk, identity, out)
            probe.assert_not_called()

    def test_valid_cached_sdk_is_probed_without_reinstallation(self):
        import official_ft
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temporary:
            sdk, out, identity = self.image_fixture(Path(temporary))
            probe = Mock(return_value=dict(versions=dict(vllm='0.23.0'), source='actual-origin-probe'))
            module = SimpleNamespace(digest=official_ft.sha,
                file_map=lambda path: {'payload': official_ft.sha(path / 'payload')}, probe=probe)
            with patch.dict(sys.modules, {'official_image': module}):
                value = official_ft.admit_image(sdk, identity, out)
            self.assertTrue(value['cached_payload_verified'])
            self.assertFalse(value['sdk_reinstalled'])
            probe.assert_called_once_with(sdk, out / 'sdk-probe', identity)

    def test_legacy_weight_stat_schema_still_hashes_unchanged_size_and_mtime(self):
        from official_ft import verify_model
        import os
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / 'shard.safetensors'
            shard.write_bytes(b'abcd')
            original = shard.stat()
            expected = dict(bytes=4, sha256=hashlib.sha256(b'abcd').hexdigest())
            metadata = dict(expected, inode=original.st_ino, mtime_ns=original.st_mtime_ns)
            shard.write_bytes(b'efgh')
            os.utime(shard, ns=(original.st_atime_ns, original.st_mtime_ns))
            with patch('full_reference_reuse.domain', return_value=dict(weights={shard.name: expected})):
                with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
                    verify_model(root, {shard.name: metadata}, root)

    def test_legacy_weight_schema_uses_actual_hash_when_bytes_match(self):
        from official_ft import verify_model
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / 'shard.safetensors'
            shard.write_bytes(b'abcd')
            original = shard.stat()
            expected = dict(bytes=4, sha256=hashlib.sha256(b'abcd').hexdigest())
            metadata = dict(expected, inode=original.st_ino, mtime_ns=original.st_mtime_ns)
            with patch('full_reference_reuse.domain', return_value=dict(weights={shard.name: expected})):
                checked = verify_model(root, {shard.name: metadata}, root)
            self.assertFalse(checked[shard.name]['unchanged_original_verified_stat'])

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
