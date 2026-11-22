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
from official_ft_inputs import classify_envelopes, envelope_dependencies, stable_lengths


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

    def test_full_query_advance_is_traced_only_to_retained_second_drafts(self):
        request = 'a' * 64
        for retained in (1, 3):
            with self.subTest(retained=retained):
                scheduler, drafts = [], []
                for wave, start, tokens, offered, kept, produced in (
                    (0, 0, [2, 3], 1, [3], [4, 5]),
                    (1, 2, [6, 7, 8], 1, [6], [9, 10]),
                    (2, 3, [9, 10, 11], 3, [9, 10, 11][:retained], [12, 13]),
                ):
                    size = len(tokens)
                    planes = [
                        dict(step=0, query_end_envelope=[size], actual_kv_lengths=[start + size]),
                        dict(step=1, query_end_envelope=[1], actual_kv_lengths=[start + size + 1]),
                    ]
                    for plane in planes:
                        plane["kv_length_envelope"] = list(plane["actual_kv_lengths"])
                    drafts.append(dict(window='w', wave=wave, call=wave, capacity=size,
                        actual_input_tokens=size, requests=[request],
                        sample_indices=[size - 1 if wave == 0 else offered - 1],
                        first_pass_positions=list(range(start, start + size)),
                        first_pass_input_ids=tokens, draft_token_ids=[produced],
                        length_planes_before=planes, length_planes_after=copy.deepcopy(planes)))
                    scheduler.append(dict(window='w', wave=wave,
                        rows=[dict(request=request, start=start, count=size, draft=0 if wave == 0 else 2,
                                   draft_token_ids=[] if wave == 0 else [-1, -1])],
                        retained=[dict(request=request, offered=offered, kept=len(kept),
                                       kept_token_ids=kept, stale=False, stopped=wave == 2)]))
                streams = [scheduler, copy.deepcopy(scheduler), copy.deepcopy(scheduler),
                           drafts, copy.deepcopy(drafts)]
                original = copy.deepcopy(streams)
                counts, widened = classify_envelopes(drafts)
                self.assertEqual(counts, dict(selected_prefix=2, full_query_advance=1))
                self.assertEqual(widened[1, request]['final_position_without_current_call_writer'], 5)
                dependency = envelope_dependencies(streams, widened)
                self.assertEqual(dependency['widened_full_attention_calls'], int(retained == 3))
                self.assertEqual(streams, original)
                # The version-specific adaptation does not accept arbitrary
                # lengths or silently accept the projected stream as physical.
                with self.assertRaisesRegex(ValueError, 'selected-prefix'):
                    stable_lengths(drafts[1])
                for boundary in ('length_planes_before', 'length_planes_after'):
                    drafts[1][boundary][1]['actual_kv_lengths'][0] += 1
                with self.assertRaisesRegex(ValueError, 'selected-prefix'):
                    classify_envelopes(drafts)


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
    parallel_drafting = False

    def __init__(self, state):
        from types import SimpleNamespace
        self.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1), speculative_config=SimpleNamespace(num_speculative_tokens=2, num_speculative_tokens_per_batch_size=None))
        self.pcp_size = self.dcp_size = 1
        self.state = state
        self.input_ids = DeferredTensor([11, 12], state)
        self.positions = DeferredTensor([0, 1], state)
        self.output = DeferredTensor([[21, 22]], state)
        self.fail = False

    def _get_positions(self, capacity):
        return self.positions[:capacity]

    def _set_positions(self, capacity, positions):
        raise AssertionError('observer must not mutate positions')

    def set_inputs_first_pass(self, **kwargs):
        raise AssertionError('observer must not prepare inputs')

    def prepare_inputs_padded(self, **kwargs):
        raise AssertionError('observer must not prepare padded inputs')

    def _runnable(self, **kwargs):
        # Real merged draft mutates first-pass buffers in place.
        self.input_ids.values[:] = [91, 92]
        self.positions.values[:] = [8, 9]
        if self.fail:
            raise RuntimeError('original draft failed')
        return self.output

    _run_merged_draft = _runnable

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
        runner = SimpleNamespace(drafter=proposer, dcp_size=1, pcp_size=1, input_batch=SimpleNamespace(
            req_ids=['request-1'], sampling_metadata=SimpleNamespace(all_greedy=True)))
        proposer.runner = runner
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


class FlaArithmeticTests(unittest.TestCase):
    def fixture(self, lengths=(1, 8, 64, 65)):
        from types import SimpleNamespace
        from gdn_partition_bounds import bounds
        import official_ft_fla
        model = SimpleNamespace(linear_key_head_dim=128, linear_value_head_dim=128,
            linear_num_value_heads=32, layer_types=['linear_attention'] * 30 + ['full_attention'] * 10)
        partition = bounds(list(lengths), allow_recurrent_prompt=True)
        multiplier = model.linear_num_value_heads * model.layer_types.count('linear_attention')
        endpoints = {side: dict(total_basic=partition[extremum+'_basic'] * multiplier + 19,
            components=dict(target_gdn_prefill=partition[extremum+'_basic'] * multiplier, other=19))
            for side, extremum in [('lower', 'minimum'), ('upper', 'maximum')]}
        geometry = dict(geometry_sha256='own-native-geometry', requests=[dict(
            request_ordinal=i, input_tokens=n) for i, n in enumerate(lengths)])
        base = dict(schema='current-public-necessary-arithmetic-interval/v1',
            requests=len(lengths), geometry_sha256=geometry['geometry_sha256'],
            F_interval=[endpoints[side]['total_basic'] for side in ('lower', 'upper')],
            endpoints=endpoints, prefill_partition_envelope=partition)
        workers = [dict(rank=i, resolved_configuration=dict(gdn_backend=dict(selected='fla-npu',
            operator_identity=dict(admitted=True, profile_sha256=official_ft_fla.PROFILE_SHA256)))) for i in (0, 1)]
        return base, model, geometry, workers

    def test_source_adjustment_keeps_full_canonical_and_reference_independent(self):
        from official_ft_arithmetic import adjust_fla
        base, model, geometry, workers = self.fixture()
        original = copy.deepcopy(base)
        result = adjust_fla(base, model, geometry, workers)
        deduction = sum(row['input_tokens'] for row in geometry['requests']) * 127 * 30 * 32
        self.assertEqual(result['F_interval'], [original['F_interval'][0] - deduction, original['F_interval'][1]])
        self.assertEqual(base, original)
        self.assertEqual(result['base_canonical_ledger'], original)
        self.assertNotIn('prefill_partition_envelope', result)
        self.assertFalse(result['source_bound_correction']['corrected_endpoint_witness_claimed'])
        self.assertFalse(result['source_bound_correction']['original_reference_denominator_modified'])
        for side in ('lower', 'upper'):
            self.assertEqual(sum(result['endpoints'][side]['components'].values()), result['endpoints'][side]['total_basic'])

    def test_bound_contains_every_source_adjusted_small_partition_including_recurrence(self):
        from official_ft_arithmetic import adjust_fla
        from gdn_prefill_reference import chunk_arithmetic
        from gdn_reference import operation_counts
        # Enumerate all ordered compositions and both admitted one-token paths;
        # this checks the conservative envelope rather than copying its formula.
        def actual_costs(n, initial):
            if n == 0:
                return [0]
            result = []
            for size in range(1, n + 1):
                costs = [chunk_arithmetic(size, initial)['total_basic'] - size * 127]
                if size == 1:
                    costs.append(operation_counts(128, 128)['total_basic'] if initial else 3*128*128+128)
                result.extend(head + tail for head in costs for tail in actual_costs(n-size, True))
            return result
        for length in range(1, 8):
            base, model, geometry, workers = self.fixture((length,))
            result = adjust_fla(base, model, geometry, workers)
            actual = actual_costs(length, False)
            self.assertLessEqual(result['F_interval'][0], min(actual)*30*32+19)
            self.assertGreaterEqual(result['F_interval'][1], max(actual)*30*32+19)

    def test_original_triton_and_mixed_backend_are_conservative_without_tp_repeat(self):
        from official_ft_arithmetic import adjust_fla
        base, model, geometry, workers = self.fixture((282265,))
        workers[0]['resolved_configuration']['gdn_backend'] = dict(selected='triton-chunk64')
        result = adjust_fla(base, model, geometry, workers)
        self.assertEqual(result['source_bound_correction']['maximum_possible_basic_deduction'], 34413748800)
        self.assertEqual(result['source_bound_correction']['extra_tensor_parallel_multiplier'], 1)
        workers[1]['resolved_configuration']['gdn_backend'] = dict(selected='triton-chunk64')
        self.assertIs(adjust_fla(base, model, geometry, workers), base)

    def test_changed_geometry_or_unadmitted_profile_cannot_correct_f(self):
        from official_ft_arithmetic import adjust_fla
        base, model, geometry, workers = self.fixture()
        geometry['geometry_sha256'] = 'another-native-repetition'
        with self.assertRaisesRegex(ValueError, 'geometry/model binding'):
            adjust_fla(base, model, geometry, workers)
        geometry['geometry_sha256'] = base['geometry_sha256']
        workers[1]['resolved_configuration']['gdn_backend']['operator_identity']['admitted'] = False
        with self.assertRaisesRegex(ValueError, 'profile admission'):
            adjust_fla(base, model, geometry, workers)


class CompletionTests(unittest.TestCase):
    def prefix_fixture(self, root, mismatch=True):
        from official_report import seal
        from official_baselines import save
        original, companion = root / 'original', root / 'companion'
        native = original / 'native-timing'
        native.mkdir(parents=True)
        (companion / 'client/raw-work').mkdir(parents=True)
        identity = json.loads((ROOT / 'engine/official-baselines/0.23.0.json').read_bytes())
        plan = dict(role='B0', formal_repetitions=3, baseline_id='official-vllm-0.23.0',
                    expected_versions=dict(vllm='0.23.0', ascend='0.23.0'))
        for path in (native, companion):
            save(path / 'native-plan.json', plan)
            save(path / 'receipt.json', dict(completed=True, quiescent=True, purpose='arithmetic-companion'))
        save(companion / 'native-plan.json', dict(plan, purpose='arithmetic-companion'))
        fixture = json.loads((ROOT / 'tests/fixtures/official-fixed200-tokens.json').read_bytes())
        population = gzip.decompress(base64.b64decode(fixture['gzip_base64']))
        self.assertEqual(hashlib.sha256(population).hexdigest(), fixture['prompt_tokens_sha256'])
        for path in (native, companion):
            save(path / 'token-workload.json', dict(request_token_ids=json.loads(population)))
        save(companion / 'client/companion-receipt.json', dict(completed=True, timing_eligible=False,
            requests=200, observers_detached=True, native_counts_or_times_replaced=False, window='a'*32))
        raw = companion / 'client/raw-work'
        save(raw / 'core-end.json', dict(completed=True, window='a'*32, waves=6))
        save(raw / 'workers-end.json', [dict(completed=True, window='a'*32, rank=i, waves=6, draft_calls=6) for i in (0, 1)])
        rows = []
        for call in range(6):
            positions = [call] if call < 5 or not mismatch else [1381, 1382, 1383]
            planes = [dict(step=0, actual_kv_lengths=[positions[-1]+1], query_end_envelope=[len(positions)]),
                      dict(step=1, actual_kv_lengths=[positions[-1]+2], query_end_envelope=[1])]
            rows.append(dict(window='a'*32, call=call, wave=call, requests=['cpu-fixture-request'],
                actual_input_tokens=len(positions), capacity=len(positions), sample_indices=[0],
                first_pass_positions=positions, first_pass_input_ids=[10]*len(positions),
                draft_token_ids=[[11, 12]], length_planes_before=planes, length_planes_after=planes))
        for rank in (0, 1):
            (raw / f'rank-{rank}-drafts.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
        seal(native)
        save(original / 'campaign.json', dict(status='measurements_collected', started=1, ended=2,
            stages=[], baseline_image=identity, sources={}, qualification=dict(F_interval=None)))
        seal(original)
        seal(companion)
        (root / 'empty-model').mkdir()
        config = dict(model=str(root / 'empty-model'), cache=str(root / 'absent-cache'),
            out=str(root / 'out'), physical_devices=[0, 7], versions=['0.23.0'])
        save(root / 'evaluation.json', config)
        return original, native, companion, config

    def test_public_cli_reports_actual_prefix_counterexample_without_sdk_or_model(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, native, companion, config = self.prefix_fixture(root)
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'run_all.py'), '--official-baselines',
                '--config', str(root / 'evaluation.json'), '--complete-ft', str(original),
                '--source-version-seal-sha256', hashlib.sha256((original / 'checksums.json').read_bytes()).hexdigest(),
                '--reuse-ft-companion', str(companion), '--companion-seal-sha256',
                hashlib.sha256((companion / 'checksums.json').read_bytes()).hexdigest()],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
            self.assertIn('expected=1383, observed=1385', result.stderr)
            out = Path(config['out'])
            record = Evidence(out).read('continuation.json')
            report = Evidence(out).read('necessary-work/audit/producer-prefix-rejection.json')
            self.assertEqual(record['status'], 'failed')
            self.assertFalse(record['sdk_and_model_admission_performed'])
            self.assertIsNone(report['F_interval'])
            self.assertEqual(report['stage'], 'sealed-json-preflight')
            self.assertEqual(report['first_counterexample']['call'], 5)
            self.assertEqual(report['first_counterexample']['first_pass_query_positions'], [1381, 1382, 1383])
            self.assertFalse(Path(config['cache']).exists())
            self.assertEqual(list((root / 'empty-model').iterdir()), [])

    def test_changed_sealed_prefix_cannot_produce_a_counterexample_report(self):
        from official_ft_audit import prefix_preflight
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, native, companion, config = self.prefix_fixture(root)
            (companion / 'client/raw-work/rank-0-drafts.jsonl').write_bytes(b'{}\n')
            with self.assertRaisesRegex(ValueError, 'Unsealed or changed'):
                prefix_preflight(Evidence(native), Evidence(companion))
            self.assertFalse(Path(config['out']).exists())

    def test_no_mismatch_preflight_never_claims_full_qualification(self):
        from official_ft_audit import prefix_preflight
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, native, companion, config = self.prefix_fixture(root, mismatch=False)
            value = prefix_preflight(Evidence(native), Evidence(companion))
            self.assertEqual(value['status'], 'no_prefix_mismatch_found')
            self.assertFalse(value['full_qualification_performed'])
            self.assertTrue(value['requires_full_sdk_and_model_admission'])

    def test_different_same_version_population_is_not_reported_as_this_counterexample(self):
        from official_ft_audit import prefix_preflight
        from official_report import seal
        from official_baselines import save
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, native, companion, config = self.prefix_fixture(root)
            tokens = json.loads((companion / 'token-workload.json').read_bytes())
            tokens['request_token_ids'][0][0] += 1
            save(companion / 'token-workload.json', tokens)
            seal(companion)
            with self.assertRaisesRegex(ValueError, 'same fixed200 token population'):
                prefix_preflight(Evidence(native), Evidence(companion))
            self.assertFalse(Path(config['out']).exists())

    def test_preflight_without_mismatch_still_requires_actual_sdk_admission(self):
        import official_ft
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, native, companion, config = self.prefix_fixture(root, mismatch=False)
            identity = json.loads((ROOT / 'engine/official-baselines/0.23.0.json').read_bytes())
            image = Path(config['cache']) / 'official-sdks' / ('0.23.0-'+identity['manifest_digest'].split(':')[1][:16])
            image.mkdir(parents=True)
            (image / 'installation.json').write_text('{}')
            admission = Mock(side_effect=ValueError('actual SDK admission sentinel'))
            modules = dict(official_image=SimpleNamespace(read_lock=lambda path: identity,
                prepare=Mock(), finish_restoration=Mock()), official_suite=SimpleNamespace(source_bindings=Mock()),
                resource_guard=SimpleNamespace(acquire=Mock(), release=Mock(), verify_mapping=Mock()))
            with patch.dict(sys.modules, modules), patch('platform.system', return_value='Linux'), \
                    patch('platform.machine', return_value='aarch64'), patch.object(official_ft, 'admit_image', admission):
                with self.assertRaisesRegex(ValueError, 'actual SDK admission sentinel'):
                    official_ft.continue_closed(config, original, official_ft.sha(original / 'checksums.json'),
                        companion_source=companion, companion_seal=official_ft.sha(companion / 'checksums.json'))
            admission.assert_called_once()
            self.assertFalse((Path(config['out']) / 'qualified-m01.json').exists())

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




class DraftContextParallelABITests(unittest.TestCase):
    def fixture(self):
        from types import SimpleNamespace as S
        from official_ft_draft import ASCEND_030_FIRST_PASS
        runner = S(dcp_size=1)
        proposer = S(dcp_size=1, runner=runner, vllm_config=S(parallel_config=S(
            prefill_context_parallel_size=1, decode_context_parallel_size=1)))
        methods = {name: dict(sha256=ASCEND_030_FIRST_PASS)
                   for name in ('set_inputs_first_pass', '_propose')}
        return proposer, runner, methods

    def test_missing_030_pcp_uses_actual_config_and_reviewed_source(self):
        from official_ft_draft import context_parallel_contract
        proposer, runner, methods = self.fixture()
        value = context_parallel_contract(proposer, runner, methods)
        self.assertFalse(value['proposer_pcp_field_present'])
        self.assertFalse(value['numeric_default_used'])
        self.assertEqual(value['prefill_context_parallel_size'], 1)
        self.assertFalse(value['managers']['dcp_manager']['attribute_present'])

    def test_no_default_for_missing_or_foreign_parallel_configuration(self):
        from official_ft_draft import context_parallel_contract
        for fault in ('missing-config', 'pcp2', 'dcp2', 'bool', 'runtime2', 'missing-runtime',
                      'proposer2', 'proposer-pcp2', 'foreign-runner', 'pcp-manager', 'dcp-manager', 'source'):
            with self.subTest(fault=fault):
                proposer, runner, methods = self.fixture()
                cfg = proposer.vllm_config.parallel_config
                if fault == 'missing-config': del cfg.prefill_context_parallel_size
                elif fault == 'pcp2': cfg.prefill_context_parallel_size = 2
                elif fault == 'dcp2': cfg.decode_context_parallel_size = 2
                elif fault == 'bool': cfg.prefill_context_parallel_size = True
                elif fault == 'runtime2': runner.dcp_size = 2
                elif fault == 'missing-runtime': del runner.dcp_size
                elif fault == 'proposer2': proposer.dcp_size = 2
                elif fault == 'proposer-pcp2': proposer.pcp_size = 2
                elif fault == 'foreign-runner': proposer.runner = object()
                elif fault == 'pcp-manager': runner.pcp_manager = object()
                elif fault == 'dcp-manager': runner.dcp_manager = object()
                else: methods['set_inputs_first_pass']['sha256'] = '0' * 64
                with self.assertRaises(ValueError): context_parallel_contract(proposer, runner, methods)


class DraftProducerABIFailures(unittest.TestCase):
    def test_opaque_actual_runnable_is_rejected_before_hooks_or_events(self):
        from types import SimpleNamespace as S
        from official_ft_draft import DraftObservation
        state = dict(updated=False, reads=0, clones=[])
        proposer = SafeDraftProposer(state)
        runner = S(drafter=proposer, dcp_size=1, input_batch=S(req_ids=['r'], sampling_metadata=S(all_greedy=True)))
        proposer.runner = runner
        original = lambda **kwargs: None
        proposer._runnable = original
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with self.assertRaisesRegex(ValueError, 'does not wrap'):
                DraftObservation(runner, path, 0, 'd'*32, lambda:0)
            self.assertEqual(list(path.iterdir()), [])
            self.assertIs(proposer._runnable, original)
            self.assertNotIn('_propose', vars(proposer))
        self.assertEqual(state['clones'], [])

    def test_missing_step_or_layer_metadata_is_not_an_empty_observation(self):
        from official_ft_draft import length_snapshot
        for value in (None, [], {}, [{}]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'metadata is absent'):
                length_snapshot(value, 1)


class Draft030Proposer(SafeDraftProposer):
    def __init__(self, state):
        super().__init__(state)
        del self.pcp_size  # Actual 0.30 ABI, not an assumed numeric default.

    def _run_merged_draft(self, num_input_tokens, batch_size, token_indices_to_sample,
                          target_positions, inputs_embeds, multi_steps_attn_metadata,
                          num_tokens, is_prefill=None, sampling_metadata=None):
        self.state['sampling'] = sampling_metadata
        return SafeDraftProposer._runnable(self)

    _runnable = _run_merged_draft

    def _propose(self, num_speculative_tokens, sentinel, **kwargs):
        self.state['width_argument'] = num_speculative_tokens
        self.state['sentinel'] = sentinel
        result = self._runnable(**kwargs)
        self.state['updated'] = True
        return result


class Draft030BoundaryTests(unittest.TestCase):
    def fixture(self, directory):
        from types import SimpleNamespace as S
        import official_ft_draft as draft
        state = dict(updated=False, reads=0, clones=[])
        proposer = Draft030Proposer(state)
        runner = S(drafter=proposer, dcp_size=1, input_batch=S(
            req_ids=['request-1'], sampling_metadata=S(all_greedy=True)))
        proposer.runner = runner
        meta = S(seq_lens=DeferredTensor([3], state), seq_lens_list=[3], actual_seq_lengths_q=[2])
        kwargs = dict(num_input_tokens=2, batch_size=1, token_indices_to_sample=DeferredTensor([1], state),
            target_positions=proposer.positions, inputs_embeds=None, multi_steps_attn_metadata=[{'layer':meta}],
            num_tokens=2, is_prefill=False, sampling_metadata=runner.input_batch.sampling_metadata)
        # The fake executable producer has this test file's SHA; production
        # only accepts the exact independently inspected official 0.30 source.
        with patch.object(draft, 'ASCEND_030_FIRST_PASS', hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
            observer = draft.DraftObservation(runner, directory, 0, 'c' * 32, lambda: 4)
        return proposer, observer, state, kwargs

    def test_real_030_keyword_shape_and_explicit_leading_width_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            proposer, observer, state, kwargs = self.fixture(path)
            sentinel = object()
            self.assertIs(proposer._propose(2, sentinel, **kwargs), proposer.output)
            self.assertIs(state['sentinel'], sentinel)
            self.assertIs(state['sampling'], kwargs['sampling_metadata'])
            self.assertEqual(state['width_argument'], 2)
            self.assertFalse(any(state['clones']))
            self.assertGreater(state['reads'], 0)
            self.assertEqual(observer.stop(), 1)
            record = json.loads((path/'rank-0-drafts.jsonl').read_bytes())
            self.assertEqual(record['host_read_boundary'], 'after_original_propose_post_replay_update')
            self.assertEqual(record['first_pass_input_ids'], [11,12])
            contract = json.loads((path/'rank-0-proposer-source.json').read_bytes())['explicit_parallel_contract']
            self.assertFalse(contract['proposer_pcp_field_present'])
            self.assertFalse(contract['numeric_default_used'])

    def test_missing_or_foreign_actual_keywords_fail_before_device_copy_or_original(self):
        for fault in ('required', 'signature-required', 'unknown'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                proposer, observer, state, kwargs = self.fixture(Path(folder))
                if fault == 'required': del kwargs['multi_steps_attn_metadata']
                elif fault == 'signature-required': del kwargs['target_positions']
                else: kwargs['unknown_draft_abi'] = object()
                with self.assertRaisesRegex(ValueError, 'keyword'):
                    proposer._propose(2, object(), **kwargs)
                self.assertEqual(state['reads'], 0)
                self.assertEqual(state['clones'], [])
                self.assertEqual(proposer.input_ids.values, [11,12])
                self.assertEqual(observer.stop(), 0)

    def test_missing_attention_fields_refuse_without_executing_draft(self):
        for field in ('seq_lens','seq_lens_list','actual_seq_lengths_q'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                proposer, observer, state, kwargs = self.fixture(Path(folder))
                delattr(kwargs['multi_steps_attn_metadata'][0]['layer'],field)
                with self.assertRaisesRegex(ValueError, 'length arguments'):
                    proposer._propose(2, object(), **kwargs)
                self.assertEqual(state['reads'], 0)
                self.assertEqual(proposer.input_ids.values, [11,12])
                observer.stop()


class CompanionFailureCleanupTests(unittest.TestCase):
    def execute(self, output, *, core_fails=False, shutdown_fails=False):
        import types
        import official_ft_entry
        calls = []
        class Core:
            def call_utility(self, name, *args):
                calls.append(name)
                if name == 'hust_m01_core_end' and core_fails: raise RuntimeError('core cleanup failed')
                return dict(completed=True)
            def shutdown(self):
                calls.append('shutdown')
                if shutdown_fails: raise RuntimeError('shutdown cleanup failed')
        class Engine:
            engine_core = Core()
            def collective_rpc(self, name, **kwargs):
                calls.append(name)
                if name == 'hust_ft_begin': raise AttributeError("AscendEagleProposer has no attribute pcp_size")
                if name == 'hust_ft_end': raise ValueError('Cannot close an absent or foreign official companion')
                return []
        engine = Engine()
        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value)+'\n')
        modules = {
            'dataset_contract':types.SimpleNamespace(save=save),
            'baseline_native_entry':types.SimpleNamespace(reset_prefix_cache=lambda *a:None),
            'native_public_client':types.SimpleNamespace(ranks=lambda value:value,
                save_result=lambda *a:None,timed_public_requests=lambda *a:(_ for _ in ()).throw(AssertionError('No requests'))),
            'native_output_geometry':types.SimpleNamespace(geometry=lambda *a:None),
            'native_state_cache':types.SimpleNamespace(wait_checkpoint_admission=lambda *a:None),
            'vllm':types.SimpleNamespace(SamplingParams=lambda **kwargs:object()),
            'vllm.sampling_params':types.SimpleNamespace(RequestOutputKind=types.SimpleNamespace(DELTA='delta')),
            'vllm.v1.engine.llm_engine':types.SimpleNamespace(LLMEngine=types.SimpleNamespace(from_engine_args=lambda args:engine)),
        }
        workload = output/'workload.json'
        workload.write_text(json.dumps(dict(request_token_ids=[[1]*(1412 if i<65 else 1411) for i in range(200)])))
        config=dict(output=str(output),workload=str(workload),sampling={},admission_root=str(output),max_tokens=256)
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(AttributeError,'pcp_size'):
            official_ft_entry.run(config,object())
        return calls,json.loads((output/'companion-receipt.json').read_bytes())

    def test_worker_cleanup_failure_does_not_mask_begin_or_skip_core_and_shutdown(self):
        with tempfile.TemporaryDirectory() as folder:
            calls,receipt=self.execute(Path(folder))
            self.assertLess(calls.index('hust_ft_end'),calls.index('hust_m01_core_end'))
            self.assertEqual(calls[-1],'shutdown')
            self.assertIn('pcp_size',receipt['error'])
            self.assertFalse(receipt['completed'])
            self.assertEqual(receipt['cleanup_errors'][0]['action'],'workers')

    def test_all_cleanup_failures_are_retained_and_primary_error_survives(self):
        with tempfile.TemporaryDirectory() as folder:
            calls,receipt=self.execute(Path(folder),core_fails=True,shutdown_fails=True)
            self.assertEqual(calls[-1],'shutdown')
            self.assertEqual([row['action'] for row in receipt['cleanup_errors']],['workers','core'])
            self.assertIn('shutdown cleanup failed',receipt['shutdown_error'])
            self.assertIn('pcp_size',receipt['error'])
            self.assertFalse(receipt['completed'])


if __name__ == '__main__':
    unittest.main()
