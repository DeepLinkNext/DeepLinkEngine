"""Same serving corrections as r6, without diagnostic journals or route ops."""
import os
from hust_prefill_microflow.school028.worker import Worker as MicroflowWorker
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from dual_clock_boundary import DualClockWorkerBoundary
from m01_progress_source_snapshot import ProgressSnapshotRunnerMixin, ProgressSourceSnapshots
from host_prefill_aggregate import PrefillAggregate


class CorrectedRunner(ProgressSnapshotRunnerMixin, NPUModelRunner):
    def initialize_kv_cache(self, *args, **kwargs):
        # Resolve the late-installed State initializer at invocation, as in K04/K03.
        import hashlib
        import inspect
        from pathlib import Path
        from types import SimpleNamespace
        import torch
        from betterscale.models.qwen35 import position_reuse
        from guarded_source_lifecycle import initialize
        from owned_optimistic_operand import OwnedOptimisticOperand
        initializer = super().initialize_kv_cache
        implementation = initializer.__func__
        source = Path(inspect.getsourcefile(implementation)).resolve()
        expected = 'b8918ed6955de98c034444ef8e2bb14e2244eb75f1612b9980fe91a2cd00d564'
        if (not str(source).endswith('/betterscale/models/qwen35/state_backend.py')
                or hashlib.sha256(source.read_bytes()).hexdigest() != expected
                or implementation.__qualname__ != 'install.<locals>.initialize'):
            raise ValueError('r7 must preserve the frozen late-installed native State initializer')
        result = initializer(*args, **kwargs)
        closure = inspect.getclosurevars(position_reuse.prepare).nonlocals
        owned = closure.get('owned')
        if owned is None or closure.get('runner') is not self:
            raise ValueError('The native position wrapper is not bound to the initializing runner')
        worker = SimpleNamespace(model_runner=self, _owned_optimistic_operand=owned)
        receipt = initialize(worker, torch, ProgressSourceSnapshots, OwnedOptimisticOperand,
            OwnedOptimisticOperand.refresh, ProgressSourceSnapshots.run)
        receipt['native_initialization'] = dict(path=str(source), sha256=expected,
            qualname=implementation.__qualname__, late_bound=True, SDK_files_modified=True)
        self._hust_source_guard_lifecycle = receipt
        return result

    def _build_attention_metadata(self, *args, **kwargs):
        result = super()._build_attention_metadata(*args, **kwargs)
        aggregate = getattr(self, '_m01_host_prefill', None)
        if aggregate is not None and aggregate.active:
            bound = self._m01_metadata_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if not bound.arguments.get('for_cudagraph_capture', False):
                count = bound.arguments['num_reqs']
                aggregate.observe(result[0], list(self.input_batch.req_ids[:count]))
        return result


class CorrectedWorker(DualClockWorkerBoundary, MicroflowWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        if type(self.model_runner) is not NPUModelRunner:
            raise ValueError('Expected qualified unmodified runner class')
        parallel = self.vllm_config.parallel_config
        if parallel.pipeline_parallel_size != 1 or parallel.tensor_parallel_size != 2:
            raise ValueError('TP2/PP1 required')
        import torch
        self.model_runner.__class__ = CorrectedRunner
        self.model_runner._m01_progress_source_snapshots = ProgressSourceSnapshots(
            torch, self.model_runner.input_batch)
        self.model_runner._m01_progress_snapshot_calls = 0
        if os.environ.get('HUST_M01_HOST_PREFILL_AGGREGATE') != '1':
            raise ValueError('Explicit host work-counting activation required')
        from pinned_metadata_signature import load_signature
        self.model_runner._m01_metadata_signature = load_signature()
        self.model_runner._m01_host_prefill = PrefillAggregate()
        from owned_optimistic_operand import install as install_operand
        from causal_draft_lengths import install as install_causal
        install_operand(self)
        install_causal(self)
        return result

    def hust_m01_sync_boundary(self, phase):
        result = super().hust_m01_sync_boundary(phase)
        aggregate = self.model_runner._m01_host_prefill
        if phase == 'begin':
            aggregate.begin()
            result['host_prefill_aggregate'] = aggregate.snapshot()
        else:
            result['host_prefill_aggregate'] = aggregate.end()
        from owned_optimistic_operand import snapshot as operand_snapshot
        from causal_draft_lengths import snapshot as causal_snapshot
        guard = self.model_runner._m01_progress_source_snapshots
        result['serving_corrections'] = dict(
            variant='owned-optimistic-causal-draft/v1',
            progress_source=dict(enabled=True, failed=guard.failed, active=guard.active,
                calls=self.model_runner._m01_progress_snapshot_calls),
            owned_operand=operand_snapshot(self), causal_lengths=causal_snapshot(self),
            diagnostic_journals=False, route_recording=False,
            native_spec_metrics=self.vllm_config.observability_config.per_request_spec_decode_metrics,
            host_prefill_counting=True,
            correction_cost_included=True)
        return result


def register():
    if os.environ.get('HUST_M01_CORRECTED_SERVING') != '1':
        raise ValueError('Corrected serving must be explicit')
    import betterscale.qwen35_worker as worker
    if worker.Worker is CorrectedWorker:
        return
    if worker.Worker is not MicroflowWorker:
        raise ValueError('Expected Manager enabled Microflow worker')
    worker.Worker = CorrectedWorker
