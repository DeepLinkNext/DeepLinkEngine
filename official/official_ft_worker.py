"""Named primitives for the separate official arithmetic companion."""
import hashlib
import inspect
import os
from pathlib import Path

from official_worker import OfficialWorker


class OfficialCompanionWorker(OfficialWorker):
    def hust_ft_identity(self):
        if os.environ.get('VLLMHUST_M01_PROC_CAPTURE') != '1':
            raise ValueError('Official arithmetic companion is not enabled')
        from runtime_identity import identity
        config = self.vllm_config
        result = identity(self)
        runner = self.model_runner
        sources = {}
        for item in (type(runner), type(runner.drafter), runner.drafter._run_merged_draft,
                     runner.drafter.set_inputs_first_pass, runner._build_attention_metadata):
            method = inspect.unwrap(item)
            path = Path(inspect.getfile(method)).resolve()
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        def select(owner, fields):
            return {name: getattr(owner, name) for name in fields}
        result.update(timing_eligible=False, model_sources=sources,
            resolved_scheduler=select(config.scheduler_config,
                ('async_scheduling', 'max_num_seqs', 'max_num_batched_tokens')),
            resolved_parallel=select(config.parallel_config, ('tensor_parallel_size',
                'pipeline_parallel_size', 'data_parallel_size', 'prefill_context_parallel_size',
                'decode_context_parallel_size', 'enable_expert_parallel')),
            resolved_speculative=select(config.speculative_config, ('method', 'num_speculative_tokens',
                'parallel_drafting')))
        return result

    def hust_ft_begin(self, directory: str, window: str):
        from m01_proc_plugin import require_enabled, window_id
        require_enabled()
        window_id(window)
        from official_ft_observer import start
        return start(self, directory, window)

    def hust_ft_end(self, window: str):
        from m01_proc_plugin import require_enabled, window_id
        require_enabled()
        window_id(window)
        from runtime_identity import synchronize
        synchronize(self)
        from official_ft_observer import stop
        return stop(self, window)
