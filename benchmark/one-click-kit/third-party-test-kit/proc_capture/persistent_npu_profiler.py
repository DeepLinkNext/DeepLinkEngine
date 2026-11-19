"""Keep raw NPU profiles in daemon workers; all parsing belongs to the parent."""

import hashlib
import json
import os
from pathlib import Path
import re


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
        # Sealing a long profile must not refill tens of GiB of clean cache
        # while the model is still resident. Evidence bytes remain unchanged.
        os.fdatasync(stream.fileno())
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


class PersistentNPUProfiler:
    def __init__(self, torch_npu, directory, *, rank, window):
        if type(rank) is not int or rank not in (0, 1):
            raise ValueError('Expected an explicit TP2 rank')
        if type(window) is not str or not re.fullmatch(r'[a-f0-9]{32}', window):
            raise ValueError('Bind the profile to the explicit capture window')
        path = Path(directory)
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            raise ValueError('Use the existing owned absolute capture directory')
        self.output, self.rank, self.window = path.resolve(), rank, window
        self.raw = self.output / f'rank-{rank}-raw-profiler'
        self.raw.mkdir(exist_ok=False)
        self.start_attempted = self.started = False
        self.closed = False
        self.receipt = None
        factory = torch_npu.profiler
        # These Level1/PipeUtilization options match the frozen Ascend profiler
        # wrapper. The callback explicitly disables analysis in this daemon.
        experimental = factory._ExperimentalConfig(
            export_type=factory.ExportType.Text,
            profiler_level=factory.ProfilerLevel.Level1,
            msprof_tx=False, aic_metrics=factory.AiCMetrics.PipeUtilization,
            l2_cache=False, op_attr=False, data_simplification=True,
            record_op_args=False, gc_detect_threshold=None)
        handler = factory.tensorboard_trace_handler(str(self.raw),
            worker_name=f'hust_m01_rank{rank}_{window}', analyse_flag=False, async_mode=False)
        self.profile = factory.profile(
            activities=[factory.ProfilerActivity.CPU, factory.ProfilerActivity.NPU],
            record_shapes=True, with_stack=False, profile_memory=False,
            experimental_config=experimental,
            schedule=factory.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=handler)
        self.profile.add_metadata_json('hust_m01_capture', json.dumps(dict(
            rank=rank, window=window, worker_pid=os.getpid(),
            arithmetic_certified=False, timing_eligible=False)))

    def start(self):
        if self.start_attempted or self.closed:
            raise ValueError('Create a fresh owner after any attempted profiler start')
        self.start_attempted = True
        self.profile.start()
        self.started = True

    def stop(self):
        if self.closed:
            return dict(self.receipt)
        if not self.started:
            raise ValueError('No successfully started profiler belongs to this owner')
        self.profile.stop()
        candidates = list(self.raw.glob('*_ascend_pt'))
        if len(candidates) != 1 or candidates[0].is_symlink():
            raise ValueError('Missing or ambiguous persistent raw profiler directory')
        raw = candidates[0]
        info = json.loads((raw / f'profiler_info_{self.rank}.json').read_bytes())
        if info['rank_id'] != self.rank:
            raise ValueError('Profiler metadata rank differs')
        metadata = json.loads((raw / 'profiler_metadata.json').read_bytes())
        window = metadata.get('hust_m01_capture')
        if isinstance(window, str):
            window = json.loads(window)
        if not isinstance(window, dict) or window.get('window') != self.window or window.get('rank') != self.rank:
            raise ValueError('Persistent profile is not bound to this window/rank')
        files = {}
        for path in sorted(raw.rglob('*')):
            if path.is_symlink():
                raise ValueError('Symlink in persistent profiler raw data')
            if path.is_file():
                files[path.relative_to(raw).as_posix()] = dict(bytes=path.stat().st_size, sha256=digest(path))
        if not any(name.startswith('FRAMEWORK/') for name in files) or not any(name.startswith('PROF_') for name in files):
            raise ValueError('Missing framework or CANN raw data')
        result = dict(rank=self.rank, window=self.window, worker_pid=os.getpid(),
            raw_directory=str(raw), files=files, collection_completed=True,
            export_deferred_to_standalone_process=True, trace_validated=False,
            arithmetic_certified=False, efficiency_percent=None,
            timing_eligible_for_performance_comparison=False)
        target = self.output / f'rank-{self.rank}-profiler-raw-receipt.json'
        with target.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        self.receipt, self.closed = result, True
        return dict(result)
