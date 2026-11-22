"""Read host memory on cgroup v1 while preserving the frozen v2 guard."""
import re
from pathlib import Path
import sys
import time

from official_baselines import save


def counters(path):
    return {name: int(value) for name, value in
            (line.split() for line in path.read_text().splitlines())}


def memory_scopes(proc):
    def unescape(value):
        return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), value)

    group = next((Path(path) for _, controllers, path in
                  (line.split(':', 2) for line in (proc / 'self/cgroup').read_text().splitlines())
                  if 'memory' in controllers.split(',')), None)
    mounts = []
    for line in (proc / 'self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        kind, _, options = right.split()
        fields = left.split()
        root = Path(unescape(fields[3]))
        if (kind == 'cgroup' and 'memory' in options.split(',')
                and group is not None and group.is_relative_to(root)):
            mounts.append((root, Path(unescape(fields[4]))))
    if not mounts:
        raise ValueError('Cannot locate the active cgroup v1 memory controller')
    root, mount = max(mounts, key=lambda item: len(item[0].parts))
    leaf = mount / group.relative_to(root)
    return (leaf, *(parent for parent in leaf.parents if parent.is_relative_to(mount)
                    and (parent / 'memory.use_hierarchy').read_text().strip() == '1'))


class CgroupV1Memory:
    def __init__(self, original, proc=Path('/proc')):
        self.original = original
        self.proc = proc
        self.scopes = memory_scopes(proc)

    def sample(self):
        meminfo = {line.split(':', 1)[0]: int(line.split()[1]) * 1024
                   for line in (self.proc / 'meminfo').read_text().splitlines()
                   if line.startswith(('MemTotal:', 'MemFree:', 'MemAvailable:'))}
        total, free = meminfo['MemTotal'], meminfo['MemFree']
        host_oom = counters(self.proc / 'vmstat').get('oom_kill', 0)
        samples = []
        for scope in self.scopes:
            raw = counters(scope / 'memory.stat')
            names = dict(file='cache', inactive_file='inactive_file', shmem='shmem',
                         file_dirty='dirty', file_writeback='writeback')
            stat = {name: raw['total_' + key] if 'total_' + key in raw else raw[key]
                    for name, key in names.items()}
            current = int((scope / 'memory.usage_in_bytes').read_text())
            limit = min(int((scope / 'memory.limit_in_bytes').read_text()),
                        raw['hierarchical_memory_limit'], total)
            oom = counters(scope / 'memory.oom_control')
            valid = all(0 <= value <= stat['file'] for value in stat.values())
            # The old guard computes exactly the same credit on valid fields.
            working = self.original.estimate(current, stat) if valid else current
            samples.append(dict(path=str(scope), limit=limit, current=current, stat=stat,
                                working_estimate=working, oom=oom, raw_stat=raw,
                                limit_hits=int((scope / 'memory.failcnt').read_text())))
        tightest = min(samples, key=lambda sample: sample['limit'] - sample['current'])
        host_limited = free <= tightest['limit'] - tightest['current']
        current = total - free if host_limited else tightest['current']
        limit = total if host_limited else tightest['limit']
        return dict(wall=time.time(), monotonic=time.monotonic(), cgroup_version=1,
                    current=current, limit=limit, stat=tightest['stat'],
                    working_estimate=current if host_limited else tightest['working_estimate'],
                    events=dict(oom=max(row['oom']['under_oom'] for row in samples),
                                oom_kill=host_oom + sum(row['oom']['oom_kill'] for row in samples)),
                    host_memory=meminfo, host_oom_kill=host_oom, scopes=samples,
                    host_raw_headroom_selected=host_limited, pressure=None)

    def check(self, value, initial):
        if value['events']['oom']:
            raise RuntimeError('Active cgroup v1 out-of-memory condition')
        self.original.check(value, initial)
        reserve = 4 * 2**30
        if (value['host_memory']['MemFree'] <= reserve
                or any(row['limit'] - row['working_estimate'] <= reserve for row in value['scopes'])):
            raise RuntimeError('4GiB conservative working-set memory reserve reached')


def memory_guard(original):
    if sys.platform != 'linux' or Path('/sys/fs/cgroup/memory.stat').is_file():
        return original
    return CgroupV1Memory(original)


def admit_startup(guard, model, weights, root):
    from run_b1_r4 import admit_startup_memory, check_memory

    if not isinstance(guard, CgroupV1Memory):
        return admit_startup_memory(guard, model, weights, root)
    from checkpoint_cache import release_loaded_checkpoint_cache

    initial = guard.sample()
    save(root / 'memory-initial.json', initial)
    receipt = root / 'checkpoint-cache-startup.json'
    # Use the existing checkpoint-only hint. Do not run its cgroup v2 pressure
    # allocator on a shared cgroup v1 host.
    record = release_loaded_checkpoint_cache(model, weights, receipt, guard.sample)
    required = 36 * 2**30
    record.update(required_raw_headroom_bytes=required,
                  reason='Read-only checkpoint hint and 36GiB raw cgroup/host headroom gate')
    save(receipt, record)
    if record['after']['limit'] - record['after']['current'] < required:
        raise RuntimeError('Insufficient raw cgroup/host headroom after checkpoint cache admission')
    check_memory(guard, initial, root)
    return initial
