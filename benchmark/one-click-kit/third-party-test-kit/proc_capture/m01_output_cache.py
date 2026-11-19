"""Release only this observation's recorded file cache, without crediting it.

These read-only descriptors never alter evidence bytes or globally flush other
workloads. fdatasync makes recorded pages eligible for a per-file DONTNEED hint.
The parent still checks actual cgroup headroom with unchanged reserve gates.
This helper is exclusive to instrumented observation, outside clean native T.
"""

import os
from pathlib import Path
import stat
import time


def release(path, directory):
    path, directory = Path(path), Path(directory).resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(directory):
        raise ValueError("Observation cache hint escaped its owned output directory")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode):
            raise ValueError("Observation cache hints require regular owned files")
        os.fdatasync(descriptor)
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        after = os.fstat(descriptor)
        if (identity.st_dev, identity.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("Observation output identity changed during cache hint")
        return dict(path=str(path.relative_to(directory)), bytes_requested=identity.st_size,
                    inode=identity.st_ino, file_content_written=False,
                    global_drop_caches=False, reclaimed_bytes_claimed=False)
    finally:
        os.close(descriptor)


class ProfilerCache:
    """Bound each guard tick; revisit live writers and incrementally scan new files."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.previous = {}
        self.pending = None

    def paths(self):
        for rank in range(2):
            root = self.directory / f"rank-{rank}-raw-profiler"
            if not root.exists():
                continue
            if root.is_symlink():
                raise ValueError("Unexpected profiler output symlink")
            for parent, dirs, files in os.walk(root, followlinks=False):
                for name in dirs:
                    if (Path(parent) / name).is_symlink():
                        raise ValueError("Unexpected profiler subtree symlink")
                for name in files:
                    yield Path(parent) / name

    def active(self, owners):
        for pid in owners:
            folder = Path(f"/proc/{pid}/fd")
            try:
                descriptors = list(folder.iterdir())
            except FileNotFoundError:
                continue
            for descriptor in descriptors:
                try:
                    path = descriptor.readlink()
                except FileNotFoundError:
                    continue
                if path.is_absolute() and path.is_relative_to(self.directory):
                    relative = path.relative_to(self.directory)
                    if relative.parts and relative.parts[0] in ("rank-0-raw-profiler", "rank-1-raw-profiler"):
                        yield path

    def sweep(self, owners=(), *, limit=64, seconds=1.0):
        if type(limit) is not int or limit < 1 or seconds <= 0:
            raise ValueError("A bounded cache maintenance tick is required")
        results, visited = [], set()
        deadline = time.monotonic() + seconds
        active = iter(self.active(owners))
        if self.pending is None:
            self.pending = iter(self.paths())
        for _ in range(limit):
            if time.monotonic() >= deadline:
                break
            try:
                path = next(active)
            except StopIteration:
                try:
                    path = next(self.pending)
                except StopIteration:
                    self.pending = None
                    break
                # Closed profiler packets are immutable. Growing live writers
                # are revisited from owned /proc descriptors above.
                if str(path) in self.previous:
                    continue
            if str(path) in visited:
                continue
            visited.add(str(path))
            identity = path.lstat()
            if not stat.S_ISREG(identity.st_mode):
                raise ValueError("Unexpected profiler output type")
            key = (identity.st_dev, identity.st_ino, identity.st_size, identity.st_mtime_ns)
            if self.previous.get(str(path)) != key and identity.st_size:
                results.append(release(path, self.directory))
                self.previous[str(path)] = key
        return results
