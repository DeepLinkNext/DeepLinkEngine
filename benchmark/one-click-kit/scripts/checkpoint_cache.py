"""Release checkpoint cache and bounded pre-service file cache; never globally flush."""

import json
import os
from pathlib import Path
import stat


GIB = 2**30


class StartupMemoryReserve:
    """Hold real anonymous headroom until idle or the working-set gate needs it.

    Driver pinned allocations may fail before the kernel reclaims otherwise clean
    file cache. Reserve pages before starting a model, then release those pages
    at the healthy, idle service boundary. No pressure allocation runs against a
    live model and no buffer survives into warmup or a measured request.
    """

    def __init__(self, path, sample, check):
        self.path, self.sample, self.check = path, sample, check
        self.buffers = []
        self.record = dict(schema="one-click-startup-memory-reserve/v1",
            bytes=16 * GIB, before=sample(), model_started_at_allocation=False,
            global_drop_caches=False, memory_limit_changed=False, released=False)
        try:
            if self.record["before"]["limit"] - self.record["before"]["current"] < 36 * GIB:
                raise RuntimeError("Less than36GiB raw memory immediately before startup reserve allocation; model not started")
            # Startup admission has already verified 36GiB raw headroom. This
            # sixteen-GiB reserve is included in the unchanged working-set guard.
            for _ in range(256):
                self.buffers.append(allocate_private_chunk(64 * 2**20))
                check()
            self.record["held"] = sample()
            self.save()
        except BaseException:
            self.close("allocation-failed")
            raise

    def save(self):
        self.path.write_text(json.dumps(self.record, indent=2) + "\n")

    def release_chunks(self, count, reason):
        if self.record['released']:
            return
        count=min(count,len(self.buffers))
        for _ in range(count):
            self.buffers.pop().close()
        self.record.update(released=not self.buffers, release_reason=reason,
            remaining_bytes=len(self.buffers)*64*2**20, after=self.sample())
        self.record.setdefault('releases',[]).append(dict(reason=reason,
            released_bytes=count*64*2**20,remaining_bytes=self.record['remaining_bytes'],
            after=self.record['after']))
        self.save()

    def close(self, reason):
        self.release_chunks(len(self.buffers),reason)

    def release_for_working_set(self):
        # Return only8GiB at a time to the unchanged SDK guard. Retain the other
        # half for actual pinned State allocations at the healthy boundary.
        # No new pages are allocated while a model exists.
        self.release_chunks(128,'release-owned-buffer-for-original-working-set-reserve')

    def before_warmup(self):
        self.close('healthy-idle-service-before-first-warmup')
        self.check()
        value=self.sample()
        self.record['warmup_admission']=value
        self.record['required_warmup_raw_bytes']=12*GIB
        self.record['warmup_raw_basis']='two frozen4GiB/rank State budgets plus original4GiB raw reserve'
        self.save()
        if value['limit']-value['current'] < 12*GIB:
            raise RuntimeError('Less than12GiB raw memory before warmup after startup reserve release; no formal request submitted')


def reclaim_plan(value, required):
    """Bound temporary anonymous memory by conservatively reclaimable cache."""
    current, limit = value.get("current"), value.get("limit")
    fields = value.get("stat", {})
    names = ("file", "inactive_file", "shmem", "file_dirty", "file_writeback", "file_mapped", "unevictable")
    if (type(current) is not int or type(limit) is not int or not 0 <= current <= limit
            or any(type(fields.get(n)) is not int for n in names)
            or fields.get("file", 0) > current
            or any(not 0 <= fields[n] <= fields["file"] for n in names)):
        return dict(eligible=False, reason="Complete valid finite cache counters required")
    # Repeated SDK/checkpoint hashing promotes unused cache to the active LRU.
    # This is a pre-service reclamation bound, not the model working-set gate:
    # clean active pages are reclaimable too. Exclude every mapped/locked page,
    # shared memory and all dirty/writeback bytes, even where those overlap.
    clean = max(0, fields["file"] - sum(fields[n] for n in (
        "shmem", "file_dirty", "file_writeback", "file_mapped", "unevictable")))
    # Existing raw free bytes are available too. Bound the combined raw+clean
    # capacity, preserving the original4GiB safety; subtracting it once again
    # from file cache discards usable raw memory before the model even exists.
    working = current - min(current, clean)
    budget = min(40 * GIB, limit - working - 4 * GIB - 64 * 2**20)
    budget = max(0, budget // (64 * 2**20) * (64 * 2**20))
    return dict(eligible=limit - current < required and budget >= required + GIB,
                reason="Pre-service allocation bounded by clean unmapped active/inactive file cache; original model gates rechecked after every temporary page is released",
                clean_unmapped_file_cache_bytes=clean, non_file_estimate=working, inactive_file_observed_bytes=fields["inactive_file"], temporary_bytes=budget)


def allocate_private_chunk(size):
    import mmap
    chunk = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try:
        view = memoryview(chunk)
        try:
            view[::4096] = b"\1" * ((size + 4095) // 4096)
        finally:
            view.release()
    except BaseException:
        chunk.close()
        raise
    return chunk


def allocation_status():
    # Full memory.stat reads can flush per-CPU counters and are expensive on
    # large hosts. During ordinary temporary allocation only observe raw usage
    # and OOM events; full counter snapshots bracket the complete operation.
    root = Path("/sys/fs/cgroup")
    return dict(current=int((root / "memory.current").read_text()),
                events={k: int(v) for k, v in (line.split() for line in (root / "memory.events").read_text().splitlines())})


def reclaim_before_service(sample, required, *, monitor=None):
    monitor = monitor or sample
    before = sample()
    plan = reclaim_plan(before, required)
    attempts = [before]
    # The kernel may briefly expose a writeback-counter underflow. Never credit
    # that sample as clean cache; use the same bounded observation retries as
    # the original memory guard, and decline if counters remain invalid.
    for _ in range(2):
        if plan.get("reason") != "Complete valid finite cache counters required":
            break
        import time
        time.sleep(0.02)
        before = sample()
        attempts.append(before)
        plan = reclaim_plan(before, required)
    result = dict(sampling_attempts=attempts, status="declined", before=before, plan=plan,
                  global_drop_caches=False, memory_limit_changed=False,
                  model_service_started=False, temporary_memory_released=True)
    if not plan["eligible"]:
        result["after"] = before
        return result
    buffers = []
    try:
        for _ in range(plan["temporary_bytes"] // (64 * 2**20)):
            buffers.append(allocate_private_chunk(64 * 2**20))
            value = monitor()
            result["last_allocation_sample"] = value
            if value["current"] > result.get("peak", {"current": -1})["current"]:
                result["peak"] = value
            if any(value.get("events", {}).get(n, 0) > before.get("events", {}).get(n, 0)
                   for n in ("oom", "oom_kill", "oom_group_kill")):
                raise RuntimeError("New cgroup OOM event during pre-service cache recovery")
        result.update(status="completed")
    except Exception as error:
        result.update(status="failed", error=type(error).__name__ + ": " + str(error))
    finally:
        for chunk in buffers:
            chunk.close()
    result["after"] = sample()
    if any(result["after"].get("events", {}).get(n, 0) > before.get("events", {}).get(n, 0)
           for n in ("oom", "oom_kill", "oom_group_kill")):
        result.update(status="failed", error="New cgroup OOM event during pre-service cache recovery")
    return result


def release_loaded_checkpoint_cache(model, weights, out, sample, *, before_service=False):
    model = model.resolve()
    before = sample()
    files = []
    for name, expected in weights.items():
        path = (model / name).resolve()
        if (
            not path.is_relative_to(model)
            or Path(name).name != name
            or not name.endswith(".safetensors")
        ):
            raise ValueError("Cache hint must stay within the admitted checkpoint")
        with path.open("rb") as stream:
            identity = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(identity.st_mode)
                or identity.st_size != expected["bytes"]
            ):
                raise ValueError("Checkpoint file identity changed: " + name)
            for key, actual in (
                ("mtime_ns", identity.st_mtime_ns),
                ("inode", identity.st_ino),
            ):
                if key in expected and expected[key] != actual:
                    raise ValueError("Checkpoint file identity changed: " + name)
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            after_identity = os.fstat(stream.fileno())
            if (identity.st_ino, identity.st_size, identity.st_mtime_ns) != (
                after_identity.st_ino,
                after_identity.st_size,
                after_identity.st_mtime_ns,
            ):
                raise ValueError("Checkpoint changed during cache admission: " + name)
            files.append(
                {
                    "name": name,
                    "bytes": identity.st_size,
                    "inode": identity.st_ino,
                    "mtime_ns": identity.st_mtime_ns,
                    "admitted_sha256": expected["sha256"],
                }
            )
    after = sample()
    record = {
        "operation": "POSIX_FADV_DONTNEED on read-only admitted checkpoint descriptors",
        "file_content_written": False,
        "global_drop_caches": False,
        "files": files,
        "before": before,
        "after": after,
        "required_raw_headroom_bytes": 36 * 2**30 if before_service else 0,
        "reserve_policy": "36GiB raw before model allocation; original 4GiB working-set/OOM guard after allocation",
        "reason": "Before model creation clear unused clean cache and reserve16GiB controller-owned startup pages; all temporary reclaim pages released before model spawn; after allocation clean inactive file cache is governed by the unchanged SDK working-set/OOM check",
    }
    if before_service and after["limit"] - after["current"] < record["required_raw_headroom_bytes"]:
        # SDK hashing can refill cache even after installation archives were
        # released. Overlay filesystems may ignore DONTNEED. Use ordinary bounded
        # private anonymous pages to let this cgroup reclaim its clean file cache,
        # then release every page before a model process can exist. Never run this
        # fallback at the after-load call sites or during native/request timing.
        record["bounded_pre_service_reclaim"] = reclaim_before_service(sample, record["required_raw_headroom_bytes"], monitor=allocation_status)
        after = record["after"] = record["bounded_pre_service_reclaim"]["after"]
    out.write_text(json.dumps(record, indent=2) + "\n")
    if record.get("bounded_pre_service_reclaim", {}).get("status") == "failed":
        raise RuntimeError("Pre-service cache recovery failed; original memory/OOM gates remain unchanged")
    if before_service and after["limit"] - after["current"] < record["required_raw_headroom_bytes"]:
        raise RuntimeError(
            "Insufficient raw cgroup headroom after startup cache admission; budgets remain unchanged"
        )
    return record
