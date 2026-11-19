"""Release checkpoint cache and bounded pre-service file cache; never globally flush."""

import json
import os
from pathlib import Path
import stat


GIB = 2**30


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
    working = current - min(current, clean)
    budget = min(16 * GIB, clean - 4 * GIB, limit - working - 4 * GIB - 64 * 2**20)
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
        "required_raw_headroom_bytes": 12 * 2**30 if before_service else 0,
        "reserve_policy": "12GiB raw before model allocation; original 4GiB working-set/OOM guard after allocation",
        "reason": "Before model creation reserve8GiB host State plus4GiB raw; after allocation clean inactive file cache is governed by the unchanged SDK working-set/OOM check",
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
