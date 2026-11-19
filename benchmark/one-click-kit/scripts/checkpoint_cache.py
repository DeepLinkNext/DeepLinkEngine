"""Release clean cache of this checkpoint only; never write or globally flush."""

import json
import os
from pathlib import Path
import stat


def release_loaded_checkpoint_cache(model, weights, out, sample):
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
        "required_raw_headroom_bytes": 12 * 2**30,
        "reason": "8GiB total host State budget plus 4GiB raw cgroup allocation reserve; cached bytes are not guaranteed reclaimable by the driver",
    }
    out.write_text(json.dumps(record, indent=2) + "\n")
    if after["limit"] - after["current"] < record["required_raw_headroom_bytes"]:
        raise RuntimeError(
            "Insufficient raw cgroup headroom after checkpoint-only cache hint; budgets remain unchanged"
        )
    return record
