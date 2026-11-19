"""Use the original pinned018/CANN8.5.1 image SDK from a relocated rootfs."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

from restore_b0_image import MANIFEST


def environment(root, variant="text"):
    root = Path(root).resolve()
    rootfs = root / "rootfs"
    config = json.loads((root / "image-config.json").read_bytes())

    def relocate(value):
        result = []
        for item in value.split(":"):
            if not item:
                continue
            if item.startswith("/usr/local/") and not item.startswith(
                "/usr/local/Ascend/driver/"
            ):
                item = str(rootfs) + item
            result.append(item)
        return ":".join(result)

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("ASCEND", "ATB", "BETTERSCALE", "HUST", "MTP", "KVMAT", "VLLM", "HCCL")
        )
        and key not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD")
    }
    for item in config["config"]["Env"]:
        key, value = item.split("=", 1)
        env[key] = (
            relocate(value)
            if key in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "CMAKE_PREFIX_PATH")
            or value.startswith("/usr/local/")
            else value
        )
    p = rootfs / "usr/local/python3.11.14"
    sources = [rootfs / "vllm-workspace/vllm", rootfs / "vllm-workspace/vllm-ascend"]
    if variant == "text":
        sources = [
            root / "text-only-overlay/vllm",
            root / "text-only-overlay/vllm-ascend",
        ]
    elif variant != "multimodal":
        raise ValueError("Unknown baseline image variant")
    env["PYTHONPATH"] = (
        ":".join(map(str, sources + [p / "lib/python3.11/site-packages"]))
        + ":"
        + env.get("PYTHONPATH", "")
    )
    env["LD_LIBRARY_PATH"] = (
        env.get("LD_LIBRARY_PATH", "")
        + ":"
        + ":".join(
            map(
                str,
                [
                    p / "lib/python3.11/site-packages/torch/lib",
                    p / "lib/python3.11/site-packages/torch_npu/lib",
                    root / "system-libraries",
                ],
            )
        )
    )
    # npu-smi belongs to the host driver, not the relocated image SDK. Do not
    # append the host Python/bin directories to an otherwise isolated SDK PATH.
    env["PATH"] = (
        env.get("PATH", "") + ":/usr/local/sbin:/usr/local/Ascend/driver/tools"
    )
    env.update(
        PYTHONHOME=str(p),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        VLLMHUST_NO_SITE="1",
        VLLM_PLUGINS="ascend,ascend_model_loader,ascend_kv_connector",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="4",
    )
    for key in (
        "VLLMHUST_ENGINE_LOCKS",
        "VLLM_HUST_TEST_OUTPUT",
        "VLLM_HUST_PHYSICAL_DEVICES",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def python(root):
    return Path(root).resolve() / "rootfs/usr/local/python3.11.14/bin/python3"


def probe(root, variant):
    code = """
import importlib.metadata as m,importlib.util as u,json,sys
versions={name:m.version(name) for name in ('vllm','vllm-ascend','torch','torch-npu','transformers','triton-ascend')}
origins={name:u.find_spec(name).origin for name in ('vllm','vllm_ascend','torch','torch_npu')}
print(json.dumps({'versions':versions,'origins':origins,'prefix':sys.prefix,'python':sys.executable,'no_site':sys.flags.no_site}))
"""
    result = subprocess.run(
        [str(python(root)), "-S", "-c", code],
        env=environment(root, variant),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise ValueError(
            "Pinned018 image metadata probe failed: " + result.stderr[-2000:]
        )
    return json.loads(result.stdout)


def file_map(root):
    root = Path(root).resolve()
    result = {}
    for directory in (
        "rootfs/usr/local",
        "rootfs/vllm-workspace/vllm",
        "rootfs/vllm-workspace/vllm-ascend",
        "rootfs/etc/Ascend",
        "text-only-overlay",
        "system-libraries",
    ):
        for path in sorted((root / directory).rglob("*")):
            name = str(path.relative_to(root))
            if path.is_symlink():
                if not path.resolve().is_relative_to(root):
                    raise ValueError("SDK link escaped recovered image")
                result[name] = {"symlink": os.readlink(path)}
            elif path.is_file():
                with path.open("rb") as stream:
                    result[name] = {
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                    }
                    # Admission reads the13GiB SDK before the separate checkpoint
                    # scan. Do not retain that clean cache inside the64GiB cgroup.
                    # The unchanged raw-headroom guard still measures actual release.
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return result


def map_digest(files):
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_image(root):
    root = Path(root).resolve()
    raw = (root / "image-manifest.json").read_bytes()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != MANIFEST:
        raise ValueError("Original018 image manifest changed")
    manifest = json.loads(raw)
    config = (root / "image-config.json").read_bytes()
    if "sha256:" + hashlib.sha256(config).hexdigest() != manifest["config"]["digest"]:
        raise ValueError("Original018 image configuration changed")
    return manifest
