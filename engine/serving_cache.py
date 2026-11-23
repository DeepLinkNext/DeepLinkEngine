"""Select and relocate trusted, persistent compilation caches for serving."""

import ast
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sysconfig
import tempfile
import uuid


POOL = ".deeplink-cache-v1"
RECORD = "identity.json"
LOCK = "writer.lock"
VLLM = Path("on/vllm/torch_compile_cache")
TRITON = Path("on/triton")
MODEL_INFOS = Path("on/vllm/modelinfos")
MODEL_INFO_NAME = re.compile(r"vllm-model_executor-models-([a-zA-Z_][a-zA-Z_0-9]*)-([a-zA-Z_][a-zA-Z_0-9]*)\.json")
ENV_PREFIXES = ("VLLM_", "VLLMHUST_", "TORCH_", "PYTORCH_", "TORCHINDUCTOR_", "TRITON_",
                "ASCEND_", "CANN_", "ATB_", "HCCL_", "BETTERSCALE_", "HUST_", "MTP_")
# These locations and IDs describe the current process, not the compiled recipe.
LOCAL_ENV = {
    "ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME", "ASCEND_TOOLKIT_LATEST_HOME", "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH", "ASCEND_CACHE_PATH", "ASCEND_PROCESS_LOG_PATH",
    "ATB_HOME_PATH", "HCCL_IF_BASE_PORT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
    "VLLM_CACHE_ROOT", "VLLM_HUST_PHYSICAL_DEVICES", "VLLM_RPC_BASE_PATH",
    "VLLMHUST_ENGINE_BOOTSTRAPPED", "VLLMHUST_ENGINE_LOCKS",
    "VLLM_LOGGING_CONFIG_PATH", "VLLM_LOGGING_LEVEL",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_device_identity(table, physical):
    version = re.search(r"\bVersion:\s*(\S+)", table)
    names = {}
    for row in table.splitlines():
        match = re.match(r"\|\s*(\d+)\s+(\S+)\s*\|\s*(?:OK|Warning|Alarm|Fault)\b", row, re.I)
        if match:
            names[int(match[1])] = match[2]
    if not version or any(card not in names for card in physical):
        raise ValueError("Could not identify the selected NPU models and driver")
    return {"models": [names[card] for card in physical], "driver": version[1]}


def fingerprint(installation, receipt, model, hardware, environ):
    sdk = installation / "sdk"
    profile = json.loads((sdk / "deployment.json").read_bytes())
    deps = receipt["dependencies"]
    packages = dict(deps["metadata"])
    for package in ("triton", "triton-ascend", "npugraph-ex", "torchair"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    cann = Path(environ.get("ASCEND_HOME_PATH", ""))
    cann_files = [cann / "version.cfg", cann / "version.info",
                  cann / "aarch64-linux/ascend_toolkit_install.info"]
    cann_version = [file_digest(path) for path in cann_files if path.is_file()]
    if not cann_version:
        raise ValueError("Could not identify the installed CANN toolkit")
    relevant_env = {key: val for key, val in environ.items()
                    if key.startswith(ENV_PREFIXES) and key not in LOCAL_ENV}
    return {
        "release": receipt["release"],
        "package": receipt["package_version"],
        "sources": file_digest(sdk / "provenance/installed-files.json"),
        "runtime_config": profile.get("runtime_configuration_sha256"),
        "python": deps["python_version"],
        "packages": packages,
        "platform": [platform.machine(), *platform.libc_ver()],
        "hardware": hardware,
        "cann": cann_version,
        "model": {path.name: file_digest(path)
                  for path in (model / "config.json", *sorted(model.glob("*.index.json")))},
        "environment_sha256": digest(relevant_env),
    }


def normal_file(path):
    return path.is_file() and not path.is_symlink()


def ordered_dirs(path):
    try:
        if path.is_dir() and not path.is_symlink():
            return sorted(child for child in path.iterdir() if child.is_dir() and not child.is_symlink())
    except OSError:
        pass
    return []


def read_record(path):
    try:
        seal = path / RECORD
        if not path.is_symlink() and normal_file(seal):
            value = json.loads(seal.read_bytes())
            if isinstance(value, dict):
                return value
    except (OSError, ValueError):
        pass
    return {}


@contextmanager
def cache_lease(path, *, writer):
    import fcntl

    try:
        stream = (path / LOCK).open("a+b" if writer else "rb")
    except OSError:
        yield False
        return
    with stream:
        try:
            mode = fcntl.LOCK_EX if writer else fcntl.LOCK_SH
            fcntl.flock(stream, mode | fcntl.LOCK_NB)
        except OSError:
            yield False
        else:
            if writer:
                os.set_inheritable(stream.fileno(), True)
            yield True


def triton_groups(source, origin, staged, target):
    groups = 0
    for directory in ordered_dirs(source / TRITON):
        for index in sorted(directory.glob("__grp__*.json")):
            prefix = TRITON / directory.name
            try:
                if not normal_file(index):
                    continue
                entry = json.loads(index.read_bytes())
                files = entry["child_paths"]
                if (not isinstance(files, dict) or not files
                        or any(not isinstance(name, str) or Path(name).name != name
                               or not isinstance(path, str) or Path(path) != origin / prefix / name
                               or not normal_file(directory / name) for name, path in files.items())):
                    continue
                out = staged / prefix
                out.mkdir(parents=True, exist_ok=True)
                for name in files:
                    shutil.copyfile(directory / name, out / name)
                entry["child_paths"] = {name: str(target / prefix / name) for name in files}
                (out / index.name).write_text(json.dumps(entry), encoding="utf-8")
                groups += 1
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return groups


def ascend_graphs(source, origin, staged, target):
    graphs = 0
    for cache_key in ordered_dirs(source / VLLM):
        if not re.fullmatch(r"[0-9a-f]+", cache_key.name):
            continue
        for rank in ordered_dirs(cache_key):
            for model in ordered_dirs(rank):
                index = model / "vllm_compile_cache.py"
                if not normal_file(index):
                    continue
                prefix = model.relative_to(source)
                output_index = staged / prefix / index.name
                if os.path.lexists(output_index):
                    continue
                try:
                    entries = ast.literal_eval(index.read_text(encoding="utf-8"))
                    if not isinstance(entries, dict):
                        continue
                    admitted = {}
                    for key, entry in entries.items():
                        if not isinstance(key, tuple) or len(key) != 3 or key[2] != "AscendCompiler":
                            continue
                        name, path = entry["graph_handle"]
                        if not isinstance(name, str) or Path(name).name != name:
                            continue
                        graph = model / name
                        if path != str(origin / prefix / name) or not normal_file(graph):
                            continue
                        code = graph.read_bytes()
                        if not code or b"triton_kernel_wrapper" in code:
                            continue
                        artifact = staged / prefix / name
                        if not copy_new(artifact, code) and (
                            not normal_file(artifact) or artifact.read_bytes() != code
                        ):
                            continue
                        admitted[key] = dict(entry, graph_handle=(name, str(target / prefix / name)))
                    if admitted and copy_new(output_index, repr(admitted).encode("utf-8")):
                        graphs += len(admitted)
                except (OSError, SyntaxError, ValueError, KeyError, TypeError):
                    continue
    return graphs


def copy_new(path, content):
    """Publish only complete entries, preserving any already present."""
    if os.path.lexists(path):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".seed-", dir=path.parent) as temp:
        staged = Path(temp) / path.name
        staged.write_bytes(content)
        try:
            os.link(staged, path)
        except FileExistsError:
            return False
    return True


def model_infos(source, target, models):
    if models is None:
        return 0
    count = 0
    for candidate in sorted((source / MODEL_INFOS).glob("*.json")):
        match = MODEL_INFO_NAME.fullmatch(candidate.name)
        output = target / MODEL_INFOS / candidate.name
        if not match or not normal_file(candidate) or os.path.lexists(output):
            continue
        module, architecture = match.groups()
        module_file = models / (module + ".py")
        if not normal_file(module_file):
            continue
        try:
            raw = candidate.read_bytes()
            record = json.loads(raw)
            content = module_file.name.encode() + b"\0" + module_file.read_bytes() + b"\0"
            # Match the installed registry's safe_hash(..., usedforsecurity=False).
            try:
                native_hash = hashlib.md5(content, usedforsecurity=False).hexdigest()
            except ValueError:
                native_hash = hashlib.sha256(content).hexdigest()
            if record["hash"] == native_hash and record["modelinfo"]["architecture"] == architecture:
                count += copy_new(output, raw)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return count


def runtime_helpers(source, target):
    count = 0
    ext = sysconfig.get_config_var("EXT_SUFFIX")
    names = {"npu_utils.so"}
    if ext:
        names.update("launcher_cxx11abi" + abi + ext for abi in ("0", "1"))
    for folder in ordered_dirs(source / TRITON):
        # Triton supplies the cache key. Keep it intact and copy only the
        # known driver/launcher artifacts selected by that native key.
        if not re.fullmatch(r"[A-Z2-7]+", folder.name):
            continue
        for name in names:
            path = folder / name
            output = target / TRITON / folder.name / name
            if not normal_file(path) or os.path.lexists(output):
                continue
            try:
                raw = path.read_bytes()
                if raw.startswith(b"\x7fELF"):
                    count += copy_new(output, raw)
            except OSError:
                continue
    return count


def compatible_sources(group, extra_group, target, identity):
    candidates = ordered_dirs(group)
    if extra_group is not None and extra_group != group:
        candidates += ordered_dirs(extra_group)
    for candidate in candidates:
        if candidate == target:
            continue
        saved = read_record(candidate)
        origin = saved.get("origin")
        if saved.get("identity") == identity and isinstance(origin, str):
            origin = Path(origin)
            if origin.is_absolute():
                yield candidate, origin


def seed_startup(candidates, staged, target, models, report):
    for source, origin in candidates:
        with cache_lease(source, writer=False) as available:
            if available:
                report["model_infos"] += model_infos(source, staged, models)
                report["runtime_helpers"] += runtime_helpers(source, staged)
                # One cache pool may hold native keys for several SDK paths.
                # Preserve every available key, including during backfill.
                report["ascend_graphs"] += ascend_graphs(source, origin, staged, target)


def select(pool, extra_pool, identity, location, models=None):
    key = digest(identity)[:32]
    group = pool / POOL / key
    extra_group = extra_pool / POOL / key if extra_pool is not None else None
    target = group / digest([str(pool), *location])[:24]
    record = {"identity": identity, "origin": str(target)}
    if os.path.lexists(target):
        if read_record(target) == record:
            report = {"mode": "existing"}
            candidates = list(compatible_sources(group, extra_group, target, identity))
            if candidates:
                with cache_lease(target, writer=True) as available:
                    if available:
                        startup = {"model_infos": 0, "runtime_helpers": 0, "ascend_graphs": 0}
                        seed_startup(candidates, target, target, models, startup)
                        report.update({name: count for name, count in startup.items() if count})
            return target, report
        return group / ("unverified-" + uuid.uuid4().hex), {
            "mode": "isolated", "reason": "cached identity record is missing or inconsistent",
        }
    group.mkdir(parents=True, exist_ok=True)
    report = {"mode": "cold", "triton_groups": 0, "ascend_graphs": 0,
              "model_infos": 0, "runtime_helpers": 0}
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=group) as temp:
        staged = Path(temp) / "cache"
        staged.mkdir()
        candidates = list(compatible_sources(group, extra_group, target, identity))
        for candidate, origin in candidates:
            with cache_lease(candidate, writer=False) as available:
                if not available:
                    continue
                report["triton_groups"] += triton_groups(candidate, origin, staged, target)
            if report["triton_groups"]:
                break
        seed_startup(candidates, staged, target, models, report)
        if any(count for name, count in report.items() if name != "mode"):
            report["mode"] = "relocated"
        (staged / RECORD).write_text(json.dumps(record), encoding="utf-8")
        (staged / LOCK).touch()
        try:
            staged.rename(target)
        except OSError:
            if read_record(target) != record:
                raise
            report = {"mode": "existing"}
    return target, report


def prepare(pool, extra_pool, installation, receipt, model, physical, environ):
    if extra_pool is None and any((pool / relative).is_dir() for relative in (VLLM, TRITON)):
        # Older clients wrote directly here. Let the native loaders use their
        # own keys in place; these entries have no portable identity record.
        return pool, {"mode": "legacy-native"}
    try:
        table = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                               check=True, timeout=20).stdout
        identity = fingerprint(installation, receipt, model,
                               parse_device_identity(table, physical), environ)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        # Hardware admission and service launch still belong to the normal producer path.
        return pool / POOL / ("unverified-" + uuid.uuid4().hex), {
            "mode": "isolated", "reason": str(error),
        }
    return select(pool, extra_pool, identity, [
        str((installation / "sdk").resolve()), receipt["python"], str(model),
    ], models=installation / "sdk/runtime/engine-core/vllm/model_executor/models")
