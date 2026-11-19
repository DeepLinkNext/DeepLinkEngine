"""Repair Python dependencies in a private cache; load an installed CANN toolkit."""
import hashlib
import http.client
import codecs
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
import urllib.request
import urllib.error
import sys

ROOT = Path(__file__).resolve().parent
OPTIONS = {"python", "wheelhouse", "cann_home"}
PROBE = """import importlib.metadata as m,json,platform,sys
names=json.loads(sys.argv[1]);versions={}
for n in names:
 try:versions[n]=m.version(n)
 except m.PackageNotFoundError:versions[n]=None
print(json.dumps({'python':sys.executable,'prefix':sys.prefix,'version':list(sys.version_info[:2]),'implementation':platform.python_implementation(),'machine':platform.machine(),'system':platform.system(),'packages':versions}))
"""


def probe(python, packages, env=None):
    result = subprocess.run([str(python), "-I", "-c", PROBE, json.dumps(list(packages))],
                            env=env, capture_output=True, text=True, check=True, timeout=30)
    return json.loads(result.stdout)


def compatible_python(info):
    return (info["version"] == [3, 12] and info["implementation"] == "CPython"
            and info["machine"] == "aarch64" and info["system"] == "Linux")


def differences(info, required):
    return {name: {"expected": version, "found": info["packages"].get(name)}
            for name, version in required.items() if info["packages"].get(name) != version}


def select_python(options, packages):
    candidates = [options["python"]] if options.get("python") else [sys.executable, shutil.which("python3.12")]
    for python in dict.fromkeys(str(x) for x in candidates if x):
        try:
            info = probe(python, packages)
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
        if compatible_python(info):
            return python, info
    raise ValueError("No usable CPython 3.12 for Linux aarch64; install it or set environment.python")


def cann_environment(options, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    expected = "9.1.0"
    roots = [options["cann_home"]] if options.get("cann_home") else [
        "/usr/local/Ascend/cann-9.1.0", "/usr/local/Ascend/ascend-toolkit/9.1.0",
        env.get("ASCEND_HOME_PATH"), "/usr/local/Ascend/ascend-toolkit/latest"]
    for name in dict.fromkeys(str(x) for x in roots if x):
        root = Path(name).expanduser().resolve()
        script = root / "set_env.sh"
        if not script.is_file():
            continue
        descriptors = [root / "version.cfg", root / "version.info",
                       root / "aarch64-linux/ascend_toolkit_install.info",
                       root / "ascend_toolkit_install.info"]
        evidence = root.name + "\n" + "\n".join(x.read_text(errors="replace") for x in descriptors if x.is_file())
        if expected not in evidence:
            continue
        libraries = [root / "lib64/libascendcl.so", root / "aarch64-linux/lib64/libascendcl.so"]
        if not any(path.is_file() for path in libraries):
            continue
        result = subprocess.run(["bash", "--noprofile", "--norc", "-c",
             '. "$1" >/dev/null || exit $?; env -0', "deeplinkengine-env", str(script)],
             env=env, capture_output=True, check=True, timeout=30)
        loaded = dict(item.split("=", 1) for item in result.stdout.decode().split("\0") if "=" in item)
        if not loaded.get("PATH"):
            raise ValueError("CANN setup did not return a usable environment")
        return loaded, {"home": str(root), "setup": str(script), "version": expected}
    raise ValueError("CANN 9.1.0 toolkit not found; mount/install it or set environment.cann_home. Host drivers are not replaced.")


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def run_repair_stage(args, env, log):
    from one_click import verify_kit
    verify_kit()
    sys.path.insert(0, str(ROOT / "benchmark/one-click-kit/scripts"))
    from child_runner import run_stage
    return run_stage(args, env=env, stdout=log, stderr=subprocess.STDOUT)


@contextmanager
def follow_log(path, offset):
    """Follow only this invocation's bytes while retaining the original log."""
    stopped = threading.Event()

    def follow():
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        with path.open("rb") as source:
            source.seek(offset)
            while True:
                chunk = source.read(65536)
                if chunk:
                    print(decoder.decode(chunk), end="", flush=True)
                elif stopped.is_set():
                    print(decoder.decode(b"", final=True), end="", flush=True)
                    return
                else:
                    stopped.wait(0.2)

    thread = threading.Thread(target=follow, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def command(args, env, directory, name, report, report_path):
    print("ENVIRONMENT", name, flush=True)
    record = {"name": name, "status": "running"}
    report["steps"].append(record)
    save(report_path, report)
    try:
        log_path = directory / (name + ".log")
        print("ENVIRONMENT_LOG", log_path, flush=True)
        with log_path.open("ab", buffering=0) as log, follow_log(log_path, log.tell()):
            result = run_repair_stage(args, env, log)
        record.update(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode)
        if result.returncode:
            raise ValueError(name + " failed; see " + str(directory / (name + ".log"))
                             + ". Exact vendor wheels can be supplied with environment.wheelhouse or PIP_FIND_LINKS.")
    except BaseException as error:
        if record["status"] == "running":
            record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "launch_failed")
        raise
    finally:
        save(report_path, report)


def wheel_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_wheel(source, links, cache):
    """Use only a hash-matching local wheel or fetch the fixed public artifact."""
    import fcntl
    cache.mkdir(parents=True, exist_ok=True)
    filename = source["filename"]
    if Path(filename).name != filename:
        raise ValueError("Unsafe dependency wheel filename")
    def valid(path):
        return (path.is_file() and not path.is_symlink()
                and ("bytes" not in source or path.stat().st_size == source["bytes"])
                and wheel_digest(path) == source["sha256"])
    for folder in links:
        local = Path(folder) / filename
        if valid(local):
            return local.resolve()
    target = cache / filename
    with (cache / (filename + ".lock")).open("a+b") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        if valid(target):
            return target.resolve()
        if os.path.lexists(target):
            if target.is_symlink():
                raise ValueError("Cached wheel must not be a symlink")
            target.rename(target.with_name(filename + ".invalid-" + wheel_digest(target)[:12]))
        urls = list(dict.fromkeys([source.get("api_url", source["url"]), source["url"]]))
        last_error = None
        for attempt in range(2):
            for url in urls:
                print("ENVIRONMENT downloading", filename, flush=True)
                temporary = None
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "DeepLinkEngine-environment", "Accept": "application/octet-stream"})
                    with tempfile.NamedTemporaryFile(dir=cache, prefix="wheel-", delete=False) as stream:
                        temporary = Path(stream.name)
                        with urllib.request.urlopen(request, timeout=45) as raw:
                            while block := raw.read(1024 * 1024):
                                stream.write(block)
                                if stream.tell() > source.get("bytes", 2 * 1024**3):
                                    raise ValueError("Dependency download exceeds its fixed size")
                    observed_bytes = temporary.stat().st_size
                    if "bytes" in source and observed_bytes < source["bytes"]:
                        # read(size) may return EOF without raising IncompleteRead.
                        # Retry a truncated transfer, but never admit its bytes.
                        raise OSError(f"Truncated dependency download: {filename}: "
                                      f"received {observed_bytes}, expected {source['bytes']} bytes")
                    if not valid(temporary):
                        raise ValueError("Dependency wheel size or SHA-256 mismatch: " + filename
                                         + "; bytes=" + str(observed_bytes)
                                         + "; sha256=" + wheel_digest(temporary))
                    temporary.replace(target)
                    return target.resolve()
                except (urllib.error.URLError, TimeoutError, OSError, http.client.IncompleteRead) as error:
                    last_error = error
                    print("ENVIRONMENT download retry:", str(error), flush=True)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        raise OSError("Dependency download failed: " + filename + ": " + str(last_error))


def report_state(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def distribution_integrity(name):
    """Check installed RECORD hashes, including files shared with another wheel."""
    import base64
    from importlib import metadata
    distribution = metadata.distribution(name)
    mismatches, checked = [], 0
    for entry in distribution.files or []:
        if entry.hash is None:
            continue
        checked += 1
        path = distribution.locate_file(entry)
        if not path.is_file():
            mismatches.append(str(entry))
            continue
        digest = hashlib.new(entry.hash.mode)
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        actual = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
        if actual != entry.hash.value:
            mismatches.append(str(entry))
    return dict(verified=checked > 0 and not mismatches, checked_files=checked,
                mismatches=mismatches)


def vendor_integrity(python, required, env):
    if "triton-ascend" not in required:
        return dict(verified=True, packages={})
    code = """import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('environment_setup',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
print(json.dumps(module.distribution_integrity('triton-ascend')))
"""
    observed = subprocess.run([str(python), "-I", "-c", code, str(ROOT / "environment_setup.py")],
                              env=env, capture_output=True, text=True, check=True, timeout=180)
    result = json.loads(observed.stdout)
    return dict(verified=result["verified"], packages={"triton-ascend": result})


def ensure_environment(data):
    """Return an exec handoff when repair/loading is needed; None in a verified child."""
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("The frozen engine requires Linux aarch64; an OS/architecture mismatch cannot be repaired with pip")
    options = data.get("environment", {})
    lock = json.loads((ROOT / "engine/r8-lock.json").read_text())
    required = lock["required_python_packages"]
    env, cann = cann_environment(options)
    selection = dict(options)
    if os.environ.get("DEEPLINKENGINE_MANAGED_PYTHON") == sys.executable:
        selection["python"] = sys.executable
    python, before = select_python(selection, required)
    env.setdefault("PIP_CACHE_DIR", str(Path(data["cache"]) / "pip"))
    env.setdefault("XDG_CACHE_HOME", str(Path(data["cache"]) / "xdg"))
    recipe = {"packages": required, "sources": lock.get("environment_sources", {}),
              "python": "3.12", "abi": "linux_aarch64", "cann": cann}
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
    # Separate run environments avoid repairing a runtime used by another campaign.
    run_key = hashlib.sha256(data["out"].encode()).hexdigest()[:12]
    directory = Path(data["cache"]) / "environments" / (key + "-" + run_key)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Environment cache must not be a symlink: " + str(directory))
    directory.mkdir(exist_ok=True)
    report_path = directory / "environment.json"
    tmp = directory / "tmp"
    tmp.mkdir(exist_ok=True)
    env["TMPDIR"] = str(tmp)
    marker = os.environ.get("DEEPLINKENGINE_ENVIRONMENT_REPORT")
    if marker == str(report_path) and Path(sys.executable) == Path(python) and not differences(before, required):
        current = report_state(report_path)
        if (current.get("recipe_sha256") == key and current.get("status") == "ready"
                and current.get("after", {}).get("python") == sys.executable
                and current.get("after", {}).get("prefix") == sys.prefix):
            integrity = vendor_integrity(python, required, env)
            if integrity["verified"]:
                current["vendor_integrity"] = integrity
                save(report_path, current)
                os.environ.update(env)
                return None
    import fcntl
    with (directory / "repair.lock").open("a+b") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        previous = report_state(report_path)
        reusable = previous.get("status") == "ready" and previous.get("recipe_sha256") == key
        report = {"status": "checking", "recipe_sha256": key, "before": before,
                  "cann": cann, "steps": [], "shared_environment_modified": False}
        try:
            missing = differences(before, required)
            report["differences"] = missing
            managed = directory / "venv"
            if managed.is_symlink():
                raise ValueError("Managed venv must not be a symlink")
            python = str(managed / "bin/python")
            if not Path(python).is_file():
                command([str(before["python"]), "-m", "venv", "--system-site-packages", str(managed)],
                        env, directory, "create-venv", report, report_path)
            info = probe(python, required, env)
            if not compatible_python(info):
                raise ValueError("Cached environment has the wrong Python ABI")
            tmp = directory / "tmp"
            tmp.mkdir(exist_ok=True)
            env.update(TMPDIR=str(tmp), PYTHONNOUSERSITE="1", PIP_REQUIRE_VIRTUALENV="true")
            wheelhouses = [options.get("wheelhouse"), str(Path(data["cache"]) / "wheels")]
            links = [str(Path(x)) for x in wheelhouses if x and Path(x).is_dir()]
            if options.get("wheelhouse") and not Path(options["wheelhouse"]).is_dir():
                raise ValueError("environment.wheelhouse is not a directory")
            install = [python, "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:", "--no-compile"]
            for folder in links:
                install += ["--find-links", folder]
            if differences(info, required) or not reusable:
                requirements = directory / "requirements.txt"
                lines = []
                for name, version in required.items():
                    source = recipe["sources"].get(name)
                    if source:
                        local = cached_wheel(source, links, Path(data["cache"]) / "wheels")
                        url = local.as_uri()
                        lines.append(name + " @ " + url + "#sha256=" + source["sha256"])
                    else:
                        lines.append(name + "==" + version)
                requirements.write_text("\n".join(lines) + "\n")
                # Required distributions must have real metadata in the SDK's own site-packages.
                # --ignore-installed applies only inside this private venv, never to the host.
                command([*install, "--no-deps", "--ignore-installed", "-r", str(requirements)],
                        env, directory, "repair-vendor-packages", report, report_path)
                command([*install, "-r", str(requirements)], env, directory,
                        "resolve-python-dependencies", report, report_path)
            integrity = vendor_integrity(python, required, env)
            if not integrity["verified"]:
                # triton-ascend declares the base triton distribution as a
                # dependency, but overlays its Python modules and native library.
                # Dependency resolution can install the base wheel after the
                # vendor wheel. Restore the pinned vendor bytes last.
                source = recipe["sources"]["triton-ascend"]
                local = cached_wheel(source, links, Path(data["cache"]) / "wheels")
                report["vendor_integrity_before_repair"] = integrity
                command([*install, "--no-deps", "--ignore-installed", str(local)],
                        env, directory, "restore-ascend-triton-overlay", report, report_path)
                integrity = vendor_integrity(python, required, env)
                if not integrity["verified"]:
                    raise ValueError("Ascend Triton installed files differ from their RECORD hashes")
            after = probe(python, required, env)
            if differences(after, required):
                raise ValueError("Dependency versions remain inconsistent after repair")
            report.update(status="ready", after=after, vendor_integrity=integrity)
            save(report_path, report)
            env["DEEPLINKENGINE_ENVIRONMENT_REPORT"] = str(report_path)
            env["DEEPLINKENGINE_MANAGED_PYTHON"] = str(python)
            print("ENVIRONMENT ready", report_path, flush=True)
            return {"python": str(python), "env": env, "report": str(report_path)}
        except BaseException as error:
            report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error=type(error).__name__ + ": " + str(error))
            save(report_path, report)
            raise
