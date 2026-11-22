from engine.release_selection import kit_path, runtime_lock, profile as selected_profile
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
import time
import urllib.request
import urllib.error
import sys

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = {"python", "wheelhouse", "cann_home", "official_sdk"}
PROBE = """import importlib.metadata as m,json,platform,sys
names=json.loads(sys.argv[1]);versions={}
for n in names:
 try:versions[n]=m.version(n)
 except m.PackageNotFoundError:versions[n]=None
print(json.dumps({'python':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'bytecode_writes_disabled':sys.dont_write_bytecode,'version':list(sys.version_info[:2]),'implementation':platform.python_implementation(),'machine':platform.machine(),'system':platform.system(),'packages':versions}))
"""


def probe(python, packages, env=None):
    result = subprocess.run([str(python), "-I", "-B", "-c", PROBE, json.dumps(list(packages))],
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
    if options.get('official_sdk'):
        if options.get('cann_home'):
            raise ValueError('Choose official_sdk or cann_home, not both')
        from official import enable_module_imports

        enable_module_imports()
        from official_protocol_bootstrap import cann_environment as private_cann
        return private_cann(options['official_sdk'])
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
    from benchmark.one_click import verify_kit
    verify_kit()
    sys.path.insert(0, str(kit_path() / "scripts"))
    from child_runner import run_stage
    return run_stage(args, env=env, stdout=log, stderr=subprocess.STDOUT)


def stream_log(path, offset, stopped):
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    with path.open("rb") as source:
        source.seek(offset)
        while not stopped.is_set():
            chunk = source.read(65536)
            if chunk:
                print(decoder.decode(chunk), end="", flush=True)
            else:
                stopped.wait(0.2)
        # A read may have reached the old EOF just before the producer's last
        # write and stop signal. Drain after observing stop, then flush UTF-8.
        while chunk := source.read(65536):
            print(decoder.decode(chunk), end="", flush=True)
        print(decoder.decode(b"", final=True), end="", flush=True)


@contextmanager
def follow_log(path, offset):
    """Follow only this invocation's bytes while retaining the original log."""
    stopped = threading.Event()
    thread = threading.Thread(target=stream_log, args=(path, offset, stopped), daemon=True)
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
        from engine.pip_transport import plan, audit
        args, child_env, source_audit = plan(args, env, directory)
        if source_audit:
            record['pip_source_audit'] = source_audit
        log_path = directory / (name + ".log")
        print("ENVIRONMENT_LOG", log_path, flush=True)
        with log_path.open("ab", buffering=0) as log, follow_log(log_path, log.tell()):
            result = run_repair_stage(args, child_env, log)
        if result.returncode == 0 and source_audit:
            record['status'] = 'verifying-source'
            record['publisher_admission'] = audit(source_audit, runtime_lock(ROOT))
        record.update(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode)
        if result.returncode:
            raise ValueError(name + " failed; see " + str(directory / (name + ".log"))
                             + ". Exact vendor wheels can be supplied with environment.wheelhouse or PIP_FIND_LINKS.")
    except BaseException as error:
        if record["status"] == 'verifying-source':
            record.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        elif record["status"] == "running":
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
    from benchmark.baseline_setup import download_lock
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
    with (cache / (filename + ".lock")).open("a+b") as lease, download_lock(lease):
        if valid(target):
            return target.resolve()
        if os.path.lexists(target):
            if target.is_symlink():
                raise ValueError("Cached wheel must not be a symlink")
            target.rename(target.with_name(filename + ".invalid-" + wheel_digest(target)[:12]))
        urls = list(dict.fromkeys([source.get("api_url", source["url"]), source["url"]]))
        return _download_dependency(source,target,urls,valid)


def _download_dependency(source,target,urls,valid):
    """Resume only the pinned artifact; partials remain unqualified until full SHA."""
    import re
    filename=source['filename'];size=source.get('bytes')
    partial=target.with_name(filename+'.download.partial')
    provenance=target.with_name(filename+'.download.json')
    if partial.is_symlink() or provenance.is_symlink():
        raise ValueError('Dependency partial paths must not be links')
    pin=dict(filename=filename,sha256=source['sha256'],bytes=size)
    if provenance.exists():
        journal=json.loads(provenance.read_bytes())
        if journal.get('schema')!='deeplinkengine-dependency-download/v1' or journal.get('artifact')!=pin:
            raise ValueError('Dependency partial has a different pinned source; preserve '+str(provenance))
        if partial.exists():
            if (partial.stat().st_size!=journal.get('partial_bytes')
                    or wheel_digest(partial)!=journal.get('partial_sha256')):
                raise ValueError('Dependency partial changed outside its recorded transfer; preserve '+str(partial))
    else:
        if partial.exists():raise ValueError('Unowned dependency partial retained: '+str(partial))
        journal=dict(schema='deeplinkengine-dependency-download/v1',artifact=pin,started=time.time(),attempts=[])
    if journal.get('error'):
        journal.setdefault('previous_failures',[]).append(dict(error=journal.pop('error'),ended=journal.get('ended')))
    journal.pop('ended',None)
    journal.update(status='running',last_invocation_started=time.time())
    journal.pop('completed_file',None)
    def checkpoint():
        journal.update(partial_path=str(partial),partial_bytes=partial.stat().st_size if partial.exists() else 0,
                       partial_sha256=wheel_digest(partial) if partial.exists() else hashlib.sha256(b'').hexdigest(),
                       updated=time.time(),partial_qualified=False)
        save(provenance,journal)
    checkpoint();last_error=None
    try:
        if valid(partial):
            partial.replace(target);journal.update(status='completed',completed_file=dict(path=str(target),bytes=target.stat().st_size,sha256=source['sha256']))
            return target.resolve()
        if partial.exists() and size is not None and partial.stat().st_size>=size:
            raise ValueError('Dependency partial fails its pinned whole SHA; preserve '+str(partial))
        for attempt in range(2):
            for url in urls:
                while True:
                    offset=partial.stat().st_size if partial.exists() else 0
                    event=dict(url=url,offset=offset,started=time.time());journal['attempts'].append(event);checkpoint()
                    try:
                        if offset and size is None:raise ValueError('Cannot resume a dependency without a pinned byte count')
                        headers={'User-Agent':'DeepLinkEngine-environment','Accept':'application/octet-stream'}
                        if offset:headers['Range']='bytes='+str(offset)+'-'
                        print('ENVIRONMENT downloading',filename,'offset',offset,flush=True)
                        request=urllib.request.Request(url,headers=headers)
                        with urllib.request.urlopen(request,timeout=45) as raw:
                            status=getattr(raw,'status',200);header=raw.headers.get('Content-Range') if hasattr(raw,'headers') else None
                            event.update(response_status=status,content_range=header)
                            if offset or status==206:
                                match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',header or '')
                                if (status!=206 or not match or size is None or int(match[1])!=offset
                                        or int(match[3])!=size or not offset<=int(match[2])<size):
                                    raise OSError('Dependency resume requires exact206/Content-Range/total bytes')
                                response_bytes=int(match[2])-offset+1
                            else:
                                if status!=200:raise OSError('Unexpected dependency HTTP status')
                                response_bytes=None
                            read=getattr(raw,'read1',raw.read);checked=time.monotonic();checked_bytes=offset
                            with partial.open('ab') as stream:
                                while block:=read(64*1024):
                                    if (stream.tell()+len(block)>source.get('bytes',2*1024**3)
                                            or response_bytes is not None and stream.tell()+len(block)-offset>response_bytes):
                                        raise ValueError('Dependency download exceeds its fixed size or declared range')
                                    stream.write(block);now=time.monotonic()
                                    if now-checked>=45:
                                        received=stream.tell();print('ENVIRONMENT progress',filename,received,size,flush=True)
                                        if (size is None or received<size) and (received-checked_bytes)/(now-checked)<1024:
                                            raise OSError('Dependency transfer remains below 1 KiB/s: '+filename)
                                        checked,checked_bytes=now,received
                        observed=partial.stat().st_size
                        if size is not None and observed<size:
                            message=f'Truncated dependency download: {filename}: received {observed}, expected {size}'
                            event.update(status='truncated',error=message)
                            # Large useful fragments do not consume the network-error
                            # retry budget. Every continuation advances at least1MiB.
                            if observed-offset>=1024**2:continue
                            raise OSError(message)
                        if not valid(partial):raise ValueError('Dependency wheel size or SHA-256 mismatch: '+filename)
                        partial.replace(target);event.update(status='completed',received_bytes=observed-offset)
                        journal.update(status='completed',completed_file=dict(path=str(target),bytes=observed,sha256=source['sha256']))
                        return target.resolve()
                    except (urllib.error.URLError,TimeoutError,OSError,http.client.IncompleteRead) as error:
                        last_error=error;event.update(status='retry',error=type(error).__name__+': '+str(error))
                        if isinstance(error,urllib.error.HTTPError):error.close()
                        print('ENVIRONMENT download retry:',str(error),flush=True);break
                    except BaseException as error:
                        event.update(status='failed',error=type(error).__name__+': '+str(error));raise
                    finally:
                        event.update(ended=time.time(),partial_bytes=partial.stat().st_size if partial.exists() else 0)
                        checkpoint()
        raise OSError('Dependency download failed: '+filename+': '+str(last_error)+'; preserve '+str(partial))
    except BaseException as error:
        journal.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=type(error).__name__+': '+str(error));raise
    finally:
        journal['ended']=time.time();checkpoint()


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
                mismatches=mismatches, record_sha256=hashlib.sha256((distribution.read_text("RECORD") or "").encode()).hexdigest())


def vendor_integrity(python, required, env):
    if "triton-ascend" not in required:
        return dict(verified=True, packages={})
    code = """import json,sys
sys.path.insert(0,sys.argv[1])
from engine.environment_setup import distribution_integrity
print(json.dumps(distribution_integrity('triton-ascend')))
"""
    observed = subprocess.run([str(python), "-I", "-B", "-X", "utf8", "-c", code, str(ROOT)],
                              env=env, capture_output=True, text=True, encoding="utf-8", check=True, timeout=180)
    result = json.loads(observed.stdout)
    return dict(verified=result["verified"], packages={"triton-ascend": result})


def runtime_imports(python, required, env):
    """Exercise the CPU TorchVision ABI without loading the Ascend backend."""
    if "torchvision" not in required:
        return {"status": "not_required"}
    isolated = dict(env, TORCH_DEVICE_BACKEND_AUTOLOAD="0", ASCEND_RT_VISIBLE_DEVICES="",
                    CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    code = """import json,sys
import torch,torchvision
from torchvision.ops import nms
result=nms(torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]]),torch.tensor([.9,.8]),.5)
assert result.tolist()==[0], 'TorchVision CPU operator failed'
assert not any(n in sys.modules for n in ('torch_npu','vllm','vllm_ascend'))
print(json.dumps({'status':'verified','torch':torch.__version__,'torchvision':torchvision.__version__,
                  'cpu_operator':'torchvision.ops.nms','model_loaded':False,'npu_backend_imported':False}))
"""
    try:
        observed = subprocess.run([str(python), "-I", "-B", "-c", code], env=isolated,
                                  capture_output=True, text=True, check=True, timeout=180)
        return json.loads(observed.stdout)
    except subprocess.CalledProcessError as error:
        raise ValueError("Torch/TorchVision CPU ABI check failed: " + error.stderr[-3000:]) from error


def triton_backend_smoke(python, required, env, data, options, directory, integrity):
    """Backfill a new CPU qualification without reinstalling a healthy runtime."""
    if 'triton-ascend' not in required:
        return dict(status='not_required')
    import uuid
    source = runtime_lock(ROOT)['environment_sources']['triton-ascend']
    helper = ROOT / 'tools/triton_backend_probe.py'
    record_hash = integrity.get('packages', {}).get('triton-ascend', {}).get('record_sha256')
    boot = Path('/proc/sys/kernel/random/boot_id')
    binding = dict(schema='triton-ascend-original-wheel-abi/v1', probe_source_sha256=wheel_digest(helper),
                   source_wheel_sha256=source['sha256'], installed_record_sha256=record_hash,
                   boot_id=boot.read_text().strip() if boot.is_file() else None)
    receipt = directory / 'triton-backend-probe.json'
    if receipt.is_symlink():
        raise ValueError('Triton qualification receipt must not be a symlink')
    if receipt.exists():
        previous = json.loads(receipt.read_bytes())
        if (record_hash and all(previous.get(key) == value for key, value in binding.items())
                and previous.get('verified') is True and previous.get('driver_initialized') is False
                and previous.get('kernel_compiled') is False and previous.get('model_loaded') is False
                and previous.get('implementation') == 'triton-ascend'):
            return dict(previous, reused=True, receipt=str(receipt), receipt_sha256=wheel_digest(receipt))
        if previous.get('schema') != binding['schema']:
            raise ValueError('Unknown Triton qualification receipt; preserve ' + str(receipt))
        # Preserve the earlier qualification while a new helper or source is checked.
        receipt.rename(receipt.with_name('triton-backend-probe.previous-' + uuid.uuid4().hex + '.json'))
    links = [str(Path(value)) for value in (options.get('wheelhouse'), str(Path(data['cache']) / 'wheels'))
             if value and Path(value).is_dir()]
    wheel = cached_wheel(source, links, Path(data['cache']) / 'wheels')
    isolated = dict(env, TORCH_DEVICE_BACKEND_AUTOLOAD='0', ASCEND_RT_VISIBLE_DEVICES='',
                    CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    print('ENVIRONMENT triton-backend-qualification', receipt, flush=True)
    completed = subprocess.run([str(python), '-I', '-B', str(helper), '--wheel', str(wheel),
                                '--sha256', source['sha256'], '--out', str(receipt)],
                               env=isolated, capture_output=True, text=True, timeout=180)
    if completed.returncode or not receipt.is_file():
        raise ValueError('CPU Ascend backend qualification failed; preserve ' + str(receipt))
    value = json.loads(receipt.read_bytes())
    if (not record_hash or any(value.get(key) != expected for key, expected in binding.items())
            or value.get('verified') is not True or value.get('driver_initialized') is not False):
        raise ValueError('CPU Ascend backend qualification identity differs; preserve ' + str(receipt))
    return dict(value, reused=False, receipt=str(receipt), receipt_sha256=wheel_digest(receipt))


def ensure_environment(data, *, progress=None):
    """Return an exec handoff when repair/loading is needed; None in a verified child."""
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("The frozen engine requires Linux aarch64; an OS/architecture mismatch cannot be repaired with pip")
    options = data.get("environment", {})
    lock = runtime_lock(ROOT)
    required = lock["required_python_packages"]
    if progress is not None:
        progress.detail("check-cann")
    env, cann = cann_environment(options)
    selection = dict(options)
    if cann.get('bootstrap_python') and not selection.get('python'):
        selection['python'] = cann['bootstrap_python']
    if os.environ.get("DEEPLINKENGINE_MANAGED_PYTHON") == sys.executable:
        selection["python"] = sys.executable
    python, before = select_python(selection, required)
    # Large public package indexes can pause longer than pip's 15-second default.
    env.setdefault("PIP_DEFAULT_TIMEOUT", "120")
    env.setdefault("PIP_RETRIES", "3")
    env.setdefault("PIP_CACHE_DIR", str(Path(data["cache"]) / "pip"))
    env.setdefault("XDG_CACHE_HOME", str(Path(data["cache"]) / "xdg"))
    recipe = {"packages": required, "sources": lock.get("environment_sources", {}),
              "python": "3.12", "abi": "linux_aarch64", "cann": cann,
              "system_site_packages": False}
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
    # Separate run environments avoid repairing a runtime used by another campaign.
    run_key = hashlib.sha256(data["out"].encode()).hexdigest()[:12]
    directory = Path(data["cache"]) / "environments" / (key + "-" + run_key)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Environment cache must not be a symlink: " + str(directory))
    directory.mkdir(exist_ok=True)
    report_path = directory / "environment.json"
    if progress is not None:
        progress.detail("prepare-python", receipt=report_path)
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
                original_report = report_path.read_bytes()
                from engine.package_metadata import prepare
                current["package_metadata"] = prepare(python)
                try:
                    current["runtime_imports"] = runtime_imports(python, required, env)
                except (ValueError, subprocess.SubprocessError) as error:
                    current.update(status="invalid", runtime_import_error=str(error))
                    save(report_path, current)
                else:
                    current["vendor_integrity"] = integrity
                    try:
                        previous_probe = current.get('triton_backend', {})
                        current['triton_backend'] = triton_backend_smoke(python, required, env, data, options, directory, integrity)
                        if (current['triton_backend'].get('verified') is True
                                and previous_probe.get('probe_source_sha256') != current['triton_backend'].get('probe_source_sha256')):
                            import uuid
                            prior = directory / ('environment.before-triton-probe-' + uuid.uuid4().hex + '.json')
                            with prior.open('xb') as stream:
                                stream.write(original_report)
                            current['triton_probe_backfilled_from'] = str(prior)
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        current.update(status='invalid', triton_backend_error=str(error))
                        save(report_path, current)
                        raise
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
        if progress is not None:
            report["progress_id"] = progress.run_id
        try:
            missing = differences(before, required)
            report["differences"] = missing
            managed = directory / "venv"
            if managed.is_symlink():
                from engine.python_storage import admitted, ensure as restore_python
                if not admitted(managed / "bin/python", data["cache"]):
                    raise ValueError("Managed venv must not be an unmanaged symlink")
                restore_python(managed)
            python = str(managed / "bin/python")
            if not Path(python).is_file():
                command([str(before["python"]), "-I", "-B", "-m", "venv", str(managed)],
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
            # CANN exposes its tools through PYTHONPATH for runtime use. Keep them out of pip resolution.
            install = [python, "-I", "-B", "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:", "--no-compile"]
            for folder in links:
                install += ["--find-links", folder]
            qualified_imports = None
            if reusable and not differences(info, required):
                from engine.package_metadata import prepare
                report["package_metadata"] = prepare(python)
                try:
                    qualified_imports = runtime_imports(python, required, env)
                except (ValueError, subprocess.SubprocessError) as error:
                    reusable = False
                    report["runtime_import_error_before_repair"] = str(error)
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
            from engine.package_metadata import prepare
            report["package_metadata"] = prepare(python)
            report["runtime_imports"] = qualified_imports or runtime_imports(python, required, env)
            report['triton_backend'] = triton_backend_smoke(python, required, env, data, options, directory, integrity)
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
