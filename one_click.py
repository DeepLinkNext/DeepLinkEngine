#!/usr/bin/env python3
"""Public r8 assessment entry; verify the kit and fetch pinned reference evidence."""

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import sys
import tarfile
import tempfile
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from release_selection import kit_path, runtime_lock, profile as selected_profile
KIT = kit_path()
LOCK = ROOT / "benchmark/one-click-lock.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_kit(*, legacy=False):
    global KIT
    from release_selection import active
    selected = None if legacy else active()
    if selected and selected.get('client'):
        from evaluation_client import verify
        data = verify(selected['client'])
        KIT = kit_path()
        lock = json.loads(LOCK.read_bytes())
        lock['selected_client'] = selected['client']
        lock['tools']['files'] = len(data['files'])
        return lock
    source_kit = ROOT / 'benchmark/one-click-kit' if legacy else KIT
    lock = json.loads(LOCK.read_bytes())
    manifest = source_kit / "DEPLOYMENT-TOOLS.json"
    if manifest.is_symlink() or digest(manifest) != lock["tools"]["manifest_sha256"]:
        raise ValueError("One-click source manifest differs from the public lock")
    data = json.loads(manifest.read_bytes())
    if data["commit"] != lock["tools"]["upstream_commit"]:
        raise ValueError("One-click source revision differs")
    expected = set(data["files"]) | {"DEPLOYMENT-TOOLS.json"}
    actual = {str(p.relative_to(source_kit)) for p in source_kit.rglob("*")
              if (p.is_file() or p.is_symlink()) and "__pycache__" not in p.parts}
    if actual != expected:
        raise ValueError("One-click source inventory differs")
    for name, value in data["files"].items():
        path = source_kit / name
        if path.is_symlink() or not path.resolve().is_relative_to(source_kit.resolve()) or digest(path) != value:
            raise ValueError("One-click source changed: " + name)
    return lock


def evidence_module():
    sys.path.insert(0, str(KIT / "scripts"))
    spec = importlib.util.spec_from_file_location("public_project_evidence", KIT / "scripts/project_evidence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option(arguments, flag):
    values = []
    for index, token in enumerate(arguments):
        if token == flag:
            if index + 1 == len(arguments) or arguments[index + 1].startswith("--"):
                raise ValueError(flag + " requires a value")
            values.append(arguments[index + 1])
        elif token.startswith(flag + "="):
            values.append(token.split("=", 1)[1])
    if len(values) > 1 or values == [""]:
        raise ValueError(flag + " requires exactly one non-empty value")
    return values[0] if values else None


def reference_evidence(lock):
    module = evidence_module()
    explicit = os.environ.get("VLLM_HUST_PROJECT_EVIDENCE")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        module.verify(path)
        return path
    record = lock["reference_evidence"]
    if (record["sha256"], record["bytes"]) != (module.ASSET_SHA256, module.ASSET_BYTES):
        raise ValueError("Public reference mirror differs from the fixed reference contract")
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vllm-hust-engine/project-evidence-53541118"
    cache.mkdir(parents=True, exist_ok=True)
    root = cache / "benchmark-evidence"
    with (cache / "public-download.lock").open("a+b") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        if root.exists():
            module.verify(root)
            return root
        archive = cache / "source.tar.gz"
        if not (archive.is_file() and archive.stat().st_size == record["bytes"] and digest(archive) == record["sha256"]):
            urls = list(dict.fromkeys([record.get("api_url", record["url"]), record["url"]]))
            last_error = None
            for attempt in range(2):
                for url in urls:
                    print("REFERENCE_DOWNLOAD", url, flush=True)
                    request = urllib.request.Request(url, headers={
                        "User-Agent": "DeepLinkEngine-public-one-click", "Accept": "application/octet-stream"})
                    with tempfile.NamedTemporaryFile(prefix="reference-", dir=cache, delete=False) as output:
                        partial = Path(output.name)
                        try:
                            with urllib.request.urlopen(request, timeout=30) as response:
                                while block := response.read(1024 * 1024):
                                    output.write(block)
                                    if output.tell() > record["bytes"]:
                                        raise ValueError("Reference mirror exceeds its fixed size")
                            output.flush()
                            if partial.stat().st_size != record["bytes"] or digest(partial) != record["sha256"]:
                                raise ValueError("Reference mirror size/SHA-256 mismatch")
                            partial.replace(archive)
                        except (urllib.error.URLError, TimeoutError, OSError) as error:
                            last_error = error
                            print("REFERENCE_DOWNLOAD_RETRY", type(error).__name__, flush=True)
                            continue
                        finally:
                            partial.unlink(missing_ok=True)
                    break
                else:
                    continue
                break
            else:
                raise OSError("Public reference download failed after both endpoints and retries: " + str(last_error))
        with tempfile.TemporaryDirectory(prefix="public-unpack-", dir=cache) as temp:
            with tarfile.open(archive) as source:
                seen = set()
                total = 0
                for member in source.getmembers():
                    path = PurePosixPath(member.name)
                    total += member.size
                    if (not member.isfile() or path.is_absolute() or ".." in path.parts
                            or not path.parts or path.parts[0] != "benchmark-evidence"
                            or member.name in seen or total > 512 * 1024 * 1024):
                        raise ValueError("Unsafe reference archive member")
                    seen.add(member.name)
                source.extractall(temp, filter="data")
            candidate = Path(temp) / "benchmark-evidence"
            module.verify(candidate)
            candidate.rename(root)
    return root


def validate_deployment(arguments):
    path = Path(option(arguments, "--deployment"))
    profile = path / "deployment.json"
    lock = runtime_lock(ROOT)
    if not profile.is_file() or json.loads(profile.read_bytes()).get("release") != lock["upstream_release"]:
        raise ValueError("Provide the selected stable SDK before downloading test inputs: " + str(path))


def configuration_arguments(subcommand, arguments):
    """Use the sealed parser without executing/download side effects."""
    verify_kit()
    spec = importlib.util.spec_from_file_location("public_one_click_configuration", KIT / "one_click.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(KIT / "scripts"))
    spec.loader.exec_module(module)
    module.CONFIG_FIELDS = tuple(module.CONFIG_FIELDS) + ("rerun_b0",)
    config_name = option(arguments, '--config')
    data = json.loads(Path(config_name).read_bytes()) if config_name else {}
    from b0_run_policy import boolean
    rerun = boolean(data.get('rerun_b0', False))
    flags = [x for x in arguments if x in ('--rerun-b0', '--no-rerun-b0')]
    if len(flags) > 1:
        raise ValueError('只指定一次 --rerun-b0 或 --no-rerun-b0')
    if flags:
        rerun = flags == ['--rerun-b0']
    configured = module.configuration(subcommand, [x for x in arguments if x not in flags])
    return configured + ['--rerun-b0'] if configured is not None and rerun else configured


def prepare_data(arguments):
    from dataset_bundle import prepare
    validate_deployment(arguments)
    cache = option(arguments, "--cache") or str(Path.home() / ".cache/vllm-hust-engine/third-party")
    if option(arguments, "--baseline-deployment") is None:
        from baseline_bundle import ensure_deployment
        baseline = ensure_deployment(cache)
        if baseline is not None:
            arguments += ["--baseline-deployment", str(baseline)]
    context = "--context-tenant-only" in arguments
    selected = [] if context else (option(arguments, "--datasets") or "mmlu,hle,frontier,swe,terminal").split(",")
    if "--compute-reference-only" in arguments:
        selected = ["mmlu"]
    root = prepare(cache, selected, assets=option(arguments, "--dataset-assets"),
                   context=context, deployment=option(arguments, "--deployment"))
    if not context:
        existing = option(arguments, "--dataset-assets")
        if existing is None:
            arguments += ["--dataset-assets", str(root)]
        else:
            for index, value in enumerate(arguments):
                if value == "--dataset-assets":
                    arguments[index + 1] = str(root)
                elif value.startswith("--dataset-assets="):
                    arguments[index] = "--dataset-assets=" + str(root)
    from artifact_cache import release
    release(cache, option(arguments, "--baseline-deployment"))
    return arguments


def prepare_baseline(arguments):
    if (option(arguments, "--baseline-deployment") is None
            and "--context-tenant-only" not in arguments
            and "--compute-reference-only" not in arguments):
        from baseline_setup import prepare
        cache = option(arguments, "--cache") or str(Path.home() / ".cache/vllm-hust-engine/third-party")
        prepare(cache)
    return arguments


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    read_only = not arguments or any(x in ('--help','-h') for x in arguments) or arguments[0] in ('verify','fetch-reference','status','dashboard')
    if not read_only:
        from release_selection import active, select, bind
        if not active():
            config_name = option(arguments, '--config')
            config = json.loads(Path(config_name).read_bytes()) if config_name else {}
            deployment = option(arguments, '--deployment') or config.get('deployment')
            saved = Path(deployment).resolve().parent / 'engine-selection.json' if deployment else None
            if saved is not None and saved.is_file():
                bind(saved)
            else:
                select(option(arguments, '--cache') or config.get('cache') or Path.home()/'.cache/deeplinkengine')
    lock = verify_kit()
    if arguments == ["verify"]:
        print("PASS: public one-click source inventory and SHA-256 (" + str(lock["tools"]["files"]) + " files)")
        return 0
    if arguments == ["fetch-reference"]:
        print("REFERENCE_EVIDENCE", reference_evidence(lock))
        return 0
    help_only = not arguments or any(arg in ("--help", "-h") for arg in arguments)
    read_only = arguments and arguments[0] in ("status", "dashboard")
    if not help_only and not read_only:
        # Validate configuration before any download; public runs require the
        # SDK installed from this repository's frozen public r8 distribution.
        spec = importlib.util.spec_from_file_location("public_one_click_kit", KIT / "one_click.py")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(KIT / "scripts"))
        spec.loader.exec_module(module)
        effective = arguments
        if arguments[0] in ("prepare", "run"):
            effective = configuration_arguments(arguments[0], arguments[1:])
        if "--context-tenant-only" in effective:
            if option(effective, "--engine-profile") is not None:
                raise ValueError("The context/tenant controller reads the SDK identity; omit --engine-profile")
        else:
            if option(effective, "--engine-profile") is None:
                effective += ["--engine-profile", selected_profile()]
            if option(effective, "--engine-profile") != selected_profile():
                raise ValueError("The requested profile differs from the selected engine release")
        if option(effective, "--deployment") is None:
            raise ValueError("Provide deployment: /path/to/installation/sdk from the selected stable installation; see engine/README.md")
        from b0_run_policy import public_arguments, enabled
        effective = public_arguments(effective, prepare_only=arguments[0] == 'prepare' or '--prepare-only' in effective)
        print('是否重跑 B0：' + ('是' if enabled() else '否（复用已封存记录）'), flush=True)
        effective = prepare_data(effective)
        effective = prepare_baseline(effective)
        os.environ["VLLM_HUST_PROJECT_EVIDENCE"] = str(reference_evidence(lock))
        arguments = effective
    if help_only:
        print("B0 选项：--rerun-b0 显式重跑；--no-rerun-b0 默认复用旧记录。配置项 rerun_b0 默认 false。", flush=True)
    sys.dont_write_bytecode = True
    sys.argv = [str(KIT / "one_click.py"), *arguments]
    runpy.run_path(str(KIT / "one_click.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print("PUBLIC_ONE_CLICK_ERROR:", error, file=sys.stderr)
        raise SystemExit(2)
