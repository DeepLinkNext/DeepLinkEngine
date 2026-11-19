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
KIT = ROOT / "benchmark/one-click-kit"
LOCK = ROOT / "benchmark/one-click-lock.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_kit():
    lock = json.loads(LOCK.read_bytes())
    manifest = KIT / "DEPLOYMENT-TOOLS.json"
    if manifest.is_symlink() or digest(manifest) != lock["tools"]["manifest_sha256"]:
        raise ValueError("One-click source manifest differs from the public lock")
    data = json.loads(manifest.read_bytes())
    if data["commit"] != lock["tools"]["upstream_commit"]:
        raise ValueError("One-click source revision differs")
    expected = set(data["files"]) | {"DEPLOYMENT-TOOLS.json"}
    actual = {str(p.relative_to(KIT)) for p in KIT.rglob("*")
              if (p.is_file() or p.is_symlink()) and "__pycache__" not in p.parts}
    if actual != expected:
        raise ValueError("One-click source inventory differs")
    for name, value in data["files"].items():
        path = KIT / name
        if path.is_symlink() or not path.resolve().is_relative_to(KIT.resolve()) or digest(path) != value:
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


def prepare_data(arguments):
    from dataset_bundle import prepare
    cache = option(arguments, "--cache") or str(Path.home() / ".cache/vllm-hust-engine/third-party")
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
            effective = module.configuration(arguments[0], arguments[1:])
        if "--context-tenant-only" in effective:
            if option(effective, "--engine-profile") is not None:
                raise ValueError("The context/tenant controller reads the SDK identity; omit --engine-profile")
        else:
            if option(effective, "--engine-profile") is None:
                effective += ["--engine-profile", "r8"]
            if option(effective, "--engine-profile") != "r8":
                raise ValueError("This public entry requires engine_profile r8")
        if option(effective, "--deployment") is None:
            raise ValueError("Provide deployment: /path/to/r8-installation/sdk from the public r8 installation; see engine/README.md")
        effective = prepare_data(effective)
        effective = prepare_baseline(effective)
        os.environ["VLLM_HUST_PROJECT_EVIDENCE"] = str(reference_evidence(lock))
        arguments = effective
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
