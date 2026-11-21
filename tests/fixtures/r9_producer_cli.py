"""Verify, install and evaluate the exact B1 r9 runtime; no implicit model load."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import uuid
from . import __version__

sys.dont_write_bytecode = True
PAYLOAD = Path(__file__).resolve().parent / "payload"
RELEASE = "full-answer-b1-20261119-r9"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def info():
    return json.loads((PAYLOAD / "frozen.json").read_bytes())


def verify():
    record = info()
    if record["release"] != RELEASE or record["version"] != __version__:
        raise ValueError("Wrong frozen package identity")
    for name, expected in record["archives"].items():
        if (
            sha(PAYLOAD / name) != expected["sha256"]
            or (PAYLOAD / name).stat().st_size != expected["bytes"]
        ):
            raise ValueError("Frozen package archive changed: " + name)
    return record


def extract(archive, target, prefix=None):
    seen = set()
    with tarfile.open(archive) as source:
        for member in source:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("Unsafe archive path")
            if prefix:
                if path.parts[0] != prefix:
                    raise ValueError("Wrong tools archive root")
                path = PurePosixPath(*path.parts[1:])
            if member.isdir():
                continue
            if not member.isfile() or str(path) in seen or str(path) == ".":
                raise ValueError("Non-regular or duplicate archive member")
            seen.add(str(path))
            dest = target / str(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as raw, dest.open("xb") as stream:
                shutil.copyfileobj(raw, stream)
            dest.chmod(member.mode & 0o777)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def audit_install(sdk, tools):
    code = """
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[2])/'scripts'))
from restore_b1_r4 import verify_installed_sources
from dependency_admission import admit_dependencies
root=Path(sys.argv[1])
sources=verify_installed_sources(root)
dependencies=admit_dependencies(root)
assert not {'torch','torch_npu','vllm','vllm_ascend'} & set(sys.modules)
print(json.dumps({'verified_runtime_files':len(sources),'dependencies':dependencies}))
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="")
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(sdk), str(tools)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    return json.loads(result.stdout)


def verify_tools(tools):
    manifest = tools / "DEPLOYMENT-TOOLS.json"
    expected = info()["tools_manifest_sha256"]
    if sha(manifest) != expected:
        raise ValueError("Installed tools manifest changed")
    data = json.loads(manifest.read_bytes())
    for path, digest in data["files"].items():
        target = tools / path
        if (
            not target.resolve().is_relative_to(tools.resolve())
            or target.is_symlink()
            or sha(target) != digest
        ):
            raise ValueError("Installed tools changed: " + path)


def install(out, python):
    metadata = verify()
    if (
        sys.platform != "linux"
        or platform.machine() != "aarch64"
        or sys.version_info[:2] != (3, 12)
    ):
        raise ValueError("The frozen binary ABI is CPython3.12/Linux aarch64")
    out = out.expanduser().absolute()
    python = python.expanduser().absolute()
    if os.path.lexists(out):
        raise ValueError(
            "Installation destination already exists; select a new directory"
        )
    attempt = out.with_name(out.name + ".attempt-" + uuid.uuid4().hex)
    attempt.mkdir(parents=True)
    sdk, tools = attempt / "sdk", attempt / "tools"
    sdk.mkdir()
    tools.mkdir()
    extract(PAYLOAD / "sdk-r9.tar.gz", sdk)
    extract(PAYLOAD / "tools-r9.tar.gz", tools, "vllm-hust-engine")
    layout = metadata["pythonpath_layout"]
    profile = json.loads((sdk / "deployment.json").read_bytes())
    profile.update(
        root=str(sdk.resolve()),
        python=str(python),
        pythonpath=[str((sdk / p).resolve()) for p in layout],
        model_started=False,
    )
    (sdk / "deployment.json").write_text(json.dumps(profile, indent=2) + "\n")
    audit = audit_install(sdk, tools)
    if (
        audit["verified_runtime_files"] != metadata["runtime_files"]
        or profile["release"] != RELEASE
    ):
        raise ValueError("Installed runtime differs from frozen r9")
    attempt.rename(out)
    sdk, tools = out / "sdk", out / "tools"
    profile.update(
        root=str(sdk.resolve()), pythonpath=[str((sdk / p).resolve()) for p in layout]
    )
    (sdk / "deployment.json").write_text(json.dumps(profile, indent=2) + "\n")
    # Relocation preserves every frozen runtime and overlay source byte.
    audit = audit_install(sdk, tools)
    assert audit["verified_runtime_files"] == metadata["runtime_files"]
    receipt = dict(
        release=RELEASE,
        package_version=__version__,
        installation=str(out),
        deployment=str(sdk),
        tools=str(tools),
        python=str(python),
        verified_runtime_files=audit["verified_runtime_files"],
        native_packages_imported=False,
        model_loaded=False,
        dependencies=audit["dependencies"],
    )
    (out / "installation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def evaluate(installation, forwarded):
    root = installation.expanduser().resolve()
    receipt = json.loads((root / "installation.json").read_bytes())
    if receipt["release"] != RELEASE:
        raise ValueError("Evaluation requires this exact r9 installation")
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    tools = root / "tools"
    verify_tools(tools)
    entry = module(tools / "one_click.py", "frozen_r9_one_click")
    if forwarded and forwarded[0] in ("prepare", "run"):
        forwarded = entry.configuration(forwarded[0], forwarded[1:])
        if forwarded is None:
            return 0
    profile = [
        x.split("=", 1)[1] if x.startswith("--engine-profile=") else forwarded[i + 1]
        for i, x in enumerate(forwarded)
        if x == "--engine-profile" or x.startswith("--engine-profile=")
    ]
    if profile and profile != ["r9"]:
        raise ValueError("This package evaluates frozen r9 only")
    if not profile:
        forwarded += ["--engine-profile", "r9"]
    if not any(x == "--deployment" or x.startswith("--deployment=") for x in forwarded):
        forwarded += ["--deployment", str(root / "sdk")]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="")
    os.execve(
        sys.executable, [sys.executable, str(tools / "one_click.py"), *forwarded], env
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info")
    sub.add_parser("verify")
    setup = sub.add_parser("install")
    setup.add_argument("--out", type=Path, required=True)
    setup.add_argument("--python", type=Path, default=Path(sys.executable))
    serving = sub.add_parser("serve")
    serving.add_argument("--installation", type=Path, required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--installation", type=Path, required=True)
    args, forwarded = parser.parse_known_args(argv)
    try:
        if args.command == "serve":
            root = args.installation.expanduser().resolve()
            receipt = json.loads((root / "installation.json").read_bytes())
            if receipt["release"] != RELEASE:
                raise ValueError("Serving requires this exact R9 installation")
            verify_tools(root / "tools")
            if forwarded and forwarded[0] == "--":
                forwarded = forwarded[1:]
            if any(x == "--deployment" or x.startswith("--deployment=") for x in forwarded):
                raise ValueError("The installed R9 SDK owns --deployment")
            os.execve(sys.executable, [sys.executable,
                str(root / "tools/scripts/serve_frozen.py"),
                "--deployment", str(root / "sdk"), *forwarded],
                dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=""))
        if args.command == "evaluate":
            return evaluate(args.installation, forwarded)
        if forwarded:
            parser.error("Unexpected arguments: " + " ".join(forwarded))
        result = (
            info()
            if args.command == "info"
            else verify()
            if args.command == "verify"
            else install(args.out, args.python)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print("FROZEN_R9_ERROR " + str(error), file=sys.stderr)
        return 2
