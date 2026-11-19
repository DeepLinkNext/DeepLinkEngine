#!/usr/bin/env python3
"""Build a checksum-sealed tools candidate and both standalone entrypoints.

Run acceptance against the extracted archive, not merely the checkout. Asset ID
is zero in CI-only artifacts; CD supplies GitHub's actual uploaded asset ID and
rechecks the generated entrypoints before publishing a release.
"""

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

REPOSITORY = "vLLM-HUST/vllm-hust-engine"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(root, out, commit, asset_id=0):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A full source commit SHA is required")
    if (root / ".git").exists():
        actual = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != commit:
            raise ValueError("Build commit differs from checkout HEAD")
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "--exit-code",
                commit,
                "--",
                *json.loads((root / "publication-files.json").read_bytes()),
            ],
            check=True,
        )
    elif asset_id:
        raise ValueError("Publishing requires a Git checkout bound to its commit")
    out.mkdir(parents=True, exist_ok=False)
    paths = json.loads((root / "publication-files.json").read_bytes())
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate publication inventory path")
    sources = {}
    for name in sorted(paths):
        path = root / name
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or path.is_symlink():
            raise ValueError("Unsafe publication path: " + name)
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Publication path outside checkout: " + name)
        sources[name] = path.read_bytes()
    manifest = (
        json.dumps(
            {
                "repository": REPOSITORY,
                "commit": commit,
                "files": {p: sha(b) for p, b in sources.items()},
            },
            sort_keys=True,
            indent=2,
        ).encode()
        + b"\n"
    )
    sources["DEPLOYMENT-TOOLS.json"] = manifest
    archive = out / "engine-tools.tar.gz"
    # Stable file metadata makes repeated builds reproducible within Python.
    import gzip

    with (
        archive.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
    ):
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for name, body in sorted(sources.items()):
                info = tarfile.TarInfo("vllm-hust-engine/" + name)
                info.size, info.mode, info.mtime = len(body), 0o644, 0
                tar.addfile(info, io.BytesIO(body))
    identities = {
        "COMMIT": commit,
        "ASSET_ID": asset_id,
        "ASSET_BYTES": archive.stat().st_size,
        "ASSET_SHA256": sha(archive.read_bytes()),
        "MANIFEST_SHA256": sha(manifest),
        "PREFIX": "vllm-hust-engine",
    }
    for source, target in (
        ("download_and_run.py", "download-and-run.py"),
        ("runners/r8/download_and_run.py", "download-and-run-r8.py"),
    ):
        text = (root / source).read_text()
        for key, value in identities.items():
            text, count = re.subn(
                r"^" + key + r" = .*$",
                lambda _: key + " = " + repr(value),
                text,
                flags=re.MULTILINE,
            )
            if count != 1:
                raise ValueError("Missing/duplicate bootstrap pin: " + key)
        ast.parse(text)
        (out / target).write_text(text)
    # Extraction uses the very downloader implementation shipped to users.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "candidate_bootstrap", out / "download-and-run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extracted = out / "extracted"
    extracted.mkdir()
    module.extract(archive, extracted)
    tools = module.verify_tools(extracted)
    record = {
        "source_commit": commit if (root / ".git").exists() else None,
        "source_context": commit,
        "build_kind": "commit" if (root / ".git").exists() else "local-source-snapshot",
        "asset_id": asset_id,
        "archive": archive.name,
        "archive_sha256": identities["ASSET_SHA256"],
        "bytes": identities["ASSET_BYTES"],
        "manifest_sha256": identities["MANIFEST_SHA256"],
        "files": len(paths),
        "extracted_tools": str(tools),
        "downloadable": asset_id > 0,
    }
    (out / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--asset-id", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.out, args.commit, args.asset_id), indent=2))
