#!/usr/bin/env python3
"""Download and verify the pinned private B1 r4 release; never run a model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import urllib.parse
import urllib.request


REPOSITORY = "vLLM-HUST/vllm-hust-engine"
TAG = "full-answer-b1-20261030-r4"
ASSETS = {
    "full-answer-frozen-sources-20261030-r4.tar.gz": (
        735845919,
        4036202,
        "3b13f565c2b94c4c1ba7dbfa1a1a3ac6af222085ef4e74fefe50906a3bd9c961",
    ),
    "joint-b1-r4-runtime-config.zip": (
        735846108,
        15990,
        "2a8e0371fdf122f9d0bdc705ee5284ba03cb50779e67c5a433c748fe75688e87",
    ),
    "b1-r4-recovered-components.tar.gz": (
        735855644,
        118877244,
        "565c3a3e73333b54af2cad969db1f15d8743f627dc9e6c28de55178767db8c33",
    ),
    "SHA256SUMS": (
        735856246,
        309,
        "331325eed48267b2767c7844173e7583e951c41cf94b96dbbe40eb87f59d5f88",
    ),
}


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            before = urllib.parse.urlparse(req.full_url)
            after = urllib.parse.urlparse(newurl)
            if after.scheme != "https":
                raise ValueError("Refusing non-HTTPS asset redirect")
            if before.netloc != after.netloc:
                redirected.remove_header("Authorization")
        return redirected


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_token(env_file=None):
    """Read only GitHub keys; do not execute .env or import unrelated secrets."""
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        if value := os.environ.get(key):
            return value, "environment:" + key
    if env_file is not None:
        candidates = [Path(env_file)]
        if not candidates[0].is_file():
            raise FileNotFoundError("Credential file does not exist: " + str(env_file))
    else:
        candidates = []
        for start in (Path.cwd(), Path(__file__).resolve().parent):
            candidates.extend(p / ".env" for p in (start, *start.parents))
        candidates.append(Path.home() / ".env")
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        values = {}
        for line in path.read_text().splitlines():
            match = re.match(
                r"^\s*(?:export\s+)?(GH_TOKEN|GITHUB_TOKEN)\s*=\s*(.*)$", line
            )
            if match:
                try:
                    parts = shlex.split(match[2], comments=True, posix=True)
                except ValueError:
                    raise ValueError(
                        "Invalid GitHub credential quoting in " + str(path)
                    ) from None
                if len(parts) == 1 and parts[0]:
                    values[match[1]] = parts[0]
        for key in ("GH_TOKEN", "GITHUB_TOKEN"):
            if key in values:
                return values[key], str(path) + ":" + key
    raise ValueError(
        "No GitHub credential: set GH_TOKEN/GITHUB_TOKEN, use --env-file, or add the key to .env"
    )


def fetch(out, token=None, env_file=None):
    out.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(SafeRedirect())

    def request(path, accept):
        return urllib.request.Request(
            "https://api.github.com/repos/" + REPOSITORY + path,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": accept,
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "vllm-hust-engine-pinned-release-fetch",
            },
        )

    complete_cache = all(
        (out / name).is_file()
        and (out / name).stat().st_size == size
        and digest(out / name) == expected
        for name, (_, size, expected) in ASSETS.items()
    )
    advertised = None
    credential_source = None
    release_id = 441819441
    if not complete_cache:
        if token is None:
            token, credential_source = resolve_token(env_file)
        else:
            credential_source = "caller"
        with opener.open(
            request("/releases/tags/" + TAG, "application/vnd.github+json"), timeout=60
        ) as stream:
            release = json.load(stream)
        if release["id"] != release_id:
            raise ValueError("Pinned release ID changed")
        advertised = {asset["name"]: asset for asset in release["assets"]}
    receipt = {
        "repository": REPOSITORY,
        "tag": TAG,
        "release_id": release_id,
        "source": "verified-local-cache" if complete_cache else "authenticated-release",
        "credential_source": credential_source,
        "model_started": False,
        "assets": {},
    }
    for name, (asset_id, size, expected) in ASSETS.items():
        if advertised is not None:
            metadata = advertised[name]
            if (metadata["id"], metadata["size"], metadata["digest"]) != (
                asset_id,
                size,
                "sha256:" + expected,
            ):
                raise ValueError("Release identity changed for " + name)
        path = out / name
        if not (
            path.is_file() and path.stat().st_size == size and digest(path) == expected
        ):
            partial = path.with_name(name + ".partial")
            with (
                opener.open(
                    request(
                        "/releases/assets/" + str(asset_id), "application/octet-stream"
                    ),
                    timeout=60,
                ) as stream,
                partial.open("wb") as target,
            ):
                while chunk := stream.read(1024 * 1024):
                    target.write(chunk)
                    if target.tell() > size:
                        raise ValueError("Asset exceeds pinned size: " + name)
            if partial.stat().st_size != size or digest(partial) != expected:
                raise ValueError("Asset bytes do not match: " + name)
            partial.replace(path)
        receipt["assets"][name] = {"bytes": size, "sha256": expected, "verified": True}
        print("VERIFIED", name, size, expected, flush=True)
    (out / "download-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    fetch(args.out, env_file=args.env_file)


if __name__ == "__main__":
    main()
