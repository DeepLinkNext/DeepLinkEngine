#!/usr/bin/env python3
"""Publish an engine-tools-v* tag only after round-trip artifact acceptance.

Creates/retains a draft on failure. Never changes historical engine releases.
Uses only the job's repository-scoped GitHub token; no model/NPU credentials.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

sys.dont_write_bytecode = True
REPO = "vLLM-HUST/vllm-hust-engine"
ROOT = Path(__file__).resolve().parents[1]


def request(url, body=None, method=None, binary=False):
    data = body if binary else None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/octet-stream"
            if binary
            else "application/json",
            "User-Agent": "vllm-hust-tools-release-ci",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def main():
    tag = os.environ["GITHUB_REF_NAME"]
    if os.environ["GITHUB_REPOSITORY"] != REPO or not re.fullmatch(
        r"engine-tools-v[0-9A-Za-z._-]+", tag
    ):
        raise ValueError("Refusing non-tools tag or other repository")
    source = os.environ["GITHUB_SHA"]
    # The tag must name a commit already merged into main. A passing PR alone
    # is not permission to publish a branch's tools as an official release.
    subprocess.run(["git", "fetch", "--no-tags", "origin", "main"], check=True)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source, "FETCH_HEAD"], check=True
    )
    base = "https://api.github.com/repos/" + REPO
    candidate = ROOT / "dist/engine-tools.tar.gz"
    build = json.loads((ROOT / "dist/build.json").read_bytes())
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != build["archive_sha256"]:
        raise ValueError("Candidate changed after CI")
    if build["source_commit"] != source:
        raise ValueError("Candidate source differs from tag")
    release = request(
        base + "/releases",
        {
            "tag_name": tag,
            "target_commitish": source,
            "name": tag,
            "draft": True,
            "make_latest": "false",
            "body": "CPU-validated tools release: fixed five-profile full test, MMLU200 quick test, reference verification enabled. Both standalone entrypoints and the uploaded archive pass the same contract. Frozen engine versions and NPU results are unchanged.",
        },
    )
    upload = release["upload_url"].split("{")[0]

    def asset(path):
        return request(
            upload + "?name=" + urllib.parse.quote(path.name),
            path.read_bytes(),
            "POST",
            True,
        )

    uploaded = asset(candidate)
    # Download exactly what GitHub stored, strip credentials on cross-host
    # redirects with the shipped downloader, then rerun the real entry routes.
    spec = importlib.util.spec_from_file_location(
        "release_downloader", ROOT / "dist/download-and-run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="engine-tools-release-") as tmp:
        tmp = Path(tmp)
        req = urllib.request.Request(
            base + "/releases/assets/" + str(uploaded["id"]),
            headers={
                "Authorization": "Bearer " + os.environ["GH_TOKEN"],
                "Accept": "application/octet-stream",
            },
        )
        with urllib.request.build_opener(module.SafeRedirect()).open(
            req, timeout=120
        ) as response:
            blob = response.read(50 * 1024**2 + 1)
        if hashlib.sha256(blob).hexdigest() != build["archive_sha256"]:
            raise ValueError("Uploaded release archive differs")
        archive = tmp / "engine-tools.tar.gz"
        archive.write_bytes(blob)
        tools = tmp / "tools"
        tools.mkdir()
        module.extract(archive, tools)
        reports = []
        for filename, profile in (
            ("download-and-run.py", "r6"),
            ("download-and-run-r8.py", "r8"),
        ):
            runner = tmp / filename
            text = (ROOT / "dist" / filename).read_text()
            text, count = re.subn(
                r"^ASSET_ID = .*$",
                "ASSET_ID = " + str(uploaded["id"]),
                text,
                flags=re.MULTILINE,
            )
            if count != 1:
                raise ValueError("Cannot pin uploaded asset ID")
            runner.write_text(text)
            report = tmp / (filename + ".contract.json")
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "ci/evaluation_contract.py"),
                    "--root",
                    str(tools),
                    "--bootstrap",
                    str(runner),
                    "--runner-profile",
                    profile,
                    "--report",
                    str(report),
                ],
                check=True,
            )
            asset(runner)
            asset(report)
            reports.append(json.loads(report.read_bytes()))
        receipt = tmp / "release-acceptance.json"
        receipt.write_text(
            json.dumps(
                {
                    "source_commit": source,
                    "asset_id": uploaded["id"],
                    "archive_sha256": build["archive_sha256"],
                    "checks": reports,
                },
                indent=2,
            )
            + "\n"
        )
        asset(receipt)
    request(
        base + "/releases/" + str(release["id"]),
        {"draft": False, "make_latest": "false"},
        "PATCH",
    )
    print("Published validated tools release: " + release["html_url"])


if __name__ == "__main__":
    main()
