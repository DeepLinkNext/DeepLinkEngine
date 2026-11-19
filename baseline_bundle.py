"""Restore byte-identical B0 OCI blobs from repository assets or verified local cache."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "data/b0-oci.json"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def inventory():
    from one_click import verify_kit
    verify_kit()
    sys.path.insert(0, str(ROOT / "benchmark/one-click-kit/scripts"))
    from restore_b0_image import MANIFEST
    data = json.loads(INDEX.read_bytes())
    if (data["schema"] != "deeplinkengine-original018-oci/v1"
            or data["repository"] != "DeepLinkNext/DeepLinkEngine"
            or data["manifest_digest"] != MANIFEST):
        raise ValueError("B0 archive index differs from the frozen image")
    names, blobs = set(), set()
    for blob in data["blobs"]:
        value = blob["digest"]
        if (not value.startswith("sha256:") or len(value) != 71
                or any(c not in "0123456789abcdef" for c in value[7:]) or value in blobs):
            raise ValueError("Invalid or duplicate B0 blob identity")
        blobs.add(value)
        if not blob["parts"] or sum(p["bytes"] for p in blob["parts"]) != blob["bytes"]:
            raise ValueError("B0 part sizes differ from their original blob")
        for part in blob["parts"]:
            name = part["name"]
            if Path(name).name != name or name in names or not name.startswith("b0-oci-"):
                raise ValueError("Unsafe or duplicate B0 archive filename")
            names.add(name)
            expected = "https://github.com/DeepLinkNext/DeepLinkEngine/releases/download/" + data["release"] + "/" + name
            if part["url"] != expected:
                raise ValueError("B0 archive is outside the repository release")
            if part.get("api_url") and not part["api_url"].startswith(
                    "https://api.github.com/repos/DeepLinkNext/DeepLinkEngine/releases/assets/"):
                raise ValueError("B0 API archive is outside the repository")
    return data


def verify_release(cache):
    """Verify public assets and original manifest/config without fetching 6 GB in CI."""
    data = inventory()
    url = "https://api.github.com/repos/DeepLinkNext/DeepLinkEngine/releases/tags/" + data["release"]
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                  "User-Agent": "DeepLinkEngine-B0-verification"})
    with urllib.request.urlopen(request, timeout=45) as response:
        release = json.load(response)
    if release["draft"]:
        raise ValueError("The fixed B0 archive release is not public")
    assets = {row["name"]: row for row in release["assets"]}
    for blob in data["blobs"]:
        for part in blob["parts"]:
            asset = assets[part["name"]]
            if (asset["size"] != part["bytes"] or asset.get("digest") != "sha256:" + part["sha256"]
                    or part.get("api_url") != "https://api.github.com/repos/DeepLinkNext/DeepLinkEngine/releases/assets/" + str(asset["id"])):
                raise ValueError("Published B0 archive differs from its locked index: " + part["name"])
    from environment_setup import cached_wheel
    small = {}
    for blob in data["blobs"]:
        if blob["kind"] not in ("manifest", "config"):
            continue
        if blob["bytes"] > 1024**2:
            raise ValueError("Unexpected B0 manifest/config size")
        raw = b"".join(cached_wheel(dict(part, filename=part["name"]), [], Path(cache) / "b0-parts").read_bytes()
                       for part in blob["parts"])
        if hashlib.sha256(raw).hexdigest() != blob["digest"][7:]:
            raise ValueError("B0 manifest/config digest differs")
        small[blob["kind"]] = json.loads(raw)
    manifest = small["manifest"]
    expected = {data["manifest_digest"]: next(r["bytes"] for r in data["blobs"] if r["kind"] == "manifest"),
                manifest["config"]["digest"]: manifest["config"]["size"]}
    expected.update({r["digest"]: r["size"] for r in manifest["layers"]})
    if expected != {r["digest"]: r["bytes"] for r in data["blobs"]}:
        raise ValueError("B0 release inventory differs from the original manifest")
    if (small["config"]["architecture"], small["config"]["os"]) != ("arm64", "linux"):
        raise ValueError("B0 image architecture differs")
    print("B0_RELEASE_VERIFIED", len(expected), "original OCI blobs", flush=True)


def valid(path, size, checksum):
    return path.is_file() and not path.is_symlink() and path.stat().st_size == size and digest(path) == checksum


def prepare(cache):
    data = inventory()
    cache = Path(cache).expanduser().resolve()
    out = cache / "b0-oci"
    out.mkdir(parents=True, exist_ok=True)
    raw_sources = []
    if os.environ.get("DEEPLINKENGINE_B0_OCI_CACHE"):
        raw_sources.append(Path(os.environ["DEEPLINKENGINE_B0_OCI_CACHE"]).expanduser().resolve())
    raw_sources.append(Path.home() / ".cache/vllm-hust-engine/b0-image-018")
    offline = os.environ.get("DEEPLINKENGINE_B0_ARCHIVES")
    from environment_setup import cached_wheel
    with (out / "repository-bundle.lock").open("a+b") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        for blob in data["blobs"]:
            name = blob["digest"].replace(":", "-")
            checksum = blob["digest"][7:]
            target = out / name
            if valid(target, blob["bytes"], checksum):
                continue
            if os.path.lexists(target):
                raise ValueError("Cached B0 OCI blob differs: " + str(target))
            original = next((folder / name for folder in raw_sources
                             if valid(folder / name, blob["bytes"], checksum)), None)
            partial = None
            try:
                with tempfile.NamedTemporaryFile(dir=out, prefix="b0-", delete=False) as stream:
                    partial = Path(stream.name)
                    if original:
                        print("B0_LOCAL_VERIFIED", name, flush=True)
                        with original.open("rb") as source:
                            shutil.copyfileobj(source, stream, 1024**2)
                    else:
                        for part in blob["parts"]:
                            if offline:
                                path = Path(offline).expanduser().resolve() / part["name"]
                                if not valid(path, part["bytes"], part["sha256"]):
                                    raise ValueError("Offline B0 archive missing or changed: " + str(path))
                            else:
                                record = dict(part, filename=part["name"])
                                path = cached_wheel(record, [], cache / "b0-parts")
                            with path.open("rb") as source:
                                shutil.copyfileobj(source, stream, 1024**2)
                            if stream.tell() > blob["bytes"]:
                                raise ValueError("B0 reassembly exceeds its frozen size")
                if not valid(partial, blob["bytes"], checksum):
                    raise ValueError("Reassembled B0 OCI blob differs from the original image: " + name)
                partial.replace(target)
                print("B0_OCI_VERIFIED", name, blob["bytes"], flush=True)
            finally:
                if partial is not None:
                    partial.unlink(missing_ok=True)
        manifest = json.loads((out / data["manifest_digest"].replace(":", "-")).read_bytes())
        expected = {data["manifest_digest"]: (out / data["manifest_digest"].replace(":", "-")).stat().st_size,
                    manifest["config"]["digest"]: manifest["config"]["size"]}
        expected.update({r["digest"]: r["size"] for r in manifest["layers"]})
        if expected != {r["digest"]: r["bytes"] for r in data["blobs"]}:
            raise ValueError("B0 archive inventory differs from the original OCI manifest")
    print("B0_READY", out, flush=True)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--verify-release", action="store_true")
    args = parser.parse_args()
    verify_release(args.cache) if args.verify_release else prepare(args.cache)
