"""Restore the pinned historical018 ARM64 SDK without a Docker daemon."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

REGISTRY = "https://quay.io/v2/ascend/vllm-ascend/"
MANIFEST = "sha256:2467d259ff068ca4edd7b50b849152910f3c05e435f314639fc06bcc7a00f9fa"
PREFIXES = (
    "usr/local/",
    "etc/Ascend/",
    "vllm-workspace/vllm/",
    "vllm-workspace/vllm-ascend/",
)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(cache, kind, digest, size=None):
    target = cache / digest.replace(":", "-")
    expected = digest.removeprefix("sha256:")
    if target.is_file():
        if (size is None or target.stat().st_size == size) and sha(target) == expected:
            return target
        raise ValueError("Cached OCI blob changed: " + digest)
    request = urllib.request.Request(
        REGISTRY + kind + "/" + digest,
        headers={
            "Accept": "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json",
            "User-Agent": "vllm-hust-fixed018-restoration",
        },
    )
    partial = target.with_suffix(".partial")
    with (
        urllib.request.urlopen(request, timeout=180) as response,
        partial.open("wb") as stream,
    ):
        while block := response.read(4 * 1024 * 1024):
            stream.write(block)
            if size is not None and stream.tell() > size:
                raise ValueError("OCI blob exceeds pinned size")
    if (size is not None and partial.stat().st_size != size) or sha(
        partial
    ) != expected:
        raise ValueError("OCI digest mismatch: " + digest)
    partial.replace(target)
    print("OCI_VERIFIED", digest, target.stat().st_size, flush=True)
    return target


def selected(name):
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("Unsafe OCI member")
    name = str(pure)
    return any((name + "/").startswith(prefix) for prefix in PREFIXES)


def destination(root, name):
    path = root / str(PurePosixPath(name))
    if not path.parent.resolve().is_relative_to(root):
        raise ValueError("OCI parent escaped destination")
    return path


def remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def extract_layer(blob, root):
    with tarfile.open(blob) as archive:
        members = [m for m in archive.getmembers() if selected(m.name)]
        # OCI whiteouts apply to earlier layers, before this layer's additions.
        for member in members:
            path = destination(root, member.name)
            if path.name == ".wh..wh..opq":
                if path.parent.exists():
                    for old in path.parent.iterdir():
                        remove(old)
            elif path.name.startswith(".wh."):
                remove(path.with_name(path.name[4:]))
        links = []
        for member in members:
            path = destination(root, member.name)
            if path.name.startswith(".wh."):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                if path.is_symlink() or (path.exists() and not path.is_dir()):
                    remove(path)
                path.mkdir(exist_ok=True)
            elif member.isfile():
                remove(path)
                with archive.extractfile(member) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target, 4 * 1024 * 1024)
                path.chmod(member.mode & 0o777)
            elif member.issym():
                remove(path)
                linked = (
                    root / member.linkname.lstrip("/")
                    if member.linkname.startswith("/")
                    else path.parent / member.linkname
                )
                if not linked.resolve().is_relative_to(root):
                    raise ValueError("OCI symlink escaped image root")
                path.symlink_to(os.path.relpath(linked, path.parent))
            elif member.islnk():
                links.append((path, destination(root, member.linkname)))
        for path, linked in links:
            if not linked.resolve().is_relative_to(root) or not linked.is_file():
                raise ValueError("Unresolved OCI hard link")
            remove(path)
            os.link(linked, path)


def restore(cache, out):
    cache, out = cache.resolve(), out.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(out).free < 25 * 2**30:
        raise RuntimeError("Pinned SDK restoration requires25GiB free disk")
    manifest_path = fetch(cache, "manifests", MANIFEST)
    manifest = json.loads(manifest_path.read_bytes())
    config = manifest["config"]
    config_path = fetch(cache, "blobs", config["digest"], config["size"])
    image = json.loads(config_path.read_bytes())
    if image["architecture"] != "arm64" or image["os"] != "linux":
        raise ValueError("Pinned image architecture differs")
    layers = manifest["layers"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        blobs = list(
            pool.map(lambda x: fetch(cache, "blobs", x["digest"], x["size"]), layers)
        )
    rootfs = out / "rootfs"
    rootfs.mkdir()
    for index, blob in enumerate(blobs):
        extract_layer(blob, rootfs)
        print("OCI_LAYER_APPLIED", index + 1, len(blobs), flush=True)
    shutil.copyfile(manifest_path, out / "image-manifest.json")
    shutil.copyfile(config_path, out / "image-config.json")
    data = {
        "schema": "vllm-hust-fixed018-image/v1",
        "image": "quay.io/ascend/vllm-ascend:v0.18.0",
        "manifest_digest": MANIFEST,
        "cann": "8.5.1",
        "rootfs": str(rootfs),
        "layers": layers,
        "image_config_digest": config["digest"],
        "extracted_prefixes": PREFIXES,
        "scope": "Historical pinned image SDK restored; host driver retained; no Docker daemon or host-root changes. Model inference qualification remains separate.",
    }
    (out / "restoration.json").write_text(json.dumps(data, indent=2) + "\n")
    restore_system_libraries(cache, out)
    return data


def restore_system_libraries(cache, out):
    """Restore image ABI libraries while retaining the host glibc/loader pair."""
    cache, out = Path(cache).resolve(), Path(out).resolve()
    raw = (out / "image-manifest.json").read_bytes()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != MANIFEST:
        raise ValueError("System libraries require the pinned original image")
    dest = out / "system-libraries"
    dest.mkdir(exist_ok=False)
    included = (
        "libbz2.so",
        "libnuma.so",
        "libgomp.so",
        "libffi.so",
        "liblzma.so",
        "libsqlite3.so",
        "libssl.so",
        "libcrypto.so",
        "libreadline.so",
        "libtinfo.so",
        "libncursesw.so",
        "libuuid.so",
        "libz.so",
        "libstdc++.so",
        "libgcc_s.so",
        "libexpat.so",
        "libcrypt.so",
        "libatomic.so",
    )
    excluded = (
        "ld-linux",
        "libc.so",
        "libpthread",
        "libdl.so",
        "librt.so",
        "libm.so",
        "libmvec",
        "libresolv",
        "libnss_",
        "libutil.so",
        "libanl.so",
        "libBrokenLocale",
        "libthread_db",
    )
    for layer in json.loads(raw)["layers"]:
        blob = fetch(cache, "blobs", layer["digest"], layer["size"])
        with tarfile.open(blob) as archive:
            for item in archive:
                path = PurePosixPath(item.name)
                if str(path.parent) not in (
                    "lib/aarch64-linux-gnu",
                    "usr/lib/aarch64-linux-gnu",
                ):
                    continue
                name = path.name
                if not name.startswith(included) or name.startswith(excluded):
                    continue
                target = dest / name
                remove(target)
                if item.isfile():
                    with (
                        archive.extractfile(item) as source,
                        target.open("wb") as output,
                    ):
                        shutil.copyfileobj(source, output)
                    target.chmod(item.mode & 0o777)
                elif item.issym() or item.islnk():
                    linked = PurePosixPath(item.linkname).name
                    if linked.startswith(excluded):
                        continue
                    target.symlink_to(linked)
    if not (dest / "libbz2.so.1.0").is_file():
        raise ValueError("Pinned image bzip2 ABI library was not restored")
    print("IMAGE_SYSTEM_LIBRARIES_RESTORED", len(list(dest.iterdir())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = restore(args.cache, args.out)
    print("BASELINE_IMAGE_RESTORED", data["rootfs"], data["manifest_digest"])


if __name__ == "__main__":
    main()
