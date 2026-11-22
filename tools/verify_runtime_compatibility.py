"""Verify that the NumPy1.26 compatibility wheel preserves the official payload."""
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(cache):
    from engine.environment_setup import cached_wheel
    lock = json.loads((ROOT / "engine/r8-lock.json").read_text())
    record = lock["runtime_compatibility_sources"]["opencv-python-headless"]
    adapted = cached_wheel(record, [], Path(cache) / "wheels")
    source = dict(filename=unquote(Path(urlsplit(record["upstream_url"]).path).name),
                  url=record["upstream_url"], sha256=record["upstream_sha256"], bytes=record["upstream_bytes"])
    original = cached_wheel(source, [], Path(cache) / "original")
    with ZipFile(original) as before, ZipFile(adapted) as after:
        old = next(name.split("/")[0] for name in before.namelist() if name.endswith(".dist-info/METADATA"))
        new = next(name.split("/")[0] for name in after.namelist() if name.endswith(".dist-info/METADATA"))
        expected = set()
        unchanged = 0
        for info in before.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace(old + "/", new + "/", 1)
            expected.add(name)
            if name.endswith("/RECORD"):
                continue
            left, right = before.read(info.filename), after.read(name)
            if name.endswith("/METADATA"):
                def retained(raw):
                    return [line for line in raw.splitlines() if not line.startswith(
                        (b"Version: ", b"Requires-Python: ", b"Requires-Dist: numpy"))]
                if retained(left) != retained(right):
                    raise ValueError("Compatibility metadata changed unrelated fields")
                if (("Version: " + record["version"]).encode() not in right
                        or b"Requires-Dist: numpy==1.26.4\n" not in right
                        or b"Requires-Python: >=3.12,<3.13\n" not in right):
                    raise ValueError("Compatibility NumPy/version declaration differs")
                continue
            if left != right:
                raise ValueError("OpenCV payload changed: " + name)
            unchanged += 1
        if expected != set(after.namelist()) or len(after.namelist()) != len(expected):
            raise ValueError("Compatibility wheel file inventory differs")
        for name, encoded, size in csv.reader(io.StringIO(after.read(new + "/RECORD").decode())):
            if not encoded:
                if name != new + "/RECORD":
                    raise ValueError("Unexpected unhashed compatibility entry")
                continue
            raw = after.read(name)
            checksum = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
            if encoded != checksum or int(size) != len(raw):
                raise ValueError("Compatibility RECORD hash differs: " + name)
    print("COMPATIBILITY_VERIFIED", unchanged, "official payload files unchanged", flush=True)
    vision = dict(lock["environment_sources"]["torchvision"], version=lock["required_python_packages"]["torchvision"])
    wheel = cached_wheel(vision, [], Path(cache) / "wheels")
    with ZipFile(wheel) as archive:
        metadata = archive.read(next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))).decode()
        expected_torch = lock["required_python_packages"]["torch"].split("+", 1)[0]
        if ("Version: " + vision["version"] not in metadata
                or "Requires-Dist: torch (==" + expected_torch + ")" not in metadata):
            raise ValueError("TorchVision wheel is not paired with the frozen Torch")
    print("TORCHVISION_VERIFIED", vision["version"], "paired with Torch", expected_torch, flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    verify(parser.parse_args().cache)
