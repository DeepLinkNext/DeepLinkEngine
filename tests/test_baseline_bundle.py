"""Exercise offline OCI reassembly and fail-closed original-image integrity."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import baseline_bundle as bundle


class BaselineBundleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.parts = self.root / "parts"
        self.parts.mkdir()
        config, layer = b'{"architecture":"arm64","os":"linux"}', b"original compressed OCI layer bytes"
        record = lambda b: dict(digest="sha256:" + hashlib.sha256(b).hexdigest(), size=len(b))
        manifest = json.dumps(dict(config=record(config), layers=[record(layer)])).encode()
        self.originals = {record(b)["digest"]: b for b in [manifest, config, layer]}
        self.index = dict(manifest_digest=record(manifest)["digest"], blobs=[])
        for identity, body in self.originals.items():
            parts = []
            for i, offset in enumerate(range(0, len(body), 11)):
                data = body[offset:offset+11]
                name = "b0-oci-" + identity[7:] + f".part-{i+1:03d}"
                (self.parts / name).write_bytes(data)
                parts.append(dict(name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            self.index["blobs"].append(dict(digest=identity, bytes=len(body), parts=parts))
        self.cache = self.root / "cache"

    def prepare(self, extra=None):
        environment = dict(DEEPLINKENGINE_B0_ARCHIVES=str(self.parts))
        environment.update(extra or {})
        with patch.object(bundle, "inventory", return_value=self.index), \
             patch.object(Path, "home", return_value=self.root / "empty-home"), \
             patch.dict(os.environ, environment, clear=True), \
             patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            return bundle.prepare(self.cache)

    def test_offline_parts_reassemble_to_all_original_digest_pinned_blobs(self):
        out = self.prepare()
        for identity, body in self.originals.items():
            self.assertEqual((out / identity.replace(":", "-")).read_bytes(), body)
        self.prepare()

    def test_corrupt_part_is_not_admitted_and_temporary_bytes_are_removed(self):
        part = self.index["blobs"][0]["parts"][0]
        (self.parts / part["name"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Offline B0 archive"):
            self.prepare()
        self.assertFalse(list((self.cache / "b0-oci").glob("sha256-*")))
        self.assertFalse(list((self.cache / "b0-oci").glob("b0-*")))

    def test_individually_valid_parts_in_wrong_order_fail_the_original_blob_hash(self):
        self.index["blobs"][-1]["parts"].reverse()
        with self.assertRaisesRegex(ValueError, "Reassembled B0 OCI blob differs"):
            self.prepare()
        name = self.index["blobs"][-1]["digest"].replace(":", "-")
        self.assertFalse((self.cache / "b0-oci" / name).exists())

    def test_corrupted_existing_blob_is_retained_and_rejected(self):
        out = self.prepare()
        name = self.index["blobs"][-1]["digest"].replace(":", "-")
        (out / name).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Cached B0 OCI blob differs"):
            self.prepare()
        self.assertEqual((out / name).read_bytes(), b"corrupt")

    def test_hash_matching_raw_oci_cache_needs_neither_parts_nor_network(self):
        raw = self.root / "raw"
        raw.mkdir()
        for identity, body in self.originals.items():
            (raw / identity.replace(":", "-")).write_bytes(body)
        self.prepare(dict(DEEPLINKENGINE_B0_OCI_CACHE=str(raw),
                          DEEPLINKENGINE_B0_ARCHIVES=str(self.root / "absent")))

    def test_index_cannot_add_a_blob_outside_the_original_manifest(self):
        body = b"an individually valid blob absent from the original image"
        checksum = hashlib.sha256(body).hexdigest()
        name = "b0-oci-" + checksum + ".part-001"
        (self.parts / name).write_bytes(body)
        self.index["blobs"].append(dict(digest="sha256:" + checksum, bytes=len(body),
            parts=[dict(name=name, bytes=len(body), sha256=checksum)]))
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
