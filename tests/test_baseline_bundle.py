"""Exercise offline OCI reassembly and fail-closed original-image integrity."""
import io
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

    def test_successful_restore_removes_only_local_staging_and_retains_all_originals(self):
        sys.path.insert(0, str(ROOT / "benchmark/one-click-kit/scripts"))
        raw = self.root / "persistent-originals"
        raw.mkdir()
        for identity, body in self.originals.items():
            (raw / identity.replace(":", "-")).write_bytes(body)
        captured = []
        def restored(folder, name, explicit, build):
            captured.append(folder)
            (folder / "oci").mkdir(parents=True)
            for identity, body in self.originals.items():
                (folder / "oci" / identity.replace(":", "-")).write_bytes(body)
            (folder / "oci/unrelated").write_bytes(b"retained")
            sdk = folder / "original018-sdk"
            sdk.mkdir()
            return sdk
        with patch.object(bundle, "prepare", return_value=raw), \
             patch.object(bundle, "inventory", return_value=self.index), \
             patch.object(bundle, "local_runtime_cache", return_value=self.root / "local"), \
             patch("third_party.cached_runtime", side_effect=restored):
            sdk = bundle.ensure_deployment(self.cache)
        self.assertTrue(sdk.is_dir())
        for identity, body in self.originals.items():
            self.assertFalse((captured[0] / "oci" / identity.replace(":", "-")).exists())
            self.assertEqual((raw / identity.replace(":", "-")).read_bytes(), body)
        self.assertEqual((captured[0] / "oci/unrelated").read_bytes(), b"retained")
        self.assertTrue(json.loads((captured[0] / "staging-cleanup.json").read_text())["persistent_originals_retained"])

    def test_failed_restore_does_not_remove_local_staging(self):
        sys.path.insert(0, str(ROOT / "benchmark/one-click-kit/scripts"))
        captured = []
        def failed(folder, name, explicit, build):
            (folder / "oci").mkdir(parents=True)
            target = folder / "oci" / next(iter(self.originals)).replace(":", "-")
            target.write_bytes(b"retained failed-stage original")
            captured.append(target)
            raise RuntimeError("restore failed")
        with patch.object(bundle, "prepare", return_value=self.root / "raw"), \
             patch.object(bundle, "local_runtime_cache", return_value=self.root / "local"), \
             patch("third_party.cached_runtime", side_effect=failed):
            with self.assertRaisesRegex(RuntimeError, "restore failed"):
                bundle.ensure_deployment(self.cache)
        self.assertEqual(captured[0].read_bytes(), b"retained failed-stage original")

    def test_metadata_credential_is_never_forwarded_to_anonymous_original_downloads(self):
        self.index["release"] = "fixture-v1"
        assets, bodies = [], {}
        for kind, blob in zip(["manifest", "config", "layer"], self.index["blobs"]):
            blob["kind"] = kind
            for part in blob["parts"]:
                number = len(assets) + 1
                part["url"] = "https://fixture.invalid/" + part["name"]
                part["api_url"] = "https://api.github.com/repos/DeepLinkNext/DeepLinkEngine/releases/assets/" + str(number)
                assets.append(dict(name=part["name"], size=part["bytes"], digest="sha256:" + part["sha256"], id=number))
                bodies[part["api_url"]] = (self.parts / part["name"]).read_bytes()
        for credential in ("", "fixture-read-only-token"):
            with self.subTest(authenticated_metadata=bool(credential)):
                downloads = []
                def fetch(request, **kwargs):
                    headers = {key.lower(): value for key, value in request.header_items()}
                    if "/releases/tags/" in request.full_url:
                        self.assertEqual(headers.get("authorization"), "Bearer " + credential if credential else None)
                        return io.BytesIO(json.dumps(dict(draft=False, assets=assets)).encode())
                    self.assertNotIn("authorization", headers)
                    downloads.append(request.full_url)
                    return io.BytesIO(bodies[request.full_url])
                with patch.object(bundle, "inventory", return_value=self.index), \
                     patch.dict(os.environ, {"GH_TOKEN": credential}, clear=True), \
                     patch("urllib.request.urlopen", side_effect=fetch):
                    bundle.verify_release(self.root / ("auth" if credential else "anon"))
                self.assertTrue(downloads)

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
