"""CPU fixtures exercise source lineage, never benchmark outputs."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import restore_b1_r4 as restore


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(root, name, raw):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


@contextmanager
def source_fixture():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        helper = "mods/correction/pinned_metadata_signature.py"
        contract = "mods/microflow/hust_prefill_microflow/school028/host-contract.json"
        runtime = "runtime/ascend/vllm_ascend/worker/model_runner_v1.py"
        old_helper = ("from pathlib import Path\n" + restore.ABI_OLD + "\n").encode()
        new_helper = old_helper.replace(
            restore.ABI_OLD.encode(), restore.ABI_NEW.encode()
        )
        original_contract = {
            "versions": {
                "vllm-ascend": "0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909"
            },
            "sources": {"fixture": "pin"},
        }
        raw_contract = (json.dumps(original_contract, indent=2) + "\n").encode()
        runtime_bytes = b"pinned-runtime-fixture"
        source = {
            "/root/m01-storage-timing-runtime-20261024/pinned_metadata_signature.py": sha(
                old_helper
            )
        }
        components = {
            "files": {
                restore.RUNTIME_PREFIX
                + "microflow-recovery-payload-r2/hust_prefill_microflow/school028/host-contract.json": sha(
                    raw_contract
                ),
                restore.RUNTIME_PREFIX
                + "runtime/vllm_ascend/worker/model_runner_v1.py": sha(runtime_bytes),
            }
        }
        manifests = {
            "source-manifest.json": json.dumps(source).encode(),
            "component-manifest.json": json.dumps(components).encode(),
        }
        for name, raw in manifests.items():
            save(root, "provenance/" + name, raw)
        save(root, helper, new_helper)
        save(root, contract, raw_contract)
        save(root, runtime, runtime_bytes)
        save(
            root,
            restore.ABI_SOURCE,
            (
                Path(restore.__file__).resolve().parents[1] / restore.ABI_SOURCE
            ).read_bytes(),
        )
        files = {
            helper: sha(new_helper),
            contract: sha(raw_contract),
            runtime: sha(runtime_bytes),
            restore.ABI_SOURCE: restore.ABI_SHA256,
        }
        save(root, "provenance/installed-files.json", json.dumps(files).encode())
        save(
            root,
            "deployment.json",
            json.dumps(
                {
                    "root": str(root),
                    "release": "full-answer-b1-20261030-r4",
                    "pythonpath": [
                        str(root / name) for name in restore.PYTHONPATH_LAYOUT
                    ],
                }
            ).encode(),
        )
        original_plan = {"synthetic_fixture": True}
        save(root, "provenance/original-plan.json", json.dumps(original_plan).encode())
        with patch.multiple(
            restore,
            MANIFEST_SHA256={name: sha(raw) for name, raw in manifests.items()},
            ORIGINAL_PLAN=sha(json.dumps(original_plan, indent=2).encode()),
        ):
            yield root, files


class SourceAdmissionTests(unittest.TestCase):
    def test_original_manifest_derivation_accepts_only_the_pinned_files(self):
        with source_fixture() as (root, files):
            self.assertEqual(restore.verify_installed_sources(root), files)

    def test_modified_runtime_cannot_be_blessed_by_editing_its_local_manifest(self):
        with source_fixture() as (root, files):
            name = "runtime/ascend/vllm_ascend/worker/model_runner_v1.py"
            save(root, name, b"changed-runtime")
            files[name] = sha(b"changed-runtime")
            save(root, "provenance/installed-files.json", json.dumps(files).encode())
            with self.assertRaisesRegex(
                ValueError, "does not derive from the pinned original"
            ):
                restore.verify_installed_sources(root)

    def test_mutating_original_manifest_is_rejected_even_if_json_is_equivalent(self):
        with source_fixture() as (root, _):
            path = root / "provenance/source-manifest.json"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Pinned source manifest changed"):
                restore.verify_installed_sources(root)

    def test_abi_helper_must_only_change_the_path(self):
        with source_fixture() as (root, _):
            path = root / "mods/correction/pinned_metadata_signature.py"
            path.write_bytes(path.read_bytes() + b"# unreviewed change\n")
            with self.assertRaisesRegex(ValueError, "ABI helper differs beyond"):
                restore.verify_installed_sources(root)

    def test_host_contract_date_label_can_change_but_source_pins_cannot(self):
        with source_fixture() as (root, files):
            name = "mods/microflow/hust_prefill_microflow/school028/host-contract.json"
            value = json.loads((root / name).read_bytes())
            value["versions"]["vllm-ascend"] = value["versions"][
                "vllm-ascend"
            ].removesuffix(".d20260909")
            raw = (json.dumps(value, indent=2) + "\n").encode()
            save(root, name, raw)
            files[name] = sha(raw)
            save(root, "provenance/installed-files.json", json.dumps(files).encode())
            self.assertEqual(restore.verify_installed_sources(root), files)
            value["sources"] = {}
            save(root, name, (json.dumps(value, indent=2) + "\n").encode())
            with self.assertRaisesRegex(ValueError, "Host contract differs beyond"):
                restore.verify_installed_sources(root)

    def test_python_search_paths_cannot_point_outside_the_deployment(self):
        with source_fixture() as (root, _):
            path = root / "deployment.json"
            value = json.loads(path.read_bytes())
            value["pythonpath"].insert(0, "/unrelated-runtime")
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(
                ValueError, "Deployment paths or release identity differ"
            ):
                restore.verify_installed_sources(root)


if __name__ == "__main__":
    unittest.main()
