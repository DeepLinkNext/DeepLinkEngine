"""The baseline must not inherit optimized Engine plugins or lose ownership leases."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from baseline_entry import baseline_environment, model_view
from baseline_admission import validate_identity


class BaselineEnvironmentTests(unittest.TestCase):
    def test_optimized_plugins_are_removed_while_ownership_is_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inherited = {
                "BETTERSCALE_GDN_LIBRARY": "foreign-optimized-library",
                "HUST_MICROFLOW_ARM": "on",
                "MTP_GDN_LAYOUT_FUSION": "1",
                "KVMAT_RUN_DIR": "optimized-state",
                "VLLM_HUST_EXT_CONFIG": "optimized-manager",
                "VLLM_KV_POLICY_MODE": "controller",
                "VLLM_HUST_TEST_OUTPUT": "owned-run",
                "VLLMHUST_ENGINE_LOCKS": "inherited-fd-proof",
                "PYTHONPATH": "cann-only",
            }
            (root / "out").mkdir()
            with patch("baseline_entry.environment", return_value=inherited):
                env = baseline_environment(root, root / "out", root / "cache", "2,3")
            self.assertEqual(
                env["VLLM_PLUGINS"], "ascend,ascend_model_loader,ascend_kv_connector"
            )
            self.assertEqual(env["VLLMHUST_ENGINE_LOCKS"], "inherited-fd-proof")
            self.assertEqual(env["VLLM_HUST_TEST_OUTPUT"], str(root / "out"))
            self.assertFalse(
                any(
                    k.startswith(("BETTERSCALE", "HUST", "MTP", "KVMAT", "VLLM_KV"))
                    for k in env
                )
            )

    def test_model_view_links_are_outside_sealed_evidence(self):
        import json

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, cache, out = root / "model", root / "cache", root / "run"
            for path in (model, cache, out):
                path.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "text_config": {
                            "model_type": "qwen3_5_moe_text",
                            "rope_parameters": {
                                "mrope_section": [11, 11, 10],
                                "mrope_interleaved": True,
                                "rope_theta": 10000000,
                            },
                        }
                    }
                )
            )
            (model / "weights.safetensors").write_bytes(
                b"synthetic fixture; not model weights"
            )
            view = model_view(model, cache, out, False)
            self.assertTrue((view / "weights.safetensors").is_symlink())
            self.assertFalse(any(p.is_symlink() for p in out.rglob("*")))
            rope = json.loads((view / "config.json").read_bytes())["rope_parameters"]
            self.assertEqual(rope, {"rope_theta": 10000000})
            self.assertEqual(
                json.loads((out / "model-config-normalization.json").read_bytes())[
                    "removed_text_only_rope_fields"
                ]["mrope_section"],
                [11, 11, 10],
            )
            self.assertEqual(
                json.loads((out / "model-config-view.json").read_bytes())[
                    "architectures"
                ],
                ["Qwen3_5MoeForCausalLM"],
            )

    def test_lib64_alias_is_accepted_but_another_package_origin_is_not(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "lib").mkdir()
            (root / "lib64").symlink_to("lib", target_is_directory=True)
            for package in ("vllm", "vllm_ascend"):
                (root / "lib" / package).mkdir()
                (root / "lib" / package / "__init__.py").write_text("# synthetic")
            origins = {
                p: str(root / "lib" / p / "__init__.py")
                for p in ("vllm", "vllm_ascend")
            }
            actual = {
                "role": "B0",
                "vllm_version": "0.18.0",
                "ascend_version": "0.18.0",
                "vllm_source": str(root / "lib64/vllm/__init__.py"),
                "ascend_source": str(root / "lib64/vllm_ascend/__init__.py"),
            }
            validate_identity(actual, {"origins": origins})
            with self.assertRaisesRegex(ValueError, "family"):
                validate_identity(
                    {**actual, "vllm_version": "0.28.1"}, {"origins": origins}
                )
            with self.assertRaisesRegex(ValueError, "import origin"):
                validate_identity(
                    {**actual, "vllm_source": str(root / "another/vllm/__init__.py")},
                    {"origins": origins},
                )


if __name__ == "__main__":
    unittest.main()
