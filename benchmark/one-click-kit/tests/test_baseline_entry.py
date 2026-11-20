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


class BaselineSourcePreparationTests(unittest.TestCase):
    def _arguments(self, root):
        return ["baseline_entry.py", "--deployment", str(root / "sdk"),
                "--out", str(root / "service"), "--cache", str(root / "cache"),
                "--devices", "0,1", "--model", str(root / "model"),
                "--baseline-deployment", str(root / "baseline"),
                "--phase", "serve", "--kind", "continuous"]

    def test_real_entry_keeps_slow_verification_before_lease_and_model_exec(self):
        import json
        import os
        import baseline_entry
        from run_b1_r4 import service_admission_phase
        from contextlib import ExitStack

        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            out = root / "service"
            seen = []

            def verify(deployment, receipt, variant):
                self.assertEqual(deployment, root / "baseline")
                self.assertEqual(receipt, out / "baseline-source-admission.json")
                self.assertEqual(service_admission_phase(out, "B0", 100, 1800, clock=lambda: 2401),
                                 "verifying-baseline")
                self.assertFalse((out / "resource-admission.json").exists())
                seen.append("full-source-verification")
                return {"python": "/fixture/original-python"}

            def acquire(out, timeout):
                self.assertEqual(service_admission_phase(out, "B0", 100, timeout, clock=lambda: 2502),
                                 "admitting")
                seen.append("resource-acquisition")

            stack.enter_context(patch.object(sys, "argv", self._arguments(root)))
            stack.enter_context(patch.dict(os.environ, {"VLLMHUST_BASELINE_BOOTSTRAPPED": ""}))
            stack.enter_context(patch("baseline_entry.time.monotonic", side_effect=[101, 2501]))
            stack.enter_context(patch("baseline_entry.verify_installed_sources"))
            stack.enter_context(patch("baseline_entry.admit", side_effect=verify))
            stack.enter_context(patch("baseline_entry.acquire", side_effect=acquire))
            stack.enter_context(patch("baseline_entry.baseline_environment", return_value={}))
            execute = stack.enter_context(patch("baseline_entry.os.execve", side_effect=RuntimeError("fixture-exec-boundary")))
            with self.assertRaisesRegex(RuntimeError, "fixture-exec-boundary"):
                baseline_entry.main()
            self.assertEqual(seen, ["full-source-verification", "resource-acquisition"])
            self.assertEqual(execute.call_args.args[0], "/fixture/original-python")
            progress = json.loads((out / "source-preparation.json").read_bytes())
            self.assertEqual(progress["status"], "completed")
            self.assertEqual(progress["completed_monotonic"], 2501)
            self.assertFalse((out / "source-preparation.json.tmp").exists())

    def test_source_mismatch_still_prevents_resource_acquisition_and_exec(self):
        import os
        import baseline_entry
        from contextlib import ExitStack

        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            stack.enter_context(patch.object(sys, "argv", self._arguments(root)))
            stack.enter_context(patch.dict(os.environ, {"VLLMHUST_BASELINE_BOOTSTRAPPED": ""}))
            stack.enter_context(patch("baseline_entry.verify_installed_sources"))
            stack.enter_context(patch("baseline_entry.admit", side_effect=ValueError("Original018 SDK source changed")))
            acquire = stack.enter_context(patch("baseline_entry.acquire"))
            execute = stack.enter_context(patch("baseline_entry.os.execve"))
            with self.assertRaisesRegex(ValueError, "SDK source changed"):
                baseline_entry.main()
            acquire.assert_not_called()
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
