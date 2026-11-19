import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compile_boundary
import runtime_overlay
import serving_warmup
from reuse_campaign import identity


class RuntimeRefreezeTests(unittest.TestCase):
    def boundary(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        handler = SimpleNamespace(start_callbacks=[])
        dynamo = SimpleNamespace(config=SimpleNamespace(suppress_errors=False))
        dynamo.on_compile_start = handler.start_callbacks.append
        worker = SimpleNamespace(
            rank=0, _compilation_scope=SimpleNamespace(check=lambda: None)
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "torch": SimpleNamespace(_dynamo=dynamo),
                    "torch._dynamo": dynamo,
                    "torch._dynamo.callback": SimpleNamespace(callback_handler=handler),
                },
            ),
            patch.dict("os.environ", {"VLLM_HUST_TEST_OUTPUT": self.temp.name}),
        ):
            scope = compile_boundary.Boundary(worker)
        scope.event_path = self.temp.name
        return scope, handler

    def test_post_ready_torch_failure_latches(self):
        scope, _ = self.boundary()
        with patch.dict("os.environ", {"VLLM_HUST_TEST_OUTPUT": self.temp.name}):
            scope.on_torch()
            self.assertEqual(scope.status()["observed_compilations"], 1)
            scope.sealed = True
            with self.assertRaisesRegex(RuntimeError, "after measurement"):
                scope.on_torch()
            with self.assertRaisesRegex(RuntimeError, "after measurement"):
                scope.status()

    def test_callback_replacement_rejected(self):
        scope, handler = self.boundary()
        handler.start_callbacks.clear()
        with self.assertRaisesRegex(RuntimeError, "replaced"):
            scope.status()

    def test_eager_fallback_rejected(self):
        scope, _ = self.boundary()
        scope.dynamo.config.suppress_errors = True
        with self.assertRaisesRegex(RuntimeError, "replaced"):
            scope.status()

    def test_native_triton_owner_is_preserved(self):
        scope, _ = self.boundary()
        self.assertEqual(scope.status()["triton_guard_owner"], "runtime")
        scope.native.check = lambda: (_ for _ in ()).throw(RuntimeError("native miss"))
        with self.assertRaisesRegex(RuntimeError, "native miss"):
            scope.status()

    def test_overlay_does_not_admit_modified_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = runtime_overlay.manifest()
            (root / "engines").mkdir()
            (root / "engines/b1-r5-overlay.json").write_text(json.dumps(value))
            target = (
                root
                / "overlays/b1-r5/mods/betterscale/betterscale/compilation_scope.py"
            )
            target.parent.mkdir(parents=True)
            target.write_text("# changed")
            with (
                patch.object(runtime_overlay, "ROOT", root),
                patch.object(
                    runtime_overlay, "MANIFEST", root / "engines/b1-r5-overlay.json"
                ),
            ):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    runtime_overlay.manifest()

    def test_r4_and_r5_warmup_evidence_cannot_be_reused(self):
        common = dict(
            model="m",
            physical_cards=[5, 6],
            baseline_source_map_sha256="b",
            optimized_deployment="/same-path",
            optimized_dependencies={},
        )
        self.assertNotEqual(
            identity(common),
            identity(
                dict(
                    common,
                    engine_release=runtime_overlay.RELEASE,
                    serving_warmup=serving_warmup.protocol(),
                )
            ),
        )

    def test_historical_profile_does_not_silently_change(self):
        self.assertFalse(
            serving_warmup.selected({"release": "full-answer-b1-20261030-r4"})
        )
        self.assertTrue(serving_warmup.selected({"release": runtime_overlay.RELEASE}))

    def test_warmup_preserves_frozen_chat_path_and_original_case(self):
        case = {
            "request": {
                "model": "qwen3.5-35b",
                "messages": [{"role": "user", "content": "synthetic"}],
                "seed": 17,
                "temperature": 0,
                "max_tokens": 256,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        }
        body = serving_warmup.warmup_body(case, "fresh-synthetic-salt")
        self.assertEqual(body["messages"], case["request"]["messages"])
        self.assertNotIn("prompt", body)
        self.assertEqual(body["seed"], 17)
        self.assertEqual(body["max_tokens"], 32)
        self.assertEqual(case["request"]["max_tokens"], 256)
        self.assertEqual(body["cache_salt"], "fresh-synthetic-salt")

    def test_completion_path_cannot_masquerade_as_chat_warmup(self):
        with self.assertRaisesRegex(ValueError, "chat control path"):
            serving_warmup.warmup_body({"request": {"prompt": [1000]}}, "salt")


if __name__ == "__main__":
    unittest.main()
