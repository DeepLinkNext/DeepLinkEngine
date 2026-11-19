"""Synthetic reuse guards: no model startup or performance claims."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reuse_campaign as reuse


def plan():
    return dict(
        model="/synthetic/model",
        physical_cards=[0, 1],
        baseline_source_map_sha256="base",
        optimized_deployment="/synthetic/runtime",
        optimized_dependencies={"metadata": "fixed"},
        mmlu_inputs={"quality_plan_sha256": "tasks", "scorer_sha256": "scorer"},
    )


class ReuseTests(unittest.TestCase):
    def test_task_warmup_revision_cannot_reuse_old_task_performance(self):
        previous = plan()
        current = dict(previous, task_serving_warmup={"policy": "task-v3"})
        with self.assertRaisesRegex(ValueError, "task warmup protocol differs"):
            reuse.compatible("mmlu-pair", previous, current, Path("/synthetic"))
        reuse.compatible("continuous-B1", previous, current, Path("/synthetic"))

    def test_b1_only_fix_reuses_sealed_baseline_without_relabeling(self):
        import serving_warmup

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "continuous-B0").mkdir()
            recipe = root / "new-runtime/runtime/entrypoints/prepare_continuous.py"
            recipe.parent.mkdir(parents=True)
            recipe.write_text("# same frozen C8 recipe\n")
            reuse.save(root / "continuous-B0/receipt.json", {
                "original_input_recipe": {"recipe_sha256": reuse.sha(recipe)}})
            old_source = root / "execution-source/scripts/serving_warmup.py"
            old_source.parent.mkdir(parents=True)
            old_source.write_text((Path(reuse.__file__).parent / "serving_warmup.py").read_text().replace(
                'return profile["release"] in ("full-answer-b1-20261113-r5", "full-answer-b1-20261113-r6")',
                'return profile["release"] == "full-answer-b1-20261113-r5"'))
            previous = dict(plan(), engine_release="full-answer-b1-20261113-r5",
                            baseline_deployment="/same/b0", logical_devices="0,1",
                            serving_warmup=serving_warmup.protocol(),
                            sources={n: "same" for n in (
                                "scripts/compile_boundary.py", "scripts/baseline_entry.py",
                                "scripts/baseline_admission.py", "scripts/continuous_inputs.py")})
            current = copy.deepcopy(previous)
            current.update(optimized_deployment=str(recipe.parents[2]),
                           engine_release="full-answer-b1-20261113-r6")
            saved_previous = copy.deepcopy(previous)
            reuse.compatible("continuous-B0", previous, current, root)
            self.assertEqual(previous, saved_previous)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                reuse.compatible("continuous-B1", previous, current, root)
            for key in ("baseline_deployment", "physical_cards", "logical_devices",
                        "baseline_source_map_sha256", "model", "serving_warmup"):
                changed = copy.deepcopy(current)
                changed[key] = "different"
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity differs"):
                    reuse.compatible("continuous-B0", previous, changed, root)
            recipe.write_text("# different recipe\n")
            with self.assertRaisesRegex(ValueError, "input recipe differs"):
                reuse.compatible("continuous-B0", previous, current, root)

    def test_reject_runtime_or_task_change(self):
        previous = plan()
        for key in ["physical_cards", "optimized_dependencies", "model"]:
            current = copy.deepcopy(previous)
            current[key] = "different"
            with self.assertRaisesRegex(ValueError, "identity differs"):
                reuse.compatible("mmlu-pair", previous, current, Path("/synthetic"))
        current = copy.deepcopy(previous)
        current["mmlu_inputs"]["quality_plan_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "inputs or scorer"):
            reuse.compatible("mmlu-pair", previous, current, Path("/synthetic"))

    def test_judge_B0_is_not_a_baseline_service(self):
        previous = dict(plan(), engine_release="old", task_serving_warmup={"same":True})
        current = dict(previous, engine_release="new")
        for name in ("hle-judge-B0", "frontier-judge-B0"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "identity differs"):
                reuse.compatible(name, previous, current, Path("/synthetic"))

    def test_manual_arm_selection_overrides_automatic_whole_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, origin = Path(tmp) / "new", Path(tmp) / "old"
            out.mkdir()
            origin.mkdir()
            data = plan()
            receipt = {"status":"completed", "stages":[{"name":"mmlu-pair", "status":"completed"}]}
            stages = [{"name":"mmlu-pair", "args":["--reuse-baseline-arm", "/explicit/B0"]}]
            with (
                patch.object(reuse, "source", return_value=(origin, data, receipt, {})),
                patch.object(reuse, "sha", return_value="synthetic"),
                patch.object(reuse, "audit_stage") as audit,
            ):
                self.assertEqual(reuse.prepare([origin], out, data, stages), {})
                audit.assert_not_called()
            self.assertEqual(reuse.read(out / "reused-stages.json")["stages"], {})

    def test_partial_mmlu_B0_reference_is_frozen_before_new_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            reuse.save(out / "reused-stages.json", {"paired_arms":{
                "B0":{"binding":{"run":"/original/B0", "seal_sha256":"original-seal"}}}})
            stages = [{"name":"mmlu-pair", "args":[]}]
            reuse.wire_dependencies(out, stages, {})
            self.assertEqual(stages[0]["args"], ["--reuse-baseline-arm", "/original/B0",
                "--reuse-baseline-seal", "original-seal"])
            self.assertEqual(reuse.read(out / "frozen-stages.json"), stages)

    def test_duplicate_completed_candidates_are_not_best_run_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new"
            out.mkdir()
            origin = Path(tmp) / "old"
            origin.mkdir()
            data = plan()
            receipt = {
                "status": "failed",
                "stages": [{"name": "mmlu-pair", "status": "completed"}],
            }
            with (
                patch.object(reuse, "source", return_value=(origin, data, receipt, {})),
                patch.object(reuse, "audit_stage"),
                patch.object(reuse, "sha", return_value="synthetic"),
            ):
                with self.assertRaisesRegex(ValueError, "Ambiguous"):
                    reuse.prepare([origin, origin], out, data, [{"name": "mmlu-pair"}])

    def test_failed_stage_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new"
            out.mkdir()
            origin = Path(tmp) / "old"
            origin.mkdir()
            receipt = {
                "status": "failed",
                "stages": [{"name": "mmlu-pair", "status": "failed"}],
            }
            with (
                patch.object(
                    reuse, "source", return_value=(origin, plan(), receipt, {})
                ),
                patch.object(reuse, "sha", return_value="synthetic"),
            ):
                with self.assertRaisesRegex(ValueError, "no compatible completed"):
                    reuse.prepare([origin], out, plan(), [{"name": "mmlu-pair"}])

    def test_reused_trace_baseline_is_wired_before_new_stage_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            spec = out / "b1.json"
            reuse.save(
                spec,
                {
                    "stage": "trace",
                    "service_role": "B1",
                    "dataset": "swe",
                    "baseline_trace_run": "not-created",
                },
            )
            stages = [
                {
                    "name": "swe-trace-B1",
                    "args": [
                        "--dataset-spec",
                        str(spec),
                        "--qualified-continuous",
                        "not-created",
                    ],
                }
            ]
            reuse.wire_dependencies(
                out,
                stages,
                {
                    "continuous-B1": {"source_root": "/synthetic/c8"},
                    "swe-trace-B0": {"source_root": "/synthetic/baseline"},
                },
            )
            self.assertEqual(
                reuse.read(spec)["baseline_trace_run"],
                "/synthetic/baseline/swe-trace-B0",
            )
            self.assertEqual(stages[0]["args"][-1], "/synthetic/c8/continuous-B1")
