"""Grammar routing and reuse tests; no generated answers or model scores."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import judge_grammar as grammar
import reuse_campaign as reuse
import third_party


class JudgeGrammarTests(unittest.TestCase):
    def spec(self):
        return {
            "dataset": "hle",
            "stage": "judge",
            "service_role": "B1",
            "structured_output_policy": grammar.BOUNDED,
        }

    def test_new_cli_defaults_to_fixed_policy_and_retains_explicit_history(self):
        for extra, expected in (
            ([], "bounded-v2"),
            (["--hle-judge-whitespace", "unrestricted"], "unrestricted"),
        ):
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "third_party.py",
                        "--model",
                        "/synthetic/model",
                        "--out",
                        "/synthetic/out",
                        *extra,
                    ],
                ),
                patch.object(third_party, "run", return_value=0) as run,
            ):
                third_party.main()
            self.assertEqual(run.call_args.args[0].hle_judge_whitespace, expected)

    def test_historical_missing_field_means_original_unrestricted_policy(self):
        self.assertEqual(grammar.selected({}), grammar.LEGACY)
        self.assertEqual(
            grammar.selected({"hle_judge_whitespace": "bounded-v1"}),
            grammar.WHITESPACE_ONLY,
        )
        spec = self.spec()
        del spec["structured_output_policy"]
        self.assertEqual(grammar.requested(spec), grammar.LEGACY)
        command, plan = ["serve"], {}
        grammar.configure(spec, command, plan)
        self.assertEqual(command, ["serve"])
        self.assertEqual(
            grammar.audit(spec, plan, {"command": command})["policy"], grammar.LEGACY
        )

    def test_configuration_must_actually_reach_the_server(self):
        spec = self.spec()
        command, plan = ["serve"], {}
        grammar.configure(spec, command, plan)
        self.assertEqual(json.loads(command[-1]), grammar.CONFIG)
        self.assertTrue(
            grammar.audit(spec, plan, {"command": command})[
                "backend_configuration_verified"
            ]
        )
        with self.assertRaisesRegex(ValueError, "launch differs"):
            grammar.audit(spec, plan, {"command": ["serve"]})
        with self.assertRaisesRegex(ValueError, "serving plan"):
            grammar.audit(spec, {}, {"command": command})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            grammar.configure(spec, command, plan)

    def test_confidence_generation_matches_existing_parser_domain_only(self):
        original = {
            "type": "json_schema",
            "json_schema": {
                "name": "ExtractedAnswer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "confidence": {"type": "integer", "title": "Confidence"},
                        "reasoning": {"type": "string"},
                    },
                },
            },
        }
        effective = grammar.effective_format(self.spec(), original)
        confidence = effective["json_schema"]["schema"]["properties"]["confidence"]
        self.assertEqual(
            (confidence.pop("minimum"), confidence.pop("maximum")), (0, 100)
        )
        self.assertEqual(effective, original)
        self.assertNotIn(
            "minimum", original["json_schema"]["schema"]["properties"]["confidence"]
        )
        for policy in (grammar.LEGACY, grammar.WHITESPACE_ONLY):
            spec = dict(self.spec(), structured_output_policy=policy)
            self.assertEqual(grammar.effective_format(spec, original), original)

    def test_unrecognized_confidence_schema_is_not_silently_rewritten(self):
        with self.assertRaisesRegex(ValueError, "Unexpected original"):
            grammar.effective_format(
                self.spec(),
                {
                    "json_schema": {
                        "name": "ExtractedAnswer",
                        "schema": {"properties": {"confidence": {"type": "number"}}},
                    }
                },
            )

    def test_solver_and_other_grading_protocols_are_unchanged(self):
        for name, stage in (("hle", "solver"), ("frontier", "judge")):
            spec = {"dataset": name, "stage": stage, "service_role": "B1"}
            command, plan = ["serve"], {}
            grammar.configure(spec, command, plan)
            self.assertEqual((command, plan), (["serve"], {}))
            spec["structured_output_policy"] = grammar.BOUNDED
            with self.assertRaisesRegex(ValueError, "HLE B1 judge only"):
                grammar.configure(spec, command, plan)

    def test_changing_grammar_skips_only_old_judges_and_keeps_solver_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            out, origin = Path(temp) / "new", Path(temp) / "old"
            out.mkdir()
            origin.mkdir()
            old = dict(model="/synthetic/model", physical_cards=[1, 6], baseline_source_map_sha256="synthetic", optimized_deployment="/synthetic/sdk", optimized_dependencies={})
            new = dict(old, hle_judge_whitespace="bounded-v2")
            names = ["hle-solver-B0", "hle-solver-B1", "hle-judge-B0", "hle-judge-B1"]
            receipt = {
                "status": "completed",
                "stages": [{"name": n, "status": "completed"} for n in names],
            }
            with (
                patch.object(reuse, "source", return_value=(origin, old, receipt, {})),
                patch.object(reuse, "compatible") as check,
                patch.object(reuse, "audit_stage"),
                patch.object(reuse, "sha", return_value="synthetic"),
                patch.object(reuse, "verify_weights", return_value={}),
            ):
                accepted = reuse.prepare(
                    [origin], out, new, [{"name": n} for n in names]
                )
            self.assertEqual(set(accepted), {"hle-solver-B0", "hle-solver-B1"})
            self.assertEqual(
                [c.args[0] for c in check.call_args_list],
                ["hle-solver-B0", "hle-solver-B1"],
            )
            self.assertEqual(
                reuse.read(out / "reused-stages.json")["new_inference_stages"],
                ["hle-judge-B0", "hle-judge-B1"],
            )

    def test_both_judge_arms_receive_the_same_version_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "plans").mkdir()
            args = SimpleNamespace(
                devices="0,1",
                physical_devices="5,6",
                model=root,
                report_python=None,
                continuous_evidence=None,
                smoke_evaluators=False,
            )
            frozen = {
                "optimized_deployment": "synthetic",
                "baseline_deployment": "synthetic",
                "datasets": ["hle"],
                "hle_judge_whitespace": "bounded-v2",
            }

            def descriptor(
                dataset, plan, role, stage, solver_role, out, diagnostic, responses=None
            ):
                return {
                    "dataset": dataset,
                    "stage": stage,
                    "service_role": role,
                    "solver_role": solver_role,
                }

            with patch.object(third_party, "descriptor", side_effect=descriptor):
                stages = third_party.build_stages(args, root, frozen)
            self.assertEqual(len(stages), 6)
            for role in ("B0", "B1"):
                spec = reuse.read(root / "plans" / ("hle-judge-" + role + ".json"))
                self.assertEqual(spec["structured_output_policy"], grammar.BOUNDED)
                self.assertEqual(spec["solver_role"], role)


if __name__ == "__main__":
    unittest.main()
