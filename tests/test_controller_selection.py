import argparse
import unittest
from benchmark import unified_run


class ControllerSelection(unittest.TestCase):
    def test_public_parser_accepts_comma_separated_types(self):
        parser = argparse.ArgumentParser()
        unified_run.arguments(parser)
        args = parser.parse_args(['--engine', 'b1-r8', '--config', 'workload', '--site', 'site',
                                  '--controllers', 'kv-control,instance-scaling'])
        self.assertEqual(set(args.controllers), {'kv-control', 'instance-scaling'})

    def plan(self, names=None, **extra):
        from benchmark.control_plane import ControllerPlan, controller_types
        types = ["instance-scaling", "kv-control", "request-throttling", "future-policy"]
        site = {"ride": {"management_endpoint": "http://ride:19100", "controllers": {
            name: {"endpoint": f"http://control:{9000 + index}"} for index, name in enumerate(types)}}}
        args = argparse.Namespace(controllers=controller_types(names) if names else None,
                                  route=None, controller_config=None)
        for name, value in extra.items():
            setattr(args, name, value)
        return ControllerPlan.resolve(args, site, ["http://engine:8000"])

    def test_every_subset_has_the_expected_inference_target(self):
        from itertools import combinations
        types = ("instance-scaling", "kv-control", "request-throttling")
        for size in range(4):
            for names in combinations(types, size):
                with self.subTest(names=names):
                    plan = self.plan(",".join(names))
                    throttled = "request-throttling" in names
                    self.assertEqual(plan.describe()["controllers"], list(names))
                    self.assertEqual(plan.program_release, throttled)
                    self.assertEqual(plan.endpoints, ["http://control:9002" if throttled else "http://engine:8000"])

    def test_new_policy_uses_existing_contract(self):
        self.assertEqual(self.plan("future-policy").describe()["route"], "direct")

    def test_invalid_selection(self):
        for names in ("kv-control,", "kv-control,kv-control", "kv-control/../x"):
            with self.assertRaises(argparse.ArgumentTypeError):
                self.plan(names)
        with self.assertRaisesRegex(ValueError, "no site configuration"):
            self.plan("unknown-controller")
        with self.assertRaisesRegex(ValueError, "legacy"):
            self.plan("kv-control", controller_config="invalid.json")

    def test_partial_start_failure_releases_first_controller(self):
        from contextlib import contextmanager
        from unittest.mock import patch
        events = []

        @contextmanager
        def start(management, spec, folder):
            events.append(("start", spec["type"]))
            if spec["type"] == "kv-control":
                raise ValueError("controller failed")
            try:
                yield lambda: None
            finally:
                events.append(("stop", spec["type"]))

        with patch("benchmark.control_plane.running_controller", start):
            with self.assertRaisesRegex(ValueError, "controller failed"):
                with self.plan("kv-control,instance-scaling").running(None):
                    self.fail("workload started before controllers were ready")
        self.assertEqual(events, [("start", "instance-scaling"), ("start", "kv-control"),
                                  ("stop", "instance-scaling")])
