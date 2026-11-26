"""Resolve experiment controller types against the site's RIDE installation."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import re

from benchmark import legacy_ride_local
from benchmark.benchmark_io import load
from benchmark.ride_management import running_controller


def controller_types(value):
    names = value.split(",")
    if any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name) for name in names):
        raise argparse.ArgumentTypeError("Expected comma-separated RIDE controller types")
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("Repeated controller type")
    return sorted(names)


@dataclass
class ControllerPlan:
    specs: list
    endpoints: list
    management: str = ""
    legacy: dict | None = None
    program_release: bool = False

    @classmethod
    def resolve(cls, args, site, backends):
        if args.route == "request_throttling":
            if not args.controller_config:
                raise ValueError("Legacy request_throttling requires --controller-config")
            settings = load(args.controller_config)
            spec = bind_inference_pool(settings["controller"], backends)
            return cls([spec], [spec["endpoint"]], settings["management_endpoint"], settings, True)
        if args.controller_config:
            raise ValueError("--controller-config belongs to the legacy request_throttling route")
        if not args.controllers:
            return cls([], backends)

        ride = site.get("ride", {})
        catalog = ride.get("controllers", {})
        management = ride.get("management_endpoint")
        if not management:
            raise ValueError("Configure ride.management_endpoint in the site file")
        specs, endpoints = [], backends
        occupied = {management, *backends}
        ingress = False
        for name in args.controllers:
            if name not in catalog:
                raise ValueError(f"Controller {name!r} has no site configuration")
            spec = dict(catalog[name])
            if spec.setdefault("type", name) != name:
                raise ValueError(f"RIDE spec type differs from selected type {name}")
            identifier = spec.setdefault("id", name)
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", identifier):
                raise ValueError(f"Invalid RIDE instance name for {name}")
            if any(other["id"] == identifier for other in specs):
                raise ValueError(f"Duplicate RIDE instance name: {identifier}")
            endpoint = spec.get("endpoint")
            if not endpoint or endpoint in occupied:
                raise ValueError(f"Missing or conflicting endpoint for {name}")
            occupied.add(endpoint)
            is_throttling = name in ("request-throttling", "request-throttling-managed")
            protocol = spec.pop("client_protocol", "program-release" if is_throttling else None)
            if is_throttling and protocol != "program-release":
                raise ValueError("The request throttling controller requires program-release")
            if protocol is not None:
                if protocol != "program-release" or ingress:
                    raise ValueError("Unsupported or ambiguous inference endpoint protocol")
                spec = bind_inference_pool(spec, backends)
                endpoints, ingress = [endpoint], True
            specs.append(spec)
        return cls(specs, endpoints, management, program_release=ingress)

    def describe(self):
        return dict(controllers=[spec["type"] for spec in self.specs],
                    controller_instances=[dict(type=s["type"], id=s["id"], endpoint=s["endpoint"])
                                          for s in self.specs],
                    management_endpoint=self.management or None, client_endpoints=self.endpoints,
                    route="request_throttling" if self.program_release else "direct")

    @contextmanager
    def running(self, folder):
        with ExitStack() as stack:
            if self.legacy:
                spec = self.specs[0]
                stack.enter_context(legacy_ride_local.running(spec, self.legacy, folder))
            else:
                waiters = [stack.enter_context(running_controller(self.management, spec, folder))
                           for spec in self.specs]
                for wait_ready in waiters:
                    wait_ready()
            yield


def bind_inference_pool(spec, backends):
    controller = dict(spec)
    if controller["endpoint"] in backends or any(key in controller for key in ("backend", "backends", "model_service")):
        raise ValueError("Controller requires a separate endpoint and derives its backends from the selected pool")
    if controller.get("scheduling_enabled", True) is not True:
        raise ValueError("The request_throttling route requires scheduling")
    controller["scheduling_enabled"] = True
    if len(backends) == 1:
        controller["backend"] = backends[0]
    else:
        controller["backends"] = backends
    return controller
