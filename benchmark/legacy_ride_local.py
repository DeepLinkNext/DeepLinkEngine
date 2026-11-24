"""Compatibility with the original isolated RIDE local runner."""
from contextlib import contextmanager
import subprocess

from benchmark.benchmark_io import save
from benchmark.engine_pool import require_free_endpoint


@contextmanager
def running(controller, settings, folder):
    endpoint = controller["endpoint"]
    require_free_endpoint(endpoint)
    require_free_endpoint(settings["management_endpoint"])
    config = folder / "ride.json"
    plan = folder / "ride-plan.json"
    save(config, dict(deployment="local", workspace=folder.name, state_file=str(folder / "ride-state.json"),
                      controllers=[controller]))
    save(plan, dict(config=str(config), management_endpoint=settings["management_endpoint"],
                    readiness_timeout_seconds=120))
    command = [settings["python"], settings["runner"]]
    options = ["--installation", settings["installation"], "--plan", str(plan)]
    try:
        subprocess.run(command + ["start", *options, "--output", str(folder / "controller-start")], check=True)
        yield [endpoint]
    finally:
        subprocess.run(command + ["stop", *options, "--output", str(folder / "controller-stop")], check=True)
