"""Place, start and release site-owned inference engine processes."""
from contextlib import ExitStack, contextmanager
import os
import signal
import socket
import subprocess
import time
from urllib.parse import urlsplit

from benchmark.benchmark_io import request, save


def require_free_endpoint(endpoint):
    target = urlsplit(endpoint)
    if target.scheme not in ("http", "https") or not target.hostname or target.path not in ("", "/"):
        raise ValueError("Expected a service origin: " + endpoint)
    try:
        connection = socket.create_connection(
            (target.hostname, target.port or (443 if target.scheme == "https" else 80)), timeout=2)
    except ConnectionRefusedError:
        return
    with connection:
        raise ValueError("Endpoint already in use: " + endpoint)


@contextmanager
def serve(spec, folder, cards, model):
    endpoint = spec["endpoint"]
    require_free_endpoint(endpoint)
    devices = ",".join(map(str, cards))
    argv = [arg.replace("{output}", str(folder)).replace("{cards}", devices) for arg in spec["argv"]]
    env = dict(os.environ, **spec.get("env", {}))
    env["VLLM_HUST_PHYSICAL_DEVICES"] = ",".join(map(str, cards))
    with (folder / "engine.log").open("x") as stream:
        child = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            save(folder / "engine-start.json", dict(argv=argv, pid=child.pid, physical_cards=cards))
            deadline = time.monotonic() + spec.get("startup_seconds", 1500)
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f"Engine exited during startup ({child.returncode})")
                try:
                    models = request(endpoint + "/v1/models")
                except OSError:
                    time.sleep(2)
                    continue
                if model not in [item["id"] for item in models["data"]]:
                    raise ValueError("Serving model differs from workload")
                save(folder / "model-readiness.json", models)
                yield endpoint
                return
            raise TimeoutError("Engine startup exceeded the configured operational limit")
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            save(folder / "engine-stop.json", dict(pid=child.pid, returncode=child.returncode))


def placements(site, spec, replicas, tp):
    if tp != 2 or replicas < 1:
        raise ValueError("Select at least one TP2 engine")
    if "instances" in spec:
        nodes = spec["instances"]
    else:
        nodes = [dict(spec, physical_cards=site["physical_cards"], host="localhost")]
    if len(nodes) != replicas:
        raise ValueError("The site must specify one engine endpoint per replica")
    devices, endpoints = set(), set()
    for node in nodes:
        cards = node["physical_cards"]
        if len(cards) != tp or any(type(card) is not int or card < 0 for card in cards):
            raise ValueError("Expected two assigned physical device IDs per engine")
        for card in cards:
            identity = (node["host"], card)
            if identity in devices:
                raise ValueError("Physical device assigned to multiple workers: " + str(identity))
            devices.add(identity)
        endpoint = node["endpoint"]
        lifecycle = node.get("lifecycle", "managed")
        if lifecycle not in ("managed", "existing"):
            raise ValueError("Unsupported engine lifecycle")
        if endpoint in endpoints or (lifecycle == "managed" and not node.get("argv")):
            raise ValueError("Each engine requires a unique endpoint and foreground command")
        endpoints.add(endpoint)
    return nodes

@contextmanager
def running_pool(nodes, folder, model):
    with ExitStack() as stack:
        for index, node in enumerate(nodes):
            worker = folder if len(nodes) == 1 else folder / f"engine-{index + 1}"
            worker.mkdir(exist_ok=True)
            if node.get("lifecycle") == "existing":
                models = request(node["endpoint"].rstrip("/") + "/v1/models")
                if model not in [item["id"] for item in models["data"]]:
                    raise ValueError("Existing endpoint does not serve the workload model")
                save(worker / "model-readiness.json", models)
                continue
            stack.enter_context(serve(node, worker, node["physical_cards"], model))
        yield
