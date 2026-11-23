"""Run a frozen HTTP workload against a site-owned foreground engine and optional RIDE controller."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request
from urllib.parse import urlsplit
import uuid

from benchmark import one_click, profile_streaming_transport as transport


def load(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def request(url, body=None, timeout=5):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
        return json.load(response)


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
        if endpoint in endpoints or not node.get("argv"):
            raise ValueError("Each engine requires a unique endpoint and foreground command")
        endpoints.add(endpoint)
    return nodes


def canonical_mmlu(inputs):
    ids = load(inputs / "verified-mmlu-quality-inputs.json")["cases"]
    requests = load(inputs / "mmlu-quality018-r2/quality-request-plan.json")["requests"]
    return "".join(
        json.dumps({"case_id": case["case_id"], "body": body}, ensure_ascii=False, separators=(",", ":")) + "\n"
        for case, body in zip(ids, requests, strict=True)
    ).encode()


def controller_spec(settings, backends):
    controller = dict(settings["controller"])
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


@contextmanager
def route(backends, settings, folder):
    if settings is None:
        yield backends
        return
    controller = controller_spec(settings, backends)
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


def run(args):
    config, site = load(args.config), load(args.site)
    if (args.route == "request_throttling") != bool(args.controller_config):
        raise ValueError("request_throttling requires --controller-config; direct rejects it")
    spec = site["engines"][args.engine]
    plan = dict(engine=args.engine, release=spec["release"], route=args.route, replicas=args.replicas,
                tp=args.tp, workload=config, site=str(args.site))
    nodes = placements(site, spec, args.replicas, args.tp)
    settings = load(args.controller_config) if args.controller_config else None
    backends = [node["endpoint"] for node in nodes]
    if settings:
        controller_spec(settings, backends)
    plan["backends"] = backends
    plan["client_endpoints"] = [settings["controller"]["endpoint"]] if settings else backends
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if "requests" in site:
        raw = Path(site["requests"]).read_bytes()
    else:
        from benchmark.dataset_bundle import prepare
        if config.get("dataset") != "mmlu-pro":
            raise ValueError("Supply the frozen request file for this workload")
        assets = prepare(Path(site["data_cache"]), ["mmlu"])
        raw = canonical_mmlu(assets / "mmlu/inputs")
    if hashlib.sha256(raw).hexdigest() != config["request_sha256"]:
        raise ValueError("Input file digest differs")
    cases = [json.loads(line) for line in raw.splitlines()]
    if len(cases) != config["requests"] or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Input population differs")
    lock = one_click.verify_kit()
    # The MMLU profile records reasoning-only, length-capped streams separately from final answers.
    client_sha = hashlib.sha256(Path(transport.__file__).read_bytes()).hexdigest()
    folder = (args.out or Path(site["results"]) / str(uuid.uuid4())).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "plan.json", dict(plan, kit=lock["tools"]["manifest_sha256"], source_sha256="c00dd6cbf94e220ea666bc8a85ce47213cf09597c94506a349112a2afddd25e9", execution_sha256=client_sha))
    print("BENCHMARK_OUTPUT", folder, flush=True)
    try:
        with ExitStack() as stack:
            for index, node in enumerate(nodes):
                worker = folder if len(nodes) == 1 else folder / f"engine-{index + 1}"
                worker.mkdir(exist_ok=True)
                stack.enter_context(serve(node, worker, node["physical_cards"], cases[0]["body"]["model"]))
            endpoints = stack.enter_context(route(backends, settings, folder))

            def send(item):
                index, case = item
                endpoint = endpoints[index % len(endpoints)]
                identity = f'{folder.name}-{case["case_id"]}'
                body = dict(case["body"])
                body.pop("max_completion_tokens", None)
                body.update(max_tokens=config["max_tokens"], cache_salt=identity)
                if settings:
                    body["program_id"] = identity
                path = folder / hashlib.sha256(identity.encode()).hexdigest()
                headers = {"x-delivery-request-id": identity, "x-session-id": identity}
                row = transport.call({"base_url": endpoint, "timeout": None, "trace_headers": headers}, body, path)
                row.update(case_id=case["case_id"], endpoint=endpoint)
                save(path / "result.json", row)
                if settings:
                    release = request(endpoint + "/programs/release", {"program_id": identity}, timeout=None)
                    save(path / "release.json", release)
                    if release.get("released") is not True:
                        raise RuntimeError("Program release failed: " + identity)
                return row["status"] == "ok"
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
                successful = sum(pool.map(send, enumerate(cases)))
            summary = dict(planned=len(cases), successful=successful, elapsed_seconds=time.monotonic()-started)
            save(folder / "measurement.json", summary)
            if successful != len(cases):
                raise RuntimeError("Incomplete workload")
    except BaseException as error:
        save(folder / "status.json", dict(status="failed", error=str(error)))
        raise
    save(folder / "status.json", dict(status="completed"))
    return 0


def arguments(parser):
    for name in ("config", "site"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--route", required=True, choices=("direct", "request_throttling"))
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--controller-config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
