"""Run one experiment: plan the engine pool and controllers, deploy, then measure."""
import hashlib
import json
from pathlib import Path
import uuid

from benchmark import one_click, profile_streaming_transport as transport
from benchmark.benchmark_io import load, save
from benchmark.engine_pool import placements, running_pool
from benchmark.frozen_workload import load_cases, measure
from benchmark.control_plane import ControllerPlan, controller_types


def run(args):
    config, site = load(args.config), load(args.site)
    spec = site["engines"][args.engine]
    nodes = placements(site, spec, args.replicas, args.tp)
    backends = [node["endpoint"] for node in nodes]
    control = ControllerPlan.resolve(args, site, backends)
    plan = dict(engine=args.engine, release=spec["release"], replicas=args.replicas, tp=args.tp,
                workload=config, site=str(args.site), backends=backends, **control.describe())
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    cases = load_cases(config, site)
    lock = one_click.verify_kit()
    # The MMLU profile records reasoning-only, length-capped streams separately from final answers.
    client_sha = hashlib.sha256(Path(transport.__file__).read_bytes()).hexdigest()
    folder = (args.out or Path(site["results"]) / str(uuid.uuid4())).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "plan.json", dict(plan, kit=lock["tools"]["manifest_sha256"], source_sha256="c00dd6cbf94e220ea666bc8a85ce47213cf09597c94506a349112a2afddd25e9", execution_sha256=client_sha))
    print("BENCHMARK_OUTPUT", folder, flush=True)
    try:
        model = cases[0]["body"]["model"]
        with running_pool(nodes, folder, model), control.running(folder):
            measure(cases, config, folder, control.endpoints, control.program_release)
    except BaseException as error:
        save(folder / "status.json", dict(status="failed", error=str(error)))
        raise
    save(folder / "status.json", dict(status="completed"))
    return 0


def arguments(parser):
    for name in ("config", "site"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--engine", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--controllers", type=controller_types, metavar="TYPE[,TYPE...]",
                           help="RIDE controller types; omit to run the engine baseline")
    selection.add_argument("--route", choices=("direct", "request_throttling"), help="Legacy request path")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--controller-config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
