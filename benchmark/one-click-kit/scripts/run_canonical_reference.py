"""Run one fresh original canonical sampler with owned processes and device leases.

The caller supplies current exact shapes, never historical weighted totals. The
original sampler remains byte identical. This stage certifies component samples;
full necessary-work and dependency-graph applicability are separate audits.
"""

import argparse
import contextlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

from dataset_contract import save, sha
from dependency_admission import admit_dependencies
from engine_entry import environment
from owned_processes import Owned
from resource_guard import acquire, release, cards, verify_mapping
from restore_b1_r4 import verify_installed_sources
from run_b1_r4 import check_memory, wait_before_service

KIT = Path(__file__).resolve().parents[1] / "third-party-test-kit"
sys.path.insert(0, str(KIT))
from canonical_protocol import current_plan, prepare
from canonical_results import audit


def verify_request(request, deployment, installed):
    if request.get("schema") != "current-canonical-sampler-request/v1":
        raise ValueError("Unknown current canonical sampler request")
    family, cases, bindings = request["family"], request["cases"], request["source_bindings"]
    for filename, expected in bindings.items():
        path = Path(filename).resolve()
        if not path.is_relative_to(deployment) or installed.get(str(path.relative_to(deployment))) != expected or sha(path) != expected:
            raise ValueError("Canonical recipe lacks its current frozen SDK source admission: " + filename)
    # Validate exact geometry and original controls before creating a device job.
    current_plan(family, cases, bindings)
    return family, cases, bindings


def verify_device_owners(sample, selected, known):
    for line in sample.splitlines():
        columns = [value.strip() for value in line.strip("|").split("|")]
        match = re.fullmatch(r"(\d+)\s+0", columns[0]) if columns else None
        if match and int(match[1]) in selected and len(columns) >= 4 and columns[1].isdigit():
            pid = int(columns[-1] if columns[-1].isdigit() else columns[1])
            if pid not in known:
                raise RuntimeError("Foreign selected-device owner appeared during canonical sampling")


def run(args):
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    service = root / "service"
    service.mkdir()
    owned = Owned(service)
    receipt = dict(schema="owned-current-canonical-sampler/v1", started=time.time(),
                   complete=False, quiescent=False, model_loaded=False, failure=None,
                   full_model_certificate=False, efficiency_percent=None)
    process = None
    try:
        deployment = args.deployment.resolve()
        installed = verify_installed_sources(deployment)
        save(root / "dependencies.json", admit_dependencies(deployment))
        request = json.loads(args.request.read_bytes())
        family, cases, bindings = verify_request(request, deployment, installed)
        receipt.update(family=family, request_sha256=sha(args.request))
        save(root / "request.json", request)
        prepared = prepare(root / "sampler", family, cases, bindings)
        save(root / "source-admission.json", dict(installed_files_verified=len(installed), source_bindings=bindings))
        save(root / "mapping.json", verify_mapping(args.devices))
        profile = json.loads((deployment / "deployment.json").read_bytes())
        with contextlib.ExitStack() as cleanup:
            leases = acquire(root, args.lease_timeout)
            cleanup.callback(release, leases)
            spec = importlib.util.spec_from_file_location("canonical_memory_guard", deployment / "mods/packed-host/memory_guard.py")
            guard = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(guard)
            initial = guard.sample()
            save(root / "memory-initial.json", initial)
            check_memory(guard, initial, root)
            if shutil.disk_usage(root).free < 4 * 2**30:
                raise RuntimeError("Canonical sampler startup disk reserve reached")
            env = environment(deployment, profile, service, root / "compiler-cache", args.devices)
            env.update(VLLM_HUST_TEST_OUTPUT=str(service), PYTHONUNBUFFERED="1")
            output = root / "sampler" / prepared["output"]
            command = [profile["python"], "-B", str(root / "sampler" / prepared["sampler"])]
            save(root / "launch-command.json", dict(command=command))
            wait_before_service(root, args.lease_timeout)
            try:
                with (root / "sampler.log").open("x") as log:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True, pass_fds=tuple(row["fd"] for row in leases))
                    owned.add(process.pid)
                    save(root / "process.json", owned.known[process.pid])
                    deadline, last_device, admitted = time.monotonic() + 14400, 0, False
                    while process.poll() is None:
                        now = time.monotonic()
                        known = {row["pid"] for row in owned.scan()}
                        check_memory(guard, initial, root)
                        if now - last_device >= 5:
                            sample = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=20).stdout
                            save(root / "device-latest.json", dict(wall=time.time(), raw=sample))
                            verify_device_owners(sample, cards(), known)
                            last_device = now
                        if not admitted and all((output / f"ready-{rank}.json").is_file() for rank in range(2)):
                            rows = [json.loads((output / f"ready-{rank}.json").read_bytes()) for rank in range(2)]
                            if [row["rank"] for row in rows] != [0, 1]:
                                raise ValueError("Canonical worker-ready ranks differ")
                            # The unchanged original samplers wait at this gate.
                            (output / "go").write_text("current resources/source admitted\n")
                            admitted = True
                        if now >= deadline or shutil.disk_usage(root).free < 2 * 2**30:
                            raise RuntimeError("Canonical sampler prospective deadline or disk reserve reached")
                        time.sleep(1)
                    receipt["returncode"] = process.returncode
                    if process.returncode or not admitted:
                        raise RuntimeError("Original sampler failed; every raw result and log retained")
                    summary = audit(output, root / "sampler" / prepared["plan"], cards())
                    save(root / "component-timings.json", summary)
                    receipt.update(complete=True, component_timings_sha256=sha(root / "component-timings.json"))
            finally:
                actions = owned.stop()
                if process is not None:
                    process.wait(timeout=30)
                receipt["quiescent"] = not owned.scan()
                save(root / "cleanup.json", dict(quiescent=receipt["quiescent"], actions=actions))
    except BaseException as error:
        receipt["failure"] = f"{type(error).__name__}: {error}"
        print(receipt["failure"], file=sys.stderr, flush=True)
    finally:
        if not owned.known:
            receipt["quiescent"] = True
        receipt["ended"] = time.time()
        save(root / "receipt.json", receipt)
        save(root / "checksums.json", {str(p.relative_to(root)): sha(p) for p in root.rglob("*")
             if p.is_file() and "compiler-cache" not in p.relative_to(root).parts and p.name != "checksums.json"})
    return 0 if receipt["complete"] and receipt["quiescent"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-devices", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--lease-timeout", type=float, default=86400)
    args = parser.parse_args()
    os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = args.physical_devices
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
