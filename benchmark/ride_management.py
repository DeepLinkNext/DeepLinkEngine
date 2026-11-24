"""Manage per-run controller jobs on the site's existing RIDE management service."""
from contextlib import contextmanager
import json
import time
from urllib.error import HTTPError
import uuid

from benchmark.benchmark_io import request, save
from benchmark.engine_pool import require_free_endpoint


def wait_for_job(url, job_id, endpoint, *, stopping=False):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        state = request(url)
        if state["job_id"] != job_id or state["endpoint"] != endpoint:
            raise RuntimeError("RIDE controller identity changed")
        if state["job_status"] in ("STOPPED", "SUCCEEDED", "FAILED"):
            if stopping:
                return state
            raise RuntimeError(f"Controller exited during startup: {state}")
        if not stopping and state["ready"]:
            return state
        time.sleep(1)
    raise TimeoutError(f"RIDE controller transition timed out: {url}")


@contextmanager
def running_controller(management, spec, folder):
    receipts = folder / "controllers" / spec["type"]
    receipts.mkdir(parents=True)
    identifier = spec["id"]
    url = f"{management.rstrip('/')}/api/v1/controllers/{identifier}"
    require_free_endpoint(spec["endpoint"])
    # Repeated runs may reuse a frozen controller identity after its prior job exited.
    prior_job = None
    try:
        previous = request(url)
    except HTTPError as error:
        if error.code != 404 or json.load(error).get("error") != "Unknown instance":
            raise
    else:
        prior_job = previous["job_id"]
        if prior_job and previous["job_status"] not in ("STOPPED", "SUCCEEDED", "FAILED"):
            raise ValueError(f"Controller {identifier} already has a live or unresolved job")
    intent = dict(operation_id=uuid.uuid4().hex, expected_job_id=prior_job, spec=spec)
    save(receipts / "intent.json", intent)
    try:
        started = request(url + "/start", intent, timeout=120)
    except HTTPError as error:
        if error.code < 500:
            raise
        started = recover_start(management, intent, receipts, identifier)
    except OSError:
        # A lost start response can conceal a submitted job. Query the original operation.
        started = recover_start(management, intent, receipts, identifier)
    save(receipts / "start.json", started)
    job_id = started["job_id"]
    try:
        def wait_ready():
            save(receipts / "ready.json", wait_for_job(url, job_id, spec["endpoint"]))

        yield wait_ready
    finally:
        stopped = request(url + "/stop", dict(expected_job_id=job_id), timeout=120)
        save(receipts / "stop.json", stopped)
        save(receipts / "terminal.json", wait_for_job(url, job_id, spec["endpoint"], stopping=True))


def recover_start(management, intent, receipts, identifier):
    operation = request(f"{management.rstrip('/')}/api/v1/operations/{intent['operation_id']}")
    save(receipts / "operation.json", operation)
    if operation.get("status") != "COMPLETED" or operation.get("identifier") != identifier:
        raise RuntimeError(f"Unresolved RIDE start; inspect {receipts}")
    return dict(job_id=operation["job_id"], id=identifier)
