"""Prove a stopped controller without rewriting its interrupted ledger."""
import hashlib
import json
import os
from pathlib import Path


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, start_ticks=fields[19],
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                pid_namespace=os.readlink(f"/proc/{pid}/ns/pid"))


def alive(owner):
    if (type(owner.get("pid")) is not int or owner["pid"] <= 0
            or not str(owner.get("start_ticks", "")).isdigit()):
        raise ValueError("Recovery owner lacks a valid PID/start identity")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if owner.get("boot_id") and owner["boot_id"] != boot:
        return False
    if owner.get("pid_namespace") and owner["pid_namespace"] != os.readlink("/proc/self/ns/pid"):
        raise ValueError("Recovery owner is in another PID namespace; stop cannot be verified")
    try:
        fields = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[19] == str(owner["start_ticks"]) and fields[0] not in ("Z", "X")


def stopped(source, prior):
    if prior.get("status") in ("failed", "completed"):
        return None
    if prior.get("status") != "running":
        raise ValueError("Reuse requires a stopped prior run")
    owner = prior.get("controller_owner")
    owner_file = None
    if owner is None:
        # Older launches wrote this adjacent, explicitly output-bound record.
        owner_file = source.parent / "root-process.json"
        if not owner_file.is_file():
            raise ValueError("Reuse requires a stopped prior run; controller identity is missing")
        owner = json.loads(owner_file.read_bytes())
        if Path(owner.get("out", "")).resolve() != source:
            raise ValueError("Recovery process record belongs to another output")
    if alive(owner):
        raise ValueError("Reuse requires a stopped prior run; original controller is still alive")
    stages = prior.get("stages", [])
    if (not stages or stages[-1].get("name") != "run-fixed-five"
            or stages[-1].get("status") != "running"
            or any(row.get("status") != "completed" for row in stages[:-1])):
        raise ValueError("Interrupted root is not a stopped fixed-five child failure")
    exit_path = source / "fixed-five-logs/exit.json"
    exit_record = json.loads(exit_path.read_bytes())
    if (exit_record.get("controller_reaped") is not True
            or type(exit_record.get("returncode")) is not int
            or exit_record["returncode"] == 0):
        raise ValueError("Recovery requires a reaped failed child controller")
    campaign = json.loads((source / "fixed-five/campaign.json").read_bytes())
    rows = campaign.get("stages", [])
    failed = [row for row in rows if row.get("status") == "failed"]
    if (campaign.get("status") != "failed" or not failed
            or any(row.get("status") not in ("completed", "reused_verified", "failed", "planned") for row in rows)):
        raise ValueError("Recovery requires a terminal failed campaign")
    for row in failed:
        stage = source / "fixed-five" / row["name"]
        if not stage.resolve().is_relative_to((source / "fixed-five").resolve()):
            raise ValueError("Recovery stage path escapes the source")
        from qualification import sealed_files
        sealed_files(stage)
        receipt = json.loads((stage / "receipt.json").read_bytes())
        cleanup = json.loads((stage / "cleanup.json").read_bytes())
        if (receipt.get("status") != "failed" or receipt.get("quiescent") is not True
                or cleanup.get("quiescent") is not True):
            raise ValueError("Recovery failed child is not released")
        process = stage / "service-process.json"
        if process.is_file() and alive(json.loads(process.read_bytes())):
            raise ValueError("Recovery failed service is still alive")
        for action in cleanup.get("actions", []):
            if alive(action):
                raise ValueError("Recovery failed service descendant is still alive")
    proof = dict(kind="interrupted-root-with-reaped-failed-campaign/v1", owner=owner,
                 child_exit_sha256=hashlib.sha256(exit_path.read_bytes()).hexdigest(),
                 originals_modified=False)
    if owner_file is not None:
        proof["owner_record_sha256"] = hashlib.sha256(owner_file.read_bytes()).hexdigest()
    return proof
