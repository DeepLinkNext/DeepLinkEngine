"""Let an owned B1 manager coordinate successful service shutdown."""

import hashlib
import signal
import subprocess


DIRECT_STOP = '        actions = owned.stop()\n'
MANAGED_STOP = (
    '        if role == "B1" and receipt["completed"]:\n'
    '            from managed_shutdown import stop_managed_service\n'
    '\n'
    '            actions = stop_managed_service(owned, service)\n'
    '        else:\n'
    '    ' + DIRECT_STOP
)


def without_managed_stop(body):
    managed = MANAGED_STOP.encode()
    return body.replace(managed, DIRECT_STOP.encode()) if body.count(managed) == 1 else body


def matches_pre_shutdown_controller(path, pin):
    return hashlib.sha256(without_managed_stop(path.read_bytes())).hexdigest() == pin


def stop_managed_service(owned, service, *, grace_seconds=95):
    # Capture descendants before signaling the manager so that cleanup can
    # still reach them if the manager exits and they are reparented.
    manager = owned.known[service.pid]
    active = {row["pid"]: row for row in owned.scan()}
    actions = []
    if (active.get(service.pid, {}).get("start_ticks") == manager["start_ticks"]
            and service.poll() is None):
        children = [{"pid": row["pid"], "start_ticks": row["start_ticks"]}
                    for row in active.values() if row["ppid"] == service.pid]
        service.send_signal(signal.SIGTERM)
        actions.append({"pid": manager["pid"], "start_ticks": manager["start_ticks"],
                        "signal": int(signal.SIGTERM), "managed_children": children})
        # engine_entry allows the manager 90 seconds before it escalates.
        # Give it time to finish and reap its API process after that point.
        try:
            service.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
    return actions + owned.stop()
