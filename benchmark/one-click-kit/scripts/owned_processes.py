"""Track process birth identities and stop only this run's descendants."""

import os
from pathlib import Path
import signal
import time


def state(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "state": fields[0],
        "start_ticks": fields[19],
    }


class Owned:
    def __init__(self, output):
        self.marker = ("VLLM_HUST_TEST_OUTPUT=" + str(output)).encode()
        self.known = {}
        self.first_birth = None

    def add(self, pid):
        value = state(pid)
        self.known[pid] = value
        self.first_birth = min(
            int(value["start_ticks"]), self.first_birth or int(value["start_ticks"])
        )

    def scan(self):
        snapshot, marked = {}, set()
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                pid = int(path.name)
                value = state(pid)
                if value["state"] == "Z":
                    continue
                snapshot[pid] = value
                if self.marker in (path / "environ").read_bytes().split(b"\0"):
                    marked.add(pid)
            except (OSError, ValueError):
                continue
        owned = {
            pid
            for pid, old in self.known.items()
            if pid in snapshot and snapshot[pid]["start_ticks"] == old["start_ticks"]
        }
        for pid in marked:
            if (
                self.first_birth is None
                or int(snapshot[pid]["start_ticks"]) < self.first_birth
            ):
                raise RuntimeError(
                    "Run marker belongs to a process older than this stage"
                )
            owned.add(pid)
        while True:
            children = {
                pid
                for pid, value in snapshot.items()
                if value["ppid"] in owned
                and int(value["start_ticks"]) >= self.first_birth
            }
            if children <= owned:
                break
            owned |= children
        for pid in owned:
            self.known[pid] = snapshot[pid]
        return [snapshot[pid] for pid in owned]

    def stop(self):
        actions = []
        # Network output I/O may keep a signalled owned child in D state.
        # Keep its birth identity and wait for exit before releasing leases;
        # SIGKILL cannot make an uninterruptible syscall disappear immediately.
        for sig, seconds in ((signal.SIGTERM, 15), (signal.SIGKILL, 180)):
            deadline = time.monotonic() + seconds
            signalled = set()
            while True:
                active = self.scan()
                if not active:
                    return actions
                for value in sorted(active, key=lambda row: int(row["start_ticks"])):
                    identity = (value["pid"], value["start_ticks"])
                    if identity in signalled:
                        continue
                    try:
                        if state(value["pid"])["start_ticks"] == value["start_ticks"]:
                            os.kill(value["pid"], sig)
                            actions.append(
                                {
                                    "pid": value["pid"],
                                    "start_ticks": value["start_ticks"],
                                    "signal": int(sig),
                                }
                            )
                            signalled.add(identity)
                    except (OSError, ProcessLookupError):
                        pass
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
        raise RuntimeError("Owned processes remain alive; refuse to seal output")
