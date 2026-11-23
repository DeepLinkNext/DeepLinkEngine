"""Server-wide read-only monitoring, independent of the selected acceptance run."""
from collections import deque
import os
import re
import stat
import subprocess
import threading
import time


SAMPLE_INTERVAL = 10
FRESH_FOR = 45
NPU_HEADER = re.compile(r"^\|\s*(\d+)\s+\S+\s*\|")
NPU_CHIP = re.compile(
    r"^\|\s*\d+\s*\|\s*[0-9A-Fa-f:.]+\s*\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)"
)
SERVICE_STATS = re.compile(
    rb"INFO (\d\d-\d\d \d\d:\d\d:\d\d).*?Avg generation throughput:\s*([\d.]+) tokens/s"
    rb",\s*Running:\s*(\d+)"
)
SERVICE_ENTRY = re.compile(
    rb"(?:^|[/\x00])(?:serve_b[01][\w-]*\.py|corrected_service_entry\.py|vllm(?:\x00|\.entrypoints\.)|"
    rb"[Aa][Pp][Ii][Ss]erver\b)"
)


def device_point(cards):
    snapshot = subprocess.run(
        ["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=6
    ).stdout
    card = None
    rows = []
    for line in snapshot.splitlines():
        header = NPU_HEADER.match(line)
        if header:
            card = header[1]
        chip = NPU_CHIP.match(line)
        if chip and (not cards or card in cards):
            rows.append((int(chip[1]), int(chip[2]), int(chip[3])))
    if not rows:
        return {}
    return {
        "aicore": round(sum(core for core, _, _ in rows) / len(rows), 1),
        "mem": round(sum(hbm for _, hbm, _ in rows) / len(rows) / 1024, 2),
        "mem_total": round(sum(total for _, _, total in rows) / len(rows) / 1024, 2),
    }


def service_log_paths():
    """Find current serving stdout/stderr; closed logs from past runs are never selected."""
    paths = {}
    serving = False
    with os.scandir("/proc") as processes:
        for process in processes:
            if not process.name.isdecimal():
                continue
            try:
                with open(process.path + "/cmdline", "rb") as fh:
                    command = fh.read()
            except OSError:
                continue
            if not SERVICE_ENTRY.search(command):
                continue
            serving = True
            for fd in ("1", "2"):
                descriptor = process.path + "/fd/" + fd
                try:
                    target = os.readlink(descriptor)
                    info = os.stat(descriptor)
                except OSError:
                    continue
                if target.endswith(".log") and stat.S_ISREG(info.st_mode):
                    paths[info.st_dev, info.st_ino] = descriptor
    return serving, list(paths.values())


def log_stamp(raw, now):
    year = time.localtime(now).tm_year
    for candidate_year in (year, year - 1):
        try:
            value = time.mktime(time.strptime(
                "%s-%s" % (candidate_year, raw.decode()), "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
        if value <= now + 2:
            return value
    return None


def service_point(now):
    serving, paths = service_log_paths()
    if not paths:
        return {}, "waiting" if serving else "idle"
    readings = []
    for path in paths:
        try:
            with open(path, "rb") as fh:
                size = fh.seek(0, os.SEEK_END)
                fh.seek(max(0, size - 65536))
                matches = list(SERVICE_STATS.finditer(fh.read(65536)))
        except OSError:
            continue
        if not matches:
            continue
        stamp, gen, running = matches[-1].groups()
        at = log_stamp(stamp, now)
        if at is not None and now - at <= FRESH_FOR:
            readings.append((at, float(gen), int(running)))
    if not readings:
        return {}, "waiting"
    return {
        "source_at": max(at for at, _, _ in readings),
        "gen": round(sum(gen for _, gen, _ in readings), 1),
        "running": sum(running for _, _, running in readings),
    }, "live"


class Monitor:
    def __init__(self, cards):
        self.cards = tuple(part.strip() for part in cards.split(",") if part.strip())
        self.lock = threading.Lock()
        self.device = deque(maxlen=120)
        self.service = deque(maxlen=120)
        self.device_state = "waiting"
        self.service_state = "waiting"

    def sample(self):
        now = time.time()
        try:
            device = device_point(self.cards)
        except (OSError, subprocess.SubprocessError):
            device = {}
        try:
            service, service_state = service_point(now)
        except OSError:
            service, service_state = {}, "unavailable"
        with self.lock:
            self.device.append({"at": now, **device})
            self.service.append({"at": now, **service})
            self.device_state = "live" if device else "unavailable"
            self.service_state = service_state

    def _loop(self):
        while True:
            self.sample()
            time.sleep(SAMPLE_INTERVAL)

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="live-monitor").start()

    def snapshot(self):
        with self.lock:
            return {
                "cards": self.cards,
                "device": {"state": self.device_state, "points": list(self.device)},
                "service": {"state": self.service_state, "points": list(self.service)},
            }
