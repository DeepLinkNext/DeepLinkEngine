"""Reserve the original physical NPUs and the shared model-loading budget."""

import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time


def cards():
    raw = os.environ.get("VLLM_HUST_PHYSICAL_DEVICES", "5,6")
    if not re.fullmatch(r"[0-9]+,[0-9]+", raw):
        raise ValueError("Exactly two physical NPU indices are required")
    result = [int(x) for x in raw.split(",")]
    if len(set(result)) != 2:
        raise ValueError("Physical NPU indices must differ")
    return result


def verify_mapping(logical):
    if not re.fullmatch(r"[0-9]+,[0-9]+", logical):
        raise ValueError("Exactly two logical NPU indices are required")
    sample = subprocess.run(
        ["npu-smi", "info", "-m"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout
    mapping = {}
    for line in sample.splitlines():
        words = line.split()
        if (
            len(words) >= 3
            and words[0].isdigit()
            and words[1] == "0"
            and words[2].isdigit()
        ):
            mapping[int(words[0])] = int(words[2])
    actual = [mapping.get(card) for card in cards()]
    if actual != [int(x) for x in logical.split(",")]:
        raise ValueError(
            "Configured logical NPU indices differ from actual selected physical cards"
        )
    return {
        "physical_cards": cards(),
        "logical_cards": actual,
        "npu_smi_mapping": sample,
    }


def paths():
    result = []
    for card in cards():
        result.extend(
            [
                f"/tmp/vllm-hust-engine-device-{card}.lock",
                f"/root/npu-device-{card}.lock",
                f"/run/lock/betterscale-npu-{card}.lock",
                f"/tmp/gennova-device-leases/device-{card}.lock",
                f"/tmp/gennova-physical-{card}.lock",
                f"/tmp/npu-physical-{card}.lock",
                f"/tmp/zllm2-npu-{card}.lock",
            ]
        )
    result.extend(
        [
            "/tmp/inference-project-container-model.lock",
            "/tmp/gennova-device-leases/container-model-loading.lock",
        ]
    )
    return result


def inherited():
    raw = os.environ.get("VLLMHUST_ENGINE_LOCKS")
    if not raw:
        return None
    entries = json.loads(raw)
    if [entry["path"] for entry in entries] != paths():
        raise ValueError("Inherited resource lease paths differ")
    for entry in entries:
        actual, expected = os.fstat(entry["fd"]), os.stat(entry["path"])
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("Inherited resource descriptor differs")
        fcntl.flock(entry["fd"], fcntl.LOCK_EX | fcntl.LOCK_NB)
    return entries


def release(entries):
    for entry in reversed(entries):
        os.close(entry["fd"])


def acquire(out, timeout):
    if entries := inherited():
        (out / "resource-admission.json").write_text(
            json.dumps(
                {
                    "physical_cards": cards(),
                    "exclusive_leases": paths(),
                    "inherited": True,
                    "admitted": True,
                },
                indent=2,
            )
            + "\n"
        )
        return entries
    deadline = time.monotonic() + timeout
    previous = None
    while True:
        entries, reason = [], None
        try:
            for name in paths():
                path = Path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                entries.append({"path": name, "fd": fd})
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    reason = "Shared resource lease is held: " + name
                    break
            if reason is None:
                sample = subprocess.run(
                    ["npu-smi", "info"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=20,
                ).stdout
                if not all(
                    f"No running processes found in NPU {card}" in sample
                    for card in cards()
                ):
                    reason = "Selected physical NPU pair has another running process"
                elif not all(
                    re.search(r"\|\s*" + str(card) + r"\s+910B2\b", sample)
                    for card in cards()
                ):
                    raise ValueError(
                        "Selected physical NPU pair must both be Ascend 910B2"
                    )
                else:
                    for entry in entries:
                        os.set_inheritable(entry["fd"], True)
                    os.environ["VLLMHUST_ENGINE_LOCKS"] = json.dumps(entries)
                    (out / "resource-admission.json").write_text(
                        json.dumps(
                            {
                                "physical_cards": cards(),
                                "npu_before": sample,
                                "exclusive_leases": paths(),
                                "admitted": True,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    return entries
        except BaseException:
            release(entries)
            raise
        release(entries)
        (out / "resource-wait.json").write_text(
            json.dumps(
                {
                    "physical_cards": cards(),
                    "reason": reason,
                    "model_started": False,
                    "time": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
        if reason != previous:
            print("RESOURCE_WAIT", reason, flush=True)
            previous = reason
        if time.monotonic() >= deadline:
            raise TimeoutError(reason)
        time.sleep(5)
