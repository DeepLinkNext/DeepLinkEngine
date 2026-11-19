"""Existing native State drain and checkpoint admission; outside every clock.

Adapted from production_native_helpers.py SHA256 163a92a7ca71d77b060921aaf914f3badc58091891ef70a04879b4136e192fb5.
The client argument is the public LLMEngine rather than its LLM facade.
"""

import json
import time


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def reset_state_cache(engine, output, *, timeout=120, poll=0.05):
    """Retire native TP maintenance before asking the scheduler to reset KV.

    This is outside every timed interval. Never cancel unfinished DMA, disable
    the policy, or discard host checkpoints without both worker receipts.
    """
    client = engine.engine_core
    deadline = time.monotonic() + timeout
    record = dict(schema="native-state-cache-reset/v1", drops=[], reset=False)

    def snapshot():
        if time.monotonic() >= deadline:
            raise TimeoutError("Native State-cache drain deadline reached")
        state = client.call_utility("state_cache", {"kind": "snapshot"})
        record["last_snapshot"] = state
        if any(s["owner"] is not None for s in state["seats"]):
            raise ValueError("Cannot reset State while a request owns a resident")
        if (state.get("partial_reclaim") or {}).get("paused"):
            raise ValueError("Cannot reset State while requests are paused")
        return state

    try:
        while True:
            state = snapshot()
            if state["pending"]:
                time.sleep(poll)
                continue
            if not state["host"]:
                break
            key = state["host"][0]
            operation = client.call_utility(
                "state_cache", {"kind": "drop", "key": key}
            )
            while True:
                state = snapshot()
                if operation not in state["pending"]:
                    receipt = next(
                        (r for r in state["completed"] if r["operation"] == operation),
                        None,
                    )
                    if (
                        receipt is None
                        or receipt["kind"] != "drop"
                        or receipt["cancelled"]
                        or set(receipt["ranks"]) != {0, 1}
                        or key in state["host"]
                    ):
                        raise ValueError("Missing complete TP2 host-drop receipt")
                    record["drops"].append(dict(key=key, receipt=receipt))
                    break
                time.sleep(poll)
        if not engine.reset_prefix_cache():
            raise ValueError("Native prefix-cache reset failed after State drain")
        record["reset"] = True
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(output, record)


def wait_checkpoint_admission(path, *, timeout=180, poll=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            receipt = json.loads(path.read_bytes())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(poll)
            continue
        after = receipt["after"]
        if after["limit"] - after["current"] < receipt["required_raw_headroom_bytes"]:
            raise ValueError("Checkpoint cache admission did not establish the required raw headroom")
        return receipt
    raise TimeoutError("Waited for the existing post-load checkpoint memory admission")
