"""Local HTTP control for a separately labelled raw-work observation service.

Only fixed named primitive RPCs cross the production Proc transport. Capture
adds observers, so its elapsed time is never a clean performance numerator.
"""

import asyncio
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from m01_proc_capture import ranks
from m01_proc_io import write


class Control:
    def __init__(self, client, root):
        self.client = client
        self.root = Path(root)
        self.lock = asyncio.Lock()
        self.window = None
        self.closed = self.failed = False
        self.profile = os.environ.get("VLLMHUST_M01_HTTP_PROFILE") == "1"

    async def workers(self, name, *args):
        return ranks(await self.client.collective_rpc(method=name, timeout=1800 if name == "hust_m01_stop" else 300, args=args))

    async def core(self, name, *args):
        return await self.client.engine_core.call_utility_async(name, *args)

    async def drain_state(self):
        """Retire native State operations, acknowledge both ranks, reset prefixes."""
        deadline = time.monotonic() + 120
        record = dict(completed=False, drops=[], prefix_reset=False)
        async def snapshot():
            if time.monotonic() >= deadline:
                raise TimeoutError("Native State drain deadline expired")
            state = await self.core("state_cache", {"kind": "snapshot"})
            record["last_snapshot"] = state
            if (state.get("partial_reclaim") or {}).get("paused"):
                raise ValueError("State drain encountered paused requests")
            if not state["pending"] and any(s["owner"] is not None or s.get("io_owner") is not None for s in state["seats"]):
                raise ValueError("A resident State is still request-owned")
            return state
        try:
            while True:
                state = await snapshot()
                if state["pending"]:
                    await asyncio.sleep(.05)
                    continue
                if not state["host"]:
                    break
                key = state["host"][0]
                operation = await self.core("state_cache", {"kind": "drop", "key": key})
                while True:
                    state = await snapshot()
                    if operation not in state["pending"]:
                        receipt = next((r for r in state["completed"] if r["operation"] == operation), None)
                        if (receipt is None or receipt["kind"] != "drop" or receipt["cancelled"]
                                or set(receipt["ranks"]) != {0, 1} or key in state["host"]):
                            raise ValueError("Native State drop has no complete matching TP2 receipt")
                        record["drops"].append(dict(key=key, receipt=receipt))
                        break
                    await asyncio.sleep(.05)
            if await self.client.reset_prefix_cache() is not True:
                raise ValueError("Prefix reset did not acknowledge completion after State drain")
            record.update(prefix_reset=True, completed=True)
            return record
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            write(self.root / "state-cache-drain.json", record)

    async def call(self, action):
        async with self.lock:
            if action not in ("begin", "end"):
                raise ValueError("Only a single begin/end capture is supported")
            if self.failed or self.closed:
                raise ValueError("Preserve this attempt; start a fresh service")
            if action == "begin":
                if self.window is not None:
                    raise ValueError("Capture already started")
                self.root.mkdir(exist_ok=False)
                self.window = uuid4().hex
                result = dict(window=self.window, started=time.time(), completed=False,
                              profile_requested=self.profile,
                              frontend="HTTP", timing_eligible_for_performance_comparison=False,
                              arithmetic_certified=False, efficiency_percent=None)
                try:
                    state = await self.core("hust_m01_core_status")
                    workers = await self.workers("hust_m01_status")
                    if state["active"] or state["poisoned"] or state["unfinished_requests"] or any(
                            r["active"] or r["poisoned"] or r["backend_active"] for r in workers):
                        raise ValueError("Capture requires idle, healthy TP workers and CoreProc")
                    result["source"] = await self.core("hust_m01_core_identity")
                    result["worker_sources"] = await self.workers("hust_m01_runtime_files")
                    result["cache_drain"] = await self.drain_state()
                    result["start_fence"] = await self.workers("hust_runtime_synchronize")
                    core = await self.core("hust_m01_core_begin", str(self.root), self.window)
                    result["core"] = core
                    workers = await self.workers("hust_m01_start", str(self.root), self.profile, self.window)
                    result["workers"] = workers
                    if core["window"] != self.window or not core["active"] or any(
                            r["window"] != self.window or not r["active"] for r in workers):
                        raise ValueError("A capture participant acknowledged a foreign window")
                    result["completed"] = True
                except BaseException as error:
                    self.failed = True
                    result["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    write(self.root / "capture-begin.json", result)
                return result
            if self.window is None:
                raise ValueError("No capture was begun by this service")
            # An unfinished request must be allowed to drain before detaching.
            state = await self.core("hust_m01_core_status")
            if state["unfinished_requests"]:
                raise ValueError("Requests remain active; capture has not been detached")
            result = dict(window=self.window, completed=False, ended=time.time(),
                          profile_requested=self.profile, device_trace_validated=False,
                          timing_eligible_for_performance_comparison=False,
                          arithmetic_certified=False, efficiency_percent=None)
            try:
                core = await self.core("hust_m01_core_end", self.window)
                result["core"] = core
                workers = await self.workers("hust_m01_stop", self.window)
                result["workers"] = workers
                if core["window"] != self.window or not core["completed"] or any(
                        r["window"] != self.window or not r["completed"] for r in workers):
                    raise ValueError("Capture did not close on every participant")
                result["counts"] = dict(scheduler=core["waves"], **{str(r["rank"]): r["waves"] for r in workers})
                result["restored_worker_sources"] = await self.workers("hust_runtime_identity")
                result["compile_after"] = await self.workers("hust_compile_status")
                if any(not r["sealed"] or r["failure"] is not None for r in result["compile_after"]):
                    raise ValueError("Compilation boundary changed during raw observation")
                result["completed"] = True
                self.closed = True
            except BaseException as error:
                self.failed = True
                result["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                write(self.root / "capture-stop.json", result)
            return result


def install():
    root = os.environ.get("VLLMHUST_M01_HTTP_OUTPUT")
    if not root:
        return
    if os.environ.get("VLLM_SERVER_DEV_MODE") != "1" or not Path(root).is_absolute():
        raise ValueError("HTTP observation requires explicit local dev mode and an absolute output")
    from fastapi import HTTPException, Request
    from vllm.entrypoints.serve.dev.rpc.api_router import router

    if getattr(router, "_hust_m01_http_capture", False):
        return

    @router.post("/hust_reference_observation/{action}")
    async def capture(action: str, request: Request):
        if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
            raise HTTPException(status_code=403, detail="Local observation controller required")
        owner = getattr(request.app.state, "hust_reference_observation", None)
        if owner is None:
            owner = Control(request.app.state.engine_client, root)
            request.app.state.hust_reference_observation = owner
        try:
            return await owner.call(action)
        except Exception as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    router._hust_m01_http_capture = True
