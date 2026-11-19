"""Opt-in named CoreProc capture utilities; existing State maintenance is untouched."""

import hashlib
import inspect
import os
from pathlib import Path
import re

PLUGIN = "hust-m01-proc-raw-capture/v1"
ENABLED = "VLLMHUST_M01_PROC_CAPTURE"
FIELDS = ("update_from_output", "_update_request_with_output")
METHODS = ("hust_m01_core_identity", "hust_m01_core_begin", "hust_m01_core_end", "hust_m01_core_status")


def require_enabled():
    if os.environ.get(ENABLED) != "1":
        raise ValueError("Raw Proc M01 capture is disabled; explicitly freeze the opt-in flag")


def window_id(window):
    if type(window) is not str or not re.fullmatch(r"[a-f0-9]{32}", window):
        raise ValueError("Capture window must be32 lowercase hexadecimal characters")
    return window


def snapshot_fields(owner, names):
    fields = vars(owner)
    return {name: (name in fields, fields.get(name)) for name in names}


def restore_fields(owner, fields):
    for name, (present, value) in fields.items():
        if present:
            setattr(owner, name, value)
        else:
            vars(owner).pop(name, None)


def idle(core, *, clean_cache):
    if core.scheduler.get_num_unfinished_requests() != 0:
        raise ValueError("Capture boundaries require no unfinished requests")
    cache = getattr(core.scheduler, "cache_actions", None)
    if clean_cache and cache is not None:
        data = cache.snapshot()
        if (data["pending"] or data["host"] or
                (data.get("partial_reclaim") or {}).get("paused") or
                any(s["owner"] is not None or s.get("io_owner") is not None for s in data["seats"])):
            raise ValueError("Drain State DMA, drop host checkpoints and reset prefixes before capture")


def core_identity(core):
    require_enabled()
    cls = type(core.scheduler)
    source = Path(inspect.getfile(cls)).resolve()
    return dict(plugin=PLUGIN, pid=os.getpid(), engine_core_class=type(core).__module__+"."+type(core).__name__,
                scheduler=cls.__module__+"."+cls.__name__,
                scheduler_source=str(source), scheduler_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                engine_core_mode="proc", state_cache_hook_present=callable(getattr(core, "state_cache", None)),
                arithmetic_certified=False, efficiency_percent=None)


def core_status(core):
    require_enabled()
    session = getattr(core, "_hust_m01_core_session", None)
    last = getattr(core, "_hust_m01_core_last", None)
    return dict(plugin=PLUGIN, pid=os.getpid(), active=session is not None,
                window=session["window"] if session else None, last=last,
                unfinished_requests=int(core.scheduler.get_num_unfinished_requests()),
                poisoned=bool(getattr(core, "_hust_m01_core_poisoned", False)),
                arithmetic_certified=False, efficiency_percent=None)


def core_begin(core, directory: str, window: str):
    require_enabled()
    window_id(window)
    if getattr(core, "_hust_m01_core_session", None) is not None:
        raise ValueError("Scheduler capture already active; inspect the existing window")
    if getattr(core, "_hust_m01_core_poisoned", False):
        raise ValueError("Failed capture requires a fresh engine process")
    last = getattr(core, "_hust_m01_core_last", None)
    if last is not None and last["window"] == window:
        raise ValueError("Capture window was already closed")
    idle(core, clean_cache=True)
    target = Path(directory)
    if not target.is_absolute() or target.is_symlink() or not target.is_dir():
        raise ValueError("Capture output must be an existing absolute directory")
    target = target.resolve()
    fields = snapshot_fields(core.scheduler, FIELDS)
    from m01_observer_scheduler import attach
    try:
        finish = attach(core.scheduler, target, window)
    except BaseException:
        core._hust_m01_core_poisoned = True
        restore_fields(core.scheduler, fields)
        raise
    core._hust_m01_core_session = dict(window=window, directory=str(target), finish=finish, fields=fields)
    # The cleanup closure stays in this CoreProc. Only primitive identities cross RPC.
    return dict(plugin=PLUGIN, pid=os.getpid(), window=window, active=True,
                output=str(target), arithmetic_certified=False)


def core_end(core, window: str):
    require_enabled()
    window_id(window)
    session = getattr(core, "_hust_m01_core_session", None)
    if session is None:
        last = getattr(core, "_hust_m01_core_last", None)
        if last is not None and last["window"] == window:
            return dict(last)  # Safe reconciliation of a lost stop acknowledgement.
        raise ValueError("No capture owned by this window")
    if session["window"] != window:
        raise ValueError("Cannot stop a foreign capture window")
    # DMA maintenance may still be pending; detach only between scheduler calls.
    # A new capture and all strict timing still require a clean State boundary.
    idle(core, clean_cache=False)
    result = dict(plugin=PLUGIN, pid=os.getpid(), window=window, completed=False,
                  waves=None, arithmetic_certified=False, efficiency_percent=None)
    try:
        waves = session["finish"]()
        if type(waves) is not int or waves < 0:
            raise ValueError("Invalid scheduler-wave count")
        result.update(waves=waves, completed=True)
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        core._hust_m01_core_poisoned = True
    finally:
        try:
            restore_fields(core.scheduler, session["fields"])
        except BaseException as error:
            result.update(completed=False, restoration_error=f"{type(error).__name__}: {error}")
            core._hust_m01_core_poisoned = True
        del core._hust_m01_core_session
        core._hust_m01_core_last = dict(result)
    return result


def install_on(core_class):
    functions = dict(zip(METHODS, (core_identity, core_begin, core_end, core_status), strict=True))
    prior = getattr(core_class, "_hust_m01_proc_plugin", None)
    if prior is not None and prior != PLUGIN:
        raise ValueError("Another capture plugin owns the CoreProc seam")
    for name, function in functions.items():
        existing = getattr(core_class, name, None)
        if existing is not None and existing is not function:
            raise ValueError("Refusing to replace an existing CoreProc utility: "+name)
    for name, function in functions.items():
        setattr(core_class, name, function)
    core_class._hust_m01_proc_plugin = PLUGIN
    return dict(plugin=PLUGIN, installed=True, methods=list(functions), arithmetic_certified=False)


def install():
    if os.environ.get(ENABLED) != "1":
        return dict(plugin=PLUGIN, enabled=False, installed=False)
    from vllm.v1.engine.core import EngineCoreProc
    from m01_http_capture import install as install_http

    if os.environ.get("VLLMHUST_M01_HTTP_OUTPUT"):
        from m01_route_capture import install_before_load
        install_before_load()
    install_http()
    return install_on(EngineCoreProc)
