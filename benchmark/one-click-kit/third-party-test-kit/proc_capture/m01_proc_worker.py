"""Named worker observation RPCs with exact attribute restoration and window ownership."""

import hashlib
import importlib
import inspect
import json
from pathlib import Path

from compile_boundary import WorkerExtension as BoundaryWorkerExtension
from m01_proc_plugin import require_enabled, restore_fields, snapshot_fields, window_id

FIELDS = ("execute_model", "_build_attention_metadata")


def observer_backend():
    module = importlib.import_module("m01_observer_worker")
    source = Path(module.__file__).resolve()
    expected_path = Path(__file__).resolve().with_name("m01_observer_worker.py")
    pin = json.loads(expected_path.with_name("upstream-observer-sources-v1.json").read_bytes())[source.name]
    if source != expected_path or hashlib.sha256(source.read_bytes()).hexdigest() != pin["snapshot_sha256"]:
        raise ValueError("Observation helper source differs from the frozen snapshot")
    return module


def synchronize_worker(worker):
    from runtime_identity import synchronize
    return synchronize(worker)


class WorkerSession:
    def __init__(self, worker, window, backend):
        self.worker, self.window, self.backend = worker, window, backend
        self.original = snapshot_fields(worker.model_runner, FIELDS)

    def begin(self, directory, profile):
        try:
            rank, window = self.backend.start(self.worker, directory, profile, self.window)
            if type(rank) is not int or rank != int(self.worker.rank) or window != self.window:
                raise ValueError("Worker observation returned a foreign rank or window")
        except BaseException:
            # A failed profiler/observer start never permits reuse for clean timing.
            self.worker._hust_m01_proc_poisoned = True
            try:
                self.backend.stop(self.worker)
            finally:
                restore_fields(self.worker.model_runner, self.original)
            raise
        return dict(rank=int(self.worker.rank), window=self.window, active=True,
                    arithmetic_certified=False)

    def end(self):
        result = dict(rank=int(self.worker.rank), window=self.window, completed=False,
                      waves=None, arithmetic_certified=False, efficiency_percent=None)
        errors = []
        try:
            result["end_fence"] = synchronize_worker(self.worker)
        except BaseException as error:
            errors.append(f"fence: {type(error).__name__}: {error}")
        try:
            rank, window, waves = self.backend.stop(self.worker)
            if type(rank) is not int or rank != result["rank"] or window != self.window or type(waves) is not int or waves < 0:
                raise ValueError("Invalid worker capture acknowledgement")
            result["waves"] = waves
            result["draft_calls"] = self.worker._hust_m01_raw_draft_calls
            if type(result["draft_calls"]) is not int or result["draft_calls"] < 0:
                raise ValueError("Invalid merged draft call count")
        except BaseException as error:
            errors.append(f"stop: {type(error).__name__}: {error}")
        finally:
            try:
                restore_fields(self.worker.model_runner, self.original)
            except BaseException as error:
                errors.append(f"restore: {type(error).__name__}: {error}")
        if errors:
            result["errors"] = errors
            self.worker._hust_m01_proc_poisoned = True
        else:
            result["completed"] = True
        return result


class M01WorkerExtension(BoundaryWorkerExtension):
    def hust_m01_runtime_files(self):
        require_enabled()
        module = observer_backend()
        data = module.runtime_files(self)
        data.update(active=getattr(self, "_hust_m01_proc_session", None) is not None,
                    poisoned=bool(getattr(self, "_hust_m01_proc_poisoned", False)),
                    arithmetic_certified=False, efficiency_percent=None,
                    helper_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
        # Report implementation files behind Torch decorators as additional evidence.
        effective = {}
        for name in ("execute_model", "_build_attention_metadata"):
            method = inspect.unwrap(getattr(self.model_runner, name))
            source = Path(inspect.getfile(method)).resolve()
            effective[name] = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        data["effective_method_sources"] = effective
        return data

    def hust_m01_start(self, directory: str, profile: bool, window: str):
        require_enabled()
        window_id(window)
        if type(profile) is not bool:
            raise ValueError("Profiler selection must be an explicit Boolean")
        target = Path(directory)
        if not target.is_absolute() or target.is_symlink() or not target.is_dir():
            raise ValueError("Capture output must be an existing absolute directory")
        if getattr(self, "_hust_m01_proc_session", None) is not None or hasattr(self, "_hust_m01_capture"):
            raise ValueError("Worker capture already active")
        if getattr(self, "_hust_m01_proc_poisoned", False):
            raise ValueError("Failed capture requires a fresh worker process")
        last = getattr(self, "_hust_m01_proc_last", None)
        if last is not None and last["window"] == window:
            raise ValueError("Capture window was already closed")
        owner = WorkerSession(self, window, observer_backend())
        self._hust_m01_proc_session = owner
        try:
            return owner.begin(str(target.resolve()), profile)
        except BaseException as error:
            self._hust_m01_proc_last = dict(rank=int(self.rank), window=window, completed=False,
                error=f"{type(error).__name__}: {error}", arithmetic_certified=False, efficiency_percent=None)
            del self._hust_m01_proc_session
            raise

    def hust_m01_stop(self, window: str):
        require_enabled()
        window_id(window)
        owner = getattr(self, "_hust_m01_proc_session", None)
        if owner is None:
            last = getattr(self, "_hust_m01_proc_last", None)
            if last is not None and last["window"] == window:
                return dict(last)
            raise ValueError("No worker capture owned by this window")
        if owner.window != window:
            raise ValueError("Cannot stop a foreign worker capture")
        try:
            result = owner.end()
        finally:
            del self._hust_m01_proc_session
        self._hust_m01_proc_last = dict(result)
        return result

    def hust_m01_status(self):
        require_enabled()
        owner = getattr(self, "_hust_m01_proc_session", None)
        return dict(rank=int(self.rank), active=owner is not None,
                    window=owner.window if owner else None,
                    poisoned=bool(getattr(self, "_hust_m01_proc_poisoned", False)),
                    backend_active=hasattr(self, "_hust_m01_capture"),
                    last=getattr(self, "_hust_m01_proc_last", None),
                    arithmetic_certified=False, efficiency_percent=None)
