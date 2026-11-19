"""Raw native observation through named Proc utilities; never produce an efficiency score."""

import copy
from pathlib import Path
from uuid import uuid4

from m01_proc_io import write
from m01_proc_plugin import require_enabled


def configure(config, environment):
    """Return explicit child settings without mutating a running process or plan."""
    if config.get("m01_engine_mode") != "proc" or config.get("m01_raw_only") is not True:
        raise ValueError("This prototype requires explicit proc/raw-only capture, not scoring")
    if config.get("necessary_work_certified") is True or config.get("efficiency_percent") is not None:
        raise ValueError("Raw capture is not a necessary-work or efficiency certificate")
    kwargs = copy.deepcopy(config["llm"])
    if kwargs.get("worker_extension_cls") not in (None, "compile_boundary.WorkerExtension"):
        raise ValueError("Do not silently replace another worker extension")
    kwargs["worker_extension_cls"] = "m01_proc_worker.M01WorkerExtension"
    env = dict(environment)
    if env.get("VLLM_ALLOW_INSECURE_SERIALIZATION", "0") not in ("", "0", "false", "False"):
        raise ValueError("Use named primitive RPCs without insecure serialization fallback")
    plugins = [p for p in env.get("VLLM_PLUGINS", "").split(",") if p]
    if not plugins:
        raise ValueError("Keep an explicit frozen plugin whitelist")
    if "hust_m01_proc_capture" not in plugins:
        plugins.append("hust_m01_proc_capture")
    env.update(VLLMHUST_M01_PROC_CAPTURE="1", VLLM_ENABLE_V1_MULTIPROCESSING="1",
               VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1", VLLM_PLUGINS=",".join(plugins))
    directory = str(Path(__file__).resolve().parent)
    paths = [p for p in env.get("PYTHONPATH", "").split(":") if p]
    env["PYTHONPATH"] = ":".join([directory]+[p for p in paths if p != directory])
    return kwargs, env


def ranks(rows):
    if (not isinstance(rows, (list, tuple)) or len(rows) != 2 or
            any(type(r["rank"]) is not int for r in rows) or {r["rank"] for r in rows} != {0, 1}):
        raise ValueError("Both unique TP2 rank acknowledgements are required")
    return rows


class RawObservation:
    def __init__(self, llm, output, *, profile=False):
        require_enabled()
        if type(profile) is not bool:
            raise ValueError("Profiler selection must be Boolean")
        self.llm = llm
        self.client = llm.llm_engine.engine_core
        if getattr(self.client, "engine_core", None) is not None or not callable(getattr(self.client, "call_utility", None)):
            raise ValueError("Raw Proc observation requires the production multiprocess client")
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.profile = profile
        self.window = uuid4().hex
        self.begin_attempted = self.core_attempted = self.workers_attempted = False
        self.closed = False
        self.result = None
        self.stop_attempts = 0
        write(self.output/"capture-protocol.json", dict(schema="raw-proc-m01-observation/v1",
              window=self.window, profile=profile, arithmetic_certified=False,
              efficiency_percent=None, timing_eligible_for_performance_comparison=False))

    def begin(self):
        if self.begin_attempted or self.closed:
            raise ValueError("Retain this attempt; create a new explicit observation instead of restarting it")
        self.begin_attempted = True
        try:
            core = self.client.call_utility("hust_m01_core_status")
            workers = ranks(self.llm.collective_rpc("hust_m01_status", timeout=120))
            if core["active"] or core["poisoned"] or any(r["active"] or r["poisoned"] or r["backend_active"] for r in workers):
                raise ValueError("The model already has an active or failed observation")
            identity = self.client.call_utility("hust_m01_core_identity")
            if identity["engine_core_mode"] != "proc":
                raise ValueError("Wrong Core mode")
            write(self.output/"core-source-identity.json", identity)
            write(self.output/"worker-source-identity.json", ranks(self.llm.collective_rpc("hust_m01_runtime_files", timeout=120)))
            ranks(self.llm.collective_rpc("hust_runtime_synchronize", timeout=120))
            self.core_attempted = True
            started = self.client.call_utility("hust_m01_core_begin", str(self.output), self.window)
            if started["window"] != self.window or started["active"] is not True:
                raise ValueError("Core did not acknowledge the current capture window")
            self.workers_attempted = True
            started_workers = ranks(self.llm.collective_rpc("hust_m01_start", timeout=120,
                args=(str(self.output), self.profile, self.window)))
            if any(r["window"] != self.window or r["active"] is not True for r in started_workers):
                raise ValueError("Workers did not acknowledge this capture window")
            write(self.output/"capture-begin.json", dict(core=started, workers=started_workers))
            return dict(window=self.window, active=True, arithmetic_certified=False)
        except BaseException as error:
            write(self.output/"capture-begin-error.json", dict(error=f"{type(error).__name__}: {error}",
                  core_attempted=self.core_attempted, workers_attempted=self.workers_attempted))
            if self.core_attempted or self.workers_attempted:
                self.stop()
            raise

    def stop(self):
        if self.closed:
            return self.result
        if not self.core_attempted:
            raise ValueError("No capture was started by this owner")
        self.stop_attempts += 1
        result = dict(window=self.window, completed=False, counts={}, errors=[],
                      arithmetic_certified=False, efficiency_percent=None,
                      profile_requested=self.profile, device_trace_validated=False,
                      timing_eligible_for_performance_comparison=False)
        try:
            core = self.client.call_utility("hust_m01_core_end", self.window)
            result["core"] = core
            if core["window"] != self.window or core["completed"] is not True:
                raise ValueError("Scheduler capture did not close successfully")
            result["counts"]["scheduler"] = core["waves"]
        except BaseException as error:
            result["errors"].append(f"core: {type(error).__name__}: {error}")
            try:
                status = self.client.call_utility("hust_m01_core_status")
            except BaseException:
                status = None
            if status and status["active"] and status["window"] == self.window and status["unfinished_requests"]:
                write(self.output/f"stop-attempt-{self.stop_attempts}.json", result)
                raise ValueError("Requests are still active; finish or abort them before detaching workers") from error
        if self.workers_attempted:
            try:
                workers = ranks(self.llm.collective_rpc("hust_m01_stop", timeout=300, args=(self.window,)))
                result["workers"] = workers
                if any(r["window"] != self.window or r["completed"] is not True for r in workers):
                    raise ValueError("Worker capture did not close successfully")
                for row in workers:
                    result["counts"][str(row["rank"])] = row["waves"]
            except BaseException as error:
                result["errors"].append(f"workers: {type(error).__name__}: {error}")
        result["completed"] = not result["errors"] and set(result["counts"]) == {"scheduler", "0", "1"}
        write(self.output/"capture-stop.json", result)
        self.result, self.closed = result, True
        return result

    def partial_work(self, outputs, model):
        result = self.stop()
        if not result["completed"]:
            raise ValueError("Raw observation is incomplete; preserve its stop receipt")
        from m01_observer_worker import key
        from m01_observer_work import collect
        counts = dict(scheduler=result["counts"]["scheduler"])
        counts.update({rank:result["counts"][str(rank)] for rank in (0, 1)})
        data = collect(self.output, self.window, counts, {row.request_id:key(row.request_id) for row in outputs}, outputs, model,
                       draft_counts={r["rank"]:r["draft_calls"] for r in result["workers"]})
        if data["effective_flops"] is not None:
            raise ValueError("A raw partial collector unexpectedly claimed complete arithmetic")
        data.update(arithmetic_certified=False, efficiency_percent=None,
                    timing_eligible_for_performance_comparison=False)
        write(self.output/"raw-partial-work.json", data)
        return data

    def close(self):
        if self.core_attempted:
            return self.stop()
        return None
