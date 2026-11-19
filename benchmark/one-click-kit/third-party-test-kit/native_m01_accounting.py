"""Built-in live scheduler, TP worker and Ascend device capture for M01."""

import hashlib
import inspect
import os
from pathlib import Path
from uuid import uuid4

from native_m01_scheduler import attach
from native_m01_timeline import export
from native_m01_work import collect
from native_m01_worker import key, runtime_files, start, stop
from reference_capture_worker import canonical_sha, load_backend, write


def prepare(config, phase):
    # Expose EngineCore to the collector while retaining separate TP workers.
    if config.get("m01_in_process_engine") is not True:
        raise ValueError("Freeze m01_in_process_engine=true for the in-process engine collector")
    llm = config.get("llm") or {}
    additional = llm.get("additional_config") or {}
    if (llm.get("scheduler_cls") == "betterscale.models.qwen35.seat_scheduler.LiveStateScheduler"
            and additional.get("state_cache_host_bytes", 0) > 0):
        raise ValueError("This frozen B1 State-cache runtime requires EngineCoreProc maintenance. Use a proc-capable native collector; do not disable cache maintenance to force in-process capture.")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"


def engine_config(llm):
    client = llm.llm_engine.engine_core
    core = getattr(client, "engine_core", None)
    if core is None:
        raise ValueError("Built-in collector requires vLLM V1 in-process EngineCore and two executor workers")
    return core


def implementation_digest(llm, config, module_hashes):
    core = engine_config(llm)
    path = Path(inspect.getfile(type(core.scheduler))).resolve()
    workers = llm.collective_rpc(runtime_files, timeout=60)
    if len(workers) != 2 or {item["rank"] for item in workers} != {0, 1}:
        raise ValueError("Need effective implementation identities for two workers")
    sources = config["implementation_files"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Freeze engine, plugin and model configuration source digests")
    for name, expected in sources.items():
        source = Path(name)
        if not source.is_absolute() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen runtime source missing or changed: {name}")
    identity = dict(scheduler=hashlib.sha256(path.read_bytes()).hexdigest(), packages=module_hashes,
                    workers=sorted(workers, key=lambda row: row["rank"]),
                    frozen_sources=sorted(sources.values()))
    return canonical_sha(identity)


class Observation:
    def __init__(self, llm, config, phase, output):
        self.llm, self.phase, self.output = llm, phase, output.resolve()
        self.core = engine_config(llm)
        self.model = llm.model_config.hf_text_config
        self.window = uuid4().hex
        self.finish = None
        self.workers_started = False
        self.trace = phase in ("target", "reference-work", "observe")
        path = os.environ.get("VLLM_HUST_M01_CERTIFIER")
        self.certifier = load_backend(Path(path)) if path else None
        self.config = config
        self.graph = None

    def begin_capture(self):
        if self.phase == "operators":
            return
        self.finish = attach(self.core.scheduler, self.output, self.window)
        self.workers_started = True
        arrivals = self.llm.collective_rpc(start, timeout=120, args=(str(self.output), self.trace, self.window))
        if sorted(arrivals) != [(0, self.window), (1, self.window)]:
            raise ValueError("Both TP workers must attest to this capture window")

    def stop(self):
        counts = {}
        try:
            if self.finish:
                finish, self.finish = self.finish, None
                counts["scheduler"] = finish()
        finally:
            if self.workers_started:
                self.workers_started = False
                replies = self.llm.collective_rpc(stop, timeout=300)
                if len(replies) != 2 or {row[0] for row in replies} != {0, 1}:
                    raise ValueError("The two TP worker captures did not close")
                for rank, window, waves in replies:
                    if window != self.window:
                        raise ValueError("Worker returned a foreign capture window")
                    counts[rank] = waves
        return counts

    def effective_work(self, outputs):
        counts = self.stop()
        trace_failure = None
        if self.trace:
            try:
                export(self.output)
            except (OSError, ValueError, KeyError, TypeError) as error:
                trace_failure = error
                write(self.output / "device-trace-error.json", {"reason": str(error)})
        partial = collect(self.output, self.window, counts, {row.request_id: key(row.request_id) for row in outputs},
                          outputs, self.model)
        if trace_failure is not None:
            raise ValueError(f"Device dependency export failed: {trace_failure}") from trace_failure
        if self.phase == "observe":
            return partial
        if self.certifier is None:
            raise ValueError("Raw engine capture saved. Full M01 scoring requires certified arithmetic and reference bindings")
        certified = self.certifier.certify(self.llm, self.config, self.phase, self.output, outputs, partial)
        if certified.get("schedule_signature") != partial["schedule_signature"]:
            raise ValueError("Certified arithmetic must bind to the current native schedule")
        families = certified.get("covered_families", {})
        required = {"projection", "full_attention", "vocabulary", "scalar", "convolution",
                    "gdn_prefill", "gdn_decode", "draft_dependency"}
        if not required <= {name for name, covered in families.items() if covered is True}:
            raise ValueError("Certified arithmetic omitted at least one required family")
        self.graph = certified.get("nodes")
        write(self.output / "certified-necessary-work.json", certified)
        return certified

    def dependency_graph(self, outputs):
        if not self.graph:
            raise ValueError("Required dependency graph was not certified for this capture")
        return self.graph

    def run_operation(self, name):
        if self.certifier is None:
            raise ValueError("Independent native operator calibration requires the bound reference implementation")
        return self.certifier.run_operation(self.llm, self.config, name)

    def close(self):
        self.stop()


def bind(llm, config, phase, output):
    return Observation(llm, config, phase, output)


def measurement_method():
    return "Host monotonic time with native two-card fences; host accounting cost retained"
