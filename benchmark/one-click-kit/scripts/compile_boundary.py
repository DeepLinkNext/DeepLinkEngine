"""Worker-local observed warmup and fail-closed Torch/Triton serving boundary.

CANN compilation is not covered. No eager fallback or graph policy changes.
Only local controller RPCs call this extension; existing worker math is intact.
"""

import json
import os
from pathlib import Path
import time


class Boundary:
    def __init__(self, worker):
        import torch._dynamo as dynamo
        from torch._dynamo.callback import callback_handler

        self.worker = worker
        self.sealed = False
        self.failure = None
        self.events = []
        self.dynamo = dynamo
        self.callbacks = callback_handler
        self.torch_hook = self.on_torch
        dynamo.config.suppress_errors = False
        dynamo.on_compile_start(self.torch_hook)
        self.native = getattr(worker, "_compilation_scope", None)
        self.target = self.attribute = self.triton_hook = None
        if self.native is not None:
            self.native.check()
        else:
            import triton
            from triton.runtime.jit import JITFunction

            runtime = getattr(getattr(triton, "knobs", None), "runtime", None)
            if runtime is not None and hasattr(runtime, "jit_cache_hook"):
                self.target, self.attribute = runtime, "jit_cache_hook"
            elif hasattr(JITFunction, "cache_hook"):
                self.target, self.attribute = JITFunction, "cache_hook"
            else:
                raise RuntimeError("Unsupported Triton miss hook")
            if getattr(self.target, self.attribute) is not None:
                raise RuntimeError("Triton hook already owned")
            self.triton_hook = self.on_triton
            setattr(self.target, self.attribute, self.triton_hook)
        self.event("boundary-installed")

    def event(self, compiler, **detail):
        row = dict(
            at=time.time(),
            monotonic=time.monotonic(),
            pid=os.getpid(),
            compiler=compiler,
            sealed=self.sealed,
            **detail,
        )
        self.events.append(row)
        root = Path(os.environ["VLLM_HUST_TEST_OUTPUT"])
        with (root / ("compilation-events-" + str(os.getpid()) + ".jsonl")).open(
            "a"
        ) as stream:
            stream.write(json.dumps(row, default=str) + "\n")
        if self.sealed and compiler in ("torch-dynamo", "triton"):
            self.failure = "Compilation after measurement boundary: " + compiler
            raise RuntimeError(self.failure)

    def on_torch(self, *args, **kwargs):
        self.event("torch-dynamo", detail=repr(args))

    def on_triton(self, *, fn, compile, **kwargs):
        self.event("triton", kernel=str(fn), specialization=str(compile))
        return None

    def status(self):
        if self.failure:
            raise RuntimeError(self.failure)
        if (
            self.torch_hook not in self.callbacks.start_callbacks
            or self.dynamo.config.suppress_errors
        ):
            raise RuntimeError("Torch compilation boundary replaced")
        if self.native is not None:
            self.native.check()
        elif getattr(self.target, self.attribute) is not self.triton_hook:
            raise RuntimeError("Triton compilation boundary replaced")
        return dict(
            pid=os.getpid(),
            rank=int(self.worker.rank),
            sealed=self.sealed,
            torch_dynamo_guard=True,
            triton_guard=True,
            cann_guard=False,
            triton_guard_owner="runtime" if self.native else "engine-controller",
            observed_compilations=sum(
                e["compiler"] in ("torch-dynamo", "triton") for e in self.events
            ),
            failure=self.failure,
        )


class WorkerExtension:
    def hust_runtime_identity(self):
        from runtime_identity import identity

        return identity(self)

    def hust_runtime_synchronize(self):
        from runtime_identity import synchronize

        return synchronize(self)

    def hust_compile_prepare(self):
        if hasattr(self, "_hust_compile_boundary"):
            raise RuntimeError("Compilation boundary already installed")
        self._hust_compile_boundary = Boundary(self)
        return self._hust_compile_boundary.status()

    def hust_compile_seal(self):
        scope = self._hust_compile_boundary
        scope.status()
        if scope.sealed:
            raise RuntimeError("Cannot reseal or reopen serving boundary")
        scope.sealed = True
        scope.event("measurement-ready")
        return scope.status()

    def hust_compile_status(self):
        return self._hust_compile_boundary.status()

    def hust_capacity_snapshot(self):
        from capacity_snapshot import snapshot

        return snapshot(self)
