"""One-way Triton warmup -> serving boundary, local to each worker process.

Reject in-memory JIT misses, including disk-cache loads: a disk artifact does
not prove this process warmed the specialization. This is not a Torch/Dynamo
or CANN compiler guard. Native cache hits do not invoke the miss hook.
"""


class UnwarmedKernelError(RuntimeError):
    """Serving reached a specialization absent from process-local warmup."""


class CompilationScope:
    def __init__(self):
        self._target = None
        self._attribute = None
        self._hook = self._reject
        self._failure = None

    def seal(self):
        """Called only after all native and owned warmup has succeeded."""
        if self._target is not None:
            self.check()
            return
        import triton
        from triton.runtime.jit import JITFunction

        runtime = getattr(getattr(triton, "knobs", None), "runtime", None)
        if runtime is not None and hasattr(runtime, "jit_cache_hook"):
            target, attribute = runtime, "jit_cache_hook"
        elif hasattr(JITFunction, "cache_hook"):
            target, attribute = JITFunction, "cache_hook"
        else:
            raise RuntimeError("Unsupported Triton JIT miss hook; cannot seal serving")
        # Do not silently replace another owner's compiler hook. A conflict is
        # a startup failure, not permission to serve without protection.
        if getattr(target, attribute) is not None:
            raise RuntimeError("Triton JIT miss hook already owned; cannot seal serving")
        self._target, self._attribute = target, attribute
        setattr(target, attribute, self._hook)

    def check(self):
        if self._failure is not None:
            raise self._failure
        if self._target is None:
            raise RuntimeError("Compilation scope is not sealed; warmup has not completed")
        if getattr(self._target, self._attribute) is not self._hook:
            self._failure = RuntimeError("Triton serving guard was replaced; restart worker")
            raise self._failure

    def _reject(self, *, fn, compile, **_):
        kernel = fn.jit_function.fn
        code = getattr(kernel, "__code__", None)
        location = "unknown" if code is None else f"{code.co_filename}:{code.co_firstlineno}"
        if self._failure is None:
            self._failure = UnwarmedKernelError(
                "Triton JIT miss after warmup: "
                f"{fn.module}.{fn.name} at {location}; "
                f"device={compile.get('device')}, "
                f"signature={compile.get('signature')}, "
                f"constants={compile.get('constants')}, "
                f"specialization={compile.get('key')}. "
                "Compilation/disk-cache loading blocked. Complete startup warmup "
                "and restart; this serving/benchmark run is invalid."
            )
        raise self._failure
