"""Wait for an owned stage's cleanup before allowing its parent to seal evidence."""

from contextlib import contextmanager
import os
import signal
import subprocess


def _lease_descriptors(env):
    environment = os.environ if env is None else env
    raw = environment.get("VLLMHUST_ENGINE_LOCKS")
    if not raw:
        return ()
    if (raw != os.environ.get("VLLMHUST_ENGINE_LOCKS")
            or environment.get("VLLM_HUST_PHYSICAL_DEVICES", "5,6")
            != os.environ.get("VLLM_HUST_PHYSICAL_DEVICES", "5,6")):
        raise ValueError("Stage resource leases must match the validated parent leases")
    from resource_guard import inherited
    return tuple(entry["fd"] for entry in inherited())



@contextmanager
def cancellation_handlers():
    def cancel(signum, frame):
        raise KeyboardInterrupt("Execution cancelled by signal " + str(signum))

    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_stage(command, *, env=None, stdout=None, stderr=None):
    """Only signal our direct child; it owns and cleans its service descendants.

    A new session keeps terminal interrupts from bypassing the stage controller's
    cleanup. Defer cancellation during Popen so a pending signal cannot orphan a
    just-created child. Repeated interrupts cannot interrupt the cleanup wait.
    """
    cancelled = []

    def remember(signum, frame):
        cancelled.append(signum)

    previous = {
        signum: signal.signal(signum, remember)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    child = None
    try:
        child = subprocess.Popen(
            command, env=env, start_new_session=True, stdout=stdout, stderr=stderr,
            pass_fds=_lease_descriptors(env),
        )
        while True:
            if cancelled:
                raise KeyboardInterrupt(
                    "Execution cancelled by signal " + str(cancelled[0])
                )
            try:
                return subprocess.CompletedProcess(command, child.wait(timeout=0.2))
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        if child is not None:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
            # Do not declare quiescence or seal a parent while a child controller
            # can still be writing results or holding its NPU leases.
            child.wait()
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
