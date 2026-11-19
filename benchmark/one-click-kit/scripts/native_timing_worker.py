"""Named pre/post timing audit RPCs; no per-step worker observation."""

from compile_boundary import WorkerExtension


class NativeTimingWorker(WorkerExtension):
    def hust_native_observer_audit(self):
        from native_observer_removal import audit_loaded
        from runtime_identity import identity

        runner = self.model_runner
        aggregate = getattr(runner, "_m01_host_prefill", None)
        if aggregate is None or aggregate.active:
            raise ValueError("Native clock requires an inactive host aggregate")
        if getattr(runner, "_hust_m01_route_session", None) is not None:
            raise ValueError("Route observation is active in the native timing worker")
        proof = audit_loaded()
        return dict(identity(self), observer_removal=proof,
                    host_aggregate_active=False, native_per_step_observer=False)
