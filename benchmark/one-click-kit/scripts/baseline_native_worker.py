"""Original018 named identity/fence RPCs without a per-step observer."""

import os

from compile_boundary import WorkerExtension


class BaselineNativeWorker(WorkerExtension):
    def hust_native_observer_audit(self):
        from runtime_identity import identity

        if any(os.environ.get(key) == "1" for key in
               ("VLLMHUST_M01_PROC_CAPTURE", "MTP_PROFILE", "HUST_TOKEN_OBSERVER")):
            raise ValueError("Baseline native observer flag is enabled")
        runner = self.model_runner
        if any(getattr(runner, key, None) is not None for key in
               ("_m01_host_prefill", "_hust_m01_route_session", "_hust_m01_capture")):
            raise ValueError("A current-engine observer was installed in original018")
        return dict(identity(self), role="B0", native_per_step_observer=False,
                    observer_removal_required=False,
                    observer_basis="Full original018 source admission; no B1 MODs or counted-run hooks loaded")
