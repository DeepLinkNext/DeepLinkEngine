"""Observed compiler/fence RPCs; upstream model math is unchanged."""
from compile_boundary import WorkerExtension


class OfficialWorker(WorkerExtension):
    def hust_native_observer_audit(self):
        from runtime_identity import identity
        return dict(identity(self), role='B0', native_per_step_observer=False,
                    observer_removal_required=False,
                    observer_basis='Unmodified official image; no optimized-engine MODs or per-step counting hooks')
