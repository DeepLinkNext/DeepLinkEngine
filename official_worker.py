"""Observed compiler/fence RPCs; upstream model math is unchanged."""
from compile_boundary import WorkerExtension


def inventory_collector(record):
    import hashlib
    import importlib.util
    from pathlib import Path
    if set(record) != {'path', 'sha256'}:
        raise ValueError('A pinned inventory source is required')
    path = Path(record['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Pinned model inventory helper changed')
    spec = importlib.util.spec_from_file_location('deeplink_official_model_inventory', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect


class OfficialWorker(WorkerExtension):
    def hust_official_kernel_prepare(self):
        from official_kernel_preparation import prepare
        return prepare(self)

    def hust_native_observer_audit(self):
        preparation = self.hust_official_kernel_prepare()
        from runtime_identity import identity
        import os
        from official_runtime_identity import runtime_identity, observer_basis
        runtime = runtime_identity(os.environ['VLLMHUST_OFFICIAL_SDK'])
        result = dict(identity(self), role='B0', native_per_step_observer=False,
                    kernel_preparation=preparation,
                    observer_removal_required=False,
                    runtime_identity=runtime, observer_basis=observer_basis(runtime))
        # Preserve actual loaded dimensions, dispatch and method sources once,
        # outside every timed interval. This observes no tensor values or routes
        # and does not install forward hooks or replace upstream methods.
        import hashlib
        import json
        import os
        from pathlib import Path
        try:
            runner = self.model_runner
            target, draft = runner.model, runner.drafter.model
            object_ids = (id(target), id(draft))
            if hasattr(self, '_deeplink_official_inventory'):
                previous_ids, captured = self._deeplink_official_inventory
                if previous_ids != object_ids:
                    raise ValueError('Official target/draft objects changed after inventory')
            else:
                collect = inventory_collector(json.loads(os.environ['VLLMHUST_OFFICIAL_INVENTORY']))
                captured = dict(schema='deeplinkengine-official-loaded-inventory/v1', rank=int(self.rank),
                                target=collect(target), draft=collect(draft),
                                tensor_values_observed=False, per_step_observer_installed=False,
                                arithmetic_certified=False)
                path = Path(os.environ['VLLM_HUST_TEST_OUTPUT']) / ('rank-' + str(self.rank) + '-loaded-inventory.json')
                raw = (json.dumps(captured, indent=2) + '\n').encode()
                with path.open('xb') as stream:
                    stream.write(raw)
                captured = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                arithmetic_certified=False, tensor_values_observed=False)
                self._deeplink_official_inventory = (object_ids, captured)
            result['loaded_inventory'] = captured
        except Exception as error:
            result['loaded_inventory'] = dict(available=False,
                error=type(error).__name__ + ': ' + str(error), arithmetic_certified=False)
        return result
