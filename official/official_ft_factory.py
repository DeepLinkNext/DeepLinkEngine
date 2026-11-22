"""Read the028 Factory-owned shared-expert dispatch outside timing intervals.

The existing frozen inventory still owns all tensor shapes and scalars. This
small sidecar binds the plain (non-nn.Module) Ascend shared executor to that
same loaded shared MLP, without reading values or invoking any forward path.
"""
import enum
import hashlib
import inspect
from pathlib import Path


def capture(runner):
    sources = {}

    def describe(value):
        if value is None:
            return None
        cls = type(value)
        path = Path(inspect.getfile(cls)).resolve()
        if str(path) not in sources:
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return dict(type=cls.__module__+'.'+cls.__qualname__, source=str(path))

    def enum_value(value):
        if isinstance(value, enum.Enum):
            return dict(type=type(value).__module__+'.'+type(value).__qualname__, name=value.name)
        return None

    roles = {}
    for role, model in (('target', runner.model), ('draft', runner.drafter.model)):
        rows = []
        for name, block in model.named_modules():
            if not hasattr(block, 'n_routed_experts') or not hasattr(block, 'shared_expert'):
                continue
            executor = getattr(block, 'experts', None)
            routed = getattr(executor, 'routed_experts', None)
            shared = getattr(executor, 'ascend_shared_experts', None)
            rows.append(dict(block=name, runner=describe(executor), routed=describe(routed),
                shared=describe(shared), shared_layer_is_inventory_mlp=(shared is not None
                    and getattr(shared, 'layer', None) is block.shared_expert),
                n_shared_experts=getattr(routed, 'n_shared_experts', None),
                quant_type=enum_value(getattr(executor, 'quant_type', None)),
                shared_quant_type=enum_value(getattr(shared, 'quant_type', None)),
                weights_replicated=getattr(shared, 'weights_replicated', None),
                shared_sequence_parallel=getattr(shared, 'is_sequence_parallel', None),
                multistream_overlap=getattr(shared, 'multistream_overlap', None)))
        roles[role] = rows
    return dict(schema='official028-loaded-moe-factory-dispatch/v1', roles=roles,
        source_sha256=sources, tensor_values_observed=False, forward_invoked=False,
        observer_installed=False)


def validate(value):
    if (value.get('schema') != 'official028-loaded-moe-factory-dispatch/v1'
            or any(value.get(key) is not False for key in
                   ('tensor_values_observed', 'forward_invoked', 'observer_installed'))):
        raise ValueError('028 factory dispatch must be read-only loaded metadata')
    expected = dict(runner='vllm_ascend.ops.fused_moe.fused_moe.AscendMoERunner',
        routed='vllm_ascend.ops.fused_moe.routed_experts.AscendRoutedExperts',
        shared='vllm_ascend.ops.fused_moe.shared_experts.AscendSharedExperts')
    for role, count in (('target', 40), ('draft', 1)):
        rows = value['roles'][role]
        if len(rows) != count or len({row['block'] for row in rows}) != count:
            raise ValueError('028 factory loaded block population differs')
        for row in rows:
            if any(not isinstance(row.get(key), dict) or row[key].get('type') != name
                   for key, name in expected.items()):
                raise ValueError('028 actual shared/routed executor dispatch differs')
            if (row['shared_layer_is_inventory_mlp'] is not True
                    or type(row['n_shared_experts']) is not int or row['n_shared_experts'] != 0
                    or row['weights_replicated'] is not False
                    or row['shared_sequence_parallel'] is not False
                    or row['quant_type'] is None or row['shared_quant_type'] != row['quant_type']
                    or row['quant_type'] != dict(type='vllm_ascend.quantization.quant_type.QuantType', name='NONE')
                    or type(row['multistream_overlap']) is not bool):
                raise ValueError('028 shared MLP identity, layout or arithmetic kind differs')
            for key in expected:
                if row[key]['source'] not in value['source_sha256']:
                    raise ValueError('028 actual executor source is absent')
    return dict(actual_factory_shared_alias_passed=True, target_blocks=40, draft_blocks=1,
                missing030_flags_filled=False, tensor_values_observed=False)
