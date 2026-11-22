"""Preserve the naturally selected CANN path without running a probe kernel.

Loaded libraries identify the runtime; they do not prove which kernel or
tiling parameters executed. That missing arithmetic link is recorded rather
than inferred from an API name, source filename, or base-image version.
"""
import inspect
import os
from pathlib import Path


LIBRARIES = {'libtorch_npu.so', 'libopapi.so', 'libopapi_transformer.so',
             'liboptiling.so', 'libopsproto.so', 'libopmaster_ct.so'}


def observe(entry):
    from official_ft_fla import digest
    sdk = Path(os.environ['VLLMHUST_OFFICIAL_SDK']).resolve()
    filename = Path(inspect.getfile(inspect.unwrap(entry))).resolve()
    if not filename.is_relative_to(sdk):
        raise ValueError('CANN caller escaped the actual SDK')
    libraries = {}
    foreign = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        path = Path(fields[5])
        if path.name not in LIBRARIES:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(sdk):
            foreign.append(str(resolved))
            continue
        relative = resolved.relative_to(sdk).as_posix()
        if relative not in libraries:
            libraries[relative] = digest(resolved)
    return dict(schema='official-cann-natural-backend-identity/v1', sdk=str(sdk),
        caller=filename.relative_to(sdk).as_posix(), caller_sha256=digest(filename),
        loaded_libraries=libraries, foreign_libraries=sorted(set(foreign)),
        custom_opp_paths=os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':'),
        kernel_execution_identity_observed=False, host_tiling_parameters_observed=False,
        probe_kernel_executed=False, backend_modified=False, tensor_values_observed=False)
