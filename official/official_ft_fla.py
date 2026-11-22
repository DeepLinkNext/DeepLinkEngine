"""Bind the natural A2 FLA choice to its reviewed WY64 implementation.

All observation is outside native clocks. No probe kernel, backend override,
tensor-value read, or model call is performed by this module.
"""
import ast
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import textwrap

ROOT = Path(__file__).resolve().parents[1]
PROFILE_SHA256 = 'b7a8d2460df7892c60d41c2957f4f1cf2091652993836321c602860a99ae65b6'
_FILES = {}


def digest(path):
    path = Path(path)
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    previous = _FILES.get(str(path))
    if previous is not None and previous[0] == identity:
        return previous[1]
    with path.open('rb') as stream:
        pin = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != identity:
        raise ValueError('FLA implementation changed during observation')
    _FILES[str(path)] = identity, pin
    return pin


def profile():
    path = ROOT / 'engine/official-ft-fla-a2-v1.json'
    if digest(path) != PROFILE_SHA256:
        raise ValueError('Reviewed FLA arithmetic domain changed')
    return json.loads(path.read_bytes())


def flags(operator, entry, expected):
    """Combine actual callable defaults with source-fixed caller keywords."""
    actual = {name: parameter.default for name, parameter in inspect.signature(operator).parameters.items()
              if parameter.default is not inspect.Parameter.empty}
    tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(entry))))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'fused_fwd']
    if len(calls) != 1:
        raise ValueError('Unreviewed FLA caller seam')
    for keyword in calls[0].keywords:
        if keyword.arg in expected:
            actual[keyword.arg] = ast.literal_eval(keyword.value)
    selected = {name: actual.get(name) for name in expected}
    if any(type(selected[name]) is not type(value) or selected[name] != value for name, value in expected.items()):
        raise ValueError('Actual FLA flags differ from the reviewed WY64 family')
    return selected


def files(domain):
    prefix = 'rootfs/usr/local/python3.12.16/lib/python3.12/site-packages/'
    expected = dict(domain['python_sources'])
    expected.update({prefix + name: value['sha256'] for name, value in domain['runtime_files'].items()})
    expected.update({prefix + name: value['sha256'] for name, value in domain['packaged_kernel_sources'].items()})
    return expected


def observe(operator, entry):
    domain = profile()
    sdk = Path(os.environ['VLLMHUST_OFFICIAL_SDK']).resolve()
    expected = files(domain)
    observed = {}
    for name, pin in expected.items():
        path = sdk / name
        if path.is_symlink() or not path.resolve().is_relative_to(sdk) or digest(path) != pin:
            raise ValueError('FLA source/library/kernel differs: ' + name)
        observed[name] = pin
    defaults = dict(domain['call_flags'], use_gate_in_kernel=False,
                    use_beta_sigmoid_in_kernel=False, return_intermediate_states=False,
                    a_log=None, dt_bias=None)
    actual_flags = flags(operator, entry, defaults)
    operator_file = Path(inspect.getfile(operator)).resolve()
    entry_file = Path(inspect.getfile(inspect.unwrap(entry))).resolve()
    if (str(operator_file.relative_to(sdk)).replace('\\', '/') not in expected
            or str(entry_file.relative_to(sdk)).replace('\\', '/') not in expected):
        raise ValueError('Actual FLA callable is outside its reviewed files')
    from fla_npu.ops.ascendc import _stable
    loaded = getattr(_stable, '_loaded_path', None)
    mapped = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or not parts[5].startswith('/'):
            continue
        name = parts[5]
        if ('/fla_npu/' not in name and '/fla_npu_transformer/' not in name) or not name.endswith('.so'):
            continue
        path = Path(name).resolve()
        if not path.is_relative_to(sdk):
            raise ValueError('FLA loaded a foreign SDK library')
        relative = str(path.relative_to(sdk)).replace('\\', '/')
        if relative not in expected or digest(path) != expected[relative]:
            raise ValueError('Actual loaded FLA library differs: ' + relative)
        mapped[relative] = expected[relative]
    package = sdk / 'rootfs/usr/local/python3.12.16/lib/python3.12/site-packages/fla_npu'
    vendor = (package / 'opp/vendors/fla_npu_transformer').resolve()
    opp = [str(Path(value).resolve()) for value in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':') if value]
    mandatory = [name for name in expected if name.endswith(('/libfla_npu_stable.so', '/op_api/lib/libcust_opapi.so'))]
    admitted = (loaded is not None and len(mandatory) == 2 and all(name in mapped for name in mandatory)
                and bool(opp) and opp[0] == str(vendor))
    stamp = None
    if admitted:
        path = Path(loaded).resolve()
        if path != package / 'libfla_npu_stable.so':
            raise ValueError('FLA actual stable handle differs')
        # RTLD_NOLOAD refuses to load a new library; this is a pure build-stamp
        # accessor of the already naturally loaded, byte-verified library.
        handle = ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD | os.RTLD_NOW)
        handle.fla_npu_stable_source_hash.restype = ctypes.c_char_p
        stamp = handle.fla_npu_stable_source_hash().decode('utf-8')
        admitted = stamp == domain['source_stamp']
    return dict(profile_sha256=PROFILE_SHA256, sdk=str(sdk), actual_operator_file=str(operator_file),
        actual_entry_file=str(entry_file), observed_files=observed, loaded_libraries=mapped,
        actual_library_source_stamp=stamp, actual_opp_paths=opp, admitted=admitted,
        flags=actual_flags, flags_evidence='Source-fixed caller keywords and actual loaded callable defaults',
        probe_kernel_executed=False, backend_modified=False, tensor_values_observed=False)


def validate(binding, sdk):
    domain = profile()
    expected = files(domain)
    if (binding['profile_sha256'] != PROFILE_SHA256 or binding['observed_files'] != expected
            or binding['admitted'] is not True or binding['actual_library_source_stamp'] != domain['source_stamp']
            or binding['probe_kernel_executed'] is not False or binding['backend_modified'] is not False
            or binding['tensor_values_observed'] is not False):
        raise ValueError('Actual FLA runtime admission is incomplete')
    prefix = 'rootfs/usr/local/python3.12.16/lib/python3.12/site-packages/fla_npu/'
    mandatory = {prefix + 'libfla_npu_stable.so', prefix + 'opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so'}
    loaded = binding['loaded_libraries']
    if (not mandatory.issubset(loaded) or any(expected.get(name) != pin for name, pin in loaded.items())):
        raise ValueError('Actual FLA loaded library set differs')
    original_sdk = PurePosixPath(binding['sdk'])
    if (PurePosixPath(binding['actual_operator_file']).relative_to(original_sdk).as_posix()
            != 'rootfs/usr/local/python3.12.16/lib/python3.12/site-packages/fla_npu/ops/ascendc/_stable.py'
            or PurePosixPath(binding['actual_entry_file']).relative_to(original_sdk).as_posix()
            != 'rootfs/vllm-workspace/vllm-ascend/vllm_ascend/device/device_op.py'
            or not binding['actual_opp_paths']
            or PurePosixPath(binding['actual_opp_paths'][0]).relative_to(original_sdk).as_posix()
            != prefix + 'opp/vendors/fla_npu_transformer'):
        raise ValueError('Actual FLA callable/OPP path differs')
    for name, pin in expected.items():
        if digest(Path(sdk) / name) != pin:
            raise ValueError('FLA frozen source changed after observation: ' + name)
    defaults = dict(domain['call_flags'], use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False, return_intermediate_states=False, a_log=None, dt_bias=None)
    if binding['flags'] != defaults:
        raise ValueError('Actual FLA arithmetic flags differ')
    return domain['id']
