"""Verify the locked Ascend namespace and load its compiler without a device."""
import base64
import configparser
import csv
from email.parser import BytesParser
import hashlib
import importlib
from importlib import metadata
import io
import os
from pathlib import Path
import zipfile

PROBE_ID = 'triton-ascend-original-wheel-abi/v1'

FILES = {
    'triton': 'triton/__init__.py',
    'triton.backends': 'triton/backends/__init__.py',
    'triton.backends.ascend': 'triton/backends/ascend/__init__.py',
    'triton.backends.ascend.compiler': 'triton/backends/ascend/compiler.py',
    'triton.backends.ascend.driver': 'triton/backends/ascend/driver.py',
    'triton._C.libtriton': 'triton/_C/libtriton.so',
    'triton.runtime.driver': 'triton/runtime/driver.py',
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def check(wheel, sha256, *, load=importlib.import_module):
    wheel = Path(wheel)
    if wheel.is_symlink() or digest(wheel) != sha256:
        raise ValueError('Ascend backend source wheel differs from the selected lock')
    dist = metadata.distribution('triton-ascend')
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.count('/') == 1 and name.endswith('.dist-info/METADATA')]
        if len(names) != 1:
            raise ValueError('Ambiguous Ascend source wheel metadata')
        message = BytesParser().parsebytes(archive.read(names[0]))
        if message['Name'].lower().replace('_', '-') != 'triton-ascend' or message['Version'] != dist.version:
            raise ValueError('Installed Ascend distribution differs from its locked wheel')
        prefix = names[0].rsplit('/', 1)[0]
        raw = archive.read(prefix + '/RECORD')
        records = {row[0]: row for row in csv.reader(io.StringIO(raw.decode()))}
        entrypoints = configparser.ConfigParser()
        entrypoints.read_string(archive.read(prefix + '/entry_points.txt').decode())
        expected_entry = entrypoints['triton.backends']['ascend']
        if expected_entry != 'triton.backends.ascend':
            raise ValueError('Unexpected locked Ascend backend entry point')
    actual_entry = [entry.value for entry in dist.entry_points if entry.group == 'triton.backends' and entry.name == 'ascend']
    if actual_entry != [expected_entry]:
        raise ValueError('Installed Ascend backend entry point differs')
    files = {}
    for module, name in FILES.items():
        path = Path(dist.locate_file(name))
        expected = records.get(name, [None, '', None])
        if not path.is_file() or path.is_symlink() or not expected[1].startswith('sha256='):
            raise ValueError('Missing original Ascend compiler file: ' + name)
        actual = digest(path)
        encoded = base64.urlsafe_b64encode(bytes.fromhex(actual)).rstrip(b'=').decode()
        if encoded != expected[1].split('=', 1)[1] or path.stat().st_size != int(expected[2]):
            raise ValueError('Ascend compiler file was overwritten: ' + name)
        files[module] = dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=actual)
    # Importing these classes does not select the lazy driver or launch a kernel.
    modules = {name: load(name) for name in FILES}
    for name, module in modules.items():
        if Path(module.__file__).resolve() != Path(files[name]['path']):
            raise ValueError('Ascend backend import resolved outside the locked files: ' + name)
    backend = modules['triton.backends'].backends.get('ascend')
    if (backend is None or backend.compiler.__module__ != expected_entry + '.compiler'
            or backend.driver.__module__ != expected_entry + '.driver'):
        raise ValueError('Actual Ascend compiler/driver classes were not registered')
    library = modules['triton._C.libtriton']
    if not all(hasattr(library, name) for name in ('ascend', 'buffer_ir')) or not hasattr(library.ascend, 'ir'):
        raise ValueError('Loaded Triton library lacks the required Ascend compiler ABI')
    lazy = vars(modules['triton.runtime.driver'].driver)
    if not {'_default', '_active'} <= lazy.keys() or lazy['_default'] is not None or lazy['_active'] is not None:
        raise ValueError('CPU backend qualification unexpectedly initialized a device driver')
    plain = metadata.version('triton')
    if plain != dist.version or modules['triton'].__version__ != dist.version:
        raise ValueError('Plain and Ascend Triton dependency versions differ')
    installed_record = dist.read_text('RECORD')
    if not installed_record:
        raise ValueError('Installed Ascend RECORD is unavailable')
    boot = Path('/proc/sys/kernel/random/boot_id')
    return dict(schema=PROBE_ID, verified=True, probe_source_sha256=digest(__file__),
                process_id=os.getpid(), boot_id=boot.read_text().strip() if boot.is_file() else None,
                source_wheel=str(wheel), source_wheel_sha256=sha256,
                installed_record_sha256=hashlib.sha256(installed_record.encode()).hexdigest(),
                source_record_sha256=hashlib.sha256(raw).hexdigest(), triton_ascend=dist.version,
                plain_triton=plain, implementation='triton-ascend', entry_point=expected_entry, files=files,
                compiler_class=backend.compiler.__qualname__, driver_class=backend.driver.__qualname__,
                required_compiler_symbols=['ascend.ir', 'buffer_ir'], driver_initialized=False,
                kernel_compiled=False, model_loaded=False)


if __name__ == '__main__':
    import argparse
    import json
    import traceback
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        value = check(args.wheel, args.sha256)
    except Exception:
        value = dict(schema=PROBE_ID, verified=False, error=traceback.format_exc(),
                     probe_source_sha256=digest(__file__), source_wheel_sha256=args.sha256,
                     driver_initialized=None, model_loaded=False)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    raise SystemExit(0 if value['verified'] else 1)
