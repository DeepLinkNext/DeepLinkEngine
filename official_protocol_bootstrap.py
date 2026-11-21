"""Reuse a verified private official CANN 9.1 image for protocol preparation."""
import json
from pathlib import Path
import platform


def host_identity():
    files = {}
    for name in ('/etc/os-release', '/proc/sys/kernel/random/boot_id',
                 '/usr/local/Ascend/driver/version.info', '/usr/local/Ascend/firmware/version.info'):
        path = Path(name)
        files[name] = path.read_text(errors='replace') if path.is_file() else None
    result = dict(hostname=platform.node(), architecture=platform.machine(),
                  kernel=platform.release(), source_files=files)
    from official_execution import context_identity
    context = context_identity()
    if context is not None:
        result.update(execution_context=context, host_os_release=context['host']['os_release'],
                      execution_os_release=context['execution_os_release'])
    return result


def cann_environment(sdk):
    from official_image import environment, digest, read_lock
    sdk = Path(sdk).resolve()
    lock = read_lock(sdk / 'image-lock.json')
    receipt = json.loads((sdk / 'installation.json').read_bytes())
    if (lock['manifest_digest'] != 'sha256:4ee78def8f33d59d48f116d1dfa793332c23c99ecab4f0d7dd5cd62d0fb4e6c1'
            or receipt['image_manifest'] != lock['manifest_digest']
            or digest(sdk / 'installed-files.json') != receipt['installed_files_sha256']):
        raise ValueError('Protocol preparation requires the verified official CANN 9.1 SDK')
    python, env = environment(sdk, sdk.parent.parent / 'protocol-bootstrap-cache')
    # The protocol gets its own venv and dependencies. Retain the image's
    # private Python shared library, CANN and system libraries only.
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    env.pop('VLLMHUST_NO_SITE', None)
    env['LD_LIBRARY_PATH'] = ':'.join(p for p in env['LD_LIBRARY_PATH'].split(':')
        if not p.endswith(('/torch/lib', '/torch_npu/lib')))
    return env, dict(home=env['ASCEND_HOME_PATH'], version='9.1.0',
        bootstrap_python=str(python), official_image_manifest=lock['manifest_digest'],
        installed_files_sha256=receipt['installed_files_sha256'],
        source='private-official-image', host_driver_modified=False)
