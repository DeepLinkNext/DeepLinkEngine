"""Place the API's native Ascend logger in its service output directory."""
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys

MODULE = 'vllm_ascend.logger'
# Exact unmodified logger shared by the admitted R8 SDK and upstream 0.28rc1/0.30.
LOGGER_SHA256 = 'f7907aaf0055f0cb6c0f9c917c93447da94f40a122acc3cfe03e2f6aa5633581'
SCOPE = ('API interpreter only; Python spawn does not inherit this import hook. '
         'Workers retain native additional-config logging; their early HOME fallback is not covered.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def inside(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def prepare(sdk, output, env):
    """Plan using only stdlib; safe in a service dry-run, without installing hooks."""
    sdk, output = Path(sdk).resolve(), Path(output).resolve()
    dirs = dict(logs=output / 'vllm-ascend-logs',
                XDG_CONFIG_HOME=output / 'runtime-config', VLLM_CONFIG_ROOT=output / 'vllm-config')
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    for key in ('XDG_CONFIG_HOME', 'VLLM_CONFIG_ROOT'):
        env[key] = str(dirs[key])
    paths = env.get('PYTHONPATH', '').split(os.pathsep) + list(sys.path)
    spec = importlib.machinery.PathFinder.find_spec('vllm_ascend', [p for p in paths if p])
    if spec is None or spec.origin is None or not inside(spec.origin, sdk):
        raise ValueError('Ascend logging requires an SDK-bound package origin')
    source = Path(spec.origin).resolve().parent / 'logger.py'
    applicable = source.is_file()
    if applicable and (not inside(source, sdk) or digest(source) != LOGGER_SHA256):
        raise ValueError('Unsupported Ascend logger source; refusing HOME fallback')
    plan = dict(schema='deeplink-service-logging/v1', sdk=str(sdk), output=str(output),
                module=MODULE, vendor_source=str(source),
                vendor_sha256=digest(source) if applicable else None,
                adapter_sha256=digest(__file__), applicable=applicable,
                logs=str(dirs['logs']), config_environment={k: str(dirs[k]) for k in
                    ('XDG_CONFIG_HOME', 'VLLM_CONFIG_ROOT')},
                receipt=str(output / 'service-logging-receipt.json'), scope=SCOPE,
                upstream_source_patched=False, adapter_applied=False,
                state='planned' if applicable else 'not-applicable-module-absent')
    save(output / 'service-logging-plan.json', plan)
    save(plan['receipt'], plan)
    return plan


def install(plan):
    """Use the verified native setup function immediately after its module loads."""
    if plan.get('schema') != 'deeplink-service-logging/v1' or plan.get('module') != MODULE:
        raise ValueError('Unknown service logging plan')
    if digest(__file__) != plan['adapter_sha256']:
        raise ValueError('Service logging adapter source differs')
    if 'vllm' in sys.modules or 'vllm_ascend' in sys.modules or MODULE in sys.modules:
        raise ValueError('Service logging must be installed before framework imports')
    for key, value in plan['config_environment'].items():
        if key not in ('XDG_CONFIG_HOME', 'VLLM_CONFIG_ROOT') or not inside(value, plan['output']):
            raise ValueError('Unexpected service configuration directory')
        os.environ[key] = value
    if not plan['applicable']:
        save(plan['receipt'], dict(plan, state='not-applicable-module-absent'))
        return None
    source = Path(plan['vendor_source']).resolve()
    if (not inside(source, plan['sdk']) or digest(source) != LOGGER_SHA256
            or plan['vendor_sha256'] != LOGGER_SHA256 or not inside(plan['logs'], plan['output'])):
        raise ValueError('Ascend logging source or destination differs')

    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            create = getattr(self.original, 'create_module', None)
            return create(spec) if create else None

        def exec_module(self, module):
            try:
                if digest(source) != LOGGER_SHA256:
                    raise ValueError('Ascend logger changed before import')
                self.original.exec_module(module)
                native = getattr(module, '_setup_file_logging', None)
                if not callable(native) or module._file_logging_configured:
                    raise ValueError('Unsupported native Ascend logger initialization')
                native(log_dir=plan['logs'])
                handler = Path(module._file_handler.baseFilename).resolve()
                if not inside(handler, plan['logs']) or not handler.is_file():
                    raise ValueError('Native Ascend logger did not use the service directory')
                save(plan['receipt'], dict(plan, state='applied', adapter_applied=True,
                    native_helper='_setup_file_logging', actual_handler=str(handler),
                    pid=os.getpid(), upstream_source_patched=False))
            except BaseException as error:
                save(plan['receipt'], dict(plan, state='failed', error=repr(error)))
                raise
            finally:
                if finder in sys.meta_path:
                    sys.meta_path.remove(finder)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != MODULE:
                return None
            try:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                if spec is None or spec.origin is None or Path(spec.origin).resolve() != source or spec.loader is None:
                    raise ValueError('Ascend logger imported outside the admitted SDK source')
                spec.loader = Loader(spec.loader)
                return spec
            except BaseException as error:
                if self in sys.meta_path:
                    sys.meta_path.remove(self)
                save(plan['receipt'], dict(plan, state='failed', error=repr(error)))
                raise

    finder = Finder()
    sys.meta_path.insert(0, finder)
    return finder
