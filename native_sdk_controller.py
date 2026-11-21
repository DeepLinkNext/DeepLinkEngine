"""Native-libc CPU controller handoff; never mount, chroot or load a model."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENV = 'DEEPLINKENGINE_NATIVE_CONTROLLER'
CAPABILITY = """import ctypes,json,os
library=ctypes.CDLL(None,use_errno=True)
result=library.unshare(0x00020000)
print(json.dumps({'allowed':result==0,'errno':ctypes.get_errno() if result else 0,'pid':os.getpid(),'mount_performed':False}))
"""


def native_scope():
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        return False
    import official_execution as execution
    if os.environ.get('DEEPLINK_EXECUTION_CONTEXT'):
        execution.context_identity()
        return False
    version = execution.libc_version()
    return version is not None and tuple(map(int, version.split('.')[:2])) >= (2,35)


def runtime_version():
    return tuple(sys.version_info[:2])


def state():
    import official_execution as execution
    root = os.stat('/')
    return dict(libc=execution.libc_version(), root_dev_inode=[root.st_dev,root.st_ino],
        os_release_sha256=hashlib.sha256(Path('/etc/os-release').read_bytes()).hexdigest(),
        pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def admit_interpreter(python, *, sdk=None):
    """Verify bootstrap bytes before executing them, using only the host stdlib."""
    binary=Path(python).resolve(strict=True)
    if sdk is None:
        parents=[p for p in binary.parents if (p/'installed-files.json').is_file() and (p/'image-lock.json').is_file()]
        if len(parents)!=1:raise ValueError('Controller interpreter is not bound to one admitted official SDK')
        sdk=parents[0]
    sdk=Path(sdk).resolve(strict=True)
    metadata=json.loads((sdk/'installation.json').read_bytes())
    lock=json.loads((sdk/'image-lock.json').read_bytes())
    matches=[p for p in (ROOT/'engine/official-baselines').glob('*.json') if p.read_bytes()==(sdk/'image-lock.json').read_bytes()]
    if (not matches or metadata.get('schema')!='deeplinkengine-official-sdk/v1'
            or metadata.get('image_manifest')!=lock.get('manifest_digest')
            or digest(sdk/'installed-files.json')!=metadata.get('installed_files_sha256')):
        raise ValueError('Controller SDK manifest/installation identity differs')
    files=json.loads((sdk/'installed-files.json').read_bytes())
    prefix=binary.parent.parent
    library=(prefix/'lib/libpython3.12.so.1.0').resolve(strict=True)
    admitted={}
    for path in (binary,library):
        if not path.is_relative_to(sdk/'rootfs') or not path.is_file():
            raise ValueError('Controller interpreter/library escaped its SDK')
        key=path.relative_to(sdk).as_posix()
        expected=files.get(key)
        # SDK inventories use paths relative to rootfs, not to the SDK home.
        if expected is None:expected=files.get(path.relative_to(sdk/'rootfs').as_posix())
        actual=digest(path)
        if not isinstance(expected,dict) or expected.get('sha256')!=actual or expected.get('bytes')!=path.stat().st_size:
            raise ValueError('Controller Python binary/library SHA differs')
        admitted[str(path)]=actual
    return dict(sdk=str(sdk),installed_files_sha256=metadata['installed_files_sha256'],
        image_manifest=metadata['image_manifest'],binary_and_libpython_sha256=admitted)


def maybe_enter(sdk, script, arguments, *, cache):
    if not native_scope():
        return False
    previous = os.environ.get(ENV)
    if previous:
        recorded = json.loads(previous)
        actual = state()
        if any(recorded[k] != actual[k] for k in ('libc','root_dev_inode','os_release_sha256')):
            raise ValueError('Native controller libc/root changed; no implicit chroot is permitted')
    if runtime_version() == (3,12):
        return False
    sdk = Path(sdk).resolve()
    identity = json.loads((sdk/'installation.json').read_bytes())
    if identity.get('schema') != 'deeplinkengine-official-sdk/v1':
        return False
    from official_image import environment
    python, env = environment(sdk,Path(cache)/'native-controller')
    bootstrap=admit_interpreter(python,sdk=sdk)
    from transport_trust import configure
    configure(cache,sdk,env=env)
    record = dict(state(),schema='deeplinkengine-native-controller-python/v1',
        sdk=str(sdk),python=str(python),controller_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bootstrap_admission=bootstrap,chroot_performed=False,mount_performed=False,model_started=False)
    env[ENV] = json.dumps(record,sort_keys=True)
    os.execve(str(python),[str(python),'-B',str(Path(script).resolve()),*arguments],env)
    raise AssertionError('Native CPU controller exec unexpectedly returned')


def maybe_enter_official(config, arguments):
    if not native_scope() or runtime_version() == (3,12):
        return False
    from official_execution import prepare_anchor
    sdk = prepare_anchor(config)
    return maybe_enter(sdk,ROOT/'run_all.py',['--official-baselines',*arguments],cache=config['cache'])


def maybe_enter_deployment(deployment, script, arguments):
    """A direct installed-engine client needs only its existing Python metadata."""
    if not native_scope() or runtime_version() == (3,12):
        return False
    profile=json.loads((Path(deployment)/'deployment.json').read_bytes())
    python=Path(profile['python']).expanduser().absolute()
    installation=json.loads((Path(deployment).parent/'installation.json').read_bytes())
    if (Path(profile.get('root','')).resolve()!=Path(deployment).resolve()
            or installation.get('release')!=profile.get('release')
            or installation.get('deployment')!=str(Path(deployment).resolve())
            or installation.get('python')!=str(python)
            or installation.get('dependencies',{}).get('python')!=str(python)
            or not installation.get('dependencies',{}).get('python_version','').startswith('3.12.')):
        raise ValueError('Installed controller Python differs from its original installation receipt')
    bootstrap=admit_interpreter(python)
    env=dict(os.environ)
    env.pop('PYTHONHOME',None)
    env['LD_LIBRARY_PATH']=str(python.resolve().parent.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    # SDK venv executables may depend on the admitted base libpython. Read its
    # standard venv metadata rather than importing the runtime or framework.
    cfg=python.parent.parent/'pyvenv.cfg'
    if cfg.is_file():
        values=dict(line.split('=',1) for line in cfg.read_text().splitlines() if '=' in line)
        values={k.strip():v.strip() for k,v in values.items()}
        home=values.get('home','')
        if not home or Path(home).resolve()!=python.resolve().parent:
            raise ValueError('Controller venv home differs from the admitted SDK interpreter')
    record=dict(state(),schema='deeplinkengine-native-controller-python/v1',
        deployment=str(Path(deployment).resolve()),python=str(python),
        controller_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bootstrap_admission=bootstrap,chroot_performed=False,mount_performed=False,model_started=False)
    env[ENV]=json.dumps(record,sort_keys=True)
    os.execve(str(python),[str(python),'-B',str(Path(script).resolve()),*arguments],env)
    raise AssertionError('Native installed-client exec unexpectedly returned')


def namespace_capability():
    """Probe a short-lived child only; the parent's namespace never changes."""
    before=os.readlink('/proc/self/ns/mnt')
    child=subprocess.run([sys.executable,'-B','-c',CAPABILITY],capture_output=True,text=True,check=True,timeout=15)
    result=json.loads(child.stdout)
    if before != os.readlink('/proc/self/ns/mnt'):
        raise ValueError('Capability probe changed the controller namespace')
    return dict(result,parent_mount_namespace_unchanged=True,probe_source_sha256=hashlib.sha256(CAPABILITY.encode()).hexdigest())
