from pathlib import Path
import ctypes
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]

ASCEND = Path('/usr/local/Ascend')
COMPONENTS = ('ascend-toolkit', 'cann-9.1.0', 'nnal')
SETUP = ('cann-9.1.0/set_env.sh', 'nnal/atb/set_env.sh')


def judge_command(command, config, output):
    from official_baselines import save
    if all((ASCEND / name).is_file() for name in SETUP):
        return command
    lock = ROOT / 'engine/official-baselines/0.23.0.json'
    manifest = json.loads(lock.read_bytes())['manifest_digest']
    anchor = Path(config['cache']) / 'official-sdks' / ('0.23.0-' + manifest.split(':')[1][:16])
    request = output / 'judge-private-runtime.json'
    save(request, dict(anchor=str(anchor), manifest=manifest, command=command))
    return [command[0], '-B', str(ROOT / 'official/official_judge_runtime.py'), str(request)]


def verify_source(anchor, manifest):
    import official_execution as execution
    lock = ROOT / 'engine/official-baselines/0.23.0.json'
    identity = json.loads((anchor / 'installation.json').read_bytes())
    inventory = anchor / 'installed-files.json'
    if (identity.get('schema') != 'deeplinkengine-official-sdk/v1'
            or identity['image_manifest'] != manifest
            or (anchor / 'image-lock.json').read_bytes() != lock.read_bytes()
            or json.loads(lock.read_bytes())['manifest_digest'] != manifest
            or execution.sha(inventory) != identity['installed_files_sha256']):
        raise ValueError('Fixed judge requires the admitted CANN 9.1 image')
    root = anchor / 'rootfs'
    publisher = root / ASCEND.relative_to('/')
    registered = json.loads(inventory.read_bytes())
    hashes = {}
    for name in SETUP:
        script = publisher / name
        actual = execution.sha(script)
        key = script.relative_to(root).as_posix()
        record = registered.get(key, registered.get('rootfs/' + key, {}))
        if (record.get('sha256') != actual or record.get('bytes') != script.stat().st_size
                or not script.resolve().is_relative_to(publisher)):
            raise ValueError('Installed judge setup script differs from the official image: ' + name)
        hashes[name] = actual
    return publisher, hashes


def enter(request_path):
    import official_execution as execution
    from official_baselines import save
    request = json.loads(request_path.read_bytes())
    if not ASCEND.is_dir():
        raise ValueError('The existing Ascend driver root is required for a private judge view')
    anchor = Path(request['anchor']).resolve(strict=True)
    inherited = execution.context_identity()
    if inherited is not None:
        execution.private_mounts()
    library = ctypes.CDLL(None, use_errno=True)
    execution._call(library.unshare, execution.CLONE_NEWNS)
    if inherited is None:
        execution._mount(None, Path('/'), execution.MS_REC | execution.MS_PRIVATE)
    execution.private_mounts()
    anchor_request = dict(binds=[])
    if not execution.readonly_sdk_anchor(anchor_request, anchor):
        raise ValueError('The fixed judge CANN anchor has no installation receipt')
    for source, target, readonly in anchor_request['binds']:
        execution._bind(Path('/'), source, target, readonly)
    if not os.statvfs(anchor).f_flag & os.ST_RDONLY:
        raise ValueError('Fixed judge CANN anchor must be read-only')
    publisher, hashes = verify_source(anchor, request['manifest'])

    stage = request_path.parent / 'judge-ascend-view'
    stage.mkdir(mode=0o700)
    for name in COMPONENTS:
        if not (publisher / name).is_dir():
            raise ValueError('Missing fixed judge toolkit component: ' + name)
        execution._bind(Path('/'), publisher / name, stage / name, True)
    (stage / 'cann').symlink_to('cann-9.1.0')
    for name in ('driver', 'firmware'):
        if (ASCEND / name).is_dir():
            execution._bind(Path('/'), ASCEND / name, stage / name, True)
    execution._bind(Path('/'), stage, ASCEND, True)
    if any(execution.sha(ASCEND / name) != expected for name, expected in hashes.items()):
        raise ValueError('Fixed judge toolkit view differs from the sealed SDK')

    namespace = os.readlink('/proc/self/ns/mnt')
    if inherited is not None:
        inherited['mount_namespace'] = namespace
        os.environ['DEEPLINK_EXECUTION_CONTEXT'] = json.dumps(inherited)
    save(request_path.parent / 'judge-private-runtime-admission.json', dict(
        schema='deeplinkengine-fixed-judge-cann-view/v1',
        manifest=request['manifest'], setup_sha256=hashes, mount_namespace=namespace,
        readonly_sdk_anchor=str(anchor), global_install_modified=False))
    command = request['command']
    os.execve(command[0], command, os.environ)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from official import enable_module_imports
    enable_module_imports()
    enter(Path(sys.argv[1]))


if __name__ == '__main__':
    main()
