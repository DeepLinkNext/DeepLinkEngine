"""Restore only Python bytecode drift from the original cached OCI bytes."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def users(sdk):
    """Match native mappings by inode as well as paths across mount namespaces."""
    sdk = Path(sdk).resolve()
    prefix = str(sdk) + '/'
    identities = set()
    for path in (sdk / 'rootfs').rglob('*'):
        if path.is_file():
            info = path.stat()
            identities.add((os.major(info.st_dev), os.minor(info.st_dev), info.st_ino))
    found = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            reasons = []
            arguments = (process / 'cmdline').read_bytes().split(b'\0')
            if any(os.fsdecode(value).startswith(prefix) for value in arguments if value):
                reasons.append('sdk-command-argument')
            for key in ('exe', 'cwd'):
                try:
                    if os.readlink(process / key).startswith(prefix):
                        reasons.append(key)
                except FileNotFoundError:
                    pass
            for line in (process / 'maps').read_text().splitlines():
                row = line.split(None, 5)
                if len(row) < 5:
                    continue
                major, minor = row[3].split(':')
                if (int(major, 16), int(minor, 16), int(row[4])) in identities:
                    reasons.append('mapped-sdk-inode')
                    break
            for descriptor in (process / 'fd').iterdir():
                try:
                    info = descriptor.stat()
                    if (os.readlink(descriptor).startswith(prefix)
                            or (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino) in identities):
                        reasons.append('open-sdk-file')
                        break
                except FileNotFoundError:
                    pass
            if reasons:
                found.append(dict(pid=int(process.name), start_ticks=int(fields[19]), reasons=sorted(set(reasons))))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            found.append(dict(pid=int(process.name), inspection='unreadable; quiescence is unproven'))
    return found


def classify(expected, observed):
    changed = sorted(name for name in set(expected) | set(observed) if expected.get(name) != observed.get(name))
    for name in changed:
        pure = PurePosixPath(name)
        if (pure.is_absolute() or '..' in pure.parts or '\\' in name or not name.startswith('rootfs/')
                or not name.endswith('.pyc')):
            raise ValueError('SDK source/native/unknown drift is not automatically repaired: ' + name)
        for record in (expected.get(name), observed.get(name)):
            if record is not None and (set(record) != {'bytes', 'sha256'} or not re.fullmatch(r'[0-9a-f]{64}', record.get('sha256', ''))):
                raise ValueError('Only ordinary Python bytecode files can be restored: ' + name)
        if name not in expected:
            match = re.fullmatch(r'(.+)/__pycache__/(.+)\.cpython-\d+(?:\.opt-\d+)?\.pyc', name)
            source = match.group(1) + '/' + match.group(2) + '.py' if match else name[:-1]
            if source not in expected or expected[source] != observed.get(source):
                raise ValueError('Unknown extra bytecode has no unchanged original source: ' + name)
    return changed


def preparation_state(sdk, lock, receipt, expected):
    """A completed legacy extraction need not have a later owner marker."""
    from official_image import digest, matches_version
    from official_sdk_state import refuse_active_owner
    marker = sdk / 'preparation.json'
    if os.path.lexists(marker):
        if marker.is_symlink() or not marker.is_file():
            raise ValueError('Existing SDK preparation marker is not an ordinary file')
        # Never interpret an unreadable, unknown or live owner as legacy.
        return refuse_active_owner(json.loads(marker.read_bytes()))
    metadata = receipt.get('metadata', {})
    flags = {'host_driver_retained': True, 'native_packages_imported': False,
             'model_started': False, 'source_code_patched': False,
             'distribution_versions_retained_exactly': True}
    versions = metadata.get('versions', {}) if isinstance(metadata, dict) else {}
    required = {'vllm', 'vllm-ascend', 'torch', 'torch-npu', 'transformers', 'triton-ascend'}
    origins = metadata.get('origins', {}) if isinstance(metadata, dict) else {}
    if (receipt.get('baseline_id') != lock['id'] or type(receipt.get('files')) is not int
            or receipt['files'] != len(expected) or not expected
            or any(receipt.get(name) is not value for name, value in flags.items())
            or not isinstance(metadata, dict) or type(metadata.get('no_site')) is not int
            or metadata['no_site'] != 1 or not isinstance(versions, dict)
            or not required.issubset(versions) or any(not isinstance(versions[n], str) or not versions[n] for n in required)
            or not matches_version('vllm', versions['vllm'], lock['upstream_version'])
            or not matches_version('vllm-ascend', versions['vllm-ascend'], lock['ascend_version'])
            or not isinstance(origins, dict) or set(origins) != {'torch', 'torch_npu', 'vllm', 'vllm_ascend'}):
        raise ValueError('Missing preparation marker requires a complete original SDK installation receipt')
    root = (sdk / 'rootfs').resolve()
    for value in (metadata.get('python'), metadata.get('prefix'), *origins.values()):
        if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).resolve().is_relative_to(root):
            raise ValueError('Completed SDK metadata resolves outside its original rootfs')
    if Path(metadata['python']).resolve().parent.parent != Path(metadata['prefix']).resolve():
        raise ValueError('Completed SDK interpreter/prefix identity differs')
    state = dict(mode='legacy-completed-sdk', preparation_marker_present=False,
                 past_preparation_owner_inferred=False,
                 installation_receipt_sha256=digest(sdk / 'installation.json'),
                 original_installed_manifest_sha256=receipt['installed_files_sha256'])
    auxiliary = sdk / 'restoration-recovery.json'
    if os.path.lexists(auxiliary):
        if auxiliary.is_symlink() or not auxiliary.is_file():
            raise ValueError('Legacy completion auxiliary evidence is not an ordinary file')
        proof = json.loads(auxiliary.read_bytes())
        if (proof.get('all_original_extracted_files_reverified') is not True
                or proof.get('source_code_patched') is not False or proof.get('model_started') is not False
                or proof.get('installed_files_sha256') != receipt['installed_files_sha256']):
            raise ValueError('Legacy completion auxiliary evidence differs from the original installation')
        state['restoration_recovery_sha256'] = digest(auxiliary)
    return state


def original_bytes(lock, cache, wanted):
    """Apply the original ordered OCI overlay to selected regular files only."""
    from official_image import digest
    selected = {}
    proofs = []
    for layer in lock['layers']:
        print('SDK_BYTECODE_OCI_VERIFY', layer['digest'][:19], 'bytes', layer['size'], flush=True)
        blob = cache / layer['digest'].replace(':', '-')
        if (blob.is_symlink() or not blob.is_file() or blob.stat().st_size != layer['size']
                or digest(blob) != layer['digest'].split(':')[1]):
            raise ValueError('Original whole OCI cache is missing or changed; no download attempted: ' + str(blob))
        replacements = {}
        deleted = []
        print('SDK_BYTECODE_OCI_REPLAY', layer['digest'][:19], flush=True)
        with tarfile.open(blob, 'r|*') as archive:
            for member in archive:
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or '..' in pure.parts or '\\' in member.name:
                    raise ValueError('Unsafe original OCI member')
                name = pure.as_posix()
                if pure.name == '.wh..wh..opq':
                    deleted.append((pure.parent.as_posix().rstrip('/') + '/', True))
                elif pure.name.startswith('.wh.'):
                    deleted.append(((pure.parent / pure.name[4:]).as_posix(), False))
                elif name in wanted:
                    if not member.isfile():
                        raise ValueError('Original bytecode member is not a regular file: ' + name)
                    if member.size != wanted[name]['bytes']:
                        # A later layer can replace an earlier version of the file.
                        replacements[name] = None
                    else:
                        replacements[name] = archive.extractfile(member).read(member.size + 1)
        for name in list(selected):
            if any((name.startswith(target) if opaque else name == target or name.startswith(target + '/')) for target, opaque in deleted):
                selected.pop(name)
        selected.update(replacements)
        proofs.append(dict(path=str(blob), sha256=layer['digest'].split(':')[1], bytes=layer['size']))
    for name, expected in wanted.items():
        raw = selected.get(name)
        if raw is None or len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('Original OCI replay does not match the unchanged SDK manifest: ' + name)
    print('SDK_BYTECODE_OCI_REPLAY_VERIFIED', 'layers', len(proofs), 'selected_files', len(selected), flush=True)
    return selected, proofs


def repair_locked(sdk, lock_path, cache, observed=None):
    """Caller holds sdk_preparation for the complete verification and recovery."""
    from official_image import read_lock, digest, file_map
    from official_sdk_state import owner_info
    if Path(sdk).is_symlink():
        raise ValueError('SDK repair target must not be a symbolic link')
    sdk, cache, lock_path = Path(sdk).resolve(), Path(cache).resolve(), Path(lock_path)
    if sdk == Path('/') or (sdk / 'rootfs').is_symlink():
        raise ValueError('SDK repair requires its original ordinary rootfs directory')
    lock = read_lock(lock_path)
    names = ('image-lock.json', 'image-manifest.json', 'image-config.json', 'installed-files.json', 'installation.json')
    if any((sdk / name).is_symlink() or not (sdk / name).is_file() for name in names):
        raise ValueError('SDK recovery requires ordinary original installation/image/manifest metadata')
    original = {name: (sdk / name).read_bytes() for name in names}
    receipt = json.loads(original['installation.json'])
    if (original['image-lock.json'] != lock_path.read_bytes()
            or hashlib.sha256(original['image-manifest.json']).hexdigest() != lock['manifest_digest'].split(':')[1]
            or hashlib.sha256(original['image-config.json']).hexdigest() != lock['config']['digest'].split(':')[1]
            or receipt.get('schema') != 'deeplinkengine-official-sdk/v1'
            or receipt.get('image_manifest') != lock['manifest_digest']
            or hashlib.sha256(original['installed-files.json']).hexdigest() != receipt.get('installed_files_sha256')):
        raise ValueError('SDK recovery requires the unchanged original image and installed manifest')
    expected = json.loads(original['installed-files.json'])
    print('SDK_BYTECODE_VERIFY_START', str(sdk), 'files', len(expected), flush=True)
    observed = file_map(sdk) if observed is None else observed
    if observed == expected:
        print('SDK_BYTECODE_VERIFIED', 'files', len(expected), flush=True)
        return None
    changed = classify(expected, observed)
    print('SDK_BYTECODE_DRIFT', 'pyc_files', len(changed), flush=True)
    owner_state = preparation_state(sdk, lock, receipt, expected)
    print('SDK_BYTECODE_USERS_CHECK', 'before_staging', flush=True)
    active = users(sdk)
    if active:
        raise ValueError('SDK is in use or quiescence is unproven; do not repair: ' + json.dumps(active))
    journal = cache / 'sdk-bytecode-recovery' / uuid.uuid4().hex
    journal.mkdir(parents=True)
    record = dict(schema='deeplinkengine-original-sdk-bytecode-recovery/v1', status='running',
                  sdk=str(sdk), started=time.time(), owner=owner_info(), old_preparation_owner=owner_state,
                  original_installed_manifest_sha256=receipt['installed_files_sha256'],
                  image_manifest=lock['manifest_digest'], changes={name: dict(expected=expected.get(name), observed=observed.get(name)) for name in changed},
                  no_downloads=True, original_manifest_rewritten=False, original_full_map_verified=False)
    save(journal / 'result.json', record)
    save(journal / 'before-file-map.json', observed)
    try:
        required = sum(observed.get(name, {}).get('bytes', 0) + expected.get(name, {}).get('bytes', 0) for name in changed)
        if shutil.disk_usage(cache).free < required + 2 * 1024**3:
            raise OSError('Bytecode recovery requires its exact backup plus 2 GiB free')
        wanted = {name[len('rootfs/'):]: expected[name] for name in changed if name in expected}
        restored, layers = original_bytes(lock, cache / 'official-oci', wanted)
        record['verified_original_layers'] = layers
        print('SDK_BYTECODE_BACKUP', 'files', len(changed), flush=True)
        for name in changed:
            path = sdk / name
            if not path.resolve().is_relative_to((sdk / 'rootfs').resolve()) or path.is_symlink():
                raise ValueError('Bytecode recovery path escaped the admitted SDK')
            if name in observed:
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != observed[name]['sha256']:
                    raise ValueError('Bytecode changed during recovery preparation; no repair committed')
                saved = journal / 'original-drifted-bytecode' / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(raw)
        save(journal / 'result.json', record)
        print('SDK_BYTECODE_USERS_CHECK', 'before_commit', flush=True)
        active = users(sdk)
        if active:
            raise ValueError('SDK acquired a user during recovery staging; originals retained: ' + json.dumps(active))
        print('SDK_BYTECODE_VERIFY_BEFORE_COMMIT', 'files', len(expected), flush=True)
        if file_map(sdk) != observed:
            raise ValueError('SDK changed after staging; no restoration committed')
        print('SDK_BYTECODE_COMMIT', 'files', len(changed), flush=True)
        for name in changed:
            path = sdk / name
            if name not in expected:
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
                temporary = path.with_name(path.name + '.recovery-' + uuid.uuid4().hex)
                temporary.write_bytes(restored[name[len('rootfs/'):]])
                temporary.chmod(mode)
                temporary.replace(path)
        print('SDK_BYTECODE_VERIFY_AFTER_COMMIT', 'files', len(expected), flush=True)
        after = file_map(sdk)
        save(journal / 'after-file-map.json', after)
        if after != expected or any((sdk / name).read_bytes() != raw for name, raw in original.items()):
            raise ValueError('Restored SDK differs from its unchanged original full manifest')
        record.update(status='restored', original_full_map_verified=True, files_restored=len(changed))
    except BaseException as error:
        record.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        record['ended'] = time.time()
        save(journal / 'result.json', record)
    print('SDK_BYTECODE_RESTORED', 'files', len(changed), 'journal', str(journal / 'result.json'), flush=True)
    return dict(path=str(journal / 'result.json'), sha256=digest(journal / 'result.json'),
                status=record['status'], sdk=str(sdk), files_restored=len(changed),
                original_full_map_verified=True, original_manifest_rewritten=False,
                original_installed_manifest_sha256=receipt['installed_files_sha256'])


def verify_or_repair(sdk, cache):
    """A host-interpreter preflight before the first private SDK exec."""
    from official_sdk_state import sdk_preparation
    sdk = Path(sdk).absolute()
    if sdk.is_symlink():
        raise ValueError('SDK repair target must not be a symbolic link')
    if json.loads((sdk / 'installation.json').read_bytes()).get('schema') != 'deeplinkengine-official-sdk/v1':
        return None  # Derived SDKs retain their separate original admission.
    sdk = sdk.resolve()
    identity = json.loads((sdk / 'image-lock.json').read_bytes())
    version = identity.get('upstream_version', '')
    if not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('Unknown official SDK selection')
    lock = ROOT / 'engine/official-baselines' / (version + '.json')
    with sdk_preparation(sdk):
        return repair_locked(sdk, lock, cache)
