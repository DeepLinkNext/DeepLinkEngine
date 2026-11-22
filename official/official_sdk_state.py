"""Serialize private SDK preparation and retain live or unknown owners."""
from contextlib import contextmanager
import os
from pathlib import Path
import platform
import stat
import uuid


def _host_identity():
    boot = (Path('/proc/sys/kernel/random/boot_id').read_text()).strip()
    return dict(hostname=platform.node(), boot_id=str(uuid.UUID(boot)))


def _process_identity(pid):
    try:
        value = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(state=value[0], start_ticks=int(value[19]))


def owner_info():
    host = _host_identity()
    process = _process_identity(os.getpid())
    if process is None:
        raise ValueError('Preparation owner identity is unavailable')
    return dict(schema='deeplinkengine-sdk-preparation-owner/v1', **host,
                pid=os.getpid(), start_ticks=process['start_ticks'])


def refuse_active_owner(marker):
    """Return a terminated owner's identity; legacy or foreign state is unknown."""
    owner = marker.get('owner') if isinstance(marker, dict) else None
    if (not isinstance(owner, dict)
            or owner.get('schema') != 'deeplinkengine-sdk-preparation-owner/v1'
            or type(owner.get('pid')) is not int or owner['pid'] <= 0
            or type(owner.get('start_ticks')) is not int or owner['start_ticks'] <= 0
            or not isinstance(owner.get('hostname'), str) or not owner['hostname']
            or not isinstance(owner.get('boot_id'), str)):
        raise ValueError('Unknown or legacy SDK preparation owner; preserve original directory')
    try:
        boot = str(uuid.UUID(owner['boot_id']))
        host = _host_identity()
    except (OSError, ValueError) as error:
        raise ValueError('Cannot verify SDK preparation host; preserve original directory') from error
    if owner['hostname'] != host['hostname']:
        raise ValueError('Foreign SDK preparation owner cannot be proven stopped')
    reason = 'different_boot'
    if boot == host['boot_id']:
        try:
            process = _process_identity(owner['pid'])
        except (OSError, ValueError, IndexError) as error:
            raise ValueError('Cannot verify SDK preparation process; preserve original directory') from error
        if process is None:
            reason = 'process_absent'
        elif process['start_ticks'] != owner['start_ticks']:
            reason = 'pid_reused'
        elif process['state'] == 'Z':
            reason = 'zombie'
        else:
            raise ValueError('SDK preparation owner is still active; do not move its directory')
    return dict(owner=owner, preparation_owner_alive=False, reason=reason)


@contextmanager
def sdk_preparation(sdk):
    """Hold this lock across quarantine, extraction, verification and commit."""
    from baseline_setup import download_lock
    sdk = Path(sdk).absolute()
    if sdk.is_symlink() or (sdk.exists() and not sdk.is_dir()):
        raise ValueError('SDK preparation target must be an ordinary directory')
    parent = sdk.parent
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / (sdk.name + '.prepare.lock')
    if path.is_symlink():
        raise ValueError('SDK preparation lock must not be a link')
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('SDK preparation lock must be a regular file')
        with os.fdopen(fd, 'a+b') as stream:
            fd = None
            with download_lock(stream):
                yield dict(lock=str(path), owner=owner_info())
    finally:
        if fd is not None:
            os.close(fd)
