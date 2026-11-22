"""Admit the original device leases on hosts with protected sticky files."""

import errno
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time


def open_lease(path):
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        return os.open(path, flags)
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return os.open(path, flags)


def acquire(output, timeout, *, guard=None):
    if guard is None:
        import resource_guard as guard

    try:
        return guard.acquire(output, timeout)
    except PermissionError as error:
        # Linux protected_regular rejects O_CREAT for another user's existing
        # file in a sticky directory. Keep the original path and inode.
        if (error.errno != errno.EACCES or error.filename not in guard.paths()
                or not Path(error.filename).parent.stat().st_mode & stat.S_ISVTX
                or not stat.S_ISREG(Path(error.filename).lstat().st_mode)
                or os.environ.get('VLLMHUST_ENGINE_LOCKS')):
            raise
    return acquire_existing(output, timeout, guard)


def acquire_existing(output, timeout, guard):
    deadline = time.monotonic() + timeout
    while True:
        entries = []
        admitted = False
        try:
            for name in guard.paths():
                path = Path(name)
                fd = open_lease(path)
                entries.append(dict(path=name, fd=fd))
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                opened, published = os.fstat(fd), path.lstat()
                if (not stat.S_ISREG(opened.st_mode)
                        or (opened.st_dev, opened.st_ino) != (published.st_dev, published.st_ino)):
                    raise ValueError('Original lease path changed: ' + name)
            sample = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True,
                                    timeout=20, check=True).stdout
            cards = guard.cards()
            for card in cards:
                if not re.search(rf'\|\s*{card}\s+910B2\b', sample):
                    raise ValueError('Selected physical NPU pair must both be Ascend 910B2')
                if not re.search(rf'No running processes found in NPU {card}\b', sample):
                    raise BlockingIOError('Selected physical NPU pair has another running process')
            for entry in entries:
                os.set_inheritable(entry['fd'], True)
            os.environ['VLLMHUST_ENGINE_LOCKS'] = json.dumps(entries)
            result = guard.acquire(output, timeout)
            (output / 'resource-compatibility.json').write_text(json.dumps(dict(
                physical_cards=cards, exclusive_leases=guard.paths(), npu_before=sample,
                existing_lease_open_without_create=True, admitted=True,
            ), indent=2) + '\n')
            admitted = True
            return result
        except BlockingIOError as error:
            (output / 'resource-wait.json').write_text(json.dumps(dict(
                physical_cards=guard.cards(), reason=str(error), model_started=False, time=time.time(),
            ), indent=2) + '\n')
            if time.monotonic() >= deadline:
                raise TimeoutError(str(error)) from error
        finally:
            if not admitted:
                os.environ.pop('VLLMHUST_ENGINE_LOCKS', None)
                guard.release(entries)
        time.sleep(5)
