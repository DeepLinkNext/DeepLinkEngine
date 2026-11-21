"""Select an already-mounted NFS4 view of the exact same private directory."""
from pathlib import Path, PurePosixPath
import os
import re


def mounts(text):
    records = []
    for line in text.splitlines():
        words = line.split()
        if len(words) < 4 or words[2] not in ('nfs', 'nfs4') or ':/' not in words[0]:
            continue
        source, export = words[0].rsplit(':', 1)
        mount = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), words[1])
        records.append(dict(server=source, export=PurePosixPath(export), mount=Path(mount),
                            nfs4=words[2] == 'nfs4' or any(v.startswith('vers=4') for v in words[3].split(','))))
    return records


def candidates(path, records):
    """Map export-relative identity, never another similarly named workspace."""
    path = Path(path)
    origins = [row for row in records if path.is_relative_to(row['mount'])]
    if not origins:
        return []
    original = max(origins, key=lambda row: len(row['mount'].parts))
    if original['nfs4']:
        return []
    remote = original['export'] / path.relative_to(original['mount']).as_posix()
    result = []
    for row in records:
        if row['nfs4'] and row['server'] == original['server'] and remote.is_relative_to(row['export']):
            alias = row['mount'].joinpath(*remote.relative_to(row['export']).parts)
            result.append(dict(original=str(path), preferred=str(alias), server=row['server'],
                               exact_remote_path=str(remote), original_mount=str(original['mount']),
                               preferred_mount=str(row['mount'])))
    return result


def prefer(path):
    path = Path(path).resolve()
    for row in candidates(path, mounts(Path('/proc/mounts').read_text())):
        original, alias = path, Path(row['preferred'])
        while not os.path.lexists(original):
            original, alias = original.parent, alias.parent
        try:
            left, right = original.stat(), alias.stat()
        except OSError:
            continue
        if (left.st_ino, left.st_uid, left.st_gid, left.st_mtime_ns) != (right.st_ino, right.st_uid, right.st_gid, right.st_mtime_ns):
            continue
        row['existing_ancestor'] = str(original)
        row['existing_ancestor_inode'] = left.st_ino
        row['export_and_inode_verified'] = True
        return Path(row['preferred']), row
    return path, None
