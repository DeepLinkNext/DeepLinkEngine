"""Complete standard wheel ownership metadata in a private runtime environment."""
import base64
import csv
import hashlib
import inspect
import io
from pathlib import Path, PurePosixPath


def prepare(python):
    prefix = Path(python).expanduser().absolute().parent.parent
    if not (prefix / 'pyvenv.cfg').is_file():
        return {'status': 'not_private', 'generated': []}
    site = prefix / 'lib/python3.12/site-packages'
    generated = []
    if site.is_symlink() or not site.resolve().is_relative_to(prefix.resolve()):
        raise ValueError('Private site-packages must stay inside its venv')
    if not site.is_dir():
        return {'status': 'no_site', 'generated': []}
    import fcntl
    with (prefix / '.package-metadata.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for folder in sorted(site.glob('*.dist-info')):
            if folder.is_symlink() or not folder.is_dir():
                continue
            top = folder / 'top_level.txt'
            record = folder / 'RECORD'
            if top.is_symlink() or record.is_symlink():
                raise ValueError('Private package metadata must not be a symlink')
            if (top.exists() and top.read_text().strip()) or not record.is_file():
                continue
            rows = list(csv.reader(io.StringIO(record.read_text())))
            candidates = {}
            for row in rows:
                if not row or not row[0]:
                    continue
                relative = PurePosixPath(row[0])
                if relative.is_absolute() or '..' in relative.parts:
                    continue
                name = relative.parts[0] if len(relative.parts) > 1 else inspect.getmodulename(row[0])
                if name and name.isidentifier():
                    candidates[name] = relative.parts[0]
            names = {name for name, path in candidates.items() if (site / path).exists()}
            if not names:
                continue
            content = ('\n'.join(sorted(names)) + '\n').encode()
            checksum = hashlib.sha256(content).digest()
            temporary = top.with_suffix('.tmp')
            temporary.write_bytes(content)
            temporary.replace(top)
            name = top.relative_to(site).as_posix()
            rows = [row for row in rows if row and row[0] != name]
            rows.append([name, 'sha256=' + base64.urlsafe_b64encode(checksum).rstrip(b'=').decode(), str(len(content))])
            buffer = io.StringIO()
            csv.writer(buffer, lineterminator='\n').writerows(rows)
            temporary = record.with_suffix('.tmp')
            temporary.write_text(buffer.getvalue())
            temporary.replace(record)
            generated.append({'distribution': folder.name, 'modules': sorted(names), 'sha256': checksum.hex()})
    return {'status': 'completed', 'generated': generated}
