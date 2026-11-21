"""Restore an explicitly hashed checkpoint from immutable public revisions."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import quote


def manifest(path):
    value = json.loads(Path(path).read_bytes())
    if (value.get('schema') != 'deeplinkengine-model-download/v1'
            or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', value.get('repository', ''))
            or not isinstance(value.get('files'), dict) or not value['files']):
        raise ValueError('Invalid pinned model manifest')
    for name, row in value['files'].items():
        if (not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..')
                or type(row.get('bytes')) is not int or row['bytes'] <= 0
                or not re.fullmatch(r'[0-9a-f]{64}', row.get('sha256', ''))
                or not re.fullmatch(r'[0-9a-f]{40}', row.get('revision', ''))):
            raise ValueError('Invalid immutable model file: ' + name)
    return value


def restore(lock, output):
    from official_image import digest
    value = manifest(lock)
    output = Path(output).absolute()
    if output.is_symlink():
        raise ValueError('Model destination must not be a symlink')
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema='deeplinkengine-model-restoration/v1', started=time.time(),
                   manifest_sha256=digest(lock), repository=value['repository'], files={}, status='running')
    def one(item):
        name, row = item
        target = output / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError('Unsafe existing model file: ' + name)
        url = ('https://modelscope.cn/models/' + value['repository'] + '/resolve/'
               + row['revision'] + '/' + quote(name))
        if not target.exists():
            partial = target.with_name(name + '.download-part')
            if partial.is_symlink() or (partial.exists() and not partial.is_file()):
                raise ValueError('Unsafe model partial file')
            result = subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                '--retry', '3', '--retry-delay', '2', '--connect-timeout', '20', '--max-time', '7200',
                '--continue-at', '-', '--output', str(partial), url], timeout=7500)
            if result.returncode or partial.stat().st_size != row['bytes'] or digest(partial) != row['sha256']:
                raise ValueError('Model download failed or differs: ' + name)
            partial.replace(target)
        elif target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
            raise ValueError('Existing model file differs; preserve it and choose a new directory: ' + name)
        print('MODEL_FILE_VERIFIED', name, row['sha256'], flush=True)
        return name, dict(bytes=row['bytes'], sha256=row['sha256'], revision=row['revision'], url=url)
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipt['files'] = dict(pool.map(one, value['files'].items()))
    receipt.update(status='completed', ended=time.time())
    from official_baselines import save
    save(output / 'model-restoration.json', receipt)
    return receipt
