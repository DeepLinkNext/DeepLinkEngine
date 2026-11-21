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


def transport_base_url(value):
    from model_receiver import transport_base_url as validate
    return validate(value)


def restore(lock, output, *, transport_base_url=None):
    from model_receiver import restore as receive
    return receive(lock, output, transport_base_url=transport_base_url)
