"""Warm the pinned baseline OCI cache with bounded, resumable public downloads."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import http.client
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request


def digest(path):
    with path.open('rb') as stream:
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            hasher.update(block)
        return hasher.hexdigest()


def fetch(cache, registry, kind, identity, size=None, attempts=3):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', identity):
        raise ValueError('Invalid baseline OCI digest')
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / identity.replace(':', '-')
    expected = identity.split(':')[1]
    with target.with_suffix('.download.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_symlink():
            raise ValueError('Cached OCI blob must not be a symlink')
        if target.exists():
            if (target.is_symlink() or not target.is_file()
                    or (size is not None and target.stat().st_size != size)
                    or digest(target) != expected):
                raise ValueError('Cached OCI blob changed: ' + identity)
            return target
        partial = target.with_suffix('.download.partial')
        if partial.is_symlink():
            raise ValueError('Baseline partial must not be a symlink')
        last_error = None
        for attempt in range(attempts):
            offset = partial.stat().st_size if partial.exists() else 0
            if size is not None and offset > size:
                raise ValueError('Baseline partial exceeds pinned size')
            if size is not None and offset == size:
                if digest(partial) != expected:
                    raise ValueError('Baseline OCI SHA-256 mismatch: ' + identity)
                partial.replace(target)
                return target
            headers = {'User-Agent': 'DeepLinkEngine-baseline-setup',
                       'Accept': 'application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json'}
            if offset:
                headers['Range'] = 'bytes=' + str(offset) + '-'
            request = urllib.request.Request(registry + kind + '/' + identity, headers=headers)
            try:
                print('BASELINE_DOWNLOAD', identity, 'attempt', attempt + 1, 'offset', offset, flush=True)
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status == 206:
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                        if (not match or int(match[1]) != offset
                                or (size is not None and int(match[3]) != size)):
                            raise ValueError('Baseline HTTP range differs from requested bytes')
                        mode = 'ab'
                    elif response.status == 200:
                        mode = 'wb'
                    else:
                        raise ValueError('Unexpected baseline HTTP status: ' + str(response.status))
                    transfer_started, transferred = time.monotonic(), 0
                    read = getattr(response, 'read1', response.read)
                    with partial.open(mode) as stream:
                        while block := read(4 * 1024 * 1024):
                            stream.write(block)
                            transferred += len(block)
                            if stream.tell() > (size if size is not None else 5 * 1024 * 1024):
                                raise ValueError('Baseline download exceeds pinned size')
                            elapsed = time.monotonic() - transfer_started
                            if elapsed > 1800 or (elapsed >= 45 and transferred / elapsed < 64 * 1024):
                                raise OSError('Baseline Python transfer is slow; retain verified-contract partial and retry transport')
                if size is not None and partial.stat().st_size != size:
                    raise OSError('Incomplete baseline OCI download')
                if digest(partial) != expected:
                    raise ValueError('Baseline OCI SHA-256 mismatch: ' + identity)
                partial.replace(target)
                print('BASELINE_VERIFIED', identity, target.stat().st_size, flush=True)
                return target
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as error:
                last_error = error
                print('BASELINE_RETRY', identity, type(error).__name__, flush=True)
                if attempt + 1 < attempts:
                    time.sleep(min(attempt + 1, 3))
        # Some registry CDN TLS routes work with the system curl transport but
        # time out with Python/OpenSSL. Keep the same URL and digest contract.
        curl = shutil.which('curl')
        if curl:
            print('BASELINE_CURL_DOWNLOAD', identity, flush=True)
            result = subprocess.run(
                [curl, '--fail', '--location', '--silent', '--show-error',
                 '--connect-timeout', '15', '--speed-limit', '1024', '--speed-time', '45',
                 '--max-time', '1800', '--retry', '2', '--retry-delay', '1', '--retry-all-errors',
                 '--continue-at', '-', '--max-filesize', str(size if size is not None else 5 * 1024 * 1024),
                 '--output', str(partial), registry + kind + '/' + identity],
                capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                if ((size is not None and partial.stat().st_size != size) or digest(partial) != expected):
                    raise ValueError('Baseline OCI size/SHA-256 mismatch after curl: ' + identity)
                partial.replace(target)
                print('BASELINE_VERIFIED', identity, target.stat().st_size, flush=True)
                return target
            last_error = result.stderr[-1000:]
        raise OSError('Baseline download failed after bounded retries: ' + identity + ': ' + str(last_error))


def prepare(cache):
    # Constants remain owned by the checksum-verified original tool kit.
    from restore_b0_image import REGISTRY, MANIFEST
    cache = Path(cache)
    if (cache / 'original018-sdk').is_dir():
        return
    blobs = cache / 'b0-oci'
    manifest = json.loads(fetch(blobs, REGISTRY, 'manifests', MANIFEST).read_bytes())
    config = manifest['config']
    image = json.loads(fetch(blobs, REGISTRY, 'blobs', config['digest'], config['size']).read_bytes())
    if image['architecture'] != 'arm64' or image['os'] != 'linux':
        raise ValueError('Pinned baseline architecture differs')
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: fetch(blobs, REGISTRY, 'blobs', item['digest'], item['size']), manifest['layers']))
