"""Build an explicitly selected upstream RC source candidate in a private SDK.

This is a source derivation, never an official released image. All preparation
is owned by the public run_all controller; no model is loaded during builds.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import tarfile
import time

from official_baselines import save

VERSION = '0.28.0rc1-source'
ID = 'upstream-vllm-0.28.0-ascend-0.28.0rc1-source'
BASE_MANIFEST = 'sha256:4ee78def8f33d59d48f116d1dfa793332c23c99ecab4f0d7dd5cd62d0fb4e6c1'
SOURCES = {
    'vllm': dict(version='0.28.0', filename='vllm-0.28.0.tar.gz', bytes=39991441,
                 sha256='ac96dd0ec5be9c13f2aa4bfb50498c4727110406f6e4980b58a4097e4d18634a',
                 url='https://files.pythonhosted.org/packages/22/39/096d88cfe34ee43a8b6e7c957d1ad6338fa5256474d6ff9a32008c6f8cd6/vllm-0.28.0.tar.gz'),
    'vllm-ascend': dict(version='0.28.0rc1', filename='vllm_ascend-0.28.0rc1.tar.gz', bytes=19632653,
                        sha256='bde8bf4207d225df86d8f3b7d9b2bd89ff5e5e56af0cbe4bf8b5b691b98bbe95',
                        url='https://files.pythonhosted.org/packages/b0/bc/afe9b8d66680ff14d04b15b18f13cd7dea601c583a8d1514792448a7a75c/vllm_ascend-0.28.0rc1.tar.gz'),
}


def identity():
    return dict(schema='deeplinkengine-upstream-source-candidate/v1', id=ID,
                version=VERSION, upstream_version='0.28.0', ascend_version='0.28.0rc1',
                candidate=True, official_release_image=False, base_image_manifest=BASE_MANIFEST,
                controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                soc_version='ascend910b1', sources=SOURCES,
                recipe='vllm-project/vllm-ascend@96df623103f921df1a4488170d106f36689acb59/Dockerfile',
                published_sdists=True, upstream_source_patched=False,
                git_tag_references=dict(vllm='2cf0a6915ce544dc493a0990f2ea38d81601128a',
                    vllm_ascend='96df623103f921df1a4488170d106f36689acb59'),
                sdist_git_file_equivalence_certified=False)


def key():
    return hashlib.sha256(json.dumps(identity(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe_extract(archive_path, destination):
    """Unpack source archives into an empty private directory, without links."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    top = None
    seen = set()
    with tarfile.open(archive_path) as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or '\\' in member.name or not name.parts:
                raise ValueError('Unsafe source archive path')
            if top is None:
                top = name.parts[0]
            if name.parts[0] != top:
                raise ValueError('One source archive root is required')
            relative = PurePosixPath(*name.parts[1:])
            if str(relative) == '.':
                if not member.isdir():
                    raise ValueError('Source archive root must be a directory')
                continue
            target = destination.joinpath(*relative.parts)
            if str(relative) in seen and not member.isdir():
                raise ValueError('Duplicate source archive member')
            seen.add(str(relative))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
            else:
                raise ValueError('Source archives must not contain links or special files')
    if not (destination / 'setup.py').is_file() or not (destination / 'PKG-INFO').is_file():
        raise ValueError('Complete official Python source distribution required')


def source_archive(item, cache):
    from official_image import digest
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / item['filename']
    if target.exists():
        if target.stat().st_size != item['bytes'] or digest(target) != item['sha256']:
            raise ValueError('Existing source archive is corrupt; preserve it for inspection')
        return target
    partial = cache / (item['filename'] + '.part')
    urls = [item['url'].replace('https://files.pythonhosted.org/', 'https://pypi.tuna.tsinghua.edu.cn/'), item['url']]
    for url in urls:
        command = ['curl', '--fail', '--location', '--retry', '2', '--connect-timeout', '15',
                   '--max-time', '600', '--speed-limit', '1024', '--speed-time', '60',
                   '--continue-at', '-', '--output', str(partial), url]
        result = subprocess.run(command, capture_output=True, text=True, timeout=650)
        if result.returncode == 0 and partial.stat().st_size == item['bytes'] and digest(partial) == item['sha256']:
            partial.replace(target)
            save(cache / (item['filename'] + '.download.json'), dict(url=url, original_url=item['url'],
                bytes=item['bytes'], sha256=item['sha256'], independently_verified=True))
            return target
    raise RuntimeError('Official source archive download failed; preserve .part')


def command(argv, env, cwd, output, name, timeout=7200):
    """Keep exact build logs and terminate only this owned process group."""
    receipt = dict(command=[str(v) for v in argv], cwd=str(cwd), started=time.time(), model_started=False)
    process = None
    try:
        with (output / (name + '.log')).open('xb') as stream:
            process = subprocess.Popen(argv, env=env, cwd=cwd, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        receipt['pid'] = process.pid
        receipt['start_ticks'] = Path('/proc', str(process.pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
        save(output / (name + '.json'), receipt)
        code = process.wait(timeout=timeout)
        receipt['returncode'] = code
        if code:
            raise RuntimeError('Source build failed: ' + name + '; preserve ' + name + '.log')
    except BaseException as error:
        receipt['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        receipt['ended'] = time.time()
        receipt['quiescent'] = process is None or process.poll() is not None
        save(output / (name + '.json'), receipt)


def remove_old_editables(out, site):
    """Remove only the copied SDK's old core/Ascend editable registrations."""
    out, site = Path(out).resolve(), Path(site).resolve()
    if not site.is_relative_to(out):
        raise ValueError('Candidate package destination escaped its private SDK')
    removed = []
    for path in list(site.glob('*.dist-info')) + list(site.glob('*.egg-info')):
        metadata = path / ('METADATA' if path.suffix == '.dist-info' else 'PKG-INFO')
        if not metadata.is_file():
            continue
        names = re.findall(r'^Name: (.+)$', metadata.read_text(errors='replace'), re.MULTILINE)
        if len(names) == 1 and names[0].lower().replace('_', '-') in ('vllm', 'vllm-ascend'):
            removed.append(str(path.relative_to(out)))
            shutil.rmtree(path)
    for path in site.glob('__editable__*vllm*'):
        if path.is_file() and not path.is_symlink():
            removed.append(str(path.relative_to(out)))
            path.unlink()
    return removed


def reuse(output):
    """Reuse only an entirely completed, unmodified source derivation."""
    from official_image import digest, file_map, probe
    output = Path(output).resolve()
    installed = json.loads((output / 'installation.json').read_bytes())
    lock = json.loads((output / 'image-lock.json').read_bytes())
    expected = identity()
    expected.update(manifest_digest='sha256:' + key(), manifest_kind='source-build-derivation')
    if (lock != expected or installed.get('schema') != 'deeplinkengine-source-candidate-sdk/v1'
            or installed.get('source_candidate') != identity()
            or installed.get('image_manifest') != expected['manifest_digest']
            or digest(output / 'installed-files.json') != installed['installed_files_sha256']
            or file_map(output) != json.loads((output / 'installed-files.json').read_bytes())):
        raise ValueError('Cached source candidate identity or installed files changed')
    installed['metadata'] = probe(output, output / 'reuse-probe-cache', lock)
    return installed


def prepare(base_lock_path, cache, output):
    from official_image import prepare as prepare_image, read_lock, digest, file_map, probe, environment
    cache, output = Path(cache).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Select a new source candidate SDK; failed builds are immutable')
    base_lock = read_lock(base_lock_path)
    if base_lock['manifest_digest'] != BASE_MANIFEST:
        raise ValueError('Source candidate requires its exact official CANN9.1 base')
    base = cache / 'official-sdks/0.23.0-4ee78def8f33d59d'
    if not base.exists():
        prepare_image(base_lock_path, cache / 'official-oci', base)
    record = json.loads((base / 'installation.json').read_bytes())
    if (record['image_manifest'] != BASE_MANIFEST
            or (base / 'image-lock.json').read_bytes() != Path(base_lock_path).read_bytes()
            or digest(base / 'installed-files.json') != record['installed_files_sha256']
            or file_map(base) != json.loads((base / 'installed-files.json').read_bytes())):
        raise ValueError('Official source candidate base integrity failed')
    if shutil.disk_usage(cache).free < 100 * 1024**3:
        raise ValueError('Source builds require 100 GiB free in the private cache')
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base, output, symlinks=True)
    for name in ('installation.json', 'image-lock.json', 'installed-files.json', 'image-manifest.json'):
        (output / name).rename(output / ('base-' + name))
    selection = identity()
    selection.update(manifest_digest='sha256:' + key(), manifest_kind='source-build-derivation')
    save(output / 'image-lock.json', selection)
    save(output / 'source-candidate-lock.json', identity())
    build = output / 'source-build'
    build.mkdir()
    receipt = dict(schema='deeplinkengine-upstream-source-build/v1', candidate=identity(),
        status='building', started=time.time(), model_started=False, base_sdk=str(base),
        base_installation_sha256=digest(base / 'installation.json'), shared_environment_modified=False)
    save(build / 'receipt.json', receipt)
    try:
        for name, item in SOURCES.items():
            path = output / 'rootfs/vllm-workspace' / name
            if path.exists():
                if path.is_symlink() or not path.resolve().is_relative_to(output):
                    raise ValueError('Candidate source destination escaped private SDK')
                shutil.rmtree(path)
            source = source_archive(item, cache / 'source-candidate-archives')
            safe_extract(source, path)
            metadata = (path / 'PKG-INFO').read_text(errors='replace')
            if '\nVersion: ' + item['version'] + '\n' not in metadata:
                raise ValueError('Source distribution version differs from pinned candidate')
        python, env = environment(output, build / 'cache')
        site = Path(env['PYTHONHOME']) / 'lib/python3.12/site-packages'
        receipt['removed_copied_editable_registrations'] = remove_old_editables(output, site)
        env.update(SOC_VERSION='ascend910b1', VLLM_TARGET_DEVICE='empty',
                   ASCEND_RT_VISIBLE_DEVICES='', VLLM_BATCH_INVARIANT='1',
                   PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple',
                   PIP_DISABLE_PIP_VERSION_CHECK='1',
                   PIP_CACHE_DIR=str(build / 'cache/pip'),
                   CARGO_HOME=str(build / 'cache/cargo'), RUSTUP_HOME=str(build / 'cache/rustup'),
                   CMAKE_BUILD_PARALLEL_LEVEL='16', MAX_JOBS='16')
        env['PATH'] = str(python.parent) + ':' + env['CARGO_HOME'] + '/bin:' + env['PATH']
        core = output / 'rootfs/vllm-workspace/vllm'
        ascend = output / 'rootfs/vllm-workspace/vllm-ascend'
        command([str(python), '-S', '-m', 'pip', 'install', '-e', str(core) + '[audio]',
                 '--extra-index-url', 'https://download.pytorch.org/whl/cpu/'], env, core, build, '01-core')
        command([str(python), '-S', '-m', 'pip', 'uninstall', '-y', 'triton'], env, core, build, '02-remove-base-triton')
        command([str(python), '-S', '-m', 'pip', 'install', '-e', str(ascend),
                 '--extra-index-url', 'https://download.pytorch.org/whl/cpu/',
                 '--extra-index-url', 'https://mirrors.huaweicloud.com/ascend/repos/pypi'],
                env, ascend, build, '03-ascend')
        command([str(python), '-S', '-m', 'pip', 'uninstall', '-y', 'triton', 'triton-ascend'], env, core, build, '04-remove-triton-overlap')
        command([str(python), '-S', '-m', 'pip', 'install', 'triton-ascend==3.2.2',
                 '--extra-index-url', 'https://mirrors.huaweicloud.com/ascend/repos/pypi'], env, core, build, '05-ascend-triton')
        command([str(python), '-S', '-m', 'pip', 'install', 'concurrent-log-handler', 'setuptools-rust'],
                env, core, build, '06-build-tools')
        # The upstream script bootstraps rustup. Keep its HOME unchanged; set
        # private Rust directories and preinstall before its HOME/.cargo/env
        # fallback, so subsequent upstream build sees rustup on PATH.
        if shutil.which('rustup', path=env['PATH']) is None:
            command(['curl', '--fail', '--location', '--retry', '2', '--max-time', '180',
                     '--output', str(build / 'rustup-init.sh'), 'https://sh.rustup.rs'], env, core, build, '07-rust-bootstrap-download', timeout=200)
            receipt['rustup_bootstrap_sha256'] = digest(build / 'rustup-init.sh')
            command(['sh', str(build / 'rustup-init.sh'), '-y', '--no-modify-path', '--default-toolchain', 'none'],
                    env, core, build, '08-rust-bootstrap')
        command(['bash', str(core / 'build_rust.sh')], env, core, build, '09-upstream-rust')
        # Preserve dependency inconsistencies produced by the original ordered
        # recipe. Their exact diagnostics must remain visible to reviewers.
        result = subprocess.run([str(python), '-S', '-m', 'pip', 'check'], env=env,
                                capture_output=True, text=True, timeout=120)
        (build / 'dependency-check.log').write_text(result.stdout + result.stderr)
        receipt['dependency_check_returncode'] = result.returncode
        observed = probe(output, build / 'metadata-probe', selection)
        files = file_map(output)
        save(output / 'installed-files.json', files)
        installed = dict(schema='deeplinkengine-source-candidate-sdk/v1', baseline_id=ID,
            image_manifest=selection['manifest_digest'], manifest_kind='source-build-derivation',
            base_image_manifest=BASE_MANIFEST, installed_files_sha256=digest(output / 'installed-files.json'),
            metadata=observed, model_started=False, official_release_image=False,
            upstream_source_patched=False, source_candidate=identity())
        save(output / 'installation.json', installed)
        receipt.update(status='prepared', metadata=observed)
        return installed
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        receipt['ended'] = time.time()
        save(build / 'receipt.json', receipt)
