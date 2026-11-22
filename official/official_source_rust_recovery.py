"""Resume a stopped candidate's final Rust stage within its original recipe."""

import json
import os
from pathlib import Path
import time

import official_source_candidate as candidate
from official_baselines import save
from official_execution import context_identity, sdk_run
from official_image import digest, environment, file_map, probe
from official_sdk_state import owner_info, refuse_active_owner


COMPLETED = (
    '01-core', '02-remove-base-triton', '03-ascend', '04-remove-triton-overlap',
    '05-ascend-triton', '06-concurrent-log-handler', '06-rust-build-tools',
    '07-rust-bootstrap-download', '08-rust-bootstrap',
)
RUST = '09-upstream-rust'


def read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing original source-build record: ' + str(path))
    return json.loads(path.read_bytes())


def admit(previous, cache, sdk):
    config = read(previous / 'results.json')
    [stage] = config['stages']
    if (config.get('schema') != 'deeplinkengine-official-baseline-campaign/v1'
            or config['status'] != 'incomplete' or config['model_execution_authorized'] is not False
            or config['configuration']['cache'] != str(cache)
            or config['configuration']['out'] != str(previous)
            or stage.get('baseline_id') != candidate.ID or stage.get('status') != 'failed'
            or not str(stage.get('error', '')).startswith('RuntimeError: Source build failed: ' + RUST)
            or stage.get('model_started') is not False):
        raise ValueError('Expected an original stopped CPU candidate preparation')
    if sdk.is_symlink() or not sdk.is_dir() or os.path.lexists(sdk / 'installation.json'):
        raise ValueError('Expected an unadmitted private candidate SDK')
    marker = read(sdk / 'source-preparation.json')
    identity = candidate.identity()
    if (marker.get('schema') != 'deeplinkengine-source-sdk-preparation/v1'
            or marker.get('candidate') != identity or marker.get('destination') != str(sdk)
            or marker.get('model_started') is not False):
        raise ValueError('Original SDK preparation identity differs')
    stopped = refuse_active_owner(marker)
    lock = read(sdk / 'image-lock.json')
    if (lock != dict(identity, manifest_digest='sha256:' + candidate.key(), manifest_kind='source-build-derivation')
            or read(sdk / 'source-candidate-lock.json') != identity):
        raise ValueError('Original candidate image lock differs')
    build = sdk / 'source-build'
    if build.is_symlink() or not build.is_dir():
        raise ValueError('The original private source-build directory differs')
    receipt = read(build / 'receipt.json')
    context = context_identity()
    if (receipt.get('schema') != 'deeplinkengine-upstream-source-build/v1'
            or receipt.get('status') != 'failed' or receipt.get('candidate') != identity
            or receipt.get('model_started') is not False
            or receipt.get('shared_environment_modified') is not False
            or not context or not context['source_build_system_root']
            or receipt['build_execution_context']['execution_root_receipt_sha256']
               != context['execution_root_receipt_sha256']):
        raise ValueError('Original build receipt or private tool root differs')
    base = sdk / 'base-installation.json'
    if (digest(base) != receipt['base_installation_sha256']
            or digest(sdk / 'base-installed-files.json') != read(base)['installed_files_sha256']):
        raise ValueError('Original base installation receipt differs')
    source = sdk / 'rootfs/vllm-workspace/vllm'
    evidence = {}
    for name in (*COMPLETED, RUST):
        path = build / (name + '.json')
        command = read(path)
        if (command.get('returncode') != (1 if name == RUST else 0)
                or command.get('quiescent') is not True or command.get('model_started') is not False):
            raise ValueError('Original build stage lacks an eligible terminal receipt: ' + name)
        evidence[name] = dict(receipt_sha256=digest(path), log_sha256=digest(path.with_suffix('.log')))
    log = (build / (RUST + '.log')).read_text(errors='replace')
    if (command['cwd'] != str(source)
            or command['command'][1:] != [str(source / 'build_rust.sh')]
            or 'git repository' not in log or 'Timeout was reached' not in log):
        raise ValueError('Original Rust stage did not stop at an eligible Git transport failure')
    return marker, receipt, lock, evidence, stopped


def retry(previous, cache, sdk, result):
    marker, original, lock, evidence, stopped = admit(previous, cache, sdk)
    build = sdk / 'source-build'
    result.mkdir()
    for name in ('source-preparation.json', 'source-build/receipt.json'):
        (result / Path(name).name).write_bytes((sdk / name).read_bytes())
    original_map = file_map(sdk)
    before = result / 'original-sdk-files.json'
    save(before, original_map)
    recovery = {
        'schema': 'deeplinkengine-source-rust-recovery/v1',
        'started': time.time(),
        'status': 'running',
        'source_campaign': str(previous),
        'source_campaign_sha256': digest(previous / 'results.json'),
        'source_owner': stopped,
        'source_build_receipt_sha256': digest(result / 'receipt.json'),
        'original_sdk_files_sha256': digest(before),
        'completed_and_failed_stages': evidence,
        'execution_context': context_identity(),
        'recovery_controller_sha256': digest(Path(__file__)),
        'cargo_git_transport': 'git-cli-http1',
        'source_or_lock_modified': False,
        'model_started': False,
    }
    save(result / 'recovery.json', recovery)
    save(sdk / 'source-preparation.json', dict(marker, owner=owner_info()))
    try:
        return _retry(cache, sdk, build, original, lock, result, recovery)
    except BaseException as error:
        recovery.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        recovery['ended'] = time.time()
        save(result / 'recovery.json', recovery)


def _retry(cache, sdk, build, original, lock, result, recovery):
    core = sdk / 'rootfs/vllm-workspace/vllm'
    ascend = sdk / 'rootfs/vllm-workspace/vllm-ascend'
    rust_lock = core / 'rust/Cargo.lock'
    lock_hash = digest(rust_lock)
    python, env = environment(sdk, build / 'cache')
    env['LD_LIBRARY_PATH'] = ':'.join(part for part in env['LD_LIBRARY_PATH'].split(':')
                                      if part and not part.endswith(('/torch/lib', '/torch_npu/lib')))
    for flag in ('PIP_BUILD_CONSTRAINT', 'PIP_CONSTRAINT', 'VLLM_TARGET_DEVICE', 'VLLM_BATCH_INVARIANT'):
        env.pop(flag, None)
    shared = cache / 'source-build-downloads' / candidate.key()
    env.update(SOC_VERSION='ascend910b1', ASCEND_RT_VISIBLE_DEVICES='',
               PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple', PIP_DISABLE_PIP_VERSION_CHECK='1',
               PIP_CACHE_DIR=str(shared / 'pip'), CARGO_HOME=str(shared / 'cargo'),
               RUSTUP_HOME=str(shared / 'rustup'), CMAKE_BUILD_PARALLEL_LEVEL='16', MAX_JOBS='16',
               PROTOC_INCLUDE='/usr/include', CARGO_HTTP_MULTIPLEXING='false',
               CARGO_NET_GIT_FETCH_WITH_CLI='true')
    env['PATH'] = os.pathsep.join((str(build / 'console-launchers'), str(python.parent),
                                  str(shared / 'cargo/bin'), env['PATH']))
    entry = int(env.get('GIT_CONFIG_COUNT', '0'))
    env.update({'GIT_CONFIG_COUNT': str(entry + 1), f'GIT_CONFIG_KEY_{entry}': 'http.version',
                f'GIT_CONFIG_VALUE_{entry}': 'HTTP/1.1'})
    name = '09-upstream-rust-recovery-' + result.parent.name
    candidate.command(['bash', str(core / 'build_rust.sh')], env, core, build, name)
    if digest(rust_lock) != lock_hash:
        raise ValueError('Cargo changed the original dependency lock')
    checked = sdk_run([str(python), '-B', '-S', '-m', 'pip', 'check'], sdk=sdk, env=env,
                      capture_output=True, text=True, timeout=120)
    (build / (name + '-dependency-check.log')).write_text(checked.stdout + checked.stderr)
    observed = probe(sdk, build / 'metadata-probe', lock)
    files = file_map(sdk)
    save(sdk / 'installed-files.json', files)
    receipt = dict(original, status='prepared', ended=time.time(), metadata=observed,
                   dependency_check_returncode=checked.returncode,
                   toolchain=candidate.toolchain_evidence(sdk, env, (core, ascend), build),
                   rust_recovery=dict(path=str(result / 'recovery.json'), stage=name,
                                      cargo_lock_sha256=lock_hash))
    receipt.pop('error', None)
    save(build / 'receipt.json', receipt)
    installed = dict(schema='deeplinkengine-source-candidate-sdk/v1', baseline_id=candidate.ID,
                     image_manifest=lock['manifest_digest'], manifest_kind='source-build-derivation',
                     base_image_manifest=candidate.BASE_MANIFEST,
                     installed_files_sha256=digest(sdk / 'installed-files.json'), metadata=observed,
                     model_started=False, official_release_image=False, upstream_source_patched=False,
                     source_candidate=candidate.identity(), build_receipt_sha256=digest(build / 'receipt.json'))
    save(sdk / 'installation.json', installed)
    recovery.update(status='prepared', installation_sha256=digest(sdk / 'installation.json'),
                    rust_stage_receipt_sha256=digest(build / (name + '.json')))
    return installed
