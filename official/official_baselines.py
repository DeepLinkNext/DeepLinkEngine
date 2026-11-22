"""One-click, isolated official B0 version campaigns; preserve the 0.18 archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def configuration(path, *, download_only=False):
    path = Path(path).resolve()
    value = json.loads(path.read_bytes())
    allowed = {'model', 'physical_devices', 'cache', 'out', 'versions', 'protocol_run',
               'protocol_deployment', 'protocol_python', 'context_contract', 'judge_installation', 'model_manifest', 'model_transport_base_url'}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError('Unknown official-baseline configuration fields')
    for key in (('cache', 'out', 'versions') if download_only else ('model', 'cache', 'out', 'physical_devices', 'versions')):
        if key not in value:
            raise ValueError('Missing official-baseline field: ' + key)
    if 'model_transport_base_url' in value:
        from engine.model_restore import transport_base_url
        value['model_transport_base_url']=transport_base_url(value['model_transport_base_url'])
        if 'model_manifest' not in value:raise ValueError('Model transport mirror requires a pinned model_manifest')
    for key in allowed - {'physical_devices', 'versions', 'model_transport_base_url'}:
        if key in value:
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError('Expected a path: ' + key)
            value[key] = os.path.abspath(path.parent / Path(value[key]).expanduser())
    cards = value.get('physical_devices', [0, 1])
    if not isinstance(cards, list) or len(cards) != 2 or any(type(x) is not int or x < 0 for x in cards) or cards[0] == cards[1]:
        raise ValueError('Two distinct physical NPU indices are required')
    versions = value['versions']
    if not isinstance(versions, list) or not versions or any(v not in ('0.23.0', '0.28.0', '0.28.0rc1-source', '0.30.0') for v in versions) or len(set(versions)) != len(versions):
        raise ValueError('Select unique explicitly supported upstream versions')
    if not download_only and not Path(value['model']).is_dir() and 'model_manifest' not in value:
        raise ValueError('Model directory does not exist')
    if os.path.lexists(value['out']):
        raise ValueError('Choose a new output directory; previous failures are immutable')
    return value


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-environment', action='store_true')
    parser.add_argument('--download-only', action='store_true', help='Only download pinned official OCI artifacts; no extraction, Python repair or model execution')
    parser.add_argument('--download-protocol-dependencies', action='store_true', help='With download-only, fetch one digest-pinned common protocol asset, without installing or running it')
    parser.add_argument('--download-source-build-dependencies', action='store_true', help='With download-only, fetch one pinned source-build artifact, without installing or extracting it')
    parser.add_argument('--artifact-digest', help='Download one digest already present in the selected official locks')
    parser.add_argument('--download-transport', choices=('auto', 'curl'), default='auto', help='Transport for download-only; preserve the same official URL and digest')
    parser.add_argument('--report-only', action='store_true',
                        help='CPU-only replay of closed C8 originals or explicit sealed-report projection')
    parser.add_argument('--complete-ft', type=Path,
                        help='Append only the independent arithmetic companion and CPU qualification to a closed version, in a new output')
    parser.add_argument('--source-version-seal-sha256')
    parser.add_argument('--reuse-ft-companion', type=Path,
                        help='CPU re-audit a closed independent companion; no model requests')
    parser.add_argument('--companion-seal-sha256')
    parser.add_argument('--reuse-completed-continuous', type=Path,
                        help='In a new output, retain sealed C8 after a later pre-model failure; preserve original clocks')
    parser.add_argument('--resume-source-rust-from', type=Path,
                        help='Resume a candidate CPU build after its final Rust Git transport failed')
    args = parser.parse_args(arguments)
    if args.resume_source_rust_from and (not args.prepare_environment or args.download_only
            or args.report_only or args.complete_ft or args.reuse_completed_continuous):
        parser.error('Rust build recovery requires CPU-only source candidate environment preparation')
    if args.download_source_build_dependencies and (not args.download_only or args.download_protocol_dependencies):
        parser.error('Source-build prefetch requires download-only, separate from protocol prefetch')
    if args.download_only and (args.report_only or args.complete_ft or args.reuse_ft_companion):
        parser.error('Download-only cannot continue or replay a model campaign')
    if args.download_protocol_dependencies and (not args.download_only or not args.artifact_digest or args.download_transport!='auto'):
        parser.error('Protocol dependency prefetch requires download-only, one artifact digest and the default Python transport')
    if (args.download_only and (args.prepare_environment or args.reuse_completed_continuous)) or ((args.artifact_digest or args.download_transport != 'auto') and not args.download_only):
        parser.error('Artifact selection requires download-only, separate from environment preparation')
    try:
        if args.report_only:
            if args.prepare_environment or args.reuse_completed_continuous or args.complete_ft or args.reuse_ft_companion or args.download_only or args.artifact_digest or args.download_protocol_dependencies:
                raise ValueError('Report-only cannot prepare or resume a model campaign')
            from official_rebuild import run
            return run(args.config)
        protocol_preparation = None
        config = configuration(args.config, download_only=args.download_only)
        if args.resume_source_rust_from and config['versions'] != ['0.28.0rc1-source']:
            raise ValueError('Rust build recovery requires the 0.28 source candidate')
        if (args.source_version_seal_sha256 or args.reuse_ft_companion or args.companion_seal_sha256) and not args.complete_ft:
            raise ValueError('F/T continuation options require --complete-ft')
        if args.complete_ft:
            if args.prepare_environment or args.reuse_completed_continuous or args.download_only or len(config['versions']) != 1:
                raise ValueError('F/T completion requires one explicit closed version and a new output')
            if not args.reuse_ft_companion and (platform.system() != 'Linux' or platform.machine() != 'aarch64'):
                raise ValueError('Official Ascend F/T completion requires Linux aarch64')
            if not args.reuse_ft_companion:
                from official_execution import maybe_enter_official, needed
                if needed():
                    from engine.transport_trust import configure
                    configure(config['cache'])
                maybe_enter_official(config, list(arguments) if arguments is not None else sys.argv[1:])
            from official_ft import continue_closed
            return continue_closed(config, args.complete_ft, args.source_version_seal_sha256,
                companion_source=args.reuse_ft_companion, companion_seal=args.companion_seal_sha256)
        if args.download_only:
            if args.download_source_build_dependencies:
                from official_source_download import run
                return run(config, artifact=args.artifact_digest, transport=args.download_transport,
                           arguments=list(arguments) if arguments is not None else sys.argv[1:])
            if args.download_protocol_dependencies:
                from official_protocol_download import run
                return run(config, artifact=args.artifact_digest)
            if '0.28.0rc1-source' in config['versions']:
                raise ValueError('Source candidates require environment preparation; download-only selects official OCI images')
            from official_download import run
            return run(config, artifact=args.artifact_digest, transport=args.download_transport)
        if args.reuse_completed_continuous:
            if args.prepare_environment or len(config['versions']) != 1:
                raise ValueError('Completed C8 recovery requires one explicit full-suite version')
            config['reuse_completed_continuous'] = str(args.reuse_completed_continuous.resolve())
        if platform.system() != 'Linux' or platform.machine() != 'aarch64':
            raise ValueError('Official Ascend baselines require Linux aarch64')
        if not args.prepare_environment:
            from official_protocol_preparation import selection
            selection(config)
            from engine.native_sdk_controller import maybe_enter_official as native_controller
            native_controller(config,list(arguments) if arguments is not None else sys.argv[1:])
        if not args.prepare_environment or '0.28.0rc1-source' in config['versions']:
            from official_execution import maybe_enter_official, needed
            if needed():
                from engine.transport_trust import configure
                configure(config['cache'])
            maybe_enter_official(config, list(arguments) if arguments is not None else sys.argv[1:])
        if config.get('model_manifest'):
            from engine.model_restore import restore
            restore(config['model_manifest'], config['model'], transport_base_url=config.get('model_transport_base_url'))
        from official_storage import prefer
        storage = []
        for key in ('cache', 'out'):
            selected, identity = prefer(config[key])
            config[key] = str(selected)
            if identity:
                storage.append(dict(field=key, **identity))
        protocol_preparation = None
        if not args.prepare_environment:
            from official_protocol_preparation import ensure
            config, protocol_preparation = ensure(config)
            from official_suite import validate_configuration
            validate_configuration(config)
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))
    out = Path(config['out'])
    out.mkdir(parents=True, exist_ok=False)
    cache = Path(config['cache'])
    cache.mkdir(parents=True, exist_ok=True)
    record = dict(schema='deeplinkengine-official-baseline-campaign/v1',
                  status='preparing', started=time.time(), configuration=config,
                  model_execution_authorized=not args.prepare_environment,
                  storage_selection=storage,
                  stages=[], source_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in [ROOT / 'run_all.py', ROOT / 'official/official_baselines.py', ROOT / 'official/official_image.py', ROOT / 'official/official_storage.py']})
    from official_protocol_bootstrap import host_identity
    record['host_identity'] = host_identity()
    from engine.native_sdk_controller import ENV as native_marker
    if os.environ.get(native_marker):record['native_controller_handoff']=json.loads(os.environ[native_marker])
    if protocol_preparation is not None:
        record['protocol_preparation'] = protocol_preparation
    save(out / 'results.json', record)
    failed = False
    try:
        for version in config['versions']:
            row = dict(version=version, baseline_id='official-vllm-' + version,
                       status='preparing', started=time.time(), model_started=False)
            record['stages'].append(row)
            save(out / 'results.json', record)
            if version == '0.28.0rc1-source':
                from official_source_candidate import prepare as prepare_candidate, reuse as reuse_candidate, ID, key
                row['baseline_id'] = ID
                row['official_release_image'] = False
                row['candidate'] = True
                save(out / 'results.json', record)
                sdk = cache / 'source-candidate-sdks' / (ID + '-' + key()[:16])
                try:
                    base_lock = ROOT / 'engine/official-baselines/0.23.0.json'
                    from official_sdk_state import sdk_preparation
                    with sdk_preparation(sdk):
                        if args.resume_source_rust_from:
                            from official_source_rust_recovery import retry
                            recovery = out / 'source-rust-recovery'
                            row['rust_recovery'] = str(recovery / 'recovery.json')
                            save(out / 'results.json', record)
                            installed = retry(args.resume_source_rust_from.resolve(), cache, sdk, recovery)
                        else:
                            from official_source_candidate import quarantine_partial
                            recovered = quarantine_partial(sdk)
                            if recovered:
                                row['interrupted_source_build'] = recovered
                                save(out / 'results.json', record)
                            if sdk.exists():
                                installed = reuse_candidate(sdk)
                            else:
                                from official_source_delivery import prepare as deliver_verified_wheels
                                row['build_dependency_delivery'] = deliver_verified_wheels(cache, sdk, out / (version + '-dependency-delivery'))
                                save(out / 'results.json', record)
                                installed = prepare_candidate(base_lock, cache, sdk)
                    row.update(installation=str(sdk), installation_receipt=installed,
                               status='prepared' if args.prepare_environment else 'running',
                               model_started=False if args.prepare_environment else None)
                    save(out / 'results.json', record)
                    if not args.prepare_environment:
                        from official_suite import run
                        record['status'] = 'running'
                        save(out / 'results.json', record)
                        row.update(run(config, sdk, out / version))
                        failed |= row['status'] != 'completed'
                except Exception as error:
                    row.update(status='failed', error=type(error).__name__ + ': ' + str(error))
                    failed = True
                    print('OFFICIAL_SOURCE_CANDIDATE_FAILED', version, row['error'], flush=True)
                row['ended'] = time.time()
                save(out / 'results.json', record)
                continue
            lock = ROOT / 'engine/official-baselines' / (version + '.json')
            if not lock.is_file():
                row.update(status='blocked', reason='No pinned supported ARM64 official release image. Ascend v0.28.0.rc1 is a candidate source tag, not a verified stable image.', ended=time.time())
                failed = True
                save(out / 'results.json', record)
                continue
            from official_image import prepare, read_lock, digest, file_map, probe, finish_restoration, quarantine_partial_sdk
            identity = read_lock(lock)
            sdk = cache / 'official-sdks' / (version + '-' + identity['manifest_digest'].split(':')[1][:16])
            try:
                from official_sdk_state import sdk_preparation
                with sdk_preparation(sdk):
                    recovered = quarantine_partial_sdk(lock, sdk)
                    if recovered:
                        row['interrupted_restoration'] = recovered
                        save(out / 'results.json', record)
                    if sdk.exists():
                        was_complete = (sdk / 'installation.json').is_file()
                        if not was_complete:
                            installed = finish_restoration(lock, sdk)
                        else:
                            installed = json.loads((sdk / 'installation.json').read_bytes())
                        if installed['image_manifest'] != identity['manifest_digest'] or (sdk / 'image-lock.json').read_bytes() != lock.read_bytes():
                            raise ValueError('Cached official SDK selection differs')
                        if digest(sdk / 'installed-files.json') != installed['installed_files_sha256']:
                            raise ValueError('Cached official SDK file manifest differs')
                        if was_complete:
                            expected_files = json.loads((sdk / 'installed-files.json').read_bytes())
                            current_files = file_map(sdk)
                            if current_files != expected_files:
                                from official_sdk_bytecode import repair_locked
                                row['sdk_bytecode_recovery'] = repair_locked(sdk, lock, cache, current_files)
                                save(out / 'results.json', record)
                            installed['metadata'] = probe(sdk, out / (version + '-probe-cache'), identity)
                            if file_map(sdk) != expected_files:
                                raise ValueError('SDK files changed during cached metadata qualification')
                    else:
                        installed = prepare(lock, cache / 'official-oci', sdk)
                row.update(installation=str(sdk), installation_receipt=installed,
                           status='prepared' if args.prepare_environment else 'running',
                           model_started=False if args.prepare_environment else None)
                save(out / 'results.json', record)
                if not args.prepare_environment:
                    from official_suite import run
                    record['status'] = 'running'
                    save(out / 'results.json', record)
                    result = run(config, sdk, out / version)
                    row.update(result)
                    failed |= row['status'] != 'completed'
            except Exception as error:
                row.update(status='failed', error=type(error).__name__ + ': ' + str(error))
                failed = True
                print('OFFICIAL_BASELINE_FAILED', version, row['error'], flush=True)
            row['ended'] = time.time()
            save(out / 'results.json', record)
        record['status'] = 'incomplete' if failed else 'prepared' if args.prepare_environment else 'completed'
        return 1 if failed else 0
    except BaseException as error:
        record.update(status='interrupted', error=type(error).__name__ + ': ' + str(error))
        if record['stages'] and record['stages'][-1]['status'] in ('preparing', 'running'):
            record['stages'][-1].update(status='interrupted', ended=time.time())
        raise
    finally:
        record['ended'] = time.time()
        save(out / 'results.json', record)


if __name__ == '__main__':
    raise SystemExit(main())
