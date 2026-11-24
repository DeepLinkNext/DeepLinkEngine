"""Stable selection, integrity and actual assessment/producer adapter contracts."""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import release_selection as selection
from benchmark import evaluation_client as client
from benchmark import managed_shutdown
from benchmark.managed_shutdown import MANAGED_STOP, without_managed_stop


def row(version, **kw):
    return dict(filename='vllm_hust_engine-'+version+'.tar.gz', url='https://files.pythonhosted.org/package.tar.gz',
                digests={'sha256':'a'*64}, size=100, packagetype='sdist', yanked=False, **kw)


def test_latest_final_skips_preview_dev_yanked_and_wheel_only():
    versions = {'2026.11.17.8.post2':[row('2026.11.17.8.post2')], '2026.11.19.9':[row('2026.11.19.9')],
                '2026.11.21.10rc1':[row('2026.11.21.10rc1')], '2026.11.22.10.dev1':[row('2026.11.22.10.dev1')],
                '2026.11.23.10':[dict(row('2026.11.23.10'),yanked=True)],
                '2026.11.24.10':[dict(row('2026.11.24.10'),packagetype='bdist_wheel')]}
    index={'releases':versions}
    assert selection.stable_sdist(index)[0]=='2026.11.19.9'
    assert selection.stable_sdist(index,'2026.11.17.8.post2')[0]=='2026.11.17.8.post2'
    with pytest.raises(ValueError):selection.stable_sdist(index,'2026.11.23.10')


def test_stable_release_order_respects_numeric_builds_and_post_releases():
    versions = ('2026.11.21.9.post99', '2026.11.21.10', '2026.11.21.10.post0',
                '2026.11.21.10.post2', '2026.11.21.10.post11', '2026.11.21.10+native')
    index = {'releases': {version: [row(version)] for version in versions}}
    assert selection.stable_sdist(index)[0] == '2026.11.21.10.post11'
    assert selection.stable_sdist(index, '2026.11.21.10')[0] == '2026.11.21.10'
    index['releases']['2026.11.24.10'] = [row('2026.11.24.10')]
    assert selection.stable_sdist(index)[0] == '2026.11.24.10'
    assert selection.release_name('2026.11.24.10') == 'full-answer-b1-20261124-r10'
    assert selection.stable_sdist(index, '2026.11.21.10')[0] == '2026.11.21.10'
    assert selection.release_name('2026.11.21.10.post11') == 'full-answer-b1-20261121-r10'


def test_stable_r10_bridge_pins_qualified_sources(monkeypatch):
    kit = ROOT / 'benchmark/one-click-kit'
    proofs = {
        'qualify_dense_source': 'canonical-dense-tail-source-domain-v1',
        'qualify_necessary_work': 'necessary-arithmetic-source-domain-v1',
    }
    originals = {}
    for qualifier, proof in proofs.items():
        for name in (f'protocols/{proof}.json', f'third-party-test-kit/{qualifier}.py'):
            originals[name] = (kit / name).read_bytes()
    overlay = 'engines/b1-r10-contiguous-restore-overlay.json'
    release = client.R10_REFERENCE_RELEASE
    result = dict(originals)
    result.update({
        'scripts/runtime_overlay.py': f'R10_STABLE_RELEASE = "{release}"'.encode(),
        'engines/b1-r10-stable.json': json.dumps(dict(
            release=release, runtime_overlay=overlay,
            model_adapters={'qwen35': {'restore_gated_rms': False}})).encode(),
        overlay: b'reviewed native RMS overlay',
    })
    pins = {}
    for name, (directory, _) in client.R10_REFERENCE_SOURCES.items():
        content = name.encode()
        result[f'overlays/{directory}/{name}'] = content
        pins[name] = (directory, client.sha(content))
    monkeypatch.setattr(client, 'R10_REFERENCE_SOURCES', pins)
    monkeypatch.setattr(client, 'R10_REFERENCE_OVERLAY_SHA256', client.sha(result[overlay]))
    tampered = {**result, 'overlays/b1-r10/runtime/ascend/vllm_ascend/ops/gdn.py': b'unreviewed'}
    with pytest.raises(ValueError, match='reference implementation requires source review'):
        client.r10_reference_source_bridge(tampered)

    client.r10_reference_source_bridge(result)
    for qualifier, proof in proofs.items():
        key = f'protocols/{proof}.json'
        protocol = json.loads(result[key])
        previous = json.loads(originals[key])
        adaptation = protocol.pop('stable_r10_source_domain')
        assert protocol == previous
        assert adaptation['sources'] == {name: pin for name, (_, pin) in pins.items()
                                         if name in previous['sources']['B1']}
        code = result[f'third-party-test-kit/{qualifier}.py']
        assert b'sources = admitted_sources(' in code
        assert f'PROTOCOL_SHA256 = "{client.sha(result[key])}"'.encode() in code
        ast.parse(code)


def test_revised_r10_reference_rejects_changed_microflow(monkeypatch):
    overlay = json.dumps(dict(
        release=client.R10_3072_REFERENCE_RELEASE,
        base_manifest_sha256=client.R10_REFERENCE_OVERLAY_SHA256,
        measurement={'installed_files_canonical_sha256': client.R10_3072_RUNTIME_SHA256},
    )).encode()
    microflow = 'overlays/b1-r10-3072/' + client.R10_3072_MICROFLOW
    files = {'scripts/runtime_overlay.py':
             f'R10_STABLE_RELEASE = "{client.R10_REFERENCE_RELEASE}"'.encode(),
             client.R10_3072_OVERLAY: overlay, microflow: b'reviewed three-chunk partition'}
    monkeypatch.setattr(client, 'R10_3072_OVERLAY_SHA256', client.sha(overlay))
    monkeypatch.setattr(client, 'R10_3072_MICROFLOW_SHA256', client.sha(files[microflow]))
    assert client.R10_3072_REFERENCE_RELEASE in client.native_reference_releases(files)
    assert client.R10_REFERENCE_RELEASE in client.native_reference_releases(files)
    files[microflow] += b' tampered'
    with pytest.raises(ValueError, match='Revised R10 reference implementation'):
        client.native_reference_releases(files)


def test_stable_r10_source_admission_requires_sealed_native_rms(tmp_path):
    from benchmark.reference_source_domains import admitted_sources, verify_http_adapter

    release = client.R10_REFERENCE_RELEASE
    profile_path = tmp_path / 'deployment.json'
    overlay = tmp_path / 'provenance/runtime-overlay.json'
    overlay.parent.mkdir()
    overlay.write_bytes(b'native RMS, contiguous restore')
    overlay_pin = client.sha(overlay.read_bytes())
    profile_path.write_text(json.dumps(dict(release=release,
        measured_runtime={'runtime_overlay_sha256': overlay_pin})))
    protocol = dict(sources={'B0': {'gdn.py': 'b0'}, 'B1': {'gdn.py': 'r8', 'other.py': 'unchanged'}},
        stable_r10_source_domain=dict(release=release, overlay_sha256=overlay_pin,
                                     sources={'gdn.py': 'r10'}))
    native = {'service/native-plan.json': dict(engine_release=release, deployment=str(tmp_path),
                  files={str(profile_path): client.sha(profile_path.read_bytes())}),
              'service/launch-command.json': {'environment': {}}}
    assert admitted_sources(protocol, 'B0', tmp_path, {}.__getitem__) == {'gdn.py': 'b0'}
    assert admitted_sources(protocol, 'B1', tmp_path, native.__getitem__) == {
        'gdn.py': 'r10', 'other.py': 'unchanged'}
    http = dict(engine_release=release, model_adapter=dict(id='qwen35', gated_rms_mode='native',
                                                           restore_gated_rms=False))
    verify_http_adapter(http, release)
    http['model_adapter']['restore_gated_rms'] = True
    with pytest.raises(ValueError, match='native RMS HTTP companion'):
        verify_http_adapter(http, release)
    native['service/launch-command.json']['environment']['HUST_RESTORE_GATED_RMS'] = '1'
    with pytest.raises(ValueError, match='sealed native RMS launch'):
        admitted_sources(protocol, 'B1', tmp_path, native.__getitem__)
    overlay.write_bytes(b'different overlay')
    with pytest.raises(ValueError, match='sealed native deployment'):
        admitted_sources(protocol, 'B1', tmp_path, native.__getitem__)


def test_selected_version_survives_changed_feed_and_rejects_tampering(tmp_path):
    record=dict(schema='deeplinkengine-stable-selection/v1',version='2026.11.19.9')
    path=tmp_path/'selection.json';path.write_text(json.dumps(record))
    with patch.dict(os.environ,{},clear=False):
        selection.bind(path)
        with patch.object(selection,'read_json',side_effect=AssertionError('must not resolve again')):
            assert selection.select(tmp_path)==record
            with pytest.raises(ValueError,match='already selected'):
                selection.select(tmp_path,version='2026.11.17.8.post2')
        path.write_text('{}')
        with pytest.raises(ValueError,match='changed'):selection.active()


def test_release_selection_admits_sdist_from_configured_wheelhouse(tmp_path, monkeypatch):
    from engine import environment_setup
    import run_all

    version = '2026.11.17.8.post2'
    release = 'full-answer-b1-20261117-r8'
    source = row(version)
    wheelhouse = tmp_path / 'wheelhouse'
    wheelhouse.mkdir()
    archive = wheelhouse / source['filename']
    frozen = json.dumps(dict(version=version, release=release)).encode()
    with tarfile.open(archive, 'w:gz') as tar:
        member = tarfile.TarInfo(
            f'vllm_hust_engine-{version}/src/vllm_hust_engine/payload/frozen.json'
        )
        member.size = len(frozen)
        tar.addfile(member, io.BytesIO(frozen))
    source['size'] = archive.stat().st_size
    source['digests']['sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    lock = tmp_path / 'engine/r8-lock.json'
    lock.parent.mkdir()
    lock.write_text(json.dumps(dict(public_distribution='vllm-hust-engine==' + version,
        assets=[dict(name=archive.name, sha256=source['digests']['sha256'])])))
    monkeypatch.setattr(selection, 'ROOT', tmp_path)
    monkeypatch.delenv(selection.SELECTION, raising=False)
    monkeypatch.delenv(selection.SELECTION_SHA, raising=False)
    monkeypatch.setattr(selection, 'read_json', lambda _: {'releases': {version: [source]}})
    with patch.dict(os.environ), patch.object(environment_setup, '_download_dependency') as download:
        record = run_all.select_run_engine(dict(
            cache=str(tmp_path / 'cache'), out=str(tmp_path / 'out'), engine_version=version,
            environment=dict(wheelhouse=str(wheelhouse)),
        ))
        download.assert_not_called()
    assert record['release'] == release
    assert record['sdist']['sha256'] == source['digests']['sha256']


def test_model_client_bridge_keeps_reference_and_startup_fixes(tmp_path):
    provider={k:v.encode() for k,v in json.loads((ROOT/'tests/fixtures/producer-r9-adapter.json').read_bytes())['files'].items()}
    manifest=json.loads((ROOT/'benchmark/one-click-kit/DEPLOYMENT-TOOLS.json').read_bytes())
    original={name:(ROOT/'benchmark/one-click-kit'/name).read_bytes() for name in manifest['files']}
    combined=client.bridge(original,provider)
    for name in ('child_runner.py',
                 'compute_reference.py','reference_observation.py','reference_http_client.py'):
        assert combined['scripts/'+name]==original['scripts/'+name]
    def remove_device_admission(controller, old):
        controller = controller.replace('    parser.add_argument("--allow-device-change", action="store_true")\n', '')
        end_marker = '        save(root / "continuous-qualification.json", qualification)'
        start = controller.index('        from device_reuse import for_continuous\n')
        end = controller.index(end_marker, start)
        old_start = old.index('        if qualification["physical_cards"] != cards():\n')
        old_end = old.index(end_marker, old_start)
        return controller[:start] + old[old_start:old_end] + controller[end:]

    native = 'scripts/run_native_reference.py'
    native_controller = combined[native].decode()
    reverted = native_controller
    for before, after in client.NATIVE_SAMPLE_REPLACEMENTS:
        assert reverted.count(after) == 1
        reverted = reverted.replace(after, before, 1)
    assert remove_device_admission(reverted, original[native].decode()) == original[native].decode()
    monitor = tmp_path / "native_controller.py"
    monitor.write_bytes(combined[native])
    source = ast.parse(combined['scripts/reference_model_reuse.py'])
    matcher = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'native_monitor_matches')
    checks = {}
    exec(compile(ast.Module(body=[matcher], type_ignores=[]), native, 'exec'), checks)
    expected = client.sha(reverted.encode())
    assert checks['native_monitor_matches'](monitor, expected)
    monitor.write_bytes(combined[native].replace(b'model_service_started', b'model_service_ignored'))
    assert not checks['native_monitor_matches'](monitor, expected)
    tree = ast.parse(native_controller)
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'retired_sample_owner')
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), native, 'exec'), namespace)
    retired = namespace['retired_sample_owner']
    namespace['state'] = lambda pid: dict(start_ticks='200', state='Z')
    assert retired(17, {17: '200'})
    assert not retired(17, {17: '123'})
    assert not retired(27, {17: '200'})
    namespace['state'] = lambda pid: dict(start_ticks='200', state='R')
    assert not retired(17, {17: '200'})
    def exited(pid):
        raise FileNotFoundError(pid)
    namespace['state'] = exited
    assert retired(17, {17: '200'})
    assert not retired(27, {17: '200'})
    # Restore only the declared insertion, then compare the whole controller.
    added = ('    if startup_reserve is not None and not startup_reserve.record["released"]:\n'
             '        from startup_memory_recovery import release_startup_headroom\n'
             '        release_startup_headroom(startup_reserve, lambda: check_memory(guard, initial, root))\n')
    controller = combined['scripts/run_b1_r4.py'].decode()
    assert controller.count(added) == 1
    assert controller.count(MANAGED_STOP) == 1
    controller = without_managed_stop(controller.encode()).decode()
    old_controller = original['scripts/run_b1_r4.py'].decode()
    assert remove_device_admission(controller.replace(added, ''), old_controller) == old_controller
    # The only startup-cache change is the declared idle-before-warmup
    # boundary. Every allocator, pre-service gate and other method stays exact.
    old=ast.parse(original['scripts/checkpoint_cache.py'])
    new=ast.parse(combined['scripts/checkpoint_cache.py'])
    for tree in (old,new):
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StartupMemoryReserve')
        cls.body[0]=ast.Pass()
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='before_warmup')
        method.body=[ast.Pass()]
    assert ast.dump(old)==ast.dump(new)
    entry=combined['scripts/engine_entry.py'].decode()
    assert 'adapter = select_model(profile, args.model' in entry
    assert 'reference_observation' in entry and 'local_runtime_root' in entry
    assert 'capacity36(profile["release"])' in entry
    for name,body in combined.items():
        if name.endswith('.py'):ast.parse(body,filename=name)
    # The exact producer geometry handles R9, not merely a changed report label.
    module={};exec(combined['scripts/runtime_geometry.py'],module)
    assert module['capacity36']('full-answer-b1-20261119-r9')
    assert module['R9_CONFIG']['execution_seats']==36
    assert b'"r9"' in combined['scripts/runtime_overlay.py']


def test_shutdown_only_reference_reuse_checks_the_original_source_pin(tmp_path, monkeypatch):
    root = ROOT / 'benchmark/one-click-kit'
    keys = ['scripts/run_b1_r4.py', 'scripts/reference_evidence.py', 'scripts/reference_observation_reuse.py']
    files = {name: (root / name).read_bytes() for name in keys}
    old = tmp_path / 'original.py'
    current = tmp_path / 'current.py'
    old.write_bytes(files[keys[0]])
    pin = client.sha(old.read_bytes())
    client.managed_shutdown_bridge(files)
    current.write_bytes(files[keys[0]])
    monkeypatch.syspath_prepend(str(root / 'scripts'))
    monkeypatch.setitem(sys.modules, 'managed_shutdown', managed_shutdown)
    evidence = {}
    exec(files[keys[1]], evidence)
    compatible = evidence['http_scope_fix_only']
    assert compatible(old, current, pin)
    assert not compatible(old, current, '0' * 64)
    changed = files[keys[0]].replace(b'child.wait(timeout=10)', b'child.wait(timeout=1)')
    assert changed != files[keys[0]]
    current.write_bytes(changed)
    assert not compatible(old, current, pin)


def test_client_inventory_rejects_added_changed_and_symlinked_files(tmp_path):
    root=tmp_path/'client';root.mkdir();(root/'x.py').write_text('a=1\n')
    raw=json.dumps(dict(files=client.inventory(root))).encode()
    (root/'CLIENT-MANIFEST.json').write_bytes(raw)
    record=dict(path=str(root),manifest_sha256=client.sha(raw))
    client.verify(record)
    (root/'extra').touch()
    with pytest.raises(ValueError,match='inventory'):client.verify(record)


def test_frozen_metadata_rejects_traversal_and_version_mismatch(tmp_path):
    def archive(name,data):
        path=tmp_path/'sdist.tar.gz'
        with tarfile.open(path,'w:gz') as stream:
            member=tarfile.TarInfo(name);member.size=len(data);stream.addfile(member,io.BytesIO(data))
        return path
    with pytest.raises(ValueError,match='Unsafe'):
        selection.metadata(archive('../frozen.json',b'{}'),'2026.11.19.9')
    name='vllm_hust_engine-2026.11.19.9/src/vllm_hust_engine/payload/frozen.json'
    with pytest.raises(ValueError,match='version differs'):
        selection.metadata(archive(name,b'{"version":"wrong"}'),'2026.11.19.9')


def test_default_sources_and_evaluation_share_selected_identity(tmp_path):
    from engine import engine_install
    import run_all
    record=dict(schema='deeplinkengine-stable-selection/v1',version='2026.11.19.9',
                engine_profile='r9',legacy=False,sdist={'name':'engine'},runtime_artifacts=[{'name':n} for n in ('vllm','ascend','support')])
    path=tmp_path/'selection.json';path.write_text(json.dumps(record))
    with patch.dict(os.environ):
        selection.bind(path)
        assert [x['name'] for x in engine_install.sources()]==['engine','vllm','ascend','support']
        plan=run_all.evaluation_configuration(dict(out=str(tmp_path),engine_version='2026.11.19.9'),tmp_path/'installation')
        assert plan['engine_profile']=='r9' and 'engine_version' not in plan
        selection.save(tmp_path/'output')
        assert (tmp_path/'output/engine-selection.json').read_bytes()==path.read_bytes()


def test_new_service_delegates_intern_and_qwen_to_installed_producer(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('stable_service',ROOT/'engine/service.py')
    service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
    import deeplink
    class Handoff(Exception):pass
    for model in ('Intern-S2','Qwen3.5-35B-A3B'):
        args=['service.py','--installation',str(tmp_path/'installation'),'--model',str(tmp_path/model),
              '--physical-devices','0,7','--devices','0,1','--cache',str(tmp_path/'cache'),'--dry-run']
        with patch.object(sys,'argv',args),patch.object(deeplink,'installation',return_value=(tmp_path/'installation',
                dict(release='full-answer-b1-20261119-r9',python='/private/bin/python'))), \
             patch.object(os,'execve',side_effect=Handoff) as handoff:
            with pytest.raises(Handoff):service.main()
        command=handoff.call_args.args[1]
        assert command[:4]==['/private/bin/python','-c','from vllm_hust_engine.cli import main; raise SystemExit(main())','serve']
        assert '--dry-run' in command and str(tmp_path/model) in command


@pytest.mark.skipif(sys.platform != 'linux', reason='The serving adapter supports Linux')
@pytest.mark.parametrize('model', ['Intern-S2', 'Qwen3.5-35B-A3B'])
def test_service_really_invokes_producer_cli_main_in_a_new_process(tmp_path, model):
    import subprocess
    package = tmp_path / 'vllm_hust_engine'
    package.mkdir()
    (package / '__init__.py').write_text('')
    # Match the published entry-point contract: cli.main exists, with no -m guard.
    (package / 'cli.py').write_text(
        'import json,os,sys\nfrom pathlib import Path\n'
        'def main():\n'
        ' Path(os.environ["PRODUCER_MAIN_RECEIPT"]).write_text(json.dumps(sys.argv[1:]))\n'
        ' return 0\n'
    )
    installed = tmp_path / 'installation'
    (installed / 'sdk').mkdir(parents=True)
    release = 'full-answer-b1-20261119-r9'
    (installed / 'installation.json').write_text(json.dumps(dict(
        release=release, package_version='2026.11.19.9.post1',
        deployment=str(installed / 'sdk'), python=sys.executable)))
    (installed / 'sdk/deployment.json').write_text(json.dumps(dict(release=release)))
    receipt = tmp_path / 'producer-main.json'
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(tmp_path)]),
               PYTHONDONTWRITEBYTECODE='1', PRODUCER_MAIN_RECEIPT=str(receipt))
    completed = subprocess.run([
        sys.executable, '-B', str(ROOT / 'engine/service.py'), '--installation', str(installed),
        '--model', str(tmp_path / model), '--physical-devices', '0,7', '--devices', '0,1',
        '--cache', str(tmp_path / 'cache'), '--dry-run'],
        env=env, capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert completed.returncode == 0, completed.stderr
    arguments = json.loads(receipt.read_text())
    assert arguments[:3] == ['serve', '--installation', str(installed)]
    assert arguments[arguments.index('--model') + 1] == str(tmp_path / model)
    assert '--dry-run' in arguments


def test_post_revision_is_stable_and_keeps_frozen_release(tmp_path):
    version = '2026.11.19.9.post1'
    filename = 'vllm_hust_engine-' + version + '.tar.gz'
    row = dict(packagetype='sdist', yanked=False, filename=filename,
               url='https://files.pythonhosted.org/packages/' + filename,
               digests={'sha256': 'a' * 64}, size=1024)
    assert selection.stable_sdist({'releases': {version: [row]}})[0] == version
    frozen = dict(version=version, release='full-answer-b1-20261119-r9')
    raw = json.dumps(frozen).encode()
    path = tmp_path / filename
    with tarfile.open(path, 'w:gz') as stream:
        member = tarfile.TarInfo('vllm_hust_engine-' + version + '/src/vllm_hust_engine/payload/frozen.json')
        member.size = len(raw)
        stream.addfile(member, io.BytesIO(raw))
    assert selection.metadata(path, version) == frozen


def test_vendor_file_probe_loads_release_contract_in_real_isolated_python(tmp_path):
    import base64
    import shutil
    from engine import environment_setup
    # A real RECORD in a private fixture exercises the isolated import and hashing
    # on CPU-only test hosts; it does not install or import a vendor backend.
    package = tmp_path / 'engine'
    package.mkdir()
    for name in ('__init__.py', 'environment_setup.py', 'release_selection.py'):
        shutil.copyfile(ROOT / 'engine' / name, package / name)
    payload = tmp_path / 'probe-payload.bin'
    payload.write_bytes(b'CPU metadata probe fixture\n')
    info = tmp_path / 'triton_ascend-3.6.0.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: triton-ascend\nVersion: 3.6.0\n')
    checksum = base64.urlsafe_b64encode(hashlib.sha256(payload.read_bytes()).digest()).rstrip(b'=').decode()
    (info / 'RECORD').write_text(f'{payload.name},sha256={checksum},{payload.stat().st_size}\n')
    with patch.object(environment_setup, 'ROOT', tmp_path):
        result = environment_setup.vendor_integrity(sys.executable, {'triton-ascend': '3.6.0'}, dict(os.environ))
        assert result['verified'] is True
        assert result['packages']['triton-ascend']['checked_files'] == 1
        payload.write_bytes(b'tampered')
        result = environment_setup.vendor_integrity(sys.executable, {'triton-ascend': '3.6.0'}, dict(os.environ))
        assert result['verified'] is False
        assert result['packages']['triton-ascend']['mismatches'] == [payload.name]


def test_r9_baseline_reuse_bridge_is_pinned_and_preserves_argv_admission():
    provider={k:v.encode() for k,v in json.loads((ROOT/'tests/fixtures/producer-r9-adapter.json').read_bytes())['files'].items()}
    root=ROOT/'benchmark/one-click-kit'
    manifest=json.loads((root/'DEPLOYMENT-TOOLS.json').read_bytes())
    original={n:(root/n).read_bytes() for n in manifest['files']}
    combined=client.bridge(original,provider)
    compat=json.loads(combined['protocols/task-warmup-v3-c8-compatibility.json'])
    assert client.sha(original['scripts/serving_warmup.py']) in compat['previous_sha256']
    assert compat['current_sha256']==client.sha(combined['scripts/serving_warmup.py'])
    baseline=compat['unchanged_continuous_source_adaptations']['scripts/baseline_entry.py']
    assert client.sha(original['scripts/baseline_entry.py']) in baseline['previous_sha256']
    assert baseline['current_sha256']==client.sha(combined['scripts/baseline_entry.py'])
    assert b'verify_baseline_serve_args(stage, root / stage / "service/plan.json")' in combined['scripts/reuse_campaign.py']
    assert b'verify_relocation' in combined['scripts/reuse_campaign.py']
    provider['scripts/serving_warmup.py']+=b'\n# unreviewed producer change\n'
    with pytest.raises(ValueError,match='requires source review'):client.bridge(original,provider)


def test_public_runtime_api_mirror_is_anonymous_and_hash_bound():
    release='full-answer-b1-20261119-r9';tag='engine-wheels-r9-v1'
    prefix='https://github.com/vLLM-HUST/vllm-hust-website/releases/download/'+tag+'/'
    rows=[dict(name=n+'-1-cp312-cp312-linux_aarch64.whl',bytes=100+i,sha256=str(i+1)*64) for i,n in enumerate(('vllm','vllm_ascend','vllm_hust_runtime'))]
    for row in rows:row['url']=prefix+row['name']
    frozen=dict(release=release,native_abi=dict(python='cp312',platform='linux_aarch64',cann='9.1.0',hardware='TP2 Ascend910B2'),public_runtime=dict(schema='vllm-hust-public-runtime-artifacts/v1',release=release,mirror_repository='vLLM-HUST/vllm-hust-website',mirror_release=tag,artifacts=rows))
    assets=[dict(id=10+i,name=r['name'],size=r['bytes'],digest='sha256:'+r['sha256'],browser_download_url=r['url']) for i,r in enumerate(rows)]
    with patch.object(selection,'read_json',return_value=dict(tag_name=tag,assets=assets)) as fetch:
        result=selection.runtime_artifacts(frozen)
        fetch.assert_called_once_with('https://api.github.com/repos/vLLM-HUST/vllm-hust-website/releases/tags/'+tag)
    assert all(r['api_url'].endswith('/'+str(10+i)) for i,r in enumerate(result))
    with patch.object(selection,'read_json',side_effect=OSError('public API rate limit')):
        fallback=selection.runtime_artifacts(frozen)
    assert all('api_url' not in r for r in fallback)
    assert [r['sha256'] for r in fallback]==[r['sha256'] for r in rows]
    for field,value in [('size',999),('digest','sha256:'+'0'*64),('browser_download_url','https://wrong.invalid/asset')]:
        bad=copy.deepcopy(assets);bad[0][field]=value
        with patch.object(selection,'read_json',return_value=dict(tag_name=tag,assets=bad)):
            with pytest.raises(ValueError,match='identity differs'):selection.runtime_artifacts(frozen)


@pytest.mark.parametrize('entry',['native_timing_prepare.py','baseline_native_prepare.py'])
@pytest.mark.parametrize('release,allowed',[('full-answer-b1-20261117-r8',True),('full-answer-b1-20261119-r9',True),('full-answer-b1-20261121-r10',False)])
def test_native_reference_entry_reaches_bootstrap_admission_for_r9(tmp_path,entry,release,allowed):
    import types
    provider={k:v.encode() for k,v in json.loads((ROOT/'tests/fixtures/producer-r9-adapter.json').read_bytes())['files'].items()}
    root=ROOT/'benchmark/one-click-kit';manifest=json.loads((root/'DEPLOYMENT-TOOLS.json').read_bytes())
    original={n:(root/n).read_bytes() for n in manifest['files']};combined=client.bridge(original,provider)
    scripts=str(root/'scripts')
    deployment=tmp_path/'sdk';deployment.mkdir();(deployment/'deployment.json').write_text(json.dumps(dict(release=release)))
    argv=[entry,'--deployment',str(deployment),'--model',str(tmp_path/'model'),'--out',str(tmp_path/'out'),
          '--cache',str(tmp_path/'cache'),'--qualified-continuous',str(tmp_path/'qualified'),'--devices','0,1',
          '--reference-profile','submission-v2','--formal-repetitions','3','--preflight']
    if entry.startswith('baseline_'):argv+=['--baseline-deployment',str(tmp_path/'b0')]
    sys.path.insert(0,scripts)
    try:
        module=types.ModuleType('entry_under_test')
        exec(compile(combined['scripts/'+entry],entry,'exec'),module.__dict__)
        with patch.object(sys,'argv',argv),patch.dict(os.environ,{'VLLMHUST_ENGINE_BOOTSTRAPPED':'','VLLMHUST_BASELINE_BOOTSTRAPPED':''}):
            with pytest.raises(ValueError,match=('admitted' if allowed else 'capacity-36')):module.main()
    finally:sys.path.remove(scripts)
    # Only the target admission and its error text changed; all actual launch,
    # warmup, work counting, sampling, timing and source checks remain original.
    old=original['scripts/'+entry].decode()
    normalized=combined['scripts/'+entry].decode().replace('not in ("full-answer-b1-20261117-r8", "full-answer-b1-20261119-r9"):', '!= "full-answer-b1-20261117-r8":')
    normalized=normalized.replace('requires an admitted frozen Qwen capacity-36 configuration','requires the frozen r8 configuration').replace('requires an admitted frozen Qwen capacity-36 campaign','requires the frozen r8 campaign')
    assert normalized==old


def test_selected_client_keeps_actual_ipc_fallback_and_explicit_storage_choice(tmp_path, monkeypatch):
    import errno
    import socket
    import shutil

    provider = {k: v.encode() for k, v in json.loads((ROOT / 'tests/fixtures/producer-r9-adapter.json').read_bytes())['files'].items()}
    root = ROOT / 'benchmark/one-click-kit'
    manifest = json.loads((root / 'DEPLOYMENT-TOOLS.json').read_bytes())
    original = {n: (root / n).read_bytes() for n in manifest['files']}
    combined = client.bridge(original, provider)
    names = {'local_runtime_root', 'ipc_directory', 'temporary_directory'}
    def functions(raw):
        return {n.name: n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in names}
    before, after = functions(original['scripts/engine_entry.py']), functions(combined['scripts/engine_entry.py'])
    assert set(after) == names
    for name in names:
        assert ast.dump(before[name]) == ast.dump(after[name])
    namespace = dict(Path=Path, os=os, errno=errno, json=json, tempfile=tempfile,
                     write=lambda p, value: p.write_text(json.dumps(value)))
    exec(compile(ast.Module(body=list(after.values()), type_ignores=[]), '<selected-client-storage-fixture>', 'exec'), namespace)
    allocate = tempfile.mkdtemp
    def full_root(*args, **kwargs):
        if kwargs.get('dir') == '/tmp':
            raise OSError(errno.ENOSPC, 'fixture full root')
        return allocate(*args, **kwargs)
    monkeypatch.delenv('VLLM_HUST_LOCAL_RUNTIME_ROOT', raising=False)
    monkeypatch.setattr(tempfile, 'mkdtemp', full_root)
    folder = namespace['ipc_directory'](tmp_path)
    try:
        assert folder.parent == Path('/dev/shm')
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
            channel.bind(str(folder / 'bridge-fixture.sock'))
    finally:
        shutil.rmtree(folder)
    other = tmp_path / 'explicit'; other.mkdir()
    monkeypatch.setenv('VLLM_HUST_LOCAL_RUNTIME_ROOT', '/tmp')
    with pytest.raises(OSError, match='fixture full root'):
        namespace['ipc_directory'](other)
    monkeypatch.setenv('VLLM_HUST_LOCAL_RUNTIME_ROOT', '/invalid')
    with pytest.raises(ValueError, match='must be /tmp or /dev/shm'):
        namespace['ipc_directory'](other)
