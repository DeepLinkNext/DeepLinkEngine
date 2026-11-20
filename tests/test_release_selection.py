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
import release_selection as selection
import evaluation_client as client


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


def test_model_client_bridge_keeps_reference_and_startup_fixes():
    provider={k:v.encode() for k,v in json.loads((ROOT/'tests/fixtures/producer-r9-adapter.json').read_bytes())['files'].items()}
    manifest=json.loads((ROOT/'benchmark/one-click-kit/DEPLOYMENT-TOOLS.json').read_bytes())
    original={name:(ROOT/'benchmark/one-click-kit'/name).read_bytes() for name in manifest['files']}
    combined=client.bridge(original,provider)
    for name in ('checkpoint_cache.py','child_runner.py','run_native_reference.py','run_b1_r4.py',
                 'compute_reference.py','reference_observation.py','reference_http_client.py'):
        assert combined['scripts/'+name]==original['scripts/'+name]
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
    import engine_install,run_all
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
        assert command[:4]==['/private/bin/python','-m','vllm_hust_engine.cli','serve']
        assert '--dry-run' in command and str(tmp_path/model) in command


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
    import environment_setup
    # A real RECORD in a private fixture exercises the isolated import and hashing
    # on CPU-only test hosts; it does not install or import a vendor backend.
    for name in ('environment_setup.py', 'release_selection.py'):
        shutil.copyfile(ROOT / name, tmp_path / name)
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
