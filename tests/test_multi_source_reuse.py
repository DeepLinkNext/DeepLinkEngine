"""Recovery retains measured originals and rejects changed image bytes or protocols."""
import ast
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmark/one-click-kit/scripts')]
import evaluation_client as client
import runtime_storage as storage


def bridge():
    root=ROOT/'benchmark/one-click-kit'
    original={n:(root/n).read_bytes() for n in json.loads((root/'DEPLOYMENT-TOOLS.json').read_bytes())['files']}
    provider={k:v.encode() for k,v in json.loads((ROOT/'tests/fixtures/producer-r9-adapter.json').read_bytes())['files'].items()}
    return original,client.bridge(original,provider)


def test_exact_input_cli_compatibility_does_not_authorize_other_source_changes():
    original,result=bridge()
    compat=json.loads(result['protocols/task-warmup-v3-c8-compatibility.json'])['unchanged_continuous_source_adaptations']['scripts/continuous_inputs.py']
    assert compat['previous_sha256']==['f9787d060f4740b21d570a077652c0e7eaac7bd08b3d8882f5214e118fc6f186']
    assert compat['current_sha256']==client.sha(original['scripts/continuous_inputs.py'])
    old=original['scripts/continuous_inputs.py'].decode().replace('import argparse\n','').replace('import json\n','').replace('from pathlib import Path\n','').split('\n\ndef main(')[0].rstrip()+'\n'
    assert client.sha(old.encode())==compat['previous_sha256'][0]


def test_relocated_b0_requires_equal_full_pinned_maps_and_preserves_original_path(tmp_path):
    import hashlib
    _,result=bridge();module={};exec(result['scripts/baseline_relocation.py'],module)
    mapping={'runtime/file.py':{'sha256':'a'*64,'bytes':12}}
    digest=hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (tmp_path/'sdk-files.json').write_text(json.dumps(mapping))
    observed=dict(deployment='/old/original018',source_map_sha256=digest,files=mapping)
    with patch('baseline_admission.PINNED_SOURCE_MAP_SHA256',digest):
        accepted=module['verify'](observed,tmp_path)
        assert accepted['observed_deployment']=='/old/original018'
        assert accepted['requested_deployment']==str(tmp_path)
        (tmp_path/'sdk-files.json').write_text(json.dumps(dict(mapping,extra='changed')))
        with pytest.raises(ValueError,match='inventory differs'):module['verify'](observed,tmp_path)
        (tmp_path/'sdk-files.json').write_text(json.dumps(mapping))
        with pytest.raises(ValueError):module['verify'](dict(observed,source_map_sha256='wrong'),tmp_path)
        with pytest.raises(ValueError):module['verify'](dict(observed,files={'altered':True}),tmp_path)


def existing_runtime(tmp_path):
    source=tmp_path/'old/installation';source.mkdir(parents=True)
    frozen=dict(release='r9',version='9',archives={'sdk':{'sha256':'a'*64,'bytes':10}},runtime_files=100)
    saved=dict(verified=True,logical=str(source),physical=str(source),python=sys.executable,
               release='r9',package_version='9',archives=frozen['archives'])
    storage.marker(source).write_text(json.dumps(saved))
    (source/'installation.json').write_text('{}')
    return source,frozen,saved


def test_new_output_reuses_only_identical_fully_audited_runtime_without_install(tmp_path):
    source,frozen,saved=existing_runtime(tmp_path);target=tmp_path/'new/installation'
    cli=types.SimpleNamespace(verify=lambda:frozen)
    with patch.dict(sys.modules,{'vllm_hust_engine':types.ModuleType('vllm_hust_engine'),'vllm_hust_engine.cli':cli}),patch.object(storage,'audit',return_value={'ok':True}) as audit,patch.object(storage.subprocess,'run') as launch:
        assert storage.ensure(target,reuse_from=source)=={'ok':True}
    audit.assert_called_once_with(source,frozen);launch.assert_not_called()
    assert target.is_symlink() and target.resolve()==source
    assert json.loads(storage.marker(target).read_text())['reused_from']==str(source)


@pytest.mark.parametrize('field,value',[('package_version','r8'),('archives',{}),('python','/wrong/python'),('physical','/wrong/physical'),('verified',False)])
def test_existing_runtime_mismatch_never_publishes_alias(tmp_path,field,value):
    source,frozen,saved=existing_runtime(tmp_path);saved[field]=value
    storage.marker(source).write_text(json.dumps(saved));target=tmp_path/'new/installation'
    cli=types.SimpleNamespace(verify=lambda:frozen)
    with patch.dict(sys.modules,{'vllm_hust_engine':types.ModuleType('vllm_hust_engine'),'vllm_hust_engine.cli':cli}),patch.object(storage,'audit') as audit:
        with pytest.raises(ValueError,match='identity differs'):storage.ensure(target,reuse_from=source)
    audit.assert_not_called();assert not target.exists()


def test_new_root_forwards_all_explicit_sources_and_retains_full_scope():
    import contextlib
    import io
    import test_run_all as fixture
    case=fixture.WorkflowTests('runTest');case.setUp()
    try:
        workflow=fixture.workflow
        source=case.root/'previous';(source/'installation').mkdir(parents=True)
        (source/'installation/installation.json').write_text(json.dumps(dict(package_version='9')))
        aux=[case.root/'historical',case.root/'current-baseline']
        baseline=case.root/'mmlu-original200/B0'
        context=case.root/'context-original/B0'
        compute=case.root/'native-original/B0'
        def execute(argv,**kwargs):
            return case.execute([*argv,'run'] if '--reuse-completed-campaign' in argv else argv,**kwargs)
        with patch.object(workflow,'completed_run_source',return_value=dict(root=str(source),completed_stages=['continuous-B1'])),patch('release_selection.active',return_value=dict(version='9',client=None,engine_profile='r9')),patch('release_selection.save'),patch('one_click.configuration_arguments',return_value=['--datasets','mmlu,hle,frontier,swe,terminal']),patch('one_click.verify_kit'),patch('child_runner.run_stage',side_effect=execute),contextlib.redirect_stdout(io.StringIO()):
            assert workflow.run(case.data,reuse_completed_run=source,reuse_completed_campaigns=aux,reuse_mmlu_baseline_arm=baseline,reuse_context_baseline_arm=context,reuse_compute_baseline_arm=compute)==0
        argv=next(c for c in case.calls if '--reuse-completed-campaign' in c)
        assert [argv[i+1] for i,x in enumerate(argv) if x=='--reuse-completed-campaign']==[str(source/'fixed-five'),*[str(p) for p in aux]]
        assert argv[argv.index('--reuse-mmlu-baseline-arm')+1]==str(baseline)
        assert argv[argv.index('--reference-baseline-source')+1]==str(compute)
        assert argv[argv.index('--datasets')+1]=='mmlu,hle,frontier,swe,terminal'
        supplement=next(c for c in case.calls if '--context-tenant-only' in c)
        assert supplement[supplement.index('--reuse-b0')+1]==str(context)
        install=next(c for c in case.calls if len(c)>1 and Path(c[1]).name=='runtime_storage.py')
        assert install[install.index('--reuse-from')+1]==str(source/'installation')
        result=json.loads((Path(case.data['out'])/'results.json').read_text())
        assert result['reuse_selection']==dict(campaigns=[str(p) for p in aux],mmlu_baseline_arm=str(baseline),context_baseline_arm=str(context),compute_baseline_arm=str(compute))
    finally:case.doCleanups()


def test_failed_existing_runtime_audit_keeps_both_installations_untouched(tmp_path):
    source,frozen,saved=existing_runtime(tmp_path);target=tmp_path/'new/installation'
    cli=types.SimpleNamespace(verify=lambda:frozen)
    with patch.dict(sys.modules,{'vllm_hust_engine':types.ModuleType('vllm_hust_engine'),'vllm_hust_engine.cli':cli}),patch.object(storage,'audit',side_effect=ValueError('runtime bytes changed')):
        with pytest.raises(ValueError,match='runtime bytes changed'):storage.ensure(target,reuse_from=source)
    assert source.exists();assert not target.exists();assert not storage.marker(target).exists()


def test_bootstrap_recovery_keeps_multi_source_and_original200_selection():
    import contextlib
    import io
    import test_run_all as fixture
    case=fixture.WorkflowTests('runTest');case.setUp()
    try:
        workflow=fixture.workflow;case.failed_preflight()
        selection=dict(campaigns=[str(case.root/'historical')],mmlu_baseline_arm=str(case.root/'old-mmlu/B0'))
        ledger=Path(case.data['out'])/'results.json';old=json.loads(ledger.read_text())
        reused=dict(root=str(case.root/'prior-root'),completed_stages=['continuous-B1'])
        old.update(reused_run=reused,reuse_selection=selection);ledger.write_text(json.dumps(old))
        def execute(argv,**kwargs):
            return case.execute([*argv,'run'] if '--reuse-completed-campaign' in argv else argv,**kwargs)
        with patch.object(workflow,'completed_run_source',return_value=reused),patch('one_click.configuration_arguments',return_value=['--datasets','mmlu,hle,frontier,swe,terminal']),patch('one_click.verify_kit'),patch('child_runner.run_stage',side_effect=execute),contextlib.redirect_stdout(io.StringIO()):
            assert workflow.run(case.data,resume_bootstrap=True)==0
        result=json.loads(ledger.read_text());assert result['reuse_selection']==dict(selection,context_baseline_arm=None,compute_baseline_arm=None)
        argv=next(c for c in case.calls if '--reuse-completed-campaign' in c)
        assert selection['campaigns'][0] in argv
        assert argv[argv.index('--reuse-mmlu-baseline-arm')+1]==selection['mmlu_baseline_arm']
    finally:case.doCleanups()
