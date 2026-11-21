"""Retargeting B1 must preserve B0 provenance and refuse old B1/judge reuse."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmark/one-click-kit/scripts')]
import run_all

OLD='full-answer-b1-20261117-r8'
NEW='full-answer-b1-20261119-r9'

@pytest.mark.parametrize('stage', ['continuous-B0','hle-solver-B0','frontier-solver-B0','swe-trace-B0','terminal-trace-B0'])
def test_explicit_retarget_accepts_only_independently_audited_baseline(tmp_path,stage):
    data=dict(model='/model',physical_devices=[1,6],cache='/cache',out=str(tmp_path/'new'),engine_version='2026.11.19.9.post1')
    source=tmp_path/'old';source.mkdir()
    prior=dict(data,out=str(source));prior.pop('engine_version')
    raw=json.dumps(dict(status='failed',configuration=prior));(source/'results.json').write_text(raw)
    campaign=dict(status='failed',stages=[dict(name=stage,status='completed')])
    with patch('reuse_campaign.source',return_value=(source,dict(engine_release=OLD),campaign,{})),patch('reuse_campaign.audit_stage') as audit,patch('release_selection.active',return_value=dict(release=NEW,client=None)):
        result=run_all.completed_run_source(data,source)
    assert result['completed_stages']==[stage]
    audit.assert_called_once_with(source/'fixed-five',stage)
    assert (source/'results.json').read_text()==raw

@pytest.mark.parametrize('stage',['continuous-B1','hle-judge-B0','frontier-judge-B0','hle-solver-B1'])
def test_retarget_rejects_optimized_or_common_judge_stage(tmp_path,stage):
    data=dict(model='/model',physical_devices=[1,6],cache='/cache',out=str(tmp_path/'new'),engine_version='2026.11.19.9.post1')
    source=tmp_path/'old';source.mkdir();(source/'results.json').write_text(json.dumps(dict(status='failed',configuration=data)))
    campaign=dict(status='failed',stages=[dict(name=stage,status='completed')])
    with patch('reuse_campaign.source',return_value=(source,dict(engine_release=OLD),campaign,{})),patch('reuse_campaign.audit_stage'),patch('release_selection.active',return_value=dict(release=NEW,client=None)):
        with pytest.raises(ValueError,match='No compatible completed stages'):run_all.completed_run_source(data,source)

@pytest.mark.parametrize('field,value',[('model','/other'),('physical_devices',[2,3])])
def test_new_target_does_not_relax_shared_configuration(tmp_path,field,value):
    data=dict(model='/model',physical_devices=[1,6],cache='/cache',out=str(tmp_path/'new'),engine_version='2026.11.19.9.post1')
    source=tmp_path/'old';source.mkdir();prior=dict(data);prior[field]=value
    (source/'results.json').write_text(json.dumps(dict(status='failed',configuration=prior)))
    with patch('reuse_campaign.source') as audit:
        with pytest.raises(ValueError,match='configuration differs'):run_all.completed_run_source(data,source)
        audit.assert_not_called()

def test_explicit_new_target_is_selected_once_for_fresh_reuse(tmp_path):
    data=dict(out=str(tmp_path/'new'),cache='/cache',engine_version='2026.11.19.9.post1')
    with patch('release_selection.select',return_value={'version':data['engine_version']}) as select,patch('release_selection.bind') as bind:
        run_all.select_run_engine(data,reuse_completed_run=tmp_path/'old')
        select.assert_called_once_with('/cache',version=data['engine_version']);bind.assert_not_called()

def test_in_place_recovery_cannot_retarget(tmp_path):
    (tmp_path/'engine-selection.json').write_text('{}')
    data=dict(out=str(tmp_path),cache='/cache',engine_version='2026.11.19.9.post1')
    with patch('release_selection.bind',return_value={'version':'2026.11.17.8.post2'}),patch('release_selection.select') as select:
        with pytest.raises(ValueError,match='In-place recovery'):run_all.select_run_engine(data,recovery=True)
        select.assert_not_called()

def test_ordinary_reuse_preserves_original_selection(tmp_path):
    (tmp_path/'engine-selection.json').write_text('{}')
    original=dict(version='2026.11.17.8.post2')
    with patch('release_selection.bind',return_value=original) as bind,patch('release_selection.select') as select:
        assert run_all.select_run_engine(dict(out='/new',cache='/cache'),reuse_completed_run=tmp_path)==original
        bind.assert_called_once_with(tmp_path/'engine-selection.json');select.assert_not_called()
