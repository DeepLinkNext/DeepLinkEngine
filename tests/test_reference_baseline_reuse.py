"""A frozen B0 clock is reused only under unchanged complete measurement controls."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmark/one-click-kit/scripts')]
import reference_baseline_reuse as reuse
from reference_capture_profile import profile
from baseline_admission import PINNED_SOURCE_MAP_SHA256


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


def reseal(source):
    write(source/'checksums.json',{str(p.relative_to(source)):reuse.sha(p) for p in source.rglob('*') if p.is_file() and p.name!='checksums.json'})
    return reuse.sha(source/'checksums.json')


@pytest.fixture
def original(tmp_path):
    source=tmp_path/'old';model=tmp_path/'model';model.mkdir()
    weights={}
    for i in range(14):
        p=model/(str(i)+'.safetensors');p.write_bytes(b'fixture');s=p.stat()
        weights[p.name]=dict(bytes=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns,sha256=reuse.sha(p))
    capture=profile('submission-v2')
    native=dict(role='B0',engine_release='original018-cann8.5.1',paired_engine_release='full-answer-b1-20261117-r8',
        deployment='/original/r8-companion',baseline_deployment='/original/pinned-b0',model=str(model),files={},
        reference_capture_profile='submission-v2',formal_repetitions=3,max_tokens=256,
        sampling=dict(max_tokens=256,temperature=0,seed=0,ignore_eos=False,top_p=1.0,top_k=-1,presence_penalty=0.0,detokenize=False))
    rows=[dict(repetition=i,start_ns=i*1000000,end_ns=i*1000000+100000,T_seconds=.0001,
        ordinary_public_outputs_retained=True,boundary=[dict(rank=r,sealed=True,failure=None) for r in [0,1]]) for i in [1,2,3]]
    write(source/'service/client/timings.json',dict(role='B0',repetitions=rows))
    receipt=dict(role='B0',completed=True,quiescent=True,preflight_only=False,failure=None,returncode=0,
        formal_repetitions=3,result='service/client/timings.json',result_sha256=reuse.sha(source/'service/client/timings.json'),started=1,ended=2)
    for name,value in {'receipt.json':receipt,'service/native-plan.json':native,'device-mapping.json':dict(physical_cards=[1,6],logical_cards=[0,1]),
        'model-weights.json':weights,'service/baseline-source-admission.json':dict(source_map_sha256=PINNED_SOURCE_MAP_SHA256)}.items():write(source/name,value)
    plan=dict(capture_profile=capture,binding=dict(model=str(model)),physical_cards=[1,6],logical_devices='0,1',baseline_deployment='/relocated/same-pinned-b0')
    with patch.dict(sys.modules,full_reference_reuse=SimpleNamespace(qualify_native=lambda *a:dict(role='B0')),
        qualify_baseline_capacity=SimpleNamespace(qualify=lambda *a,**k:dict(qualified=True)),
        baseline_relocation=SimpleNamespace(verify=lambda *a:dict(same_pinned_bytes=True))):
        yield source,plan,reseal(source)


def test_complete_b0_reuse_retains_original_clock_dates_and_companion(original):
    source,plan,pin=original;proof=reuse.inspect(source,pin,plan)
    assert proof['model_inference_rerun'] is False
    assert proof['original_paired_engine_release']=='full-answer-b1-20261117-r8'
    assert proof['original_companion_deployment']=='/original/r8-companion'
    assert (proof['original_started'],proof['original_ended'])==(1,2)
    assert [x['T_seconds'] for x in proof['repetitions']]==[.0001]*3


def test_native_source_checks_follow_restored_sdk_but_keep_original_path(original):
    source, plan, pin = original
    with patch("qualify_baseline_capacity.qualify", return_value={"qualified": True}) as qualify:
        proof = reuse.inspect(source, pin, plan)
    assert proof["original_baseline_deployment"] == "/original/pinned-b0"
    assert [call.args for call in qualify.call_args_list] == [(plan["baseline_deployment"], source)] * 3
    assert [call.kwargs for call in qualify.call_args_list] == [{"repetition": n} for n in (1, 2, 3)]


@pytest.mark.parametrize('changed',['role','incomplete','budget','profile','cards','weights','clock','repetitions','fences'])
def test_resealed_but_incompatible_measurement_never_reuses(original,changed):
    source,plan,pin=original
    if changed=='cards':plan['physical_cards']=[6,1]
    elif changed=='weights':(Path(plan['binding']['model'])/'0.safetensors').write_bytes(b'changed checkpoint')
    else:
        filename='receipt.json' if changed in ('role','incomplete') else 'service/native-plan.json' if changed in ('budget','profile') else 'service/client/timings.json'
        d=reuse.read(source/filename)
        if changed=='role':d['role']='B1'
        elif changed=='incomplete':d['completed']=False
        elif changed=='budget':d['max_tokens']=128
        elif changed=='profile':d['reference_capture_profile']='full-answer-v1'
        elif changed=='clock':d['repetitions'][0]['end_ns']=0
        elif changed=='repetitions':d['repetitions'].pop()
        else:d['repetitions'][0]['boundary'][0]['sealed']=False
        write(source/filename,d)
        if filename=='service/client/timings.json':
            receipt=reuse.read(source/'receipt.json');receipt['result_sha256']=reuse.sha(source/filename);write(source/'receipt.json',receipt)
    with pytest.raises(ValueError):reuse.inspect(source,reseal(source),plan)


def test_changed_raw_or_prospective_source_seal_fails(original):
    source,plan,pin=original
    write(source/'device-mapping.json',dict(physical_cards=[6,1],logical_cards=[0,1]))
    with pytest.raises(ValueError,match='raw evidence changed'):reuse.inspect(source,pin,plan)
    with pytest.raises(ValueError,match='seal changed'):reuse.inspect(source,'0'*64,plan)


def test_cpu_report_resolves_original_path_and_rejects_edited_proof(original,tmp_path):
    source,plan,pin=original
    folder=tmp_path/'reference';plan['baseline_reuse']=dict(source=str(source),checksums_sha256=pin)
    write(folder/'automatic-plan.json',plan);proof=reuse.inspect(source,pin,plan,location=False)
    write(folder/'baseline-native-reuse.json',proof)
    assert reuse.resolve(folder)['source']==str(source)
    proof['repetitions'][0]['T_seconds']=0;write(folder/'baseline-native-reuse.json',proof)
    with pytest.raises(ValueError,match='proof changed'):reuse.resolve(folder)


@pytest.mark.parametrize('valid',[True,False])
def test_actual_reference_coordinator_never_launches_b0_when_reuse_was_selected(original,tmp_path,valid):
    from test_context_baseline_reuse import combined
    source,plan,pin=original
    out=tmp_path/'campaign';sdk=tmp_path/'sdk';sdk.mkdir();write(sdk/'deployment.json',{})
    plan.update(deployment=str(sdk),deployment_sha256=reuse.sha(sdk/'deployment.json'),files={},
        baseline_reuse=dict(source=str(source),checksums_sha256=pin if valid else '0'*64),
        stages=[dict(id='baseline-native-synchronized-timing',implementation='scripts/run_native_reference.py')])
    write(out/'plans/automatic-plan.json',plan)
    (out/'compute-reference').mkdir()
    admission=dict(plan='plans/automatic-plan.json',plan_sha256=reuse.sha(out/'plans/automatic-plan.json'))
    namespace={'__file__':str(ROOT/'benchmark/one-click-kit/scripts/automatic_reference.py')}
    exec(combined()['scripts/automatic_reference.py'],namespace)
    with patch.dict(sys.modules,reference_baseline_reuse=reuse,engine_entry=SimpleNamespace(environment=lambda *a:{})),patch.dict(namespace,run_stage=lambda *a,**k:pytest.fail('A selected fixed B0 reuse must never launch inference')):
        result=namespace['execute'](out,admission)
    rows=result['stages']
    if valid:
        assert rows[0]['status']=='completed' and rows[0]['reused_model_stage'] is True
        assert rows[0]['original_stage_source']==str(source)
        assert reuse.read(out/'compute-reference/baseline-native-reuse.json')['model_inference_rerun'] is False
    else:
        assert result['status']=='blocked'
        assert rows[0]['status']=='failed'
        assert 'seal changed' in result['blockers'][0]
        assert not (out/'compute-reference/baseline-native-reuse.json').exists()


@pytest.mark.parametrize("device_change", [False, True])
def test_http_observation_launch_receives_the_frozen_device_change_option(tmp_path, device_change):
    from test_context_baseline_reuse import combined
    out = tmp_path / "campaign"
    sdk = tmp_path / "sdk"
    write(sdk / "deployment.json", {})
    write(out / "frozen-stages.json", [dict(name="mmlu-pair",
        args=["--qualified-continuous", str(tmp_path / "original-c8")])])
    stage = dict(id="current-http-work-observation", implementation="scripts/reference_observation.py")
    plan = dict(deployment=str(sdk), deployment_sha256=reuse.sha(sdk / "deployment.json"), files={},
        stages=[stage], binding=dict(model="/model"), physical_cards=[0, 3], logical_devices="0,1",
        capture_profile=dict(id="submission-v2"), baseline_reuse={"source": "/fixed-b0"})
    if device_change:
        plan["device_reuse_mapping"] = {"physical_cards": [0, 3]}
    path = out / "plans/automatic-plan.json"
    write(path, plan)
    (out / "compute-reference").mkdir()
    namespace = {"__file__": str(ROOT / "benchmark/one-click-kit/scripts/automatic_reference.py")}
    exec(combined()["scripts/automatic_reference.py"], namespace)

    def launch(command, **kwargs):
        assert ("--allow-device-change" in command) == device_change
        assert command[command.index("--qualified-continuous") + 1] == str(tmp_path / "original-c8")
        folder = Path(command[command.index("--out") + 1])
        write(folder / "receipt.json", dict(warmup_started=False, client_started=False, failure="fixture"))
        return SimpleNamespace(returncode=2)

    admission = dict(plan="plans/automatic-plan.json", plan_sha256=reuse.sha(path))
    with patch.dict(sys.modules, engine_entry=SimpleNamespace(environment=lambda *a: {})), patch.dict(namespace, run_stage=launch):
        result = namespace["execute"](out, admission)
    assert result["blockers"] == ["Current HTTP observation failed: fixture"]
