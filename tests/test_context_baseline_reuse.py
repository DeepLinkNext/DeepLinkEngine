"""A fixed-B0 supplementary arm must retain its raw protocol and observed identity."""
import json
from pathlib import Path
from types import SimpleNamespace,ModuleType
import sys
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmark/one-click-kit/scripts')]
import evaluation_client as client


def combined():
    root=ROOT/'benchmark/one-click-kit'
    original={name:(root/name).read_bytes() for name in json.loads((root/'DEPLOYMENT-TOOLS.json').read_text())['files']}
    provider={k:v.encode() for k,v in json.loads((ROOT/'tests/fixtures/producer-r9-adapter.json').read_text())['files'].items()}
    return client.bridge(original,provider)


def source_fixture(tmp_path):
    from completion_warmup import protocol
    from baseline_profile import expected
    files=combined();namespace={};exec(files['scripts/context_baseline_reuse.py'],namespace)
    observed=json.loads((ROOT/'tests/fixtures/context-b0-controller-20261119-v1.json').read_text())
    provenance={n:client.sha(b) for n,b in files.items() if n.startswith('scripts/') and '/' not in n[len('scripts/'): ]}
    prior=tmp_path/'prior';(prior/'service').mkdir(parents=True)
    deployment=tmp_path/'r9-sdk';deployment.mkdir()
    (deployment/'deployment.json').write_text(json.dumps(dict(release='full-answer-b1-20261119-r9')))
    records={'controller-source.json':observed,'device-mapping.json':dict(logical_cards=[0,1]),
        'source-admission.json':dict(deployment='/original/r8-companion',release='full-answer-b1-20261117-r8'),
        'service/plan.json':dict(serving_warmup=protocol('continuous',acceptance=True,capacity36=True),
            baseline_variant='text',serve_args=expected('text','continuous'))}
    for name,value in records.items():(prior/name).write_text(json.dumps(value))
    # Use the actual generated release predicate. This is pure source routing;
    # loading model modules, allocating devices and generating answers are absent.
    module=ModuleType('runtime_geometry');exec(files['scripts/runtime_geometry.py'],module.__dict__)
    args=SimpleNamespace(deployment=deployment,devices='0,1')
    return namespace,prior,args,provenance,module


def test_reviewed_b0_adaptation_retains_original_companion_and_warmup(tmp_path):
    ns,prior,args,current,geometry=source_fixture(tmp_path)
    with patch.dict(sys.modules,{'runtime_geometry':geometry}):result=ns['validate'](prior,args,current)
    assert result['source_adaptation'] is True
    assert result['original_companion']['deployment']=='/original/r8-companion'
    assert result['requested_companion']==str(args.deployment)
    assert result['shared_warmup']['policy']=='observed-torch-triton-sealed-completion-warmup-v2-capacity36'


@pytest.mark.parametrize('changed',['old-source','new-source','warmup','logical','argv'])
def test_old_context_reuse_rejects_unreviewed_sources_or_changed_execution(tmp_path,changed):
    ns,prior,args,current,geometry=source_fixture(tmp_path)
    if changed=='new-source':current=dict(current,unknown='changed')
    elif changed=='logical':args.devices='1,0'
    else:
        name='controller-source.json' if changed=='old-source' else 'service/plan.json'
        path=prior/name;value=json.loads(path.read_text())
        if changed=='old-source':value['scripts/acceptance_client.py']='0'*64
        elif changed=='warmup':value['serving_warmup']['output_tokens']=1
        else:value['serve_args']+=['--max-num-seqs','256']
        path.write_text(json.dumps(value))
    with patch.dict(sys.modules,{'runtime_geometry':geometry}):
        with pytest.raises(ValueError):ns['validate'](prior,args,current)


@pytest.mark.parametrize('problem',['incomplete','wrong-role','wrong-contract'])
def test_source_adaptation_never_bypasses_complete_raw_scope(problem,tmp_path):
    files=combined();module={'__file__':str(ROOT/'benchmark/one-click-kit/scripts/acceptance_suite.py')};exec(files['scripts/acceptance_suite.py'],module)
    prior=tmp_path/'old';prior.mkdir();(prior/'receipt.json').write_text('{}');(prior/'source-admission.json').write_text('{}')
    report=dict(execution_complete=problem!='incomplete',role='B1' if problem=='wrong-role' else 'B0',contract={'changed':True} if problem=='wrong-contract' else {})
    module['inspect']=lambda p:report;module['load']=lambda p:({},None)
    with pytest.raises(ValueError,match='exact complete supplement'):
        module['admit_reuse'](prior,'B0',SimpleNamespace(),tmp_path/'contract.json')


def test_unreviewed_future_context_controller_requires_review():
    files={}
    with patch.object(client,'context_baseline_bridge',side_effect=lambda result:files.update(result)):combined()
    admitted=dict(files);client.context_baseline_bridge(admitted)
    assert 'scripts/context_baseline_reuse.py' in admitted
    files['scripts/acceptance_client.py']+=b'\n# changed protocol\n'
    with pytest.raises(ValueError,match='requires review'):client.context_baseline_bridge(files)


def test_context_b1_reuse_still_requires_exact_optimized_deployment(tmp_path):
    module={'__file__':str(ROOT/'benchmark/one-click-kit/scripts/acceptance_suite.py')};exec(combined()['scripts/acceptance_suite.py'],module)
    prior=tmp_path/'old';prior.mkdir()
    (prior/'receipt.json').write_text('{}')
    (prior/'source-admission.json').write_text(json.dumps(dict(deployment='/old/r8')))
    module['inspect']=lambda p:dict(execution_complete=True,role='B1',contract={})
    module['load']=lambda p:({},None)
    with pytest.raises(ValueError,match='different deployment'):
        module['admit_reuse'](prior,'B1',SimpleNamespace(deployment=tmp_path/'new-r9'),tmp_path/'contract')
