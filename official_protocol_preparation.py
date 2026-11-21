"""Prepare the common protocol through public no-model CLI stages when absent."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parent
FIELDS=('protocol_run','protocol_deployment','protocol_python')
VERSION='2026.11.17.8.post2'
PHASES=['prepare-datasets','install-package','verify-engine','install-engine','prepare-fixed-five']


def selection(config):
    present=[name in config for name in FIELDS]
    if any(present) and not all(present):
        raise ValueError('Provide all three protocol paths, or omit all three for automatic preparation')
    return 'explicit' if all(present) else 'automatic'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prepared(config, output):
    path=output/'results.json';data=json.loads(path.read_bytes())
    if (data.get('status')!='prepared' or data.get('protocol_preparation_complete') is not True
            or data.get('automated_acceptance_complete') is not False
            or data.get('model_service_started') is not False or data.get('formal_client_started') is not False
            or data.get('engine_selection',{}).get('version')!=VERSION
            or data.get('protocol_preparation_mode')!='official-inputs-only'
            or [r.get('name') for r in data.get('stages',[])]!=PHASES
            or any(r.get('status')!='completed' for r in data.get('stages',[]))):
        raise ValueError('Public protocol preparation did not complete its five no-model stages')
    expected=dict(protocol_run=output/'fixed-five',protocol_deployment=output/'installation/sdk')
    result=dict(config)
    for name in FIELDS:
        value=data.get(name)
        if not isinstance(value,str) or not value.strip():raise ValueError('Missing prepared protocol path: '+name)
        result[name]=str(Path(value).resolve())
    for name,value in expected.items():
        if Path(result[name])!=value.resolve():raise ValueError('Prepared protocol output escaped its own run')
    from official_suite import validate_configuration
    validate_configuration(result)
    from official_protocol_data import verify_prepared_inputs
    verify_prepared_inputs(output/'fixed-five')
    files=[path,output.parent/'evaluation.json',output/'evaluation.json',output/'fixed-five/frozen-execution-plan.json',output/'installation/sdk/provenance/installed-files.json']
    from official_suite import protocol_data_cache
    if protocol_data_cache(result)!=Path(config['cache']).resolve():
        raise ValueError('Prepared protocol data cache differs from its public configuration')
    files+=sorted((output/'fixed-five/plans').glob('*.json'))
    files+=sorted((output/'fixed-five/continuous-B0/inputs').glob('*.json'))
    files+=[output/'fixed-five/official-protocol-preparation.json',output/'fixed-five/native-token-workload.json']
    return result,{str(p):_sha(p) for p in files}


def _recipe(config):
    model=Path(config['model']).resolve()
    metadata={name:_sha(model/name) for name in ('config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja') if (model/name).is_file()}
    return dict(model=str(model),physical_devices=config['physical_devices'],protocol_engine_version=VERSION,
                preparation_schema='deeplinkengine-official-data-protocol/v1',
                tools_manifest_sha256=_sha(ROOT/'benchmark/one-click-kit/DEPLOYMENT-TOOLS.json'),model_metadata_sha256=metadata)


def ensure(config):
    if selection(config)=='explicit':return config,None
    from baseline_setup import download_lock
    base=Path(config['cache'])/'official-protocol-preparation';base.mkdir(parents=True,exist_ok=True)
    recipe=_recipe(config);key=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    lock=base/(key+'.lock')
    if lock.is_symlink():raise ValueError('Common protocol lock must not be a link')
    with lock.open('a+b') as stream,download_lock(stream):
        for path in sorted(base.glob('*/preparation.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
            try:
                previous=json.loads(path.read_bytes())
                if previous.get('status')!='prepared' or previous.get('recipe')!=recipe:continue
                result,bindings=_prepared(config,path.parent/'protocol')
                if bindings!=previous.get('source_output_sha256'):continue
            except (OSError,ValueError,KeyError):
                continue
            return result,dict(path=str(path),sha256=_sha(path),protocol_engine_version=VERSION,
                               model_service_started=False,formal_client_started=False,reused=True)
        return _prepare(config,recipe)


def _prepare(config,recipe):
    from official_baselines import save
    from official_execution import prepare_anchor
    from official_suite import clean_client_environment
    journal=Path(config['cache'])/'official-protocol-preparation'/uuid.uuid4().hex
    journal.mkdir(parents=True,exist_ok=False)
    receipt=journal/'preparation.json'
    record=dict(schema='deeplinkengine-automatic-protocol-preparation/v1',started=time.time(),status='running',
                protocol_engine_version=VERSION,outer_output=config['out'],model_service_started=False,
                formal_client_started=False,stages=[],recipe=recipe)
    save(receipt,record)
    try:
        anchor=prepare_anchor(dict(config,versions=['0.23.0']))
        record['anchor_sdk']=str(anchor)
        output=journal/'protocol'
        plan=dict(model=config['model'],physical_devices=config['physical_devices'],cache=config['cache'],
                  out=str(output),engine_version=VERSION,rerun_b0=False,environment=dict(official_sdk=str(anchor)))
        path=journal/'evaluation.json';save(path,plan)
        command=[sys.executable,'-B',str(ROOT/'run_all.py'),'--config',str(path),'--prepare-official-protocol-only']
        record['stages'].append(dict(name='prepare-common-protocol',status='running',command=command));save(receipt,record)
        sys.path.insert(0,str(ROOT/'benchmark/one-click-kit/scripts'))
        from child_runner import run_stage
        with (journal/'prepare.log').open('xb') as log:
            completed=run_stage(command,env=clean_client_environment(),stdout=log,stderr=subprocess.STDOUT)
        record['stages'][-1].update(returncode=completed.returncode,status='completed' if completed.returncode==0 else 'failed')
        if completed.returncode:
            raise ValueError('Public common-protocol preparation failed; preserve '+str(journal))
        result,bindings=_prepared(config,output)
        record.update(status='prepared',protocol_paths={key:result[key] for key in FIELDS},source_output_sha256=bindings)
    except BaseException as error:
        record.update(status='failed',error=type(error).__name__+': '+str(error))
        raise
    finally:
        record['ended']=time.time();save(receipt,record)
        print('PROTOCOL_PREPARATION_RESULT',receipt,flush=True)
    return result,dict(path=str(receipt),sha256=_sha(receipt),protocol_engine_version=VERSION,
                       model_service_started=False,formal_client_started=False)
