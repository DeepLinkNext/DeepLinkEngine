"""Freeze original inputs for official solvers; never restore or impersonate 0.18."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from official import source_path

ROOT=Path(__file__).resolve().parents[1]
TOOLS=ROOT/'benchmark/one-click-kit'
SCHEMA='deeplinkengine-official-data-protocol/v1'
ORDER=['mmlu','hle','frontier','swe','terminal']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def contracts():
    pin=json.loads((ROOT/'data/official-protocol-inputs.json').read_bytes())
    old=json.loads((TOOLS/'protocols/m01-full-reference-reuse-domain-v1.json').read_bytes())
    if any(pin[k]!=old[k] for k in ('prompt_tokens_sha256','requests','input_tokens')):
        raise ValueError('Official input contract differs from original fixed200 authority')
    return pin

def admit_model(model,pin):
    path=ROOT/'engine/models/qwen3.5-35b-a3b.json'
    if sha(path)!=pin['model_manifest_sha256']:raise ValueError('Original checkpoint manifest changed')
    rows=json.loads(path.read_bytes())['files'];observed={}
    for name,row in rows.items():
        p=model/name
        if not p.is_file() or p.is_symlink():raise ValueError('Original model file missing or unsafe: '+name)
        before=p.stat();digest=sha(p);after=p.stat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Model changed during pure input admission')
        if after.st_size!=row['bytes'] or digest!=row['sha256']:raise ValueError('Original model bytes differ: '+name)
        observed[name]=dict(bytes=after.st_size,sha256=digest,revision=row['revision'],stat_identity=[after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns])
    for name,digest in pin['tokenizer_files'].items():
        if observed[name]['sha256']!=digest:raise ValueError('Original tokenizer differs')
    return dict(manifest_sha256=pin['model_manifest_sha256'],files=observed,files_verified=len(observed),model_started=False)

def admit_anchor(anchor):
    from official_image import read_lock,file_map
    path=ROOT/'engine/official-baselines/0.23.0.json';lock=read_lock(path)
    if (anchor/'image-lock.json').read_bytes()!=path.read_bytes():raise ValueError('Protocol anchor is not the actual selected official0.23 SDK')
    raw=(anchor/'installed-files.json').read_bytes();files=json.loads(raw);installed=json.loads((anchor/'installation.json').read_bytes())
    if file_map(anchor)!=files or hashlib.sha256(raw).hexdigest()!=installed['installed_files_sha256']:
        raise ValueError('Official anchor installed source differs')
    if installed['baseline_id']!=lock['id'] or installed['image_manifest']!=lock['manifest_digest']:
        raise ValueError('Official anchor image identity differs')
    return dict(role='protocol-python-and-CANN-anchor',baseline_id=lock['id'],image_manifest=lock['manifest_digest'],installed_files_sha256=installed['installed_files_sha256'],metadata=installed['metadata'],source_code_patched=False,model_started=False)

def verify_prepared_inputs(root):
    root=Path(root);pin=contracts();frozen=json.loads((root/'frozen-execution-plan.json').read_bytes());receipt=json.loads((root/'official-protocol-preparation.json').read_bytes())
    if frozen.get('schema')!=SCHEMA or receipt.get('schema')!=SCHEMA or receipt.get('status')!='prepared' or receipt.get('model_service_started') is not False or receipt.get('formal_client_started') is not False:
        raise ValueError('Official-only preparation is incomplete or claims execution')
    if any(k in frozen for k in ('baseline_deployment','baseline_source_map_sha256')):
        raise ValueError('Official input preparation must not impersonate legacy018 admission')
    if sha(root/'continuous-B0/inputs/continuous-inputs-r1.json')!=pin['continuous_input_sha256']:
        raise ValueError('Actual original C8 input bytes differ')
    if receipt.get('continuous_inputs',{}).get('recipe_sha256')!=pin['continuous_recipe_sha256']:
        raise ValueError('Original C8 recipe authority differs')
    if receipt.get('continuous_inputs',{}).get('path_binding_adapter'):
        from official_input_adapter import verify_record
        verify_record(receipt['continuous_inputs'])
    workload=json.loads((root/'native-token-workload.json').read_bytes());ids=workload['request_token_ids']
    actual=hashlib.sha256((json.dumps(ids,separators=(',',':'))+'\n').encode()).hexdigest()
    if actual!=pin['prompt_tokens_sha256'] or len(ids)!=pin['requests'] or sum(map(len,ids))!=pin['input_tokens']:
        raise ValueError('Actual original ordered MMLU200 inputs differ')
    return dict(continuous_input_sha256=pin['continuous_input_sha256'],prompt_tokens_sha256=actual,requests=len(ids),input_tokens=sum(map(len,ids)))

def prepare(config,anchor,*,native_path_binding=False,capability=None):
    sys.path.insert(0,str(TOOLS/'scripts'))
    out=Path(config['out']).resolve();out.mkdir(parents=True,exist_ok=False)
    pin=contracts();model=Path(config['model']).resolve();sdk=Path(config['deployment']).resolve()
    record=dict(schema=SCHEMA,status='preparing',started=time.time(),model_service_started=False,formal_client_started=False,legacy018_sdk_required=False,legacy018_sdk_admitted=False)
    try:
        if config.get('engine_profile')!='r8' or config.get('test_mode')!='full':raise ValueError('Official protocol preparation requires the original full R8 judge and five inputs')
        alias=Path('/models/Qwen3.5-35B-A3B')
        if native_path_binding:
            from engine.native_sdk_controller import native_scope
            if not native_scope() or not capability or capability.get('allowed') is not False:
                raise ValueError('Capless C8 adapter requires an actual native namespace capability refusal')
            record['tokenizer_path_mode']='declared-single-constant-binding'
            record['namespace_capability']=capability
        else:
            if not alias.is_dir() or not os.path.samefile(alias,model) or not os.statvfs(alias).f_flag&os.ST_RDONLY:
                raise ValueError('Frozen tokenizer recipe requires its read-only model alias in the private child namespace')
            record['readonly_tokenizer_alias']=dict(source=str(model),target=str(alias),read_only=True,private_namespace=os.readlink('/proc/self/ns/mnt'))
        record['preparation_sources']={name:sha(source_path(ROOT, name)) for name in ('official_protocol_data.py','official_protocol_preparation.py','run_all.py','official_execution.py','sdk_dependency_probe.py','official_input_adapter.py','native_sdk_controller.py','data/official-protocol-inputs.json')}
        from dataset_contract import prepare as prepare_dataset
        from third_party import descriptor,script_sources
        from restore_b1_r4 import verify_installed_sources
        from engine.sdk_dependency_probe import admit_dependencies, receipt_path
        from mmlu_scope import verify as verify_mmlu_scope
        from continuous_inputs import prepare as prepare_continuous
        from native_timing_inputs import prepare as prepare_native
        from benchmark import dataset_bundle
        installation=json.loads((sdk.parent/'installation.json').read_bytes());profile=json.loads((sdk/'deployment.json').read_bytes())
        if installation['release']!=pin['protocol_engine_release'] or profile['release']!=pin['protocol_engine_release'] or installation['package_version']!=pin['protocol_engine_version']:
            raise ValueError('Actual R8 judge installation differs')
        verify_installed_sources(sdk)
        record['r8_dependencies']=admit_dependencies(sdk,receipt=receipt_path(out))
        save(out/'optimized-dependency-admission.json',record['r8_dependencies'])
        installed=json.loads((sdk/'provenance/installed-files.json').read_bytes())
        for name,expected in [('runtime/entrypoints/prepare_continuous.py',pin['continuous_recipe_sha256']),('inputs/mmlu-quality018-r2/quality-request-plan.json',pin['quality_plan_sha256'])]:
            if installed.get(name)!=expected or sha(sdk/name)!=expected:raise ValueError('Original R8 input source changed: '+name)
        record['official_anchor']=admit_anchor(Path(anchor).resolve())
        record['model_admission']=admit_model(model,pin)
        record['r8_judge']=dict(release=profile['release'],package_version=installation['package_version'],deployment=str(sdk),installed_source_manifest_sha256=sha(sdk/'provenance/installed-files.json'))
        assets=dataset_bundle.prepare(config['cache'],ORDER,assets=config.get('dataset_assets'),context=True,deployment=sdk)
        manifest=dataset_bundle.manifest();record['raw_datasets']=dict(standard_id=manifest['standard_id'],manifest_sha256=sha(ROOT/'data/bundles.json'),archive_sha256={n:r['sha256'] for n,r in manifest['components'].items()},all_five_materialized=True,assets=str(assets))
        save(out/'mmlu-task-scope.json',verify_mmlu_scope(sdk));plans=out/'plans';plans.mkdir();prepared={}
        for name in ORDER:
            if name=='mmlu':continue
            source=assets/'public-agent-traces' if name in ('swe','terminal') else assets/name
            prepared[name]=prepare_dataset(name,source,plans/name,model,hle_profile='fixed200-v1')
            if prepared[name]['tasks']!=pin['denominators'][name]:raise ValueError('Original full population differs: '+name)
            for role in ('B0','B1'):
                kind='trace' if name in ('swe','terminal') else 'solver'
                descriptor(name,plans/name,role,kind,role,plans/(name+'-'+kind+'-'+role+'.json'),False)
            if name in ('frontier','hle'):
                for role in ('B0','B1'):
                    target=plans/(name+'-judge-'+role+'.json');spec=descriptor(name,plans/name,'B1','judge',role,target,False,out/(name+'-solver-'+role)/'service/evaluation')
                    if name=='hle':
                        from judge_grammar import selected
                        spec['structured_output_policy']=selected(dict(hle_judge_whitespace='bounded-v2'));save(target,spec)
        c8=out/'continuous-B0/inputs';c8.parent.mkdir()
        if native_path_binding:
            from official_input_adapter import prepare as prepare_bound
            continuous=prepare_bound(sdk,c8,model,pin,record['model_admission'])
        else:continuous=prepare_continuous(sdk,c8)
        if continuous['recipe_sha256']!=pin['continuous_recipe_sha256'] or continuous['sha256']!=pin['continuous_input_sha256']:raise ValueError('Original C8 input bytes differ')
        cases=json.loads(Path(continuous['path']).read_bytes())
        if {k:len(v) for k,v in cases.items()}!=pin['continuous_cohorts']:raise ValueError('Original C8 population differs')
        # This directory supplies only an unchanged data function's input file;
        # it contains no execution, readiness, timing or qualification receipt.
        pure_source=out/'tokenizer-validation-source';(pure_source/'service').mkdir(parents=True);shutil.copyfile(continuous['path'],pure_source/'service/request-plan.json')
        workload=prepare_native(sdk,model,pure_source,out/'native-token-workload.json');ids=workload['request_token_ids'];input_hash=hashlib.sha256((json.dumps(ids,separators=(',',':'))+'\n').encode()).hexdigest()
        if len(ids)!=pin['requests'] or sum(map(len,ids))!=pin['input_tokens'] or input_hash!=pin['prompt_tokens_sha256']:
            raise ValueError('Original ordered MMLU200 prompt IDs differ')
        sources=script_sources()
        for name in sources:
            target=out/'execution-source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(TOOLS/name,target)
        frozen=dict(schema=SCHEMA,prepared_before_model_start=True,model=str(model),datasets=ORDER,smoke_evaluators=False,engine_release=profile['release'],optimized_deployment=str(sdk),prepared_datasets=prepared,hle_profile='fixed200-v1',hle_judge_whitespace='bounded-v2',sources=sources,execution_source_snapshots=True,official_anchor=record['official_anchor'],r8_judge=record['r8_judge'],legacy018_sdk_required=False,legacy018_sdk_admitted=False,continuous_input_sha256=continuous['sha256'],mmlu_inputs=dict(requests=len(ids),input_tokens=sum(map(len,ids)),prompt_tokens_sha256=input_hash,quality_plan_sha256=pin['quality_plan_sha256']),model_service_started=False,formal_client_started=False)
        save(out/'frozen-execution-plan.json',frozen)
        if continuous.get('path_binding_adapter'):frozen['input_path_binding_adapter']=continuous['path_binding_adapter'];save(out/'frozen-execution-plan.json',frozen)
        record.update(status='prepared',continuous_inputs=continuous,mmlu_inputs=frozen['mmlu_inputs'],prepared_plan_sha256=sha(out/'frozen-execution-plan.json'),source_functions_unchanged=not native_path_binding,pure_dataset_functions_unchanged=True)
    except BaseException as error:record.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:record['ended']=time.time();save(out/'official-protocol-preparation.json',record)
    return record

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--anchor-sdk',type=Path,required=True);parser.add_argument('--tokenizer-child',action='store_true');args=parser.parse_args();config=json.loads(args.config.read_bytes())
    if config.get('baseline_deployment'):raise ValueError('Official data preparation does not admit a legacy baseline deployment')
    if args.tokenizer_child:return prepare(config,args.anchor_sdk)
    from engine.native_sdk_controller import native_scope, namespace_capability
    if native_scope():
        capability=namespace_capability()
        if not capability['allowed']:
            # This process only prepares input data; no serving or timing child
            # inherits these CPU-only safeguards.
            os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
            os.environ['ASCEND_RT_VISIBLE_DEVICES']=''
            return prepare(config,args.anchor_sdk,native_path_binding=True,capability=capability)
    import official_execution as E
    env={k:v for k,v in os.environ.items() if not k.endswith(('_TOKEN','_API_KEY','_PASSWORD','_SECRET'))}
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--config',str(args.config.resolve()),'--anchor-sdk',str(args.anchor_sdk.resolve()),'--tokenizer-child']
    inherited=E.tokenizer_alias(command,model=config['model'],sdk=args.anchor_sdk,cache=config['cache'],env=env,cwd=ROOT)
    if inherited is not None:return inherited
    root=E.ensure_for_sdk(args.anchor_sdk)
    # A host entry first enters the pinned loader/root. Its next invocation
    # creates only the alias namespace, without rebinding a chroot into itself.
    request=E.request_for(command[:-1],root,env=env,cwd=ROOT,extra_paths=(Path(config['out']).parent,Path(config['cache']),Path(config['model']),Path(config['deployment']),args.anchor_sdk))
    E.readonly_sdk_anchor(request,args.anchor_sdk)
    return subprocess.run([sys.executable,'-B',str(Path(E.__file__).resolve()),'--exec-json'],input=json.dumps(request),env=os.environ.copy(),text=True).returncode

if __name__=='__main__':
    result=main();raise SystemExit(result if isinstance(result,int) else 0)
