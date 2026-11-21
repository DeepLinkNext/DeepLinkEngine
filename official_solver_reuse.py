"""Retain completed solvers only after the old judge's pre-launch bind failure.

This copies sealed originals unchanged into a new campaign. It does not retry
an incomplete solver, replace failed responses or qualify arithmetic/efficiency.
"""
import json
import math
from pathlib import Path
import shutil

from official_reuse import read, sha, verified

PHASES=('continuous','mmlu','frontier','hle','swe','terminal','frontier-judge','hle-judge','context','native-timing')
DENOMINATORS=dict(mmlu=200,hle=200,frontier=160,swe=64,terminal=48)


def process_gone(actions):
    for action in actions:
        stat=Path('/proc')/str(action['pid'])/'stat'
        if stat.exists() and stat.read_text().rsplit(')',1)[1].split()[19]==str(action['start_ticks']):
            raise ValueError('A retained phase still has a live owned process')


def complete_clock(row):
    values=[row['start']['monotonic'],row['end']['monotonic']]
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values) or values[1]<=values[0]:
        raise ValueError('Retained phase clock is incomplete or nonfinite')
    return values


def request_rows(root,seal,kind):
    result=[]
    for name in seal:
        if not name.startswith('service/') or not name.endswith('.json') or len(Path(name).parts)!=2:
            continue
        row=read(root,name,seal)
        if isinstance(row,dict) and 'success' in row and 'start' in row and 'end' in row and ('phase' in row if kind=='continuous' else 'case_id' in row):
            complete_clock(row)
            result.append(row)
    return result


def continuous_population(root,seal):
    result=read(root,'service/continuous.json',seal)
    names=['4096-warmup','4096-formal','16384-warmup','16384-formal']
    if result['status']!='completed' or [v['name'] for v in result['phases']]!=names:
        raise ValueError('Retained C8 does not contain all original windows')
    rows=request_rows(root,seal,'continuous')
    if set(row['phase'] for row in rows)!=set(names):
        raise ValueError('Retained C8 has missing or foreign request phases')
    for index,phase in enumerate(result['phases']):
        start,end=complete_clock(phase)
        cohort=[v for v in rows if v['phase']==phase['name']]
        if (phase['seconds']!=(60 if index%2==0 else 300) or phase['failed']!=0
                or phase['requests']<=0 or len(cohort)!=phase['requests']
                or any(v['success'] is not True or v['usage']['completion_tokens']!=256
                       or v['finish_reason']!='length' or not start<=v['start']['monotonic']<v['end']['monotonic']<=end for v in cohort)):
            raise ValueError('Retained C8 original budget, population or clock differs')


def joint_population(root,seal):
    phase=read(root,'service/joint.json',seal)
    start,end=complete_clock(phase)
    rows=request_rows(root,seal,'joint')
    if (len(rows)!=200 or {v['index'] for v in rows}!=set(range(200))
            or len({v['case_id'] for v in rows})!=200
            or any(not start<=v['start']['monotonic']<v['end']['monotonic']<=end for v in rows)):
        raise ValueError('Retained MMLU original200 population differs')
    good=[v for v in rows if v['success'] is True]
    if (phase['total']!=200 or phase['completed']!=len(good) or phase['failed']!=200-len(good)
            or phase['correct']!=sum(v['correct'] is True for v in rows)
            or phase['truncated']!=sum(v.get('finish_reason')=='length' for v in rows)
            or phase['successful_output_tokens']!=sum(v['usage']['completion_tokens'] for v in good)
            or not math.isclose(phase['duration_seconds'],end-start)):
        raise ValueError('Retained MMLU summary differs from its original records')


def dataset_population(root,seal,name):
    spec=read(root,'dataset-spec.json',seal)
    result=read(root,'performance-summary.json',seal)
    expected=DENOMINATORS[name]
    if (spec['dataset']!=name or spec['diagnostic'] is not False or len(spec['case_ids'])!=expected
            or len(set(spec['case_ids']))!=expected or result['dataset']!=name
            or result['planned']!=expected or result['recorded']!=expected
            or result['successful']+result['failed']!=expected or result['unreturned']!=0):
        raise ValueError('Retained fixed dataset population differs: '+name)
    records=[]
    for item in seal:
        if item.startswith('service/evaluation/requests/') and item.endswith('.json'):
            records.append(read(root,item,seal))
    if (len(records)!=expected or {v['case_id'] for v in records}!=set(spec['case_ids'])
            or sum(v['status']=='ok' for v in records)!=result['successful']):
        raise ValueError('Retained fixed dataset originals are incomplete: '+name)


def source_identity(campaign,sources,current_root,old_root):
    policy_path=current_root/'engine/official-baselines/solver-reuse-policy-v1.json'
    policy=json.loads(policy_path.read_bytes())
    if policy.get('schema')!='deeplinkengine-reviewed-solver-reuse/v1' or policy.get('scope')!='fixed-judge-prelaunch-bind-only':
        raise ValueError('Unknown retained solver source policy')
    changes={}
    expected=set()
    for name,pin in campaign['sources'].items():
        path=Path(name)
        if sha(path)!=pin:
            raise ValueError('Original solver source changed: '+name)
        if path.is_relative_to(old_root):
            relative=path.relative_to(old_root).as_posix()
            target=current_root/relative
            current=sha(target)
            if current!=pin:
                if policy['reviewed_controller_changes'].get(relative)!={'original_sha256':pin,'current_sha256':current}:
                    raise ValueError('Unreviewed solver source difference: '+relative)
                changes[relative]=dict(original_sha256=pin,current_sha256=current)
        else:
            target=path
            current=pin
        key=str(target.resolve())
        if sources.get(key)!=current:
            raise ValueError('Current solver source admission differs: '+str(target))
        expected.add(key)
    for path in (current_root/'official_solver_reuse.py',policy_path):
        if sources.get(str(path.resolve()))!=sha(path):
            raise ValueError('Retained solver controller/policy is not source-bound')
        expected.add(str(path.resolve()))
    if set(sources)!=expected or set(changes)!=set(policy['reviewed_controller_changes']):
        raise ValueError('Retained solver source set or reviewed transition differs')
    return changes


def admit(source,destination,*,config,image_lock,sources,weights,mapping,inputs,current_root,arguments):
    if Path(source).is_symlink() or Path(destination).is_symlink():
        raise ValueError('Retained solver campaigns must not be symlinks')
    source,destination,current_root=map(lambda p:Path(p).resolve(),(source,destination,current_root))
    if (source==destination or destination.is_relative_to(source) or source.is_relative_to(destination)
            or not destination.is_dir() or any((destination/name).exists() or (destination/name).is_symlink() for name in PHASES)):
        raise ValueError('Solver retention requires an original campaign and a new empty campaign')
    outer=json.loads((source/'checksums.json').read_bytes())
    campaign=read(source,'campaign.json',outer)
    outer_result=source.parent/'results.json'
    prior=json.loads(outer_result.read_bytes())
    stages=campaign['stages']
    if (prior['status']!='incomplete' or not prior.get('ended') or len(prior['stages'])!=1
            or prior['stages'][0]['status']!='failed' or len(prior['configuration']['versions'])!=1
            or campaign['schema']!='deeplinkengine-official-full-suite/v1' or campaign['status']!='failed'
            or not campaign.get('ended') or campaign['baseline_image']!=image_lock
            or campaign['physical_mapping']!=mapping or campaign['continuous_input_sha256']!=sha(inputs)
            or campaign['fixed_five_denominators']!=DENOMINATORS
            or [v['name'] for v in stages]!=list(PHASES)
            or stages[0]['status'] not in ('completed','reused')
            or any(v['status']!='completed' for v in stages[1:6])
            or stages[6]['status']!='failed' or any(v['status']!='not_attempted' for v in stages[7:])):
        raise ValueError('Only all completed solvers followed by the first fixed-judge pre-launch failure can be retained')
    failed=read(source,'frontier-judge/receipt.json',outer)
    if (failed.get('status')!='failed' or failed.get('completed') is not False or failed.get('quiescent') is not True
            or failed.get('cleanup_actions')!=[] or 'OSError: [Errno 98]' not in failed.get('error','')
            or (source/'frontier-judge/judge-service.log').exists() or (source/'frontier-judge/judge-runtime').exists()):
        raise ValueError('The first fixed judge may have launched; pre-launch-only retention refuses a repeat')
    if stages[0]['status']=='reused':
        reuse=read(source,'continuous-reuse.json',outer)
        original=Path(reuse['source'])
        if (reuse.get('inference_repeated') is not False or reuse.get('original_dates_and_outputs_preserved') is not True
                or sha(original/'checksums.json')!=reuse['original_checksums_sha256']
                or sha(original.parent/'checksums.json')!=reuse['original_campaign_checksums_sha256']):
            raise ValueError('Original C8 retention chain differs')
    fingerprint=lambda value:{n:{k:v[k] for k in ('bytes','sha256')} for n,v in value.items()}
    if fingerprint(read(source,'model-weights.json',outer))!=fingerprint(weights):
        raise ValueError('Retained solver checkpoint differs')
    mmlu_seal=read(source,'mmlu/checksums.json',outer)
    old_plan=read(source/'mmlu','service/plan.json',mmlu_seal)
    old_root=Path(old_plan['client_tools']).resolve().parents[1]
    changes=source_identity(campaign,sources,current_root,old_root)
    expected_plan=sha(Path(config['protocol_run'])/'frozen-execution-plan.json')
    selections=[]
    for name in PHASES[:6]:
        root=source/name
        seal=read(source,name+'/checksums.json',outer)
        for item in seal:
            verified(root,item,seal)
            if outer.get(name+'/'+item)!=seal[item]:
                raise ValueError('Campaign and phase seals disagree: '+name+'/'+item)
        receipt=read(root,'receipt.json',seal)
        if (any(receipt.get(key) is not True for key in ('completed','quiescent','model_ready','client_started'))
                or receipt['baseline_id']!=image_lock['id'] or receipt['phase']!=name
                or receipt['input_plan_sha256']!=expected_plan):
            raise ValueError('Retained solver receipt or protocol differs: '+name)
        process_gone(receipt['cleanup_actions'])
        plan=read(root,'service/plan.json',seal)
        kind='continuous' if name=='continuous' else 'joint' if name=='mmlu' else 'dataset'
        if (Path(plan['model']).resolve()!=Path(config['model']).resolve()
                or Path(plan['protocol_deployment']).resolve()!=Path(config['protocol_deployment']).resolve()
                or plan['serve_arguments']!=arguments(kind)):
            raise ValueError('Retained solver serving arguments or SDK differ: '+name)
        if name=='continuous':continuous_population(root,seal)
        elif name=='mmlu':joint_population(root,seal)
        else:dataset_population(root,seal,name)
        selections.append((name,root,seal,receipt))
    retained=[]
    # All originals and transitions are admitted before the first copy.
    for name,root,seal,receipt in selections:
        target=destination/name
        target.mkdir()
        for item in seal:
            copied=target/item
            copied.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/item,copied)
            verified(target,item,seal)
        shutil.copy2(root/'checksums.json',target/'checksums.json')
        retained.append(dict(name=name,source=str(root),original_checksums_sha256=sha(root/'checksums.json'),original_receipt_sha256=seal['receipt.json']))
    return dict(schema='deeplinkengine-retained-completed-solvers/v1',source=str(source),phases=retained,
                original_campaign_checksums_sha256=sha(source/'checksums.json'),original_outer_results_sha256=sha(outer_result),
                reviewed_controller_changes=changes,inference_repeated=False,original_dates_and_outputs_preserved=True,
                failed_responses_preserved=True,new_F_T_qualification=False,independent_third_party=False)
