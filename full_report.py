from release_selection import kit_path, runtime_lock, profile as selected_profile
"""Build binary milestone/final decisions from independently audited run summaries."""
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import html
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
DATASETS = ('mmlu', 'hle', 'frontier', 'swe', 'terminal')
COUNTS = dict(mmlu=200, hle=200, frontier=160, swe=64, terminal=48)
NAMES = dict(mmlu='MMLU-Pro', hle='HLE-Verified', frontier='FrontierScience',
             swe='SWE-bench-Pro trace', terminal='TerminalBench 2.1 trace')


def number(value):
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def ratio(a, b):
    x, y = number(a), number(b)
    return x / y if x is not None and y is not None and x > 0 and y > 0 else None


def decision(met, reason):
    return dict(met=bool(met), status='达成目标' if met else '未达成目标', reason=reason)


def quality_gates(fixed):
    result={n:decision(n in ('swe','terminal'),'公开轨迹只测性能。' if n in ('swe','terminal') else '评分证据不完整。') for n in DATASETS}
    pair=fixed.get('current_mmlu_pair') or {}
    arms=pair.get('arms',{})
    if all(r in arms for r in ('B0','B1')):
        qa,qb=[arms[r].get('quality') or {} for r in ('B0','B1')]
        valid=all(q.get('planned_denominator')==q.get('recorded')==200 and q.get('unreturned')==0
                  and type(q.get('correct')) is int and 0<=q['correct']<=200 for q in (qa,qb))
        if valid:
            drop=Decimal(qa['correct']-qb['correct'])/2
            result['mmlu']=dict(decision(drop<=5,'固定200题正确率绝对下降≤5个百分点。'),drop_pp=str(drop))
    scores=fixed.get('current_dataset_quality') or {}
    for name,denominators in [('hle',{'gold':74,'revision':126}),('frontier',{'olympiad':100,'research':60})]:
        views=[scores.get(name+'-'+r) or {} for r in ('B0','B1')]
        def valid(view):
            tracks=view.get('tracks',{})
            return set(tracks)==set(denominators) and all(tracks[t].get('total')==n
                and type(tracks[t].get('successes')) is int and 0<=tracks[t]['successes']<=n for t,n in denominators.items())
        if not all(valid(v) for v in views):continue
        if name=='hle':
            drops={'overall':Decimal(sum(views[0]['tracks'][t]['successes']-views[1]['tracks'][t]['successes'] for t in denominators))/2}
        else:
            drops={t:Decimal(100)*(views[0]['tracks'][t]['successes']-views[1]['tracks'][t]['successes'])/n for t,n in denominators.items()}
        result[name]=dict(decision(all(d<=5 for d in drops.values()),'固定计划分母；未决计非成功；HLE比较总体，Frontier分别比较两个赛道，绝对下降≤5个百分点。'),drop_pp={k:str(v) for k,v in drops.items()})
    return result


def coverage(util):
    missing=number(util.get('missing_seconds'));covered=number(util.get('covered_seconds'))
    return missing is not None and missing==0 and covered is not None and covered>0


HAZARD=re.compile(r'No space left on device|Disk quota exceeded|out of memory|OutOfMemory|segmentation fault|deadlock detected|Traceback \(most recent call last\)|Fatal Python error|log write failed',re.I)


def audit_context(root):
    initial=json.loads((root/'memory-initial.json').read_text())
    samples=[json.loads(line) for line in (root/'memory-samples.jsonl').read_text().splitlines() if line]
    errors=[]
    if not samples:errors.append('没有内存监测样本')
    for row in samples:
        if any(row['events'].get(k,0)>initial['events'].get(k,0) for k in ('oom','oom_kill','oom_group_kill')):
            errors.append('cgroup新增OOM事件');break
    cleanup=json.loads((root/'cleanup.json').read_text())
    if cleanup.get('quiescent') is not True:errors.append('服务未完成清理')
    logs=[]
    for path in root.rglob('*.log'):
        logs.append(str(path.relative_to(root)))
        with path.open(errors='replace') as stream:
            for index,line in enumerate(stream,1):
                if HAZARD.search(line):errors.append(str(path.relative_to(root))+':'+str(index)+': '+line.strip()[:240])
    if 'service.log' not in logs:errors.append('缺少服务日志')
    return dict(clean=not errors,errors=errors,logs=logs)


def workload_pairs(fixed):
    pairs = {}
    quality = quality_gates(fixed or {})
    mmlu = (fixed or {}).get('current_mmlu_pair')
    if mmlu:
        arms = mmlu['arms']
        complete = all(arms[r].get('recorded_requests') == arms[r].get('successful_requests') == 200
                       and arms[r].get('failed_requests') == 0 for r in ('B0', 'B1'))
        pairs['mmlu'] = dict(complete=complete, quality=quality['mmlu'], arms={r: arms[r]['groups']['joint200'] for r in ('B0','B1')})
    observed = (fixed or {}).get('current_dataset_performance', {})
    traces = (fixed or {}).get('current_trace_performance', {})
    for name in DATASETS[1:]:
        arms = (traces.get(name) or {}).get('arms', {}) if name in ('swe','terminal') else {
            role: observed.get(name+'-solver-'+role, {}) for role in ('B0','B1')}
        if not all(arms.get(role) for role in ('B0','B1')):
            continue
        complete = all(a.get('planned') == a.get('recorded') == a.get('successful') == COUNTS[name]
                       and a.get('failed') == a.get('unreturned') == 0 and not a.get('diagnostic')
                       for a in arms.values())
        if name in ('swe','terminal'):
            complete = complete and traces[name].get('paired_execution_complete') is True
        else:
            complete = complete and all(a.get('receipt', {}).get('completed') is True for a in arms.values())
        pairs[name] = dict(complete=bool(complete), quality=quality[name], arms=arms)
    return pairs


def performance(pairs, throughput, latency):
    results = {}
    for name, pair in pairs.items():
        a,b = pair['arms']['B0'],pair['arms']['B1']
        values = dict(throughput_ratio=ratio(b.get('output_tokens_per_second'),a.get('output_tokens_per_second')),
                      ttft_ratio=ratio(b.get('mean_ttft_ms'),a.get('mean_ttft_ms')),
                      tpot_ratio=ratio(b.get('mean_tpot_ms'),a.get('mean_tpot_ms')))
        passed = pair['complete'] and pair['quality']['met'] and all(v is not None for v in values.values())
        passed = passed and values['throughput_ratio'] >= Decimal(throughput) and values['ttft_ratio'] <= Decimal(latency) and values['tpot_ratio'] <= Decimal(latency)
        results[name] = dict(met=bool(passed), quality=pair['quality'], execution_complete=pair['complete'], **{k: str(v) if v is not None else None for k,v in values.items()})
    winners = [NAMES[n] for n in DATASETS if results.get(n, {}).get('met')]
    return decision(bool(winners), '同一数据集同时满足吞吐及两项平均时延：'+', '.join(winners) if winners else '没有数据集同时满足完整请求、质量及三项性能门槛。'), results


def phase_valid(phase):
    m=phase.get('metrics', {})
    return (phase.get('status')=='completed' and m.get('requests',0)>0
            and m.get('successful')==m.get('requests') and m.get('failed')==0
            and number(m.get('duration_seconds')) is not None and number(m['duration_seconds'])>0
            and number(m.get('output_tokens_per_second')) is not None and number(m['output_tokens_per_second'])>0)


def context_decisions(context, audits):
    rows={r['phase']:r for r in (context or {}).get('comparison',[])}
    milestone=False; final=False; ratios={}; windows={}
    clean = all(audits.get(r, {}).get('clean') is True for r in ('B0','B1'))
    if all(n in rows and all(phase_valid(rows[n][r]) for r in ('B0','B1')) for n in ('4096-formal','16384-formal')):
        low,high=[rows[n]['B1']['metrics'] for n in ('4096-formal','16384-formal')]
        ratios={'throughput':ratio(high.get('output_tokens_per_second'),low.get('output_tokens_per_second'))}
        for metric in ('ttft','tpot'):
            for tail in ('p95','p99'):
                ratios[metric+'_'+tail]=ratio(high[metric+'_ms'].get(tail),low[metric+'_ms'].get(tail))
        milestone=clean and all(v is not None for v in ratios.values()) and ratios['throughput']>=Decimal('.50') and all(
            ratios[m+'_'+p]<=limit for m,limit in [('ttft',Decimal('3')),('tpot',Decimal('2'))] for p in ('p95','p99'))
    row=rows.get('32768-formal')
    if row and all(phase_valid(row[r]) for r in ('B0','B1')):
        group=row['B1'].get('windows',[])
        if len(group)==5 and all(phase_valid(dict(status='completed',metrics=w['metrics'])) for w in group):
            values={'throughput':[number(w['metrics'].get('output_tokens_per_second')) for w in group]}
            for metric in ('ttft','tpot'):
                for tail in ('p95','p99'):
                    values[metric+'_'+tail]=[number(w['metrics'][metric+'_ms'].get(tail)) for w in group]
            for key,series in values.items():
                windows[key]=max(series)/min(series) if all(v is not None and v>0 for v in series) else None
            final=clean and all(v is not None for v in windows.values()) and windows['throughput']<=Decimal('1.50') and all(
                windows[m+'_'+p]<=limit for m,limit in [('ttft',Decimal('2')),('tpot',Decimal('1.25'))] for p in ('p95','p99'))
    return dict(milestone=decision(milestone,'16K/4K吞吐≥50%，TTFT P95/P99比值≤3，TPOT P95/P99比值≤2；两臂完整且无失败。'),
                final=decision(final,'32K全部五个60秒窗口：吞吐max/min≤1.50，TTFT P95/P99≤2，TPOT P95/P99≤1.25；两臂完整且无失败。'),
                blocking_errors={r:audits.get(r,{}).get('errors',[]) for r in ('B0','B1')},
                measured_ratios={k:str(v) if v is not None else None for k,v in ratios.items()},
                measured_window_ratios={k:str(v) if v is not None else None for k,v in windows.items()})


def dataset_costs(fixed, pairs):
    mmlu=fixed.get('current_mmlu_pair') or {}
    mmlu_gains=mmlu.get('gains',{})
    observations=dict(fixed.get('current_cost_utilization') or {})
    observations['mmlu']=dict(unit_cost_reduction_percent=mmlu_gains.get('equal_rate_unit_cost_reduction_pct'),
        utilization_gain_pp=mmlu_gains.get('ai_core_utilization_gain_pp'),utilization=mmlu.get('utilization',{}))
    details={}
    for name in DATASETS:
        row=observations.get(name,{})
        cost=number(row.get('unit_cost_reduction_percent'));util=number(row.get('utilization_gain_pp'))
        samples=row.get('utilization',{})
        pair=pairs.get(name,{})
        met=bool(pair.get('complete') and pair.get('quality',{}).get('met')
            and set(samples)=={'B0','B1'} and all(coverage(v) for v in samples.values())
            and cost is not None and util is not None and cost>=10 and util>=10)
        details[name]=dict(met=met,execution_complete=bool(pair.get('complete')),
            quality=pair.get('quality'),unit_cost_reduction_percent=str(cost) if cost is not None else None,
            utilization_gain_pp=str(util) if util is not None else None,utilization=samples)
    winners=[n for n in DATASETS if details[n]['met']]
    selected=winners[0] if winners else 'mmlu'
    return winners,selected,details


def sealed_dataset_utilization(root, expected_count):
    """Use raw sealed request clocks and the same physical-card sampler window."""
    from telemetry_audit import utilization
    root=Path(root).resolve();seal=json.loads((root/'checksums.json').read_bytes())
    consumed={}
    def read(path):
        relative=str(path.relative_to(root));raw=path.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if seal.get(relative)!=digest:raise ValueError('Dataset cost input has no matching seal: '+relative)
        consumed[relative]=digest
        return raw
    clocks=[]
    for path in sorted((root/'service/evaluation/requests').glob('*.json')):
        record=json.loads(read(path));clocks.append((record['start']['monotonic'],record['end']['monotonic']))
    if len(clocks)!=expected_count:raise ValueError('Dataset cost window requires every frozen request')
    devices=json.loads(read(root/'resource-admission.json'))['physical_cards']
    if len(devices)!=2 or len(set(devices))!=2:raise ValueError('Dataset cost requires the same two physical cards')
    rows=[json.loads(line) for line in read(root/'device-samples.jsonl').splitlines() if line]
    start=min(v[0] for v in clocks);end=max(v[1] for v in clocks)
    result=utilization(rows,start,end,tuple(map(str,devices)))
    result['request_window']=dict(start_monotonic=start,end_monotonic=end,requests=expected_count)
    result['sealed_inputs']=consumed
    return result


def attach_dataset_costs(fixed, root):
    pairs=workload_pairs(fixed);rows={}
    for name in DATASETS[1:]:
        pair=pairs.get(name,{})
        if not pair.get('complete'):continue
        stage=lambda role: name+('-trace-' if name in ('swe','terminal') else '-solver-')+role
        samples={role:sealed_dataset_utilization(Path(fixed.get('reused_stage_sources',{}).get(stage(role),root/stage(role))),COUNTS[name]) for role in ('B0','B1')}
        if set(samples['B0']['per_card_percent'])!=set(samples['B1']['per_card_percent']):
            raise ValueError('Dataset cost pair requires identical physical cards')
        speed=ratio(pair['arms']['B0'].get('output_tokens_per_second'),pair['arms']['B1'].get('output_tokens_per_second'))
        rows[name]=dict(unit_cost_reduction_percent=str(100*(1-speed)) if speed is not None else None,
            utilization_gain_pp=str(number(samples['B1']['mean_percent'])-number(samples['B0']['mean_percent'])),utilization=samples)
    fixed['current_cost_utilization']=rows


def targets(fixed, context, audits):
    fixed=fixed or {}; ref=fixed.get('current_compute_reference') or {}
    value=None
    sys.path.insert(0, str(kit_path()/'scripts'))
    from compute_reference import target as reference_target
    valid = ref.get('certificate_kind')=='qualified_full_reference_denominator_reuse/v1' and reference_target(ref).get('status')=='current_workload_measured'
    if valid:
        value=number(ref.get('arms',{}).get('B1',{}).get('efficiency_lower_percent'))
    compute=dict(evidence_reason=reference_target(ref)['reason'],milestone=decision(value is not None and value>=50,'本轮必要算量、同步计时和已核验同工况参考分母，效率下界≥50%。'),
                 final=decision(value is not None and value>=90,'同一效率下界≥90%。'),
                 efficiency_lower_percent=str(value) if value is not None else None, certificate_kind=ref.get('certificate_kind'))
    pairs=workload_pairs(fixed)
    milestone, details=performance(pairs,'1.10','.95'); final, final_details=performance(pairs,'2','.80')
    speed=dict(milestone=milestone,final=final,datasets=details,final_datasets=final_details)
    cost_winners,cost_selected,cost_details=dataset_costs(fixed,pairs)
    selected=cost_details[cost_selected]
    cost=number(selected['unit_cost_reduction_percent']);util=number(selected['utilization_gain_pp'])
    multi=next((r for r in (context or {}).get('comparison',[]) if r['phase']=='multi-tenant-formal'),None)
    final_met=False;reduction=None;usage=None
    if multi and all(phase_valid(multi[r]) and audits.get(r,{}).get('clean') is True and coverage(multi[r].get('utilization',{})) for r in ('B0','B1')):
        tenants=multi['B1'].get('per_tenant',{})
        reduction=number(multi.get('equal_rate_same_card_count_unit_cost_reduction_percent'))
        usage=number(multi['B1'].get('utilization',{}).get('mean_percent'))
        final_met=(reduction is not None and usage is not None and reduction>=30 and usage>=80
                   and len(tenants)==4 and all(m.get('requests',0)>0 and m.get('successful')==m.get('requests') and m.get('failed')==0 for m in tenants.values()))
    costs=dict(milestone=decision(bool(cost_winners),'同一固定数据集完整请求及质量门槛满足，单位输出token成本降低≥10%，同窗AI Core均值提高≥10个百分点。'),
               datasets=cost_details,qualifying_datasets=cost_winners,display_dataset=cost_selected,
               final=decision(final_met,'四租户：单位输出token成本降低≥30%，B1利用率≥80%，四个租户均成功且无失败。'),
               unit_cost_reduction_percent=str(cost) if cost is not None else None,
               utilization_gain_pp=str(util) if util is not None else None,
               four_tenant_cost_reduction_percent=str(reduction) if reduction is not None else None,
               four_tenant_ai_core_mean_percent=str(usage) if usage is not None else None,
               formula='C_百万token=10^6×P×k×T/(30×86400×N)；同费率同卡数降低率=100×(1-Q_B0/Q_B1)。')
    return {'4.1':compute,'4.2':speed,'4.3':context_decisions(context, audits),'4.4':costs}


LABELS={'4.1':'有效算力','4.2':'吞吐与时延','4.3':'上下文稳定性','4.4':'产能与成本'}


def collect(root):
    from release_selection import bind
    saved = Path(root) / 'engine-selection.json'
    if saved.is_file():
        bind(saved)
    """Re-audit sealed originals; generated summaries alone never authorize a pass."""
    root=Path(root).resolve()
    from runtime_storage import restore_missing
    restore_missing(root/'installation')
    sys.path.insert(0,str(kit_path()/'scripts'))
    from third_party_report import inspect
    from acceptance_suite import paired
    fixed=inspect(root/'fixed-five')
    if fixed['plan']['datasets']!=list(DATASETS):
        raise ValueError('Full report requires all five frozen datasets')
    attach_dataset_costs(fixed,root/'fixed-five')
    context=paired({r:root/'context-tenant'/r for r in ('B0','B1')})
    stored=json.loads((root/'context-tenant/paired-summary.json').read_text())
    if stored!=context:
        raise ValueError('Context summary differs from independently re-audited originals')
    stack_path=kit_path()/'mods'/('active-stack-'+fixed['plan']['engine_release'].rsplit('-',1)[1]+'.json')
    stack=json.loads(stack_path.read_text())
    if 'enabled_mods' not in stack:
        registry=json.loads((kit_path()/'mods/registry.json').read_text())
        rows=registry['entries']
        rows=[dict(m,**json.loads((kit_path()/'mods'/m['manifest']).read_text())) for m in rows]
        if isinstance(rows,dict):
            rows=[dict(value,id=key) for key,value in rows.items()]
        wanted=set(stack['inherited']+stack['additions'])
        stack['enabled_mods']=[m for m in rows if m['id'] in wanted]
        if {m['id'] for m in stack['enabled_mods']}!=wanted:
            raise ValueError('Selected MOD registry is incomplete')
    if stack['release']!=fixed['plan']['engine_release']:
        raise ValueError('MOD catalog release differs from the measured engine')
    audits={r:audit_context(root/'context-tenant'/r) for r in ('B0','B1')}
    for row in context['comparison']:
        for r in ('B0','B1'):
            if not coverage(row[r].get('utilization',{})):
                audits[r]['errors'].append(row['phase']+': 利用率采样未覆盖完整窗口')
                audits[r]['clean']=False
    hashes={}
    for name in ('fixed-five/campaign-checksums.json','fixed-five/frozen-execution-plan.json',
                 'context-tenant/B0/checksums.json','context-tenant/B1/checksums.json'):
        with (root/name).open('rb') as stream:hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(schema='deeplinkengine-full-acceptance-report/v1',run=str(root),
                execution_complete=fixed['receipt'].get('selected_execution_complete') is True and context['execution_complete'],
                engine_release=fixed['plan']['engine_release'],
                frozen_mods=[{k:m[k] for k in ('id','name','version')} for m in stack['enabled_mods']],
                mod_catalog_sha256=hashlib.sha256(stack_path.read_bytes()).hexdigest(),
                targets=targets(fixed,context,audits),workloads=workload_pairs(fixed),
                context_audit=audits,evidence_sha256=hashes,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scoring='MMLU规则评分；HLE/Frontier使用同一模型辅助评分，未决计入固定分母；SWE/Terminal为公开轨迹性能。',
                reference_method='本轮必要F及原生同步T，除以经核验复用的原同工况参考算力上界；不复用旧效率。')


def charts(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if matplotlib.__version__!='3.10.8':raise RuntimeError('Expected matplotlib3.10.8')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    names=[n for n in DATASETS if n in data['workloads']]
    if names!=list(DATASETS):raise ValueError('Missing fixed dataset performance pairs')
    for ax,key,label in zip(axes,('output_tokens_per_second','mean_ttft_ms','mean_tpot_ms'),('Output token/s','Mean TTFT (ms)','Mean TPOT (ms)'),strict=True):
        for role,offset,color in [('B0',-.18,'#64748b'),('B1',.18,'#2563eb')]:
            values=[data['workloads'][n]['arms'][role].get(key) for n in names]
            if not all(number(v) is not None for v in values):raise ValueError('Plot input is missing: '+key)
            ax.bar([i+offset for i in range(len(names))],list(map(float,values)),width=.36,color=color,label=role)
        ax.set_xticks(range(len(names)),['MMLU','HLE','Frontier','SWE trace','Terminal trace'][:len(names)],rotation=25)
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.15);ax.legend()
    fig.tight_layout();fig.savefig(out/'fixed-five.svg',metadata={'Date':None});fig.savefig(out/'fixed-five.png',dpi=160);plt.close(fig)


def render(root,out):
    data=collect(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    charts(data,out)
    (out/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    esc=html.escape
    rows=''.join('<tr><th>'+key+' '+LABELS[key]+'</th><td class="'+('yes' if value['milestone']['met'] else 'no')+'">'+value['milestone']['status']+'</td><td class="'+('yes' if value['final']['met'] else 'no')+'">'+value['final']['status']+'</td></tr>' for key,value in data['targets'].items())
    details=''.join('<details><summary>'+key+' '+LABELS[key]+'：计算与判定详情</summary><pre>'+esc(json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>' for key,value in data['targets'].items())
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeepLinkEngine B0/B1完整测试</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#17212b;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white}th,td{padding:16px;border-bottom:1px solid #cbd5e1;text-align:left}.yes{color:#087443}.no{color:#b42318}details{background:white;padding:16px;margin:14px 0;border:1px solid #cbd5e1;border-radius:8px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{width:100%;height:auto}a{color:#175cd3}</style><h1>DeepLinkEngine B0/B1 完整测试</h1><p>MMLU 200 · HLE 分层200 · Frontier 160 · SWE 64请求 · Terminal 48请求</p><table><tr><th>指标</th><th>里程碑</th><th>项目最终目标</th></tr>'''+rows+'''</table><details><summary>冻结引擎与 MOD 组合</summary><pre>'''+esc(json.dumps(dict(release=data.get('engine_release'),mods=data.get('frozen_mods',[])),ensure_ascii=False,indent=2))+'''</pre></details><h2>固定五项配对结果</h2><img src="fixed-five.svg" alt="五数据集B0/B1吞吐和时延"><p>'''+esc(data['scoring'])+'''</p>'''+details+'''<details><summary>有效算力参考计算公式</summary><p>C本轮下界=ΣF本轮下界/(2×ΣT本轮×10¹²)；C参考上界=ΣF原参考上界/(2×ΣT原参考下界×10¹²)；η下界=100×C本轮下界/C参考上界。</p><p>'''+esc(data['reference_method'])+'''</p></details><details><summary>原始报告、采样与复算记录</summary><p><a href="../fixed-five-report/report.html">五数据集与参考报告</a> · <a href="../context-tenant/paired-report/report.html">上下文与四租户报告</a> · <a href="summary.json">本页判定与原件哈希</a></p><pre>'''+esc(json.dumps(data['context_audit'],ensure_ascii=False,indent=2))+'''</pre></details></html>'''
    (out/'report.html').write_text(document)
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();data=render(args.run,args.out)
    print('FULL_ACCEPTANCE_REPORT',args.out/'report.html',flush=True)
    return 0 if data['execution_complete'] else 2


if __name__=='__main__':raise SystemExit(main())
