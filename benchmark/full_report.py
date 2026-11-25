"""Build binary milestone/final decisions from independently audited run summaries."""
import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import html
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))

from engine.release_selection import kit_path, runtime_lock, profile as selected_profile

ACTUAL_COST_SCHEMA = 'deeplinkengine-v2-actual-card-cost/v1'
ACTUAL_COST_TEMPLATE = 'actual-cost.template.json'
PUBLISHED_RENTAL_EVIDENCE = 'evidence/indicator-4.4-cost-estimate.json'
COST_MONTH_DAYS = 30
MMLU_UNIT_COST_REDUCTION_MIN = 10
PERFORMANCE_LIMITS = {'milestone': ('1.10', '.95'), 'final': ('2', '.80')}
QUALITY_TRACKS = {'hle': {'gold': 74, 'revision': 126}, 'frontier': {'olympiad': 100, 'research': 60}}
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
    for name,denominators in QUALITY_TRACKS.items():
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


API_LOG = re.compile(r'^\(APIServer pid=(\d+)\) (?:INFO|ERROR) (\d\d-\d\d \d\d:\d\d:\d\d) ')


def api_log_epoch(value, reference, offset_minutes):
    zone = timezone(timedelta(minutes=offset_minutes))
    year = datetime.fromtimestamp(reference, zone).year
    return datetime.strptime(f'{year}-{value}', '%Y-%m-%d %H:%M:%S').replace(tzinfo=zone).timestamp()


def shutdown_clock_matches(triggers, traceback, first, measured_end, receipt):
    stopped = receipt['ended']
    measured_start, ready = receipt.get('started'), receipt.get('measurement_ready')
    # Old logs can use a host-local clock. Require the owned API's first log
    # and its shutdown pair to fit independent sealed wall-clock bounds.
    has_startup_anchor = (type(measured_start) in (int, float) and type(ready) in (int, float)
                          and measured_start <= ready <= measured_end
                          and 0 < stopped - measured_end < 15 * 60)
    offsets = range(-12 * 60, 14 * 60 + 1, 15) if has_startup_anchor else (0,)
    matches = 0
    for offset in offsets:
        try:
            trace_time = api_log_epoch(traceback, measured_end, offset)
            if not measured_end <= trace_time <= stopped:
                continue
            if offset and not measured_start <= api_log_epoch(first, measured_start, offset) <= ready:
                continue
            matches += sum(measured_end <= api_log_epoch(trigger, measured_end, offset) <= trace_time
                           for trigger in triggers)
        except ValueError:
            continue
    return matches == 1


def recorded_shutdown_traces(root, cleanup, lines):
    """Recognize the owned API drain traceback after the last measured window."""
    execution = root / 'service/acceptance/execution.json'
    receipt_path = root / 'receipt.json'
    if not (execution.is_file() and receipt_path.is_file() and cleanup.get('quiescent') is True):
        return set()
    measurement = json.loads(execution.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    phases = measurement.get('phases', [])
    if (measurement.get('status') != 'completed' or not phases
            or receipt.get('completed') is not True or receipt.get('quiescent') is not True):
        return set()
    end = phases[-1]['end']['wall']
    terminated = {str(action['pid']) for action in cleanup.get('actions', []) if action.get('signal') == 15}
    process_path = root / 'service-process.json'
    if receipt.get('role') == 'B1' and process_path.is_file():
        process = json.loads(process_path.read_bytes())
        for action in cleanup.get('actions', []):
            if (action.get('signal') == 15 and action['pid'] == process['pid']
                    and action.get('start_ticks') == process['start_ticks']):
                terminated.update(str(child['pid']) for child in action.get('managed_children', []))
    first_logs, triggers, notices = {}, {}, set()

    for index, line in enumerate(lines):
        event = API_LOG.match(line)
        if not event or event[1] not in terminated:
            continue
        pid, stamp = event.groups()
        first_logs.setdefault(pid, stamp)
        if '[shutdown] API server: shutdown triggered' in line:
            triggers.setdefault(pid, []).append(stamp)
        elif (index and pid in triggers and 'Traceback (most recent call last)' in line
              and 'AsyncLLM output_handler failed.' in lines[index-1]
              and shutdown_clock_matches(triggers[pid], stamp, first_logs[pid], end, receipt)):
            prefix = f'(APIServer pid={pid}) '
            following = [item for item in lines[index+1:index+16] if item.startswith(prefix)]
            if (lines[index-1].startswith(prefix)
                    and any('vllm.v1.engine.exceptions.EngineDeadError:' in item for item in following)
                    and any('[shutdown] API server: engine client stopped' in item for item in following)):
                notices.add(index)
    return notices


def completed_context_after_error(execution, pair):
    stages = execution.get('stages', [])
    return (execution.get('status') == 'failed' and pair.get('execution_complete') is True
            and [s.get('role') for s in stages] == ['B0', 'B1']
            and all(s.get('status') == 'completed' and type(s.get('returncode')) is int
                    and s['returncode'] == 0 for s in stages))



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
    logs, teardown = [], []
    for path in root.rglob('*.log'):
        name = path.relative_to(root).as_posix()
        logs.append(name)
        lines = path.read_text(errors='replace').splitlines()
        known = recorded_shutdown_traces(root, cleanup, lines) if name == 'service.log' else set()
        for index, line in enumerate(lines):
            if HAZARD.search(line):
                item = f'{name}:{index+1}: {line.strip()[:240]}'
                (teardown if index in known else errors).append(item)
    if 'service.log' not in logs:errors.append('缺少服务日志')
    return dict(clean=not errors,errors=errors,logs=logs,post_measurement_teardown=teardown)


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


def performance_ratios(arms):
    a, b = arms['B0'], arms['B1']
    return dict(throughput_ratio=ratio(b.get('output_tokens_per_second'), a.get('output_tokens_per_second')),
                ttft_ratio=ratio(b.get('mean_ttft_ms'), a.get('mean_ttft_ms')),
                tpot_ratio=ratio(b.get('mean_tpot_ms'), a.get('mean_tpot_ms')))


def performance_clears(values, throughput, latency):
    return (all(value is not None for value in values.values())
            and values['throughput_ratio'] >= Decimal(throughput)
            and values['ttft_ratio'] <= Decimal(latency)
            and values['tpot_ratio'] <= Decimal(latency))


def performance(pairs, throughput, latency):
    results = {}
    for name, pair in pairs.items():
        values = performance_ratios(pair['arms'])
        passed = pair['complete'] and pair['quality']['met'] and performance_clears(values, throughput, latency)
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
                percentile_method='nearest_rank',
                blocking_errors={r:audits.get(r,{}).get('errors',[]) for r in ('B0','B1')},
                measured_ratios={k:str(v) if v is not None else None for k,v in ratios.items()},
                measured_window_ratios={k:str(v) if v is not None else None for k,v in windows.items()})


def metered_cost_arm(seconds, tokens, utilization, seal):
    seconds = number(seconds)
    cards = len(utilization.get('per_card_percent', {}))
    if (seconds is None or seconds <= 0 or type(tokens) is not int or tokens <= 0 or cards != 2
            or not isinstance(seal, str) or not re.fullmatch(r'[0-9a-f]{64}', seal)):
        return None
    return dict(window_seconds=str(seconds), successful_output_tokens=tokens, occupied_cards=cards,
                measurement_seal_sha256=seal)


def read_actual_cost_manifest(path):
    raw = path.read_bytes()
    manifest = json.loads(raw, parse_float=Decimal)
    if (not isinstance(manifest, dict) or set(manifest) != {'schema', 'scenarios'}
            or manifest['schema'] != ACTUAL_COST_SCHEMA):
        raise ValueError('Invalid V2 actual cost manifest')
    scenarios = manifest['scenarios']
    if not isinstance(scenarios, dict) or not scenarios or set(scenarios) - {'mmlu', 'four_tenant'}:
        raise ValueError('Actual cost manifest must identify MMLU and/or four_tenant')
    for name, arms in scenarios.items():
        if not isinstance(arms, dict) or set(arms) != {'B0', 'B1'}:
            raise ValueError(f'{name}: actual cost evidence must cover both arms')
    return raw, scenarios


def actual_cost_inputs(path, measurements=None):
    if path is None:
        return None, {}
    path = Path(path)
    raw, scenarios = read_actual_cost_manifest(path)
    admitted = {}
    for name, arms in scenarios.items():
        admitted[name] = {}
        for role, arm in arms.items():
            label = f'{name}/{role}'
            if not isinstance(arm, dict) or set(arm) != {
                'measurement_seal_sha256', 'monthly_cost_per_card_cny', 'evidence',
            }:
                raise ValueError(f'{label}: invalid actual cost record')
            seal = arm['measurement_seal_sha256']
            if not isinstance(seal, str) or not re.fullmatch(r'[0-9a-f]{64}', seal):
                raise ValueError(f'{label}: invalid measurement seal in actual cost record')
            if measurements is not None:
                meter = measurements[name][role]
                if not meter or seal != meter['measurement_seal_sha256']:
                    raise ValueError(f'{label}: actual cost record does not match the sealed measurement')
            rate = number(arm['monthly_cost_per_card_cny'])
            if rate is None or rate <= 0:
                raise ValueError(f'{label}: actual monthly cost must be positive and finite')
            records = arm['evidence']
            if not isinstance(records, list) or not records:
                raise ValueError(f'{label}: actual cost requires supporting documents')
            digests = []
            for record in records:
                if (not isinstance(record, dict) or set(record) != {'file', 'sha256'}
                        or not isinstance(record['file'], str) or not record['file']):
                    raise ValueError(f'{label}: invalid expense evidence record')
                with (path.parent / record['file']).open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != record['sha256'] or digest in digests:
                    raise ValueError(f'{label}: altered or duplicate expense evidence')
                digests.append(digest)
            admitted[name][role] = dict(monthly_cost_per_card_cny=str(rate), expense_sha256=digests)
    return hashlib.sha256(raw).hexdigest(), admitted


def published_rental():
    raw = (ROOT / PUBLISHED_RENTAL_EVIDENCE).read_bytes()
    evidence = json.loads(raw)
    quote = evidence['quote']
    rate = Fraction(quote['server_monthly_price'], quote['cards_per_server'])
    return dict(quote, publisher='上海有色网（SMM）', currency=evidence['currency'],
                monthly_cost_per_card_cny=decimal_text(rate), days_per_month=COST_MONTH_DAYS,
                evidence_path=PUBLISHED_RENTAL_EVIDENCE, evidence_sha256=hashlib.sha256(raw).hexdigest())


def cost_per_million(arms, rates):
    if not all(arms.values()):
        return None
    return {
        role: Fraction(rates[role]) * arm['occupied_cards'] * Fraction(arm['window_seconds']) * 10**6
        / (COST_MONTH_DAYS * 86400 * arm['successful_output_tokens'])
        for role, arm in arms.items()
    }


def reduction(costs):
    return 100 * (1 - costs['B1'] / costs['B0']) if costs else None


def decimal_text(value):
    if isinstance(value, Fraction):
        value = Decimal(value.numerator) / value.denominator
    return str(value) if value is not None else None


def cost_decisions(fixed, pairs, context, audits, actual_cost):
    mmlu = fixed.get('current_mmlu_pair') or {}
    mmlu_arms = mmlu.get('arms', {})
    mmlu_util = mmlu.get('utilization', {})
    multi = next((r for r in (context or {}).get('comparison', []) if r['phase'] == 'multi-tenant-formal'), {})
    context_seals = (context or {}).get('arm_seals', {})
    multi_util = {role: multi.get(role, {}).get('utilization', {}) for role in ('B0', 'B1')}
    samples = {'mmlu': mmlu_util, 'four_tenant': multi_util}
    measurements = {'mmlu': {}, 'four_tenant': {}}
    for role in ('B0', 'B1'):
        arm = mmlu_arms.get(role, {})
        group = arm.get('groups', {}).get('joint200', {})
        measurements['mmlu'][role] = metered_cost_arm(
            group.get('cohort_wall_seconds'), group.get('successful_output_tokens'),
            mmlu_util.get(role, {}), arm.get('source_seal_sha256'))
        phase = multi.get(role, {})
        metrics = phase.get('metrics', {})
        measurements['four_tenant'][role] = metered_cost_arm(
            metrics.get('duration_seconds'), metrics.get('output_tokens'),
            phase.get('utilization', {}), context_seals.get(role))

    manifest_sha, expense = actual_cost_inputs(actual_cost, measurements)
    rental = published_rental() if actual_cost is None else None
    cost_basis = 'published_rental' if rental else 'actual_expense'
    formal, estimates, reductions = {}, {}, {}
    for name, arms in measurements.items():
        estimates[name] = dict(
            reduction_percent=decimal_text(reduction(cost_per_million(arms, {'B0': Decimal(1), 'B1': Decimal(1)}))))
        fees = ({role: dict(monthly_cost_per_card_cny=rental['monthly_cost_per_card_cny']) for role in arms}
                if rental else expense.get(name))
        costs = cost_per_million(arms, {r: Decimal(f['monthly_cost_per_card_cny']) for r, f in fees.items()}) if fees else None
        reductions[name] = reduction(costs)
        evidence_status = cost_basis if rental else 'verified'
        if not costs:
            evidence_status = 'missing_measurement' if rental else 'missing'
        formal[name] = dict(
            evidence_status=evidence_status,
            reduction_percent=decimal_text(reductions[name]),
            utilization=samples[name],
            arms={r: dict(meter, **fees[r], cost_per_million_output_tokens_cny=decimal_text(costs[r]))
                  if costs else meter for r, meter in arms.items()})

    def sampled(meters, samples):
        for role, meter in meters.items():
            util = samples.get(role, {})
            if not meter or not coverage(util) or number(util.get('mean_percent')) is None:
                return False
            if abs(number(util['covered_seconds']) - Decimal(meter['window_seconds'])) > Decimal('0.000001'):
                return False
        return True

    mmlu_ready = (pairs.get('mmlu', {}).get('complete') and pairs['mmlu']['quality']['met']
                  and sampled(measurements['mmlu'], mmlu_util))
    gain = (number(mmlu_util['B1']['mean_percent']) - number(mmlu_util['B0']['mean_percent'])
            if mmlu_ready else None)

    def complete_tenants(phase):
        if not phase_valid(phase) or (phase.get('cohort'), phase.get('tenants'), phase.get('concurrency')) != ('4096', 4, 24):
            return False
        tenants = phase.get('per_tenant', {})
        return (set(tenants) == {f'tenant-{i}' for i in range(4)}
                and sum(t['requests'] for t in tenants.values()) == phase['metrics']['requests']
                and all(t['requests'] > 0 and t['successful'] == t['requests'] and t['failed'] == 0
                        for t in tenants.values()))

    multi_ready = (sampled(measurements['four_tenant'], multi_util)
                   and all(audits.get(role, {}).get('clean') is True and complete_tenants(multi[role])
                           for role in ('B0', 'B1')))
    usage = number(multi_util['B1']['mean_percent']) if multi_ready else None
    cost = reductions['mmlu']
    multi_cost = reductions['four_tenant']

    def cost_gate(name, met, reason):
        status = formal[name]['evidence_status']
        if status == 'missing':
            reason = '缺少实际费用凭据。' + reason
        elif status == 'missing_measurement':
            reason = '缺少可核验的正式计量。' + reason
        return dict(decision(met, reason), evidence_status=status)

    unit_cost = '按公开月租折算的单位成本' if rental else '实际单位成本'
    return dict(
        milestone=cost_gate('mmlu', cost is not None and gain is not None and cost >= MMLU_UNIT_COST_REDUCTION_MIN and gain >= 10,
                            f'MMLU200 完整回答满足质量和服务要求，{unit_cost}下降≥10%，双卡 AI Core 提高≥10个百分点。'),
        final=cost_gate('four_tenant', multi_cost is not None and usage is not None and multi_cost >= 30 and usage >= 80,
                        f'4K 四租户{unit_cost}下降≥30%，B1 双卡 AI Core 均值≥80%，两臂各租户完整结束。'),
        utilization_gain_pp=decimal_text(gain),
        four_tenant_ai_core_mean_percent=decimal_text(usage),
        unit_cost_reduction_percent=decimal_text(cost),
        four_tenant_cost_reduction_percent=decimal_text(multi_cost),
        cost_basis=cost_basis,
        published_rental=rental,
        actual_cost_manifest_sha256=manifest_sha,
        formal_cost=formal,
        same_rate_estimate=estimates,
        formula=f'成本/百万输出词元=10^6×每卡月费×占卡数×正式区间秒数/({COST_MONTH_DAYS}×86400×成功输出词元数)。',
    )



def targets(fixed, context, audits, actual_cost=None, *, refused_reference=None):
    from benchmark.reference_admission import MEASURED, assess

    fixed = fixed or {}
    ref = refused_reference or fixed.get('current_compute_reference') or {}
    sys.path.insert(0, str(kit_path()/'scripts'))
    result = assess(ref)
    value = (number(ref.get('arms', {}).get('B1', {}).get('efficiency_lower_percent'))
             if result['status'] == MEASURED else None)
    missing = '有效算力证据未通过核验：' + result['reason'] if value is None else None
    compute = dict(
        evidence_reason=result['reason'],
        milestone=decision(value is not None and value>=50,
                           missing or '本轮必要算量、同步计时和已核验同工况参考分母，效率下界≥50%。'),
        final=decision(value is not None and value>=90, missing or '同一效率下界≥90%。'),
        efficiency_lower_percent=str(value) if value is not None else None,
        certificate_kind=ref.get('certificate_kind'),
    )
    if missing:
        for level in ('milestone', 'final'):
            compute[level]['evidence_status'] = 'missing_measurement'
    pairs=workload_pairs(fixed)
    milestone, details = performance(pairs, *PERFORMANCE_LIMITS['milestone'])
    final, final_details = performance(pairs, *PERFORMANCE_LIMITS['final'])
    speed=dict(milestone=milestone,final=final,datasets=details,final_datasets=final_details)
    costs = cost_decisions(fixed, pairs, context, audits, actual_cost)
    return {'4.1':compute,'4.2':speed,'4.3':context_decisions(context, audits),'4.4':costs}


LABELS={'4.1':'有效算力','4.2':'吞吐与时延','4.3':'上下文稳定性','4.4':'产能与成本'}


def context_sources(folder, stored):
    """Resolve original context arms from their completed execution bindings."""
    folder = Path(folder).resolve()
    execution = json.loads((folder / 'execution.json').read_text())
    rows = execution.get('stages', [])
    if execution.get('status') != 'completed' and not completed_context_after_error(execution, stored):
        raise ValueError('Context execution failed before the plotting boundary')
    if (execution.get('schema') != 'vllm-hust-acceptance-supplement-execution/v1'
            or stored.get('schema') != 'vllm-hust-paired-context-tenant-supplement/v1'
            or len(rows) != 2 or [row.get('role') for row in rows] != ['B0', 'B1']
            or set(stored.get('arms', {})) != {'B0', 'B1'}
            or set(stored.get('arm_seals', {})) != {'B0', 'B1'}):
        raise ValueError('Context execution does not bind exactly one B0 and B1 arm')
    result = {}
    for row in rows:
        role = row['role']
        name = row.get('run')
        if (row.get('status') != 'completed' or type(row.get('returncode')) is not int
                or row['returncode'] != 0 or type(row.get('reused')) is not bool
                or not isinstance(name, str) or not name or not Path(name).is_absolute()
                or stored['arms'][role] != name):
            raise ValueError('Context arm execution or paired source differs: ' + role)
        path = Path(name)
        if not row['reused'] and path.resolve() != (folder / role).resolve():
            raise ValueError('A fresh context arm points outside its owned output: ' + role)
        seal = hashlib.sha256((path / 'checksums.json').read_bytes()).hexdigest()
        if row.get('seal_sha256') != seal or stored['arm_seals'][role] != seal:
            raise ValueError('Context arm seal differs from its execution binding: ' + role)
        receipt = json.loads((path / 'receipt.json').read_text())
        if (receipt.get('role') != role or receipt.get('completed') is not True
                or receipt.get('quiescent') is not True):
            raise ValueError('Context arm is not completed and released: ' + role)
        result[role] = path
    # paired() still independently replays sealed inputs/SSE, checks weights,
    # devices and the measured B1 plan's exact admitted B0 path.
    return result


def v2_latency_tails(metrics, observations):
    for column, metric in ((1, 'ttft_ms'), (2, 'tpot_ms')):
        values = sorted(row[column] for row in observations if row[column] is not None)
        for percentile in (95, 99):
            rank = (percentile * len(values) + 99) // 100
            metrics[metric][f'p{percentile}'] = values[rank - 1] if values else None


def apply_v2_context_tails(context, arms):
    """Project nearest-rank tails from the same audited request clocks."""
    pairs = {row['phase']: row for row in context['comparison']}
    for role, root in arms.items():
        raw_seal = (root / 'checksums.json').read_bytes()
        if hashlib.sha256(raw_seal).hexdigest() != context['arm_seals'][role]:
            raise ValueError('Context source seal changed before V2 percentile projection')
        seal = json.loads(raw_seal)

        def read(relative):
            raw = (root / relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != seal.get(relative):
                raise ValueError('Context request or clock is missing from the audited seal: ' + relative)
            return json.loads(raw)

        folder = 'service/acceptance/'
        for phase in read(folder + 'execution.json')['phases']:
            if phase['name'] not in pairs:
                continue
            observations = []
            for request in phase['request_files']:
                row = read(folder + request)
                if not row['success']:
                    continue
                start = row['start']['monotonic']
                first = row['first_token']['monotonic']
                count = row['usage']['completion_tokens']
                ttft = 1000 * (first - start)
                tpot = 1000 * (row['end']['monotonic'] - first) / (count - 1) if count > 1 else None
                observations.append((start, ttft, tpot, row['tenant']))
            arm = pairs[phase['name']][role]
            v2_latency_tails(arm['metrics'], observations)
            for tenant, metrics in arm['per_tenant'].items():
                v2_latency_tails(metrics, [row for row in observations if row[3] == tenant])
            phase_start = phase['start']['monotonic']
            for index, window in enumerate(arm['windows']):
                low = phase_start + 60 * index
                high = min(low + 60, phase['submission_deadline_monotonic'])
                v2_latency_tails(window['metrics'], [row for row in observations if low <= row[0] < high])



def workload_sources(fixed, context, b1):
    plan = fixed['plan']
    prepared = plan.get('prepared_datasets', {})
    mmlu = plan.get('mmlu_inputs', {})
    profiles = {
        'mmlu': ('MMLU-Pro', 'TIGER-Lab/MMLU-Pro test 固定 200 题'),
        'hle': ('HLE-Verified', 'SKYLENAGE-AI/HLE-Verified 固定 200 题，Gold 74、Revision 126'),
        'frontier': ('FrontierScience', 'openai/frontierscience 固定 160 题，Olympiad 100、Research 60'),
        'swe': ('SWE-bench Pro', '公开轨迹 32 项任务，64 条请求'),
        'terminal': ('Terminal-Bench 2.1', '公开轨迹 32 项任务，48 条请求'),
    }
    fixed_five = {
        key: dict(dataset=label, sample=sample, requests_per_arm=COUNTS[key],
                  scope_or_contract_sha256=(mmlu.get('scope_file_sha256') if key == 'mmlu'
                                else prepared.get(key, {}).get('contract_sha256')))
        for key, (label, sample) in profiles.items()
    }
    contract = context['contract']
    raw = (b1 / 'acceptance-inputs/acceptance-inputs.json').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != contract['input_sha256']:
        raise ValueError('Context data sources differ from the audited input digest')
    inputs = json.loads(raw)
    trace_names = {'swe': 'SWE-bench Pro', 'terminal': 'Terminal-Bench 2.1'}
    bands = {str(row['min_tokens']): row['max_tokens_exclusive'] for row in contract['cohorts']}
    cohorts = {}
    for key, rows in inputs.items():
        cases = []
        for row in rows:
            source, separator, name = row['task_id'].partition(':')
            if not separator or not name or source not in trace_names:
                raise ValueError('Unknown context source dataset')
            cases.append(dict(dataset=trace_names[source], **{
                field: row[field] for field in ('task_id', 'source_index', 'input_tokens',
                                                'source_sha256', 'source_body_sha256', 'token_ids_sha256')
            }))
        cohorts[key] = dict(datasets=list(dict.fromkeys(case['dataset'] for case in cases)),
                            input_tokens=dict(minimum=int(key), maximum_exclusive=bands[key]), cases=cases)
    phases = {row['id'] + '-formal': {
        field: row[field] for field in ('cohort', 'tenants', 'concurrency', 'warmup_seconds', 'formal_seconds')
    } for row in contract['phases']}
    capture = (fixed.get('current_compute_reference') or {}).get('capture_profile') or {}
    return dict(fixed_five=fixed_five,
                effective_compute=dict(dataset='MMLU-Pro', workload='mmlu', capture_profile=capture),
                cost=dict(milestone_workload='mmlu', final_phase='multi-tenant-formal'),
                continuous=dict(input_kind='synthetic', dataset=None, workload='4K/16K 连续合成 C8'),
                context=dict(input_sha256=digest, source_protocol_sha256=contract['source_protocol_sha256'],
                             percentile_method='nearest_rank', output_tokens=contract['output_tokens'],
                             cohorts=cohorts, phases=phases))


def workload_source_table(sources):
    esc = html.escape
    rows = [(value['dataset'], value['sample']) for value in sources['fixed_five'].values()]
    profile = sources['effective_compute']['capture_profile']
    native = 'MMLU-Pro 固定 200 题'
    if profile.get('max_tokens') and profile.get('formal_repetitions'):
        native += f"；最多输出 {profile['max_tokens']} 词元，每臂 {profile['formal_repetitions']} 轮"
    rows.append(('4.1 原生 F/T', native))
    rows.append(('连续 C8（4K／16K）', '合成输入'))
    context = sources['context']
    for name, phase in context['phases'].items():
        cohort = context['cohorts'][phase['cohort']]
        size = cohort['input_tokens']
        label = '四租户 C24' if name == 'multi-tenant-formal' else f"{size['minimum'] // 1024}K 长上下文"
        cases = '、'.join(case['task_id'] for case in cohort['cases'])
        rows.append((label, f"{'、'.join(cohort['datasets'])}；[{size['minimum']}, {size['maximum_exclusive']}) "
                    f"输入词元；{len(cohort['cases'])} 条公开任务历史；{cases}"))
    return ('<h2>测试数据集与冻结输入</h2><table><tr><th>工况</th><th>数据来源与样本</th></tr>'
            + ''.join(f'<tr><th>{esc(label)}</th><td>{esc(sample)}</td></tr>' for label, sample in rows)
            + '</table><p>4.4 里程碑使用上表 MMLU-Pro 完整回答工况，结题使用上述四租户的 4K 历史。</p>')



def measured_stack(directory, release, *, selection=None):
    from benchmark import evaluation_client

    matches = []
    for path in sorted(directory.glob('active-stack-*.json')):
        raw = path.read_bytes()
        stack = json.loads(raw)
        if stack.get('release') == release:
            matches.append((raw, stack))
    if (not matches and selection
            and release == selection.get('release') == evaluation_client.R10_3072_REFERENCE_RELEASE):
        client = selection.get('client')
        catalog = evaluation_client.R10_3072_STACK
        if client and not (directory / Path(catalog).name).exists():
            manifest = evaluation_client.verify(client)
            if manifest.get('release') == release and catalog not in manifest['files']:
                root = Path(client['path'])
                files = {name: (root / name).read_bytes() for name in manifest['files']}
                evaluation_client.r10_3072_report_catalog(files, selection['frozen'])
                raw = files[catalog]
                return raw, json.loads(raw)
    if len(matches) != 1:
        raise ValueError('Expected exactly one MOD catalog for the measured release: ' + release)
    return matches[0]


def collect(root, actual_cost=None):
    from engine.release_selection import bind
    saved = Path(root) / 'engine-selection.json'
    selection = bind(saved) if saved.is_file() else None
    """Re-audit sealed originals; generated summaries alone never authorize a pass."""
    root=Path(root).resolve()
    from engine.runtime_storage import restore_missing
    restore_missing(root/'installation')
    sys.path.insert(0,str(kit_path()/'scripts'))
    from third_party_report import inspect
    from acceptance_suite import paired
    fixed=inspect(root/'fixed-five')
    if fixed['plan']['datasets']!=list(DATASETS):
        raise ValueError('Full report requires all five frozen datasets')
    stored=json.loads((root/'context-tenant/paired-summary.json').read_text())
    context_receipt = json.loads((root/'context-tenant/execution.json').read_bytes())
    arms=context_sources(root/'context-tenant',stored)
    identity=json.loads((arms['B1']/'source-admission.json').read_text())
    if identity.get('release')!=fixed['plan']['engine_release']:
        raise ValueError('Context B1 release differs from the fixed-five candidate')
    mmlu_root=Path(fixed['current_mmlu_pair']['arms']['B1']['run'])
    def weights(path):
        value=json.loads((path/'model-weights.json').read_text())
        return {name:{key:row[key] for key in ('bytes','sha256')} for name,row in value.items()}
    if weights(arms['B1'])!=weights(mmlu_root):
        raise ValueError('Context model weight bytes differ from the fixed-five candidate')
    device_change = stored.get('device_change')
    if device_change:
        execution = json.loads((root/'context-tenant/execution.json').read_text())
        if not fixed['plan'].get('device_reuse_mapping') or execution.get('device_change') != device_change:
            raise ValueError('Context device change has no matching execution binding')
        context = paired(arms, device_change=device_change)
    else:
        context = paired(arms)
    if stored!=context:
        raise ValueError('Context summary differs from independently re-audited originals')
    apply_v2_context_tails(context, arms)
    stack_raw, stack = measured_stack(kit_path() / 'mods', fixed['plan']['engine_release'],
                                      selection=selection)
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
    audits={r:audit_context(arms[r]) for r in ('B0','B1')}
    for row in context['comparison']:
        for r in ('B0','B1'):
            if not coverage(row[r].get('utilization',{})):
                audits[r]['errors'].append(row['phase']+': 利用率采样未覆盖完整窗口')
                audits[r]['clean']=False
    hashes={}
    sources=[root/name for name in ('fixed-five/campaign-checksums.json',
        'fixed-five/frozen-execution-plan.json','context-tenant/execution.json',
        'context-tenant/paired-summary.json')]
    sources.extend(arms[r]/'checksums.json' for r in ('B0','B1'))
    for path in sources:
        name=str(path.resolve().relative_to(root)) if path.resolve().is_relative_to(root) else str(path)
        with path.open('rb') as stream:hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    refused_reference = None
    receipt = root / 'results.json'
    if receipt.is_file():
        preflight = json.loads(receipt.read_bytes()).get('reference_preflight')
        if preflight:
            from benchmark.reference_admission import archived_refusal

            refused_reference = archived_refusal(root, preflight)
            hashes[preflight['path']] = preflight['sha256']
    return dict(schema='deeplinkengine-full-acceptance-report/v1',run=str(root),
                execution_complete=fixed['receipt'].get('selected_execution_complete') is True and context['execution_complete'],
                engine_release=fixed['plan']['engine_release'],
                frozen_mods=[{k:m[k] for k in ('id','name','version')} for m in stack['enabled_mods']],
                mod_catalog_sha256=hashlib.sha256(stack_raw).hexdigest(),
                targets=targets(fixed,context,audits,actual_cost,refused_reference=refused_reference),workloads=workload_pairs(fixed),
                workload_sources=workload_sources(fixed, context, arms['B1']),
                **({'execution_devices': {
                    'fixed_five': fixed['execution_devices'],
                    'context': context.get('device_change'),
                    'native': (fixed.get('current_compute_reference') or {}).get('native_device_change'),
                }} if fixed.get('execution_devices') else {}),
                context_audit=audits, context_results=context['comparison'],
                context_receipt=dict(status=context_receipt['status'], failure=context_receipt.get('failure')),
                context_arm_sources={r:str(arms[r]) for r in ('B0','B1')},evidence_sha256=hashes,
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


def cost_template(cost):
    scenarios = {}
    for name, row in cost['formal_cost'].items():
        arms = row['arms']
        if row['evidence_status'] != 'missing' or not all(arms.values()):
            continue
        if not all(re.fullmatch(r'[0-9a-f]{64}', arm.get('measurement_seal_sha256') or '') for arm in arms.values()):
            continue
        scenarios[name] = {
            role: dict(measurement_seal_sha256=arm['measurement_seal_sha256'],
                       monthly_cost_per_card_cny=None, evidence=[dict(file=None, sha256=None)])
            for role, arm in arms.items()
        }
    return dict(schema=ACTUAL_COST_SCHEMA, scenarios=scenarios) if scenarios else None


def render(root, out, actual_cost=None):
    data=collect(root, actual_cost);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    charts(data,out)
    cost = data['targets']['4.4']
    template = cost_template(cost)
    template_link = ''
    if template is not None:
        data['actual_cost_template'] = ACTUAL_COST_TEMPLATE
        (out / ACTUAL_COST_TEMPLATE).write_text(json.dumps(template, indent=2) + '\n', encoding='utf-8')
        template_link = ('<p><a href="' + ACTUAL_COST_TEMPLATE + '">下载按测量封存填好的费用凭据模板</a>。'
                         '填写实际费率与费用文件后，可沿原封存复算。</p>')
    (out/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    esc=html.escape
    def goal_cell(goal):
        status = {'missing': '待实际费用凭据', 'missing_measurement': '测量证据不完整'}.get(
            goal.get('evidence_status'), goal['status'])
        style = 'yes' if goal['met'] else 'no'
        return f'<td class="{style}">{esc(status)}</td>'

    rows = ''.join(f'<tr><th>{key} {LABELS[key]}</th>{goal_cell(value["milestone"])}'
                   f'{goal_cell(value["final"])}</tr>' for key, value in data['targets'].items())
    details = ''.join(
        f'<details><summary>{key} {LABELS[key]}：计算与判定详情</summary><pre>'
        + esc(json.dumps({k: v for k, v in value.items() if k != 'same_rate_estimate'}, ensure_ascii=False, indent=2))
        + '</pre></details>' for key, value in data['targets'].items())
    input_sources = workload_source_table(data['workload_sources']) if 'workload_sources' in data else ''
    context_details = ''
    if 'context_results' in data:
        context_details = '<details id="context-results"><summary>长上下文与四租户复算明细</summary><pre>' + esc(
            json.dumps(dict(receipt=data['context_receipt'], results=data['context_results']),
                       ensure_ascii=False, indent=2)) + '</pre></details>'
    rental = cost['published_rental']
    if rental:
        quote_url = esc(rental['source_url'], quote=True)
        rate = Decimal(rental['monthly_cost_per_card_cny'])
        cost_section = (
            f'<p>4.4 按公开租赁报价折算。{esc(rental["model"])} {rental["cards_per_server"]} 卡服务器'
            f'月租 {rental["server_monthly_price"]:,} 元。B0 与 B1 均按 {rate:,} 元/卡月、'
            f'每月 {rental["days_per_month"]} 天核算。'
            f'<a href="{quote_url}">{esc(rental["publisher"])} {esc(rental["published_date"])} 报价</a></p>'
        )
    else:
        cost_section = '<details><summary>4.4 同费率估算（参考）</summary><pre>' + esc(
            json.dumps(cost['same_rate_estimate'], ensure_ascii=False, indent=2)) + '</pre></details>'
    reference_goal = data['targets'].get('4.1', {})
    reference_note = ''
    if reference_goal.get('milestone', {}).get('evidence_status') == 'missing_measurement':
        reference_note = '<p><strong>4.1 测量证据不完整。</strong> ' + esc(reference_goal['evidence_reason']) + '</p>'
    fixed_report = '<a href="#fixed-five-results">固定五项配对结果</a>'
    if (Path(root) / 'fixed-five-report/report.html').is_file():
        fixed_report = '<a href="../fixed-five-report/report.html">五数据集与参考报告</a>'
    device_note = ''
    if data.get('execution_devices'):
        device_note = '<p>跨卡比较保留各自的物理设备。差值包含设备更换的影响。</p><details><summary>原始与当前设备映射</summary><pre>' + esc(json.dumps(data['execution_devices'], ensure_ascii=False, indent=2)) + '</pre></details>'
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeepLinkEngine B0/B1完整测试</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#17212b;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white}th,td{padding:16px;border-bottom:1px solid #cbd5e1;text-align:left}.yes{color:#087443}.no{color:#b42318}details{background:white;padding:16px;margin:14px 0;border:1px solid #cbd5e1;border-radius:8px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{width:100%;height:auto}a{color:#175cd3}</style><h1>DeepLinkEngine B0/B1 完整测试</h1><p>MMLU 200 · HLE 分层200 · Frontier 160 · SWE 64请求 · Terminal 48请求</p>'''+device_note+'''<table><tr><th>指标</th><th>里程碑</th><th>项目最终目标</th></tr>'''+rows+'''</table>'''+reference_note+cost_section+input_sources+'''<details><summary>冻结引擎与 MOD 组合</summary><pre>'''+esc(json.dumps(dict(release=data.get('engine_release'),mods=data.get('frozen_mods',[])),ensure_ascii=False,indent=2))+'''</pre></details><h2 id="fixed-five-results">固定五项配对结果</h2><img src="fixed-five.svg" alt="五数据集B0/B1吞吐和时延"><p>'''+esc(data['scoring'])+'''</p>'''+details+context_details+template_link+'''<details><summary>有效算力参考计算公式</summary><p>C本轮下界=ΣF本轮下界/(2×ΣT本轮×10¹²)；C参考上界=ΣF原参考上界/(2×ΣT原参考下界×10¹²)；η下界=100×C本轮下界/C参考上界。</p><p>'''+esc(data['reference_method'])+'''</p></details><details><summary>原始报告、采样与复算记录</summary><p>'''+fixed_report+''' · <a href="#context-results">长上下文与四租户复算明细</a> · <a href="summary.json">本页判定与原件哈希</a></p><pre>'''+esc(json.dumps(data['context_audit'],ensure_ascii=False,indent=2))+'''</pre></details></html>'''
    (out/'report.html').write_text(document, encoding='utf-8')
    return data


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', type=Path, help='Completed local submission run')
    source.add_argument('--engine-version', help='Archived candidate version or release for CPU replay')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--actual-cost', type=Path, help='V2 actual card cost manifest and expense evidence')
    args = parser.parse_args(arguments)
    try:
        if args.engine_version is not None:
            if args.actual_cost:
                raise ValueError('Archived candidate reporting does not take a V2 cost manifest')
            from benchmark.candidate_report import render as render_candidate

            render_candidate(args.engine_version, args.out)
            print('CANDIDATE_CPU_REPORT', args.out / 'report.html', flush=True)
            return 0
        data = render(args.run, args.out, args.actual_cost)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print('FULL_ACCEPTANCE_REPORT',args.out/'report.html',flush=True)
    return 0 if data['execution_complete'] else 2


if __name__=='__main__':raise SystemExit(main())
