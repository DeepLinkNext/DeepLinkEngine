"""Acceptance regressions: quality denominators, log failures and numeric gates."""
import copy
import hashlib
import importlib.util
from unittest.mock import patch
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import full_report as report


def stats(n,good):return dict(total=n,successes=good)


def fixed():
    arms={r:dict(recorded_requests=200,successful_requests=200,failed_requests=0,
                 quality=dict(planned_denominator=200,recorded=200,unreturned=0,correct=150),
                 groups=dict(joint200=dict(output_tokens_per_second=100 if r=='B0' else 220,
                            mean_ttft_ms=100 if r=='B0' else 75,mean_tpot_ms=50 if r=='B0' else 35))) for r in ('B0','B1')}
    quality={}
    for r in ('B0','B1'):
        quality['hle-'+r]=dict(tracks=dict(gold=stats(74,50),revision=stats(126,80)))
        quality['frontier-'+r]=dict(tracks=dict(olympiad=stats(100,60),research=stats(60,40)))
    return dict(current_mmlu_pair=dict(arms=arms,gains=dict(equal_rate_unit_cost_reduction_pct=40,ai_core_utilization_gain_pp=12),
                utilization={r:dict(missing_seconds=0,covered_seconds=100) for r in ('B0','B1')}),current_dataset_quality=quality)


def phase():
    return dict(status='completed',metrics=dict(requests=10,successful=10,failed=0,duration_seconds=300,
        output_tokens_per_second=100,ttft_ms=dict(p95=100,p99=110),tpot_ms=dict(p95=20,p99=22)))


def context():
    rows=[]
    for name in ('4096-formal','16384-formal','32768-formal','multi-tenant-formal'):
        a,b=phase(),phase()
        if name=='32768-formal':b['windows']=[dict(metrics=phase()['metrics']) for _ in range(5)]
        rows.append(dict(phase=name,B0=a,B1=b))
    return dict(comparison=rows)


class Decisions(unittest.TestCase):
    def setUp(self):self.fixed=fixed();self.context=context();self.audit={r:dict(clean=True) for r in ('B0','B1')}

    def test_one_complete_dataset_can_meet_both_speed_targets(self):
        d=report.targets(self.fixed,self.context,self.audit)
        self.assertTrue(d['4.2']['milestone']['met']);self.assertTrue(d['4.2']['final']['met'])

    def test_faster_generation_with_quality_loss_cannot_pass(self):
        self.fixed['current_mmlu_pair']['arms']['B1']['quality']['correct']=139
        d=report.targets(self.fixed,self.context,self.audit)
        self.assertFalse(d['4.2']['milestone']['met']);self.assertFalse(d['4.4']['milestone']['met'])

    def test_exact_five_percentage_points_is_allowed_without_rounding(self):
        self.fixed['current_mmlu_pair']['arms']['B1']['quality']['correct']=140
        self.assertTrue(report.quality_gates(self.fixed)['mmlu']['met'])

    def test_hle_compares_all_200_cases_rather_than_each_small_stratum(self):
        self.fixed['current_dataset_quality']['hle-B1']['tracks']['gold']['successes']-=10
        self.fixed['current_dataset_quality']['hle-B1']['tracks']['revision']['successes']+=10
        self.assertTrue(report.quality_gates(self.fixed)['hle']['met'])
        self.fixed['current_dataset_quality']['hle-B1']['tracks']['revision']['total']=116
        self.assertFalse(report.quality_gates(self.fixed)['hle']['met'])

    def test_frontier_research_cannot_hide_behind_olympiad_improvement(self):
        q=self.fixed['current_dataset_quality']['frontier-B1']['tracks']
        q['olympiad']['successes']+=10;q['research']['successes']-=4
        self.assertFalse(report.quality_gates(self.fixed)['frontier']['met'])

    def test_context_numeric_pass_is_rejected_if_original_logs_fail(self):
        self.assertTrue(report.targets(self.fixed,self.context,self.audit)['4.3']['milestone']['met'])
        self.audit['B1']['clean']=False
        d=report.targets(self.fixed,self.context,self.audit)['4.3']
        self.assertFalse(d['milestone']['met']);self.assertFalse(d['final']['met'])

    def test_all_five_32k_windows_are_required(self):
        self.context['comparison'][2]['B1']['windows'].pop()
        d=report.targets(self.fixed,self.context,self.audit)['4.3']
        self.assertTrue(d['milestone']['met']);self.assertFalse(d['final']['met'])

    def test_context_tail_limit_is_inclusive_and_uses_unrounded_value(self):
        self.context['comparison'][1]['B1']['metrics']['ttft_ms']['p99']=330
        self.assertTrue(report.targets(self.fixed,self.context,self.audit)['4.3']['milestone']['met'])
        self.context['comparison'][1]['B1']['metrics']['ttft_ms']['p99']=330.001
        self.assertFalse(report.targets(self.fixed,self.context,self.audit)['4.3']['milestone']['met'])

    def test_old_efficiency_number_cannot_pass_without_current_reference_identity(self):
        self.fixed['current_compute_reference']=dict(status='measured',efficiency_lower_observed_percent=99,
            certificate_kind='partial_dense_projection_diagnostic',full_model_certificate=True)
        d=report.targets(self.fixed,self.context,self.audit)['4.1']
        self.assertFalse(d['milestone']['met']);self.assertIsNone(d['efficiency_lower_percent'])

    def test_published_certificate_bound_is_not_rounded_up_to_ninety(self):
        path=report.ROOT/'benchmark/reports/r8-compute-reference-20261119-v1/summary.json'
        self.fixed['current_compute_reference']=json.loads(path.read_text())['compute_reference']
        goal=report.targets(self.fixed,self.context,self.audit)['4.1']
        self.assertTrue(goal['milestone']['met']);self.assertFalse(goal['final']['met'])
        self.assertEqual(goal['efficiency_lower_percent'],'89.78382533250482332421774762109257529198')

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'),'Run with the pinned report interpreter for real plot rendering')
    def test_synthetic_render_has_all_five_pairs_and_four_binary_rows(self):
        pairs=report.workload_pairs(self.fixed)
        for name in report.DATASETS:pairs[name]=copy.deepcopy(pairs['mmlu'])
        data=dict(schema='deeplinkengine-full-acceptance-report/v1',run='synthetic-schema-test',execution_complete=True,
                  targets=report.targets(self.fixed,self.context,self.audit),workloads=pairs,context_audit=self.audit,
                  scoring='synthetic schema validation',reference_method='synthetic schema validation')
        with tempfile.TemporaryDirectory() as tmp,patch.object(report,'collect',return_value=data):
            out=Path(tmp)/'report';report.render(Path(tmp),out)
            document=(out/'report.html').read_text(encoding='utf-8')
            for key in report.LABELS:self.assertIn(key+' '+report.LABELS[key],document)
            self.assertNotIn('pending',document);self.assertNotIn('复核中',document)
            for name in ('fixed-five.svg','fixed-five.png'):self.assertGreater((out/name).stat().st_size,1000)
            self.assertEqual(json.loads((out/'summary.json').read_text(encoding='utf-8'))['targets'],data['targets'])

    def test_log_write_failure_and_new_oom_are_detected_from_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'memory-initial.json').write_text(json.dumps(dict(events=dict(oom=0))))
            (p/'memory-samples.jsonl').write_text(json.dumps(dict(events=dict(oom=0)))+'\n')
            (p/'cleanup.json').write_text(json.dumps(dict(quiescent=True)))
            (p/'service.log').write_text('Healthy server\n')
            self.assertTrue(report.audit_context(p)['clean'])
            (p/'service.log').write_text('OSError: [Errno 28] No space left on device\n')
            self.assertFalse(report.audit_context(p)['clean'])
            (p/'service.log').write_text('Healthy server\n');(p/'memory-samples.jsonl').write_text(json.dumps(dict(events=dict(oom=1)))+'\n')
            self.assertFalse(report.audit_context(p)['clean'])


    def test_only_owned_shutdown_trace_after_measurement_is_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'service/acceptance'
            folder.mkdir(parents=True)
            end = report.datetime(2026, 11, 21, 15, 8, 28, tzinfo=report.timezone.utc).timestamp()
            (folder/'execution.json').write_text(json.dumps(dict(status='completed', phases=[dict(end=dict(wall=end))])))
            (root/'receipt.json').write_text(json.dumps(dict(completed=True, quiescent=True, ended=end+20)))
            (root/'cleanup.json').write_text(json.dumps(dict(quiescent=True, actions=[dict(pid=37, signal=15)])))
            (root/'memory-initial.json').write_text('{"events":{"oom":0}}')
            (root/'memory-samples.jsonl').write_text('{"events":{"oom":0}}\n')
            lines = [
                '(APIServer pid=37) INFO 11-21 15:08:30 [launcher.py:129] [shutdown] API server: shutdown triggered',
                '(APIServer pid=37) ERROR 11-21 15:08:45 [async_llm.py:829] AsyncLLM output_handler failed.',
                '(APIServer pid=37) ERROR 11-21 15:08:45 [async_llm.py:829] Traceback (most recent call last):',
                '(APIServer pid=37) ERROR 11-21 15:08:45 [async_llm.py:829] vllm.v1.engine.exceptions.EngineDeadError: E',
                '(APIServer pid=37) INFO 11-21 15:08:45 [launcher.py:154] [shutdown] API server: engine client stopped',
            ]
            log = root / 'service.log'
            log.write_text('\n'.join(lines))
            original = report.audit_context(root)
            self.assertTrue(original['clean'])
            self.assertEqual(len(original['post_measurement_teardown']), 1)
            log.write_text('\n'.join(line.replace('15:08:45','15:08:27') for line in lines))
            self.assertFalse(report.audit_context(root)['clean'])
            log.write_text('\n'.join(lines+['No space left on device']))
            self.assertFalse(report.audit_context(root)['clean'])


class CostTests(unittest.TestCase):
    def setUp(self):
        self.fixed = fixed()
        self.context = context()
        self.audit = {role: dict(clean=True) for role in ('B0', 'B1')}
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.doc = self.path / 'private-billing.txt'
        self.doc.write_bytes(b'synthetic billing evidence for regression tests')
        self.digest = hashlib.sha256(self.doc.read_bytes()).hexdigest()
        self.manifest = self.path / 'cost.json'
        self.seals = dict(mmlu=dict(B0='0'*64, B1='1'*64), four_tenant=dict(B0='2'*64, B1='3'*64))
        self.context['arm_seals'] = self.seals['four_tenant']
        multi = self.context['comparison'][-1]
        for role, total, ai_core in [('B0', 10000, 33), ('B1', 20000, 85)]:
            mmlu = self.fixed['current_mmlu_pair']
            arm = mmlu['arms'][role]
            arm['source_seal_sha256'] = self.seals['mmlu'][role]
            arm['groups']['joint200'].update(cohort_wall_seconds=100, successful_output_tokens=total)
            mmlu['utilization'][role] = self.sample(100, ai_core)
            phase = multi[role]
            phase.update(cohort='4096', tenants=4, concurrency=24, utilization=self.sample(300, ai_core),
                         per_tenant={f'tenant-{i}': dict(requests=count, successful=count, failed=0)
                                     for i, count in enumerate((2, 2, 3, 3))})
            phase['metrics']['output_tokens'] = total

    def sample(self, window, ai_core):
        return dict(covered_seconds=window, missing_seconds=0, mean_percent=ai_core,
                    per_card_percent={'1': ai_core, '6': ai_core})

    def expense(self, mmlu='1500', four_tenant='1200'):
        scenarios = {}
        for name, rate in [('mmlu', mmlu), ('four_tenant', four_tenant)]:
            scenarios[name] = {
                role: dict(measurement_seal_sha256=seal,
                           monthly_cost_per_card_cny='1000' if role == 'B0' else rate,
                           evidence=[dict(file=self.doc.name, sha256=self.digest)])
                for role, seal in self.seals[name].items()
            }
        return dict(schema='deeplinkengine-v2-actual-card-cost/v1', scenarios=scenarios)

    def judge(self, expense=None):
        path = None
        if expense is not None:
            self.manifest.write_text(json.dumps(expense), encoding='utf-8')
            path = self.manifest
        return report.targets(self.fixed, self.context, self.audit, path)['4.4']

    def test_estimates_cannot_decide_formal_cost_without_bills(self):
        result = self.judge()
        for level in ('milestone', 'final'):
            self.assertFalse(result[level]['met'])
            self.assertEqual(result[level]['evidence_status'], 'missing')
        for name in ('mmlu', 'four_tenant'):
            self.assertIsNone(result['formal_cost'][name]['reduction_percent'])
            self.assertAlmostEqual(float(result['same_rate_estimate'][name]['reduction_percent']), 50)

    def test_actual_unequal_rates_and_estimates_are_independent(self):
        result = self.judge(self.expense())
        self.assertTrue(result['milestone']['met'])
        self.assertTrue(result['final']['met'])
        self.assertAlmostEqual(float(result['unit_cost_reduction_percent']), 25)
        self.assertAlmostEqual(float(result['four_tenant_cost_reduction_percent']), 40)
        self.assertAlmostEqual(float(result['same_rate_estimate']['mmlu']['reduction_percent']), 50)
        self.assertEqual(result['formal_cost']['mmlu']['arms']['B1']['expense_sha256'], [self.digest])
        self.assertEqual(result['formal_cost']['mmlu']['utilization']['B1']['per_card_percent'], {'1': 85, '6': 85})
        self.assertEqual(result['formal_cost']['four_tenant']['utilization']['B0']['missing_seconds'], 0)
        self.assertNotIn(self.doc.name, json.dumps(result))
        self.assertFalse(self.judge(self.expense(mmlu='2100', four_tenant='1800'))['milestone']['met'])

    def test_both_cost_thresholds_are_inclusive_without_rounding(self):
        result = self.judge(self.expense(mmlu='1800', four_tenant='1400'))
        self.assertTrue(result['milestone']['met'])
        self.assertTrue(result['final']['met'])
        result = self.judge(self.expense(mmlu='1800.00000000000000000000000001',
                                         four_tenant='1400.00000000000000000000000001'))
        self.assertFalse(result['milestone']['met'])
        self.assertFalse(result['final']['met'])

    def test_report_writes_only_missing_fee_template_with_sealed_arm_ids(self):
        expense = self.expense()
        expense['scenarios'].pop('four_tenant')
        cost = self.judge(expense)
        data = dict(targets={'4.4': cost}, context_audit={}, scoring='', reference_method='')
        out = self.path / 'report'
        with patch.object(report, 'collect', return_value=data), patch.object(report, 'charts'):
            report.render(self.path, out)
        template = json.loads((out / report.ACTUAL_COST_TEMPLATE).read_bytes())
        self.assertEqual(template['schema'], report.ACTUAL_COST_SCHEMA)
        self.assertEqual(set(template['scenarios']), {'four_tenant'})
        for role, arm in template['scenarios']['four_tenant'].items():
            self.assertEqual(arm['measurement_seal_sha256'], self.seals['four_tenant'][role])
            self.assertIsNone(arm['monthly_cost_per_card_cny'])
            self.assertEqual(arm['evidence'], [dict(file=None, sha256=None)])
        self.assertIn(report.ACTUAL_COST_TEMPLATE, (out / 'report.html').read_text(encoding='utf-8'))
        self.assertIsNone(report.cost_template(self.judge(self.expense())))

    def test_only_mmlu_can_qualify_for_milestone(self):
        self.fixed['current_cost_utilization'] = {'swe': dict(unit_cost_reduction_percent=65, utilization_gain_pp=30)}
        self.fixed['current_trace_performance'] = {'swe': dict(
            paired_execution_complete=True,
            arms={r: dict(planned=64, recorded=64, successful=64, failed=0, unreturned=0,
                          output_tokens_per_second=100 if r == 'B0' else 300) for r in ('B0', 'B1')})}
        result = self.judge(self.expense(mmlu='2000'))
        self.assertFalse(result['milestone']['met'])
        self.assertTrue(result['final']['met'])

    def test_quality_and_complete_metering_are_required_for_mmlu(self):
        mmlu = self.fixed['current_mmlu_pair']
        mmlu['arms']['B1']['quality']['correct'] = 139
        self.assertFalse(self.judge(self.expense())['milestone']['met'])
        mmlu['arms']['B1']['quality']['correct'] = 150
        mmlu['utilization']['B1'] = self.sample(100, 42)
        self.assertFalse(self.judge(self.expense())['milestone']['met'])
        mmlu['utilization']['B1'] = self.sample(99, 45)
        self.assertFalse(self.judge(self.expense())['milestone']['met'])

    def test_each_four_tenant_arm_must_complete_and_final_can_lack_bills_independently(self):
        expense = self.expense()
        expense['scenarios'].pop('four_tenant')
        result = self.judge(expense)
        self.assertTrue(result['milestone']['met'])
        self.assertEqual(result['final']['evidence_status'], 'missing')
        multi = self.context['comparison'][-1]
        multi['B0']['per_tenant']['tenant-0']['failed'] = 1
        self.assertFalse(self.judge(self.expense())['final']['met'])

    def test_wrong_seal_invalid_rate_and_altered_expense_are_rejected(self):
        expense = self.expense()
        expense['scenarios']['mmlu']['B0']['measurement_seal_sha256'] = self.seals['mmlu']['B1']
        with self.assertRaisesRegex(ValueError, 'sealed measurement'):
            self.judge(expense)
        expense = self.expense(mmlu='NaN')
        with self.assertRaisesRegex(ValueError, 'positive and finite'):
            self.judge(expense)
        self.doc.write_bytes(b'altered')
        with self.assertRaisesRegex(ValueError, 'altered or duplicate'):
            self.judge(self.expense())



if __name__=='__main__':unittest.main()
