"""Acceptance regressions: quality denominators, log failures and numeric gates."""
import copy
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

    def test_cost_requires_ten_utilization_points_and_full_sampler_coverage(self):
        m=self.fixed['current_mmlu_pair'];m['gains']['ai_core_utilization_gain_pp']=9.9999
        self.assertFalse(report.targets(self.fixed,self.context,self.audit)['4.4']['milestone']['met'])
        m['gains']['ai_core_utilization_gain_pp']=10;m['utilization']['B0']['missing_seconds']=.001
        self.assertFalse(report.targets(self.fixed,self.context,self.audit)['4.4']['milestone']['met'])

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
            document=(out/'report.html').read_text()
            for key in report.LABELS:self.assertIn(key+' '+report.LABELS[key],document)
            self.assertNotIn('pending',document);self.assertNotIn('复核中',document)
            for name in ('fixed-five.svg','fixed-five.png'):self.assertGreater((out/name).stat().st_size,1000)
            self.assertEqual(json.loads((out/'summary.json').read_text())['targets'],data['targets'])

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


if __name__=='__main__':unittest.main()
