"""Public report-only CPU route: selected custody and honest missing/failed state."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def raw(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def pin(value):
    return hashlib.sha256(value).hexdigest()


class PublicProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'sealed'
        self.source.mkdir()
        metric = dict(schema='full-answer-joint/v1', role='B0', total=200, completed=200, failed=0,
            correct=150, truncated=2, unparsed=0, successful_output_tokens=2000,
            duration_seconds=10, output_tokens_per_second=200)
        self.metric_raw = raw(metric)
        receipt = dict(status='completed', completed=True, quiescent=True)
        self.receipt_raw = raw(receipt)
        self.campaign = dict(schema='deeplinkengine-official-full-suite/v1', status='failed', baseline_image={'id':'official-vllm-0.23.0'},
            sources={'fixture-only.py':'a'*64}, stages=[{'name':'mmlu','status':'completed'}])
        self.report = dict(schema='deeplinkengine-official-baseline-report/v1', baseline='official-vllm-0.23.0', status='failed',
            planned_denominators={'mmlu':200,'hle':200,'frontier':160,'swe':64,'terminal':48},
            phases={'mmlu': {'status':'completed','service/joint.json':{'path':'mmlu/service/joint.json','sha256':pin(self.metric_raw),'value':metric},
                'receipt.json':{'path':'mmlu/receipt.json','sha256':pin(self.receipt_raw),'value':receipt}}},
            metrics={'mmlu':metric}, quality={}, new_F_T_qualification=False, qualified_efficiency_percent=None)
        self.extra = {'mmlu/service/joint.json':self.metric_raw,'mmlu/receipt.json':self.receipt_raw}
        self.build()

    def build(self):
        files={'campaign.json':raw(self.campaign),'report.json':raw(self.report),**self.extra}
        # Omitted raw payload deliberately does not exist; projection must not claim to verify it.
        seal={**{name:pin(value) for name,value in files.items()},'mmlu/raw/not-in-selected-closure.sse':'f'*64}
        seal_raw=raw(seal)
        for name,value in {**files,'checksums.json':seal_raw}.items():
            path=self.source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
        self.config=dict(kind='sealed-report-projection',source='sealed',out='derived',plots=False,plot_cache='plot-cache',
            report_sha256=pin(files['report.json']),original_version_seal_sha256=pin(seal_raw))

    def run_cli(self,success=True,env=None,python=None,timeout=30):
        config=self.root/'projection.json';config.write_bytes(raw(self.config))
        result=subprocess.run([str(python or sys.executable),'-B',str(ROOT/'run_all.py'),'--official-baselines','--report-only','--config',str(config)],
                              env=env,capture_output=True,text=True,timeout=timeout)
        self.assertEqual(result.returncode==0,success,result.stdout+result.stderr)
        return result

    def test_actual_public_cli_preserves_failed_values_and_partial_custody(self):
        before={p.relative_to(self.source).as_posix():p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        result=self.run_cli();self.assertIn('model_requests=0',result.stdout)
        output=self.root/'derived';report=json.loads((output/'report.json').read_bytes());derivation=json.loads((output/'derivation.json').read_bytes());preview=json.loads((output/'publication-preview.json').read_bytes())
        self.assertEqual(report['metrics'],self.report['metrics']);self.assertEqual(report['status'],'failed')
        self.assertFalse(derivation['raw_sse_recomputed']);self.assertFalse(derivation['all_original_payloads_verified'])
        self.assertEqual(derivation['unverified_omitted_original_members'],1);self.assertFalse(derivation['publication_ready'])
        self.assertEqual(derivation['model_requests_executed'],0);self.assertIsNone(derivation['F'])
        self.assertIsNone(preview['profiles'][0]['arm']['mean_ttft_ms']);self.assertIsNone(preview['profiles'][0]['arm']['mean_tpot_ms'])
        self.assertEqual(preview['missing_metric_phases'],['hle','frontier','swe','terminal'])
        self.assertEqual((output/'originals'/'report.json').read_bytes(),before['report.json'])
        self.assertEqual(before,{p.relative_to(self.source).as_posix():p.read_bytes() for p in self.source.rglob('*') if p.is_file()})
        self.assertIn('official_projection.py',derivation['projection_sources'])
        marker=json.loads((output/'projection-execution.json').read_bytes())
        self.assertEqual(marker['status'],'completed');self.assertEqual(marker['checksums_sha256'],pin((output/'checksums.json').read_bytes()))
        for name,expected in json.loads((output/'checksums.json').read_bytes()).items():
            self.assertEqual(pin((output/name).read_bytes()),expected)

    def test_bad_external_report_pin_refused(self):
        self.config['report_sha256']='0'*64;self.run_cli(False);self.assertFalse((self.root/'derived').exists())

    def test_bad_external_version_seal_pin_refused(self):
        self.config['original_version_seal_sha256']='0'*64;self.run_cli(False)

    def test_active_campaign_and_report_refused(self):
        self.campaign['status']=self.report['status']='running';self.build();self.run_cli(False)

    def set_collected_state(self):
        self.campaign.update(status='measurements_collected',ended=123.5,qualification={'complete':False,'independent_third_party':False})
        self.report['status']='measurements_collected'

    def test_ended_measurements_collected_keeps_qualification_incomplete(self):
        self.set_collected_state();self.build();self.run_cli()
        output=self.root/'derived';report=json.loads((output/'report.json').read_bytes());derivation=json.loads((output/'derivation.json').read_bytes());preview=json.loads((output/'publication-preview.json').read_bytes())
        self.assertEqual(report['status'],'measurements_collected')
        self.assertEqual(derivation['source_execution_status'],'measurements_collected')
        self.assertEqual(derivation['source_qualification_status'],'incomplete')
        self.assertEqual(preview['source_qualification_status'],'incomplete')
        self.assertFalse(report['new_F_T_qualification']);self.assertIsNone(derivation['F']);self.assertIsNone(derivation['efficiency_percent'])
        self.assertFalse(preview['publication_ready'])

    def test_collected_without_end_or_with_qualified_source_refused(self):
        self.set_collected_state();del self.campaign['ended'];self.build();self.run_cli(False)
        self.set_collected_state();self.campaign['qualification']['complete']=True;self.build();self.run_cli(False)

    def test_collected_unquiet_original_receipt_refused(self):
        self.set_collected_state();receipt={'status':'completed','completed':True,'quiescent':False};content=raw(receipt)
        self.extra['mmlu/receipt.json']=content;self.report['phases']['mmlu']['receipt.json'].update(sha256=pin(content),value=receipt)
        self.build();self.run_cli(False)

    def test_selected_original_tamper_refused(self):
        (self.source/'mmlu/service/joint.json').write_bytes(b'{}');self.run_cli(False)

    def test_embedded_metric_mismatch_refused(self):
        self.report['metrics']=copy.deepcopy(self.report['metrics']);self.report['metrics']['mmlu']['correct']=199
        self.build();self.run_cli(False)

    def test_original_phase_status_mismatch_refused(self):
        self.report['phases']['mmlu']['status']='reused';self.build();self.run_cli(False)

    def test_output_overwrite_refused(self):
        (self.root/'derived').mkdir();self.run_cli(False)

    def test_output_inside_original_refused(self):
        self.config['out']='sealed/derived';self.run_cli(False)

    def test_admitted_reuse_requires_original_seal_and_receipt(self):
        phase_seal=raw({'receipt.json':pin(self.receipt_raw),'service/joint.json':pin(self.metric_raw)})
        self.extra['mmlu/checksums.json']=phase_seal
        reuse=dict(source='/original/solver',original_checksums_sha256=pin(phase_seal),original_receipt_sha256=pin(self.receipt_raw))
        self.campaign['stages'][0].update(status='reused',reuse=reuse);self.report['phases']['mmlu']['status']='reused';self.build();self.run_cli()
        derivation=json.loads((self.root/'derived/derivation.json').read_bytes());self.assertEqual(derivation['original_reuse_sources']['mmlu'],reuse)
        self.config['out']='second';self.campaign['stages'][0]['reuse']['original_receipt_sha256']='0'*64;self.build();self.config['out']='second';self.run_cli(False)

    def test_unknown_kind_does_not_fall_through_to_projection(self):
        self.config['kind']='unknown-report';self.run_cli(False)

    def test_cpu_plot_failure_and_timeout_cannot_leave_a_completed_projection(self):
        sys.path.insert(0,str(ROOT))
        self.addCleanup(lambda:sys.path.remove(str(ROOT)))
        spec=importlib.util.spec_from_file_location('projection_failure_checks',ROOT / 'official/official_projection.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for index,outcome in enumerate((subprocess.CompletedProcess([],1,'','fixture failure'),subprocess.TimeoutExpired('fixture plot',300))):
            self.config.update(plots=True,out='failed-plot-'+str(index));config=self.root/'projection.json';config.write_bytes(raw(self.config))
            kwargs={'side_effect':outcome} if isinstance(outcome,Exception) else {'return_value':outcome}
            with patch.object(module.subprocess,'run',**kwargs):
                with self.assertRaisesRegex(ValueError,'plotting failed'):module.run(config)
            out=self.root/self.config['out'];execution=json.loads((out/'projection-execution.json').read_bytes());report=json.loads((out/'report.json').read_bytes());derivation=json.loads((out/'derivation.json').read_bytes())
            self.assertEqual(execution['status'],'failed');self.assertEqual(derivation['status'],'failed')
            self.assertEqual(report['report_projection']['status'],'failed');self.assertEqual(report['status'],'failed')
            self.assertEqual(execution['model_requests_executed'],0)

    def test_plot_cache_cannot_enter_original_or_output_seal(self):
        for location in ('sealed/cache','derived/cache'):
            self.config.update(plots=True,plot_cache=location);self.run_cli(False)

    def test_healthy_current_renderer_does_not_install_dependencies(self):
        sys.path.insert(0,str(ROOT));self.addCleanup(lambda:sys.path.remove(str(ROOT)))
        spec=importlib.util.spec_from_file_location('current_renderer_checks',ROOT / 'official/official_projection.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        ready=json.dumps({'python':sys.executable,'matplotlib':'3.10.8'})
        record={}
        with patch.object(module.subprocess,'run',return_value=subprocess.CompletedProcess([],0,ready,'')) as commands:
            python,env=module.plot_python(self.root/'cpu-cache',record)
        self.assertEqual(record['status'],'current');self.assertEqual(record['repair_status'],'not-needed')
        self.assertEqual(commands.call_count,1);self.assertEqual(python,Path(sys.executable))

    def test_pinned_environment_install_failure_and_timeout_are_recorded(self):
        sys.path.insert(0,str(ROOT));self.addCleanup(lambda:sys.path.remove(str(ROOT)))
        spec=importlib.util.spec_from_file_location('repair_failure_checks',ROOT / 'official/official_projection.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for index,timeout in enumerate((False,True)):
            cache=self.root/('repair-cache-'+str(index));record={}
            def execute(command,**kwargs):
                if '-c' in command:return subprocess.CompletedProcess(command,1,'','No matplotlib')
                if 'venv' in command:
                    prefix=Path(command[-1]);python=prefix/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
                    python.parent.mkdir(parents=True,exist_ok=True);python.write_bytes(b'fixture only')
                    return subprocess.CompletedProcess(command,0,'','')
                if timeout:raise subprocess.TimeoutExpired(command,300)
                return subprocess.CompletedProcess(command,1,'','No matching distribution found for matplotlib==3.10.8')
            with patch.object(module.subprocess,'run',side_effect=execute):
                with self.assertRaisesRegex(ValueError,'repair failed'):module.plot_python(cache,record)
            self.assertEqual(record['status'],'failed');self.assertEqual(record['repair_status'],'failed')
            self.assertEqual(record['repair_step'],'install-pinned-renderer')

    def test_existing_one_click_report_environment_is_reused(self):
        sys.path.insert(0,str(ROOT));self.addCleanup(lambda:sys.path.remove(str(ROOT)))
        spec=importlib.util.spec_from_file_location('existing_report_renderer_checks',ROOT / 'official/official_projection.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        existing=self.root/'existing-python';existing.write_bytes(b'fixture executable')
        responses=[subprocess.CompletedProcess([],1,'','Missing matplotlib'),subprocess.CompletedProcess([],0,json.dumps({'python':str(existing),'matplotlib':'3.10.8'}),'')]
        with patch.dict(os.environ,{'VLLM_HUST_REPORT_PYTHON':str(existing)}):
            with patch.object(module.subprocess,'run',side_effect=responses) as commands:
                record={};python,env=module.plot_python(self.root/'new-cache',record)
        self.assertEqual(record['status'],'existing-report');self.assertEqual(python,existing)
        self.assertEqual(record['repair_status'],'not-needed');self.assertEqual(commands.call_count,2)

    @unittest.skipUnless(os.environ.get('DEEPLINK_PLOT_TEST_WHEELS'),'Offline CPU plot wheels not requested')
    def test_real_public_cli_repairs_ordinary_python_then_reuses_cache(self):
        ordinary=self.root/'ordinary-python';subprocess.run([sys.executable,'-m','venv',str(ordinary)],check=True,timeout=60)
        python=ordinary/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        env=dict(os.environ,PIP_NO_INDEX='1',PIP_FIND_LINKS=os.environ['DEEPLINK_PLOT_TEST_WHEELS'])
        env.pop('VLLM_HUST_REPORT_PYTHON',None);env['XDG_CACHE_HOME']=str(self.root/'ordinary-cache-home')
        self.config['plots']=True;self.config.pop('plot_cache');self.run_cli(env=env,python=python,timeout=60)
        output=self.root/'derived';record=json.loads((output/'plot-environment.json').read_bytes())
        self.assertEqual(record['status'],'repaired');self.assertFalse(record['current_renderer_ready'])
        self.assertEqual(record['matplotlib'],'3.10.8');self.assertTrue((output/'fixed-five-throughput.png').is_file())
        self.assertFalse(any('site-packages/' in key or '/environment/' in key for key in json.loads((output/'checksums.json').read_bytes())))
        empty=self.root/'empty-wheels';empty.mkdir();env['PIP_FIND_LINKS']=str(empty)
        self.config['out']='cached';self.run_cli(env=env,python=python,timeout=60)
        reused=json.loads((self.root/'cached/plot-environment.json').read_bytes())
        self.assertEqual(reused['status'],'cached');self.assertEqual(reused['repair_status'],'not-needed')
        blocked=self.root/'blocked-home-cache';blocked.write_bytes(b'not a directory');env['XDG_CACHE_HOME']=str(blocked)
        self.config['out']='healthy-with-bad-home-cache';self.run_cli(env=env,python=record['python'],timeout=60)
        healthy=json.loads((self.root/self.config['out']/'plot-environment.json').read_bytes())
        self.assertEqual(healthy['status'],'current');self.assertEqual(healthy['cache_selection'],'output-parent-default')
        self.assertTrue(Path(healthy['cache']).is_relative_to(self.root))

    def check_final_output_failure(self,stage):
        # Deterministic filesystem fault at the real public CLI, without product fault flags.
        hook=self.root/'fault-hook';hook.mkdir();target=str(self.root/'derived')
        code="""import os,errno
from pathlib import Path
target=Path(TARGET)
write_text=Path.write_text
replace=os.replace
def failed_write(path,*args,**kwargs):
    if STAGE=='html' and path==target/'report.html':raise OSError(errno.ENOSPC,'fixture HTML write failure')
    return write_text(path,*args,**kwargs)
def failed_replace(source,destination,*args,**kwargs):
    if STAGE=='seal' and Path(destination)==target/'checksums.json':raise OSError(errno.EIO,'fixture final seal publication failure')
    return replace(source,destination,*args,**kwargs)
Path.write_text=failed_write
os.replace=failed_replace
""".replace('TARGET',repr(target)).replace('STAGE',repr(stage))
        (hook/'sitecustomize.py').write_text(code,encoding='utf-8')
        env=dict(os.environ,PYTHONPATH=str(hook));result=self.run_cli(False,env=env)
        self.assertNotIn('OFFICIAL_SEALED_REPORT_PROJECTION_COMPLETED',result.stdout)
        self.assertIn('OFFICIAL_SEALED_REPORT_PROJECTION_FAILED',result.stderr)
        output=self.root/'derived';execution=json.loads((output/'projection-execution.json').read_bytes())
        self.assertEqual(execution['status'],'failed');self.assertFalse((output/'checksums.json').exists())
        self.assertEqual(json.loads((output/'derivation.json').read_bytes())['status'],'failed')
        self.assertEqual(json.loads((output/'report.json').read_bytes())['report_projection']['status'],'failed')
        self.assertTrue((output/'projection-failure.log').read_text(encoding='utf-8'))
        self.assertFalse(list(output.glob('.*.pending')))

    def test_html_write_failure_cannot_publish_completion(self):
        self.check_final_output_failure('html')

    def test_final_seal_failure_cannot_publish_completion(self):
        self.check_final_output_failure('seal')


if __name__=='__main__':unittest.main()
