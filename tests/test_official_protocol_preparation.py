import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmark/one-click-kit/scripts')]
from official import enable_module_imports

enable_module_imports()
import official_protocol_preparation as preparation


class Preparation(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'model').mkdir()
        self.config=dict(model=str(self.root/'model'),physical_devices=[0,1],cache=str(self.root/'cache'),out=str(self.root/'full'),versions=['0.30.0'])
        self.commands=[];self.fixture=None
        # These cases isolate orchestration/cache sealing. Real scientific input
        # generation is verified separately by the public on-host cold run.
        self.scientific=patch('official_protocol_data.verify_prepared_inputs',return_value={'fixture_only':True})
        self.scientific.start();self.addCleanup(self.scientific.stop)

    def child(self,args,**kwargs):
        self.commands.append(args)
        self.assertIn('--prepare-official-protocol-only',args)
        self.assertNotIn('--rerun-b0',args)
        plan=json.loads(Path(args[args.index('--config')+1]).read_bytes());self.fixture=plan
        self.assertEqual(plan['engine_version'],preparation.VERSION)
        self.assertIs(plan['rerun_b0'],False)
        output=Path(plan['out']);output.mkdir()
        run=output/'fixed-five';run.mkdir();(run/'plans').mkdir();(run/'continuous-B0/inputs').mkdir(parents=True)
        (output/'evaluation.json').write_text(json.dumps(dict(model=plan['model'],out=str(run),cache=plan['cache'])))
        (run/'frozen-execution-plan.json').write_text(json.dumps(dict(datasets=['mmlu','hle','frontier','swe','terminal'],model=plan['model'])))
        (run/'continuous-B0/inputs/continuous-inputs-r1.json').write_text('{}')
        (run/'official-protocol-preparation.json').write_text('{}')
        (run/'native-token-workload.json').write_text('{}')
        for dataset,count in [('frontier',160),('hle',200),('swe',64),('terminal',48)]:
            suffix='trace' if dataset in ('swe','terminal') else 'solver'
            (run/'plans'/(dataset+'-'+suffix+'-B0.json')).write_text(json.dumps(dict(diagnostic=False,case_ids=list(range(count)))))
        sdk=output/'installation/sdk';(sdk/'provenance').mkdir(parents=True);(sdk/'provenance/installed-files.json').write_text('{}')
        python=output/'protocol-python';python.write_text('fixture-only')
        result=dict(status='prepared',protocol_preparation_complete=True,automated_acceptance_complete=False,
            protocol_preparation_mode='official-inputs-only',
            model_service_started=False,formal_client_started=False,engine_selection=dict(version=preparation.VERSION),
            stages=[dict(name=name,status='completed') for name in preparation.PHASES],
            protocol_run=str(run),protocol_deployment=str(sdk),protocol_python=str(python))
        (output/'results.json').write_text(json.dumps(result))
        return types.SimpleNamespace(returncode=0)

    def invoke(self,child=None):
        with patch('official_execution.prepare_anchor',return_value=self.root/'anchor') as anchor, \
             patch('child_runner.run_stage',side_effect=child or self.child),contextlib.redirect_stdout(io.StringIO()):
            result=preparation.ensure(self.config)
        return result,anchor

    def test_missing_protocol_is_prepared_by_public_no_model_mode(self):
        (config,proof),anchor=self.invoke()
        self.assertEqual(len(self.commands),1);self.assertEqual(anchor.call_args.args[0]['versions'],['0.23.0'])
        self.assertEqual(self.fixture['environment']['official_sdk'],str(self.root/'anchor'))
        self.assertTrue(Path(config['protocol_run']).is_dir())
        self.assertNotIn('protocol_run',self.config)
        raw=Path(proof['path']).read_bytes();self.assertEqual(proof['sha256'],hashlib.sha256(raw).hexdigest())
        journal=json.loads(raw);self.assertEqual(journal['status'],'prepared')
        self.assertFalse(journal['model_service_started']);self.assertFalse(journal['formal_client_started'])
        from official_suite import protocol_data_cache
        self.assertEqual(protocol_data_cache(config),Path(self.config['cache']).resolve())

    def test_saved_protocol_cache_rejects_a_different_model_or_campaign(self):
        from official_suite import protocol_data_cache
        (config,proof),unused=self.invoke()
        saved=Path(config['protocol_run']).parent/'evaluation.json'
        data=json.loads(saved.read_bytes());data['out']=str(self.root/'different-campaign');saved.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'differs'):protocol_data_cache(config)

    def test_legacy_protocol_without_saved_config_retains_its_original_cache(self):
        from official_suite import protocol_data_cache
        campaign=self.root/'old/run/fixed-five'
        self.assertEqual(protocol_data_cache(dict(protocol_run=str(campaign))),(self.root/'old/cache').resolve())

    def test_explicit_paths_skip_automatic_preparation(self):
        self.config.update(protocol_run='/existing/run',protocol_deployment='/existing/sdk',protocol_python='/existing/python')
        with patch('official_execution.prepare_anchor',side_effect=AssertionError('Explicit inputs must remain explicit')):
            config,proof=preparation.ensure(self.config)
        self.assertEqual(config,self.config);self.assertIsNone(proof)

    def test_prepared_paths_are_reused_only_while_their_originals_match(self):
        (first,proof),unused=self.invoke()
        with patch('official_execution.prepare_anchor',side_effect=AssertionError('Complete protocol must be reused')), \
             patch('child_runner.run_stage',side_effect=AssertionError('No repeated preparation')):
            again,reuse=preparation.ensure(self.config)
        self.assertEqual(first,again);self.assertTrue(reuse['reused']);self.assertEqual(proof['path'],reuse['path'])
        path=Path(first['protocol_run'])/'plans/frontier-solver-B0.json'
        data=json.loads(path.read_bytes());data['case_ids']=data['case_ids'][:-1];path.write_text(json.dumps(data));original=path.read_bytes()
        (repaired,newproof),unused=self.invoke()
        self.assertNotEqual(repaired['protocol_run'],first['protocol_run'])
        self.assertNotEqual(newproof['path'],proof['path']);self.assertEqual(path.read_bytes(),original)

    def test_partial_selection_is_rejected_before_any_stage(self):
        self.config['protocol_run']='/partial'
        with self.assertRaisesRegex(ValueError,'all three'):
            self.invoke()
        self.assertFalse((self.root/'cache').exists())

    def test_nonzero_child_retains_failed_journal_without_full_stage(self):
        def failed(args,**kwargs):self.commands.append(args);return types.SimpleNamespace(returncode=9)
        with self.assertRaisesRegex(ValueError,'preserve'):self.invoke(failed)
        journals=list((self.root/'cache/official-protocol-preparation').glob('*/preparation.json'))
        self.assertEqual(len(journals),1);j=json.loads(journals[0].read_bytes())
        self.assertEqual(j['status'],'failed');self.assertEqual(len(self.commands),1)

    def test_zero_exit_with_model_activity_or_wrong_denominator_is_rejected(self):
        def incorrect(args,**kwargs):
            result=self.child(args,**kwargs);output=Path(self.fixture['out']);p=output/'results.json'
            data=json.loads(p.read_bytes());data['model_service_started']=True;p.write_text(json.dumps(data));return result
        with self.assertRaisesRegex(ValueError,'no-model'):self.invoke(incorrect)

    def test_zero_exit_with_incomplete_fixed_population_is_rejected(self):
        def incorrect(args,**kwargs):
            result=self.child(args,**kwargs);output=Path(self.fixture['out']);p=output/'fixed-five/plans/hle-solver-B0.json'
            data=json.loads(p.read_bytes());data['case_ids']=data['case_ids'][:-1];p.write_text(json.dumps(data));return result
        with self.assertRaisesRegex(ValueError,'denominator'):self.invoke(incorrect)


if __name__=='__main__':unittest.main()
