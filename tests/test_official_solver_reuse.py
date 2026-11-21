"""CPU sealed-population retention and fail-closed source/identity checks."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from official_solver_reuse import admit, PHASES, DENOMINATORS
from official_reuse import sha


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def seal(root):
    values={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=root/'checksums.json'}
    save(root/'checksums.json',values)
    return values


class SolverRetention(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.old=self.root/'old-source'
        self.current=self.root/'current-source'
        self.old.mkdir();self.current.mkdir()
        for name in ('official_suite.py','official_judge.py','client.py'):
            (self.old/name).write_bytes(('old-'+name).encode())
            (self.current/name).write_bytes((('old-' if name=='client.py' else 'new-')+name).encode())
        (self.old/'benchmark/one-click-kit').mkdir(parents=True)
        (self.current/'official_solver_reuse.py').write_bytes(b'explicit CPU fixture controller')
        self.policy=self.current/'engine/official-baselines/solver-reuse-policy-v1.json'
        save(self.policy,dict(schema='deeplinkengine-reviewed-solver-reuse/v1',scope='fixed-judge-prelaunch-bind-only',
            reviewed_controller_changes={n:dict(original_sha256=sha(self.old/n),current_sha256=sha(self.current/n)) for n in ('official_suite.py','official_judge.py')}))
        self.source=self.root/'previous/0.23.0'
        self.source.mkdir(parents=True)
        self.destination=self.root/'next/0.23.0'
        self.destination.mkdir(parents=True)
        self.sdk=self.root/'sdk';self.sdk.mkdir()
        self.protocol=self.root/'protocol';self.protocol.mkdir()
        (self.protocol/'frozen-execution-plan.json').write_bytes(b'fixed CPU fixture protocol')
        self.inputs=self.root/'continuous-inputs.json';self.inputs.write_bytes(b'fixed fixture tokens')
        self.config=dict(model=str(self.root/'model'),protocol_deployment=str(self.sdk),protocol_run=str(self.protocol))
        self.lock=dict(id='official-vllm-0.23.0',manifest_digest='fixture-public-image')
        self.mapping=dict(physical=[0,7],logical=[0,1])
        self.weights={'part.safetensors':dict(bytes=8,sha256='fixture-weights')}
        save(self.source/'model-weights.json',self.weights)
        self.campaign=dict(schema='deeplinkengine-official-full-suite/v1',status='failed',ended=9999,
            baseline_image=self.lock,physical_mapping=self.mapping,continuous_input_sha256=sha(self.inputs),fixed_five_denominators=DENOMINATORS,
            stages=[dict(name=n,status='completed' if i<6 else 'failed' if i==6 else 'not_attempted') for i,n in enumerate(PHASES)],
            sources={str((self.old/n).resolve()):sha(self.old/n) for n in ('official_suite.py','official_judge.py','client.py')})
        self.sources={str((self.current/n).resolve()):sha(self.current/n) for n in ('official_suite.py','official_judge.py','client.py','official_solver_reuse.py')}
        self.sources[str(self.policy.resolve())]=sha(self.policy)
        save(self.source/'campaign.json',self.campaign)
        save(self.source.parent/'results.json',dict(status='incomplete',ended=10000,stages=[dict(status='failed')],configuration=dict(versions=['0.23.0'])))
        for name in PHASES[:6]:
            folder=self.source/name
            folder.mkdir()
            save(folder/'receipt.json',dict(completed=True,quiescent=True,model_ready=True,client_started=True,baseline_id=self.lock['id'],phase=name,
                input_plan_sha256=sha(self.protocol/'frozen-execution-plan.json'),cleanup_actions=[]))
            kind='continuous' if name=='continuous' else 'joint' if name=='mmlu' else 'dataset'
            save(folder/'service/plan.json',dict(model=self.config['model'],protocol_deployment=self.config['protocol_deployment'],
                serve_arguments=self.arguments(kind),client_tools=str(self.old/'benchmark/one-click-kit')))
            if name=='continuous':self.make_continuous(folder)
            elif name=='mmlu':self.make_joint(folder)
            else:
                ids=[name+'-'+str(i) for i in range(DENOMINATORS[name])]
                save(folder/'dataset-spec.json',dict(dataset=name,diagnostic=False,case_ids=ids))
                save(folder/'performance-summary.json',dict(dataset=name,planned=len(ids),recorded=len(ids),successful=len(ids)-1,failed=1,unreturned=0))
                for i,case in enumerate(ids):save(folder/'service/evaluation/requests'/(case+'.json'),dict(case_id=case,status='error' if i==0 else 'ok'))
            seal(folder)
        save(self.source/'frontier-judge/receipt.json',dict(status='failed',completed=False,quiescent=True,cleanup_actions=[],error='OSError: [Errno 98] Address already in use'))
        seal(self.source)

    @staticmethod
    def arguments(kind):return ['fixture-serve',kind]

    def make_continuous(self,folder):
        phases=[]
        for i,name in enumerate(('4096-warmup','4096-formal','16384-warmup','16384-formal')):
            seconds=60 if i%2==0 else 300
            start=i*400.0;end=start+seconds+1
            phases.append(dict(name=name,seconds=seconds,failed=0,requests=1,start=dict(monotonic=start),end=dict(monotonic=end)))
            save(folder/'service'/('service-'+str(i)+'.json'),dict(phase=name,success=True,start=dict(monotonic=start+.1),end=dict(monotonic=end-.1),usage=dict(completion_tokens=256),finish_reason='length'))
        save(folder/'service/continuous.json',dict(status='completed',phases=phases))

    def make_joint(self,folder):
        for i in range(200):
            save(folder/'service'/('request-'+str(i)+'.json'),dict(index=i,case_id='mmlu-'+str(i),success=True,correct=i<152,
                start=dict(monotonic=2000+i*.9),end=dict(monotonic=2000+i*.9+.2),usage=dict(completion_tokens=2),finish_reason='length' if i<19 else 'stop'))
        save(folder/'service/joint.json',dict(total=200,completed=200,failed=0,correct=152,truncated=19,successful_output_tokens=400,duration_seconds=200,start=dict(monotonic=2000),end=dict(monotonic=2200)))

    def invoke(self,**changes):
        args=dict(config=self.config,image_lock=self.lock,sources=self.sources,weights=self.weights,mapping=self.mapping,inputs=self.inputs,current_root=self.current,arguments=self.arguments)
        args.update(changes)
        return admit(self.source,self.destination,**args)

    def reseal_phase(self,name):
        seal(self.source/name)
        seal(self.source)

    def test_all_solvers_are_byte_identical_and_failures_are_retained(self):
        original=sha(self.source/'checksums.json')
        result=self.invoke()
        self.assertFalse(result['inference_repeated'])
        self.assertTrue(result['failed_responses_preserved'])
        self.assertFalse(result['new_F_T_qualification'])
        self.assertEqual(len(result['phases']),6)
        self.assertEqual(sha(self.source/'checksums.json'),original)
        for phase in result['phases']:
            old=self.source/phase['name'];new=self.destination/phase['name']
            for p in old.rglob('*'):
                if p.is_file():self.assertEqual(p.read_bytes(),(new/p.relative_to(old)).read_bytes())
        self.assertEqual(json.loads((self.destination/'swe/performance-summary.json').read_bytes())['failed'],1)

    def test_changed_original_stops_before_any_copy(self):
        (self.source/'mmlu/service/request-0.json').write_bytes(b'changed raw original')
        with self.assertRaises(ValueError):self.invoke()
        self.assertEqual(list(self.destination.iterdir()),[])

    def test_missing_original200_refuses_even_if_resigned(self):
        (self.source/'mmlu/service/request-0.json').unlink()
        self.reseal_phase('mmlu')
        with self.assertRaisesRegex(ValueError,'original200'):self.invoke()

    def test_actual_weights_mapping_and_arguments_must_match(self):
        for changes in (dict(weights={'part.safetensors':dict(bytes=9,sha256='other')}),dict(mapping=dict(physical=[1,7])),dict(arguments=lambda kind:['different','serve'])):
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.invoke(**changes)
        self.assertEqual(list(self.destination.iterdir()),[])

    def test_unreviewed_client_changes_are_rejected(self):
        (self.current/'client.py').write_bytes(b'unreviewed client change')
        self.sources[str((self.current/'client.py').resolve())]=sha(self.current/'client.py')
        with self.assertRaisesRegex(ValueError,'Unreviewed'):self.invoke()

    @unittest.skipUnless(sys.platform.startswith('linux'),'Linux birth identity')
    def test_live_original_process_is_rejected(self):
        import os
        path=self.source/'mmlu/receipt.json'
        row=json.loads(path.read_bytes())
        row['cleanup_actions']=[dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])]
        save(path,row);self.reseal_phase('mmlu')
        with self.assertRaisesRegex(ValueError,'live owned'):self.invoke()

    def test_a_judge_launch_is_never_repeated(self):
        (self.source/'frontier-judge/judge-service.log').write_bytes(b'possible original child')
        seal(self.source)
        with self.assertRaisesRegex(ValueError,'may have launched'):self.invoke()

    def test_incomplete_solver_is_not_retried(self):
        self.campaign['stages'][5]['status']='failed'
        save(self.source/'campaign.json',self.campaign);seal(self.source)
        with self.assertRaisesRegex(ValueError,'all completed solvers'):self.invoke()

    def test_a_prior_c8_reuse_chain_remains_bound(self):
        original=self.root/'older/continuous';original.mkdir(parents=True)
        (original/'checksums.json').write_bytes(b'older original phase seal')
        (original.parent/'checksums.json').write_bytes(b'older original campaign seal')
        self.campaign['stages'][0]['status']='reused'
        save(self.source/'campaign.json',self.campaign)
        save(self.source/'continuous-reuse.json',dict(source=str(original),inference_repeated=False,original_dates_and_outputs_preserved=True,
            original_checksums_sha256=sha(original/'checksums.json'),original_campaign_checksums_sha256=sha(original.parent/'checksums.json')))
        seal(self.source)
        result=self.invoke()
        self.assertEqual(len(result['phases']),6)
        self.assertFalse(result['inference_repeated'])

    def test_configured_model_and_protocol_remain_fixed(self):
        for key in ('model','protocol_deployment'):
            changed=copy.deepcopy(self.config);changed[key]=str(self.root/'different')
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'arguments or SDK'):self.invoke(config=changed)

    def test_inconsistent_recorded_failures_are_rejected(self):
        path=self.source/'swe/service/evaluation/requests/swe-0.json'
        row=json.loads(path.read_bytes());row['status']='ok';save(path,row)
        self.reseal_phase('swe')
        with self.assertRaisesRegex(ValueError,'originals are incomplete'):self.invoke()

    def test_public_entry_refuses_environment_or_other_recovery_modes(self):
        import official_baselines
        model=Path(self.config['model']);model.mkdir()
        config=self.root/'evaluation.json'
        save(config,dict(model=str(model),cache=str(self.root/'cache'),out=str(self.root/'unused-output'),physical_devices=[0,7],versions=['0.23.0']))
        for flags in (['--prepare-environment'],['--reuse-completed-continuous',str(self.source/'continuous')]):
            with self.subTest(flags=flags),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                official_baselines.main(['--config',str(config),'--reuse-completed-solvers',str(self.source),*flags])
            self.assertEqual(error.exception.code,2)
        self.assertFalse((self.root/'unused-output').exists())


if __name__=='__main__':unittest.main()
