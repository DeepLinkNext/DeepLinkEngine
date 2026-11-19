"""Launch the frozen one-click campaign once exact persistent SDKs are ready."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path('/data/shared_upload/vllm-hust-third-party-self-run-20261112')
BASE=Path('/root/vllm-hust-r5-refreeze-20261113')
WORK=ROOT/'runtime-recovery-20261113-r1'
EXEC=ROOT/'executors/engine-d0412c2ee52c0e0c5790ec55fbbee12b5bd51238'
started=time.time();deadline=started+3*3600
while True:
 receipts=[WORK/'sdk-recovery.json',WORK/'b0-recovery.json']
 if all(p.exists() and json.loads(p.read_bytes()).get('completed') for p in receipts):break
 for name in ['recovery-process.json','b0-recovery-process.json']:
  p=json.loads((WORK/name).read_bytes())['pid'];stat=Path('/proc')/str(p)/'stat'
  if not stat.exists() or stat.read_text().split()[2]=='Z':
   matching=WORK/('sdk-recovery.json' if name=='recovery-process.json' else 'b0-recovery.json')
   if not matching.exists():raise RuntimeError('Prerequisite recovery failed; inspect persistent logs; no inference launched')
 if time.time()>deadline:raise TimeoutError('Persistent SDK recovery exceeded3h; no inference launched')
 time.sleep(10)
sdk=json.loads(receipts[0].read_bytes());baseline=json.loads(receipts[1].read_bytes());name='five-datasets-r5-matched-20261113-r3';out=ROOT/'runs'/name;queue=ROOT/'queues'/name
assert not out.exists();queue.mkdir(exist_ok=False)
command=[sys.executable,str(EXEC/'one_click.py'),'--engine-profile','r5','--model','/models/Qwen3.5-35B-A3B','--b1-python',sdk['python'],'--baseline-deployment',baseline['deployment'],'--physical-devices','6,7','--cache',str(ROOT),'--out',str(out),'--report-python','/root/vllm-hust-third-party-verify-20261111-c9c974/self-run-kit/.report-venv/bin/python','--env-file','/root/.env','--datasets','mmlu,hle,frontier,swe,terminal']
with (queue/'controller.log').open('xb') as log:
 process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=''))
record={'pid':process.pid,'command':command,'source_commit':'d0412c2ee52c0e0c5790ec55fbbee12b5bd51238','output':str(out),'log':str(queue/'controller.log'),'started':time.time(),'new_stages':15,'reused_stages':0,'recovery':'Exact archived SDKs recovered to persistent shared storage after user-confirmed container restart; no completed stages existed in interrupted r2','physical_cards':[6,7]}
for p in [queue/'process.json',BASE/'r5-process.json',WORK/'resumed-campaign.json']:p.write_text(json.dumps(record,indent=2)+'\n')
print('FROZEN_ONE_CLICK_RESUMED',json.dumps(record),flush=True)
