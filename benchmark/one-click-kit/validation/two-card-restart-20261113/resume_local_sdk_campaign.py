from pathlib import Path
import json,os,sys,time,subprocess,hashlib
sys.dont_write_bytecode=True
ROOT=Path('/data/shared_upload/vllm-hust-third-party-self-run-20261112')
WORK=ROOT/'runtime-recovery-20261113-r2'
EXEC=ROOT/'executors/engine-d0412c2ee52c0e0c5790ec55fbbee12b5bd51238'
started=time.time()
while not (WORK/'sdk-local-recovery.json').exists():
 pid=json.loads((WORK/'sdk-local-process.json').read_bytes())['pid'];stat=Path('/proc')/str(pid)/'stat'
 if not stat.exists() or stat.read_text().split()[2]=='Z':raise RuntimeError('Exact local SDK restore failed; no inference started')
 if time.time()-started>1800:raise TimeoutError('Local SDK restore exceeded30m')
 time.sleep(3)
sdk=json.loads((WORK/'sdk-local-recovery.json').read_bytes());assert sdk['completed']
sys.path.insert(0,str(EXEC/'scripts'))
from restore_b1_r4 import restore,verify_installed_sources
from runtime_overlay import apply
from dependency_admission import admit_dependencies
b1=Path('/opt/vllm-hust-r5-runtime-20261113-r2/b1-r5-local-sdk')
restore(ROOT/'b1-original-assets',b1,Path(sdk['python']),Path('/root/.env'));apply(b1)
files=verify_installed_sources(b1);dep=admit_dependencies(b1)
archive=WORK/'b1-r5-local-sdk-verified-deployment.tar';partial=archive.with_suffix('.tar.partial')
subprocess.run(['tar','-cf',str(partial),'-C',str(b1.parent),b1.name],check=True)
with partial.open('rb') as stream:
 digest=hashlib.file_digest(stream,'sha256').hexdigest();os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
partial.replace(archive)
record={'completed':True,'deployment':str(b1),'files_verified':len(files),'metadata':dep,'native_qualified':False,'archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest},'ended':time.time()}
(WORK/'b1-local-sdk-deployment.json').write_text(json.dumps(record,indent=2)+'\n')
old=ROOT/'runs/five-datasets-r5-matched-20261113-r4'
receipt=json.loads((old/'campaign.json').read_bytes());assert receipt['status']=='failed'
child=json.loads((old/'continuous-B0/receipt.json').read_bytes());assert child['quiescent'] and not child['client_started']
b0=json.loads((WORK/'b0-recovery.json').read_bytes());assert b0['completed']
name='five-datasets-r5-matched-20261113-r5';out=ROOT/'runs'/name;queue=ROOT/'queues'/name
assert not out.exists();queue.mkdir(exist_ok=False)
command=[sys.executable,str(EXEC/'one_click.py'),'--engine-profile','r5','--deployment',str(b1),'--model','/models/Qwen3.5-35B-A3B','--b1-python',sdk['python'],'--baseline-deployment',b0['deployment'],'--physical-devices','1,6','--devices','0,1','--cache',str(ROOT),'--out',str(out),'--report-python','/root/vllm-hust-third-party-verify-20261111-c9c974/self-run-kit/.report-venv/bin/python','--env-file','/root/.env','--datasets','mmlu,hle,frontier,swe,terminal']
with (queue/'controller.log').open('xb') as log:
 p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=''))
record={'pid':p.pid,'command':command,'source_commit':'d0412c2ee52c0e0c5790ec55fbbee12b5bd51238','output':str(out),'log':str(queue/'controller.log'),'started':time.time(),'new_stages':15,'reused_stages':0,'physical_cards':[1,6],'logical_cards':[0,1],'cgroup_limit_bytes':68719476736,'cpu_cores':8,'storage':'Byte-identical pinned SDKs and deployments on local fast disk; immutable source archives and all raw results on NFS','reason':'Before any formal requests, replace slow NFS SDK imports with exact same archive locally restored; original r5 protocol unchanged.'}
for dest in (queue/'process.json',WORK/'resumed-local-sdk-campaign.json'):dest.write_text(json.dumps(record,indent=2)+'\n')
observer=(ROOT/'queues/five-datasets-r5-matched-20261113-r4/observe.py').read_text().replace('five-datasets-r5-matched-20261113-r4',name)
(queue/'observe.py').write_text(observer)
with (queue/'observer.log').open('xb') as log:
 q=subprocess.Popen([sys.executable,str(queue/'observe.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
(queue/'observer-process.json').write_text(json.dumps({'pid':q.pid,'started':time.time()},indent=2)+'\n')
print('ONE_CLICK_LOCAL_SDK_STARTED',json.dumps(record),flush=True)
