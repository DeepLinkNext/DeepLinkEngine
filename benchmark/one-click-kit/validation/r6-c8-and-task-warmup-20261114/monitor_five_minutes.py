"""Read-only five-minute monitoring; snapshots outside immutable campaign evidence."""
from pathlib import Path
import json,os,re,sys,time,datetime
Q=Path(__file__).resolve().parent
R=Q.parent.parent/'runs'/Q.name
EXEC=Path(json.loads((Q/'process.json').read_text())['command'][1]).parent
sys.path.insert(0,str(EXEC/'scripts'));from telemetry_audit import parse_cards
sys.path.insert(0,str(Q));from monitor_stage import locate

def read(path):
 try:return json.loads(path.read_bytes())
 except (OSError,json.JSONDecodeError):return {}

def tail(path,size=65536):
 try:
  with path.open('rb') as stream:
   stream.seek(max(0,path.stat().st_size-size));return stream.read().decode(errors='replace')
 except OSError:return ''

process=read(Q/'process.json')
while True:
 try:
  now=time.time();status=read(Q/'live-status.json');stage=status.get('current_stage');campaign=read(R/'campaign.json');child,leaf_stage=locate(R,stage,campaign)
  proc=Path('/proc')/str(process['pid']);rawstat=proc.joinpath('stat').read_text().split() if proc.exists() else None
  alive=bool(rawstat and rawstat[2]!='Z' and rawstat[21]==str(process['pid_start_ticks']))
  sample={};log='';receipt={};memory={};formal_counts={};worker_states=[];compilation={}
  if child:
   receipt=read(child/'receipt.json');memory=read(child/'memory-latest.json');log=tail(child/'service.log')
   for line in reversed(tail(child/'device-samples.jsonl',20000).splitlines()):
    try:sample=json.loads(line);break
    except json.JSONDecodeError:continue
   for owner,pid in set(re.findall(r'\((Worker_TP\d+) pid=(\d+)\)',log)):
    p=Path('/proc')/pid
    try:
     fields=p.joinpath('stat').read_text().split();worker_states.append({'owner':owner,'pid':int(pid),'state':fields[2],'cpu_ticks':int(fields[13])+int(fields[14]),'start_ticks':fields[21]})
    except OSError:pass
   for path in list((child/'service').glob('*.json'))+list((child/'service/evaluation/requests').glob('*.json')):
    row=read(path)
    if isinstance(row,dict) and ('case_id' in row or 'phase' in row):
     if 'phase' in row and not row['phase'].endswith('-formal'):continue
     if 'success' in row:key=row.get('phase','joint')+('-success' if row['success'] else '-failed')
     elif 'status' in row:key='dataset-'+str(row['status'])
     else:continue
     formal_counts[key]=formal_counts.get(key,0)+1
   for path in (child/'service').glob('compilation-events-*.jsonl'):
    events=[]
    for line in path.read_text().splitlines():
     try:events.append(json.loads(line))
     except json.JSONDecodeError:pass
    compilation[path.name]={'events':len(events),'after_boundary_compiles':sum(e.get('sealed') is True and e.get('compiler') in ('torch-dynamo','triton') for e in events)}
  fatal=read(child/'startup-failure.json') if child else {}
  campaign=read(R/'campaign.json');gates=[]
  if campaign.get('status')=='failed':gates.append('campaign_failed')
  if not alive and not campaign:gates.append('controller_missing_without_final_receipt')
  if receipt.get('failure'):gates.append('child_failure')
  if fatal:gates.append('fatal_startup')
  if any(e['after_boundary_compiles'] for e in compilation.values()):gates.append('post_boundary_compilation')
  if sample and now-sample.get('wall',0)>60 and alive and not campaign and receipt.get('status') not in ('completed','failed'):gates.append('sampler_stale_over60s')
  capture=re.findall(r'Capturing CUDA graphs[^\r\n]*',log)
  state={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_alive':alive,'campaign_status':campaign.get('status'),'current_stage':stage,'current_leaf_stage':leaf_stage,'phase':receipt.get('status',status.get('stage_phase')),'formal_client_started':receipt.get('client_started',False),'measurement_ready':receipt.get('measurement_ready',False),'last_sample_age_seconds':round(now-sample['wall'],2) if sample else None,'ai_core_percent':parse_cards(sample['raw'],('1','6')) if sample else None,'cgroup_current_gib':round(memory.get('current',0)/2**30,2) if memory else None,'workers':worker_states,'latest_graph_capture':capture[-1] if capture else None,'formal_success_counts':formal_counts,'compilation_boundary':compilation,'failure':receipt.get('failure') or status.get('failure'),'startup_failure':fatal or None,'attention':gates,'completed_stages':status.get('completed_stages'),'reused_stages':status.get('reused_stages'),'scope':'Read-only monitor; absence of alerts is not formal performance or quality acceptance.'}
  partial=Q/'monitor-latest.json.partial';partial.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');partial.replace(Q/'monitor-latest.json')
  with (Q/'monitor-history.jsonl').open('a') as stream:stream.write(json.dumps(state,ensure_ascii=False)+'\n')
  if gates:
   with (Q/'monitor-alerts.jsonl').open('a') as stream:stream.write(json.dumps(state,ensure_ascii=False)+'\n')
  print(json.dumps(state,ensure_ascii=False),flush=True)
  if campaign:break
 except Exception as exc:
  print(json.dumps({'monitor_error':type(exc).__name__+': '+str(exc),'wall':time.time()}),flush=True)
 time.sleep(300)
