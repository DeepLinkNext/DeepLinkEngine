from pathlib import Path
import json,os,sys,time
sys.dont_write_bytecode=True
ROOT=Path('/data/shared_upload/vllm-hust-third-party-self-run-20261112')
WORK=ROOT/'runtime-recovery-20261113-r2'
EXEC=ROOT/'executors/engine-d0412c2ee52c0e0c5790ec55fbbee12b5bd51238'
sys.path.insert(0,str(EXEC/'scripts'))
from resource_guard import acquire,release,verify_mapping
from engine_entry import environment,check_pinned_host_memory
out=WORK/'native-host-pin-preflight';out.mkdir(exist_ok=False)
os.environ['VLLM_HUST_PHYSICAL_DEVICES']='1,6'
verify_mapping('0,1')
entries=acquire(out,5)
try:
 deployment=Path(json.loads((WORK/'b1-local-recovery.json').read_bytes())['deployment'])
 profile=json.loads((deployment/'deployment.json').read_bytes())
 env=environment(deployment,profile,out,WORK/'preflight-cache','0,1')
 check_pinned_host_memory(out,env)
 result={'completed':True,'physical_cards':[1,6],'logical_cards':[0,1],'native_host_transfer_validated':True,'full_model_qualified':False,'ended':time.time()}
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
 print('NATIVE_TWO_DEVICE_HOST_TRANSFER_PASSED',json.dumps(result),flush=True)
finally:
 release(entries)
