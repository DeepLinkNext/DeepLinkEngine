from pathlib import Path
import sys,json,time
sys.dont_write_bytecode=True
BASE=Path('/root/vllm-hust-r5-refreeze-20261113')
sys.path.insert(0,str(BASE/'engine/scripts'))
from restore_b0_image import restore
from restore_b0_018 import prepare
from baseline_admission import PINNED_SOURCE_MAP_SHA256
ROOT=Path('/data/shared_upload/vllm-hust-third-party-self-run-20261112')
work=ROOT/'runtime-recovery-20261113-r1'
out=work/'b0-image-018'
restore(ROOT/'source-cache/b0-image-018',out)
data=prepare(out)
assert data['source_map_sha256']==PINNED_SOURCE_MAP_SHA256
receipt={'completed':True,'deployment':str(out),'source_map_sha256':data['source_map_sha256'],'file_count':data['file_count'],'ended':time.time(),'native_qualified':False}
(work/'b0-recovery.json').write_text(json.dumps(receipt,indent=2)+'\n');print('B0_RECOVERY_COMPLETE',json.dumps(receipt),flush=True)
