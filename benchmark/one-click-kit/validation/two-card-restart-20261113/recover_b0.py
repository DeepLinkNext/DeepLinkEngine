from pathlib import Path
import hashlib,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
ROOT=Path('/data/shared_upload/vllm-hust-third-party-self-run-20261112')
WORK=ROOT/'runtime-recovery-20261113-r2'
EXEC=ROOT/'executors/engine-d0412c2ee52c0e0c5790ec55fbbee12b5bd51238'
sys.path.insert(0,str(EXEC/'scripts'))
from restore_b0_image import restore
from restore_b0_018 import prepare
from baseline_admission import PINNED_SOURCE_MAP_SHA256
out=Path('/opt/vllm-hust-r5-runtime-20261113-r2/b0-image-018')
start=time.time();restore(ROOT/'source-cache/b0-image-018',out)
print('LOCAL_EXTRACTION_FINISHED',time.time()-start,flush=True)
data=prepare(out)
assert data['source_map_sha256']==PINNED_SOURCE_MAP_SHA256
print('SOURCE_MAP_VERIFIED',data['source_map_sha256'],data['file_count'],time.time()-start,flush=True)
archive=WORK/'b0-fixed018-verified-sdk.tar'
partial=archive.with_suffix('.tar.partial')
subprocess.run(['tar','-cf',str(partial),'-C',str(out.parent),out.name],check=True)
with partial.open('rb') as stream:
 digest=hashlib.file_digest(stream,'sha256').hexdigest()
 os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
partial.replace(archive)
receipt={'completed':True,'deployment':str(out),'source_map_sha256':data['source_map_sha256'],'file_count':data['file_count'],'ended':time.time(),'restore_seconds':time.time()-start,'native_qualified':False,'persistent_archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest}}
temp=WORK/'b0-recovery.json.partial';temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(WORK/'b0-recovery.json')
print('B0_RECOVERED_AND_ARCHIVED',json.dumps(receipt),flush=True)
