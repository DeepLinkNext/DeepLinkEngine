from pathlib import Path
import json,sys,time,subprocess,hashlib,os
sys.dont_write_bytecode=True
ROOT=Path('/data/shared_upload/vllm-hust-third-party-self-run-20261112')
WORK=ROOT/'runtime-recovery-20261113-r2'
EXEC=ROOT/'executors/engine-d0412c2ee52c0e0c5790ec55fbbee12b5bd51238'
sys.path.insert(0,str(EXEC/'scripts'))
from restore_b1_r4 import restore,verify_installed_sources
from runtime_overlay import apply
from dependency_admission import admit_dependencies
sdk=json.loads((ROOT/'runtime-recovery-20261113-r1/sdk-recovery.json').read_bytes());assert sdk['completed']
out=Path('/opt/vllm-hust-r5-runtime-20261113-r2/b1-r5')
started=time.time()
restore(ROOT/'b1-original-assets',out,Path(sdk['python']),Path('/root/.env'))
apply(out)
files=verify_installed_sources(out);metadata=admit_dependencies(out)
archive=WORK/'b1-r5-verified-deployment.tar';partial=archive.with_suffix('.tar.partial')
subprocess.run(['tar','-cf',str(partial),'-C',str(out.parent),out.name],check=True)
with partial.open('rb') as source:
 digest=hashlib.file_digest(source,'sha256').hexdigest();os.posix_fadvise(source.fileno(),0,0,os.POSIX_FADV_DONTNEED)
partial.replace(archive)
receipt={'completed':True,'deployment':str(out),'files_verified':len(files),'metadata':metadata,'restore_seconds':time.time()-started,'ended':time.time(),'native_qualified':False,'archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest}}
(WORK/'b1-local-recovery.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('B1_LOCAL_RECOVERED_AND_ARCHIVED',json.dumps(receipt),flush=True)
