"""Recover only the exact pinned SDK into persistent task storage."""
from pathlib import Path, PurePosixPath
import hashlib,json,subprocess,sys,tarfile,time,os
ROOT=Path('/data/shared_upload/vllm-hust-third-party-self-run-20261112')
OUT=Path('/opt/vllm-hust-r5-runtime-20261113-r2/local-sdk')
SOURCE=Path('/root/test7-gdn-cache-20261018/main-adapter-cpu-r1/sdk213-r1-direct.tgz')
EXPECTED='c03e96929d7b061e5cd66aba22d02ba1cb2df270ddb7654f562f8c6202f2e805'
PREFIX='venv-school-r1/lib/python3.12/site-packages/'
def sha(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
OUT.mkdir(parents=True,exist_ok=True)
receipt=OUT/'sdk-recovery.json'
if receipt.exists():
 d=json.loads(receipt.read_bytes());assert d['completed'] and d['sdk_sha256']==EXPECTED;print('SDK_RECOVERY_ALREADY_COMPLETE',d['python'],flush=True);raise SystemExit(0)
assert sha(SOURCE)==EXPECTED,'Pinned SDK archive changed'
build=OUT/'b1-sdk'
assert not build.exists(),'Do not reuse a partial SDK; retain it and choose a new attempt'
build.mkdir()
venv=build/'venv-school-r1'
subprocess.run([sys.executable,'-m','venv','--without-pip',str(venv)],check=True)
print('SDK_ARCHIVE_VERIFIED',EXPECTED,flush=True)
count=0;protected={}
with tarfile.open(SOURCE) as archive:
 for member in archive:
  if not member.name.startswith(PREFIX):continue
  pure=PurePosixPath(member.name)
  if pure.is_absolute() or '..' in pure.parts:raise ValueError('Unsafe pinned SDK member')
  archive.extract(member,build,filter='data')
  count+=1
  if member.isfile() and member.name[len(PREFIX):].split('/')[0] in ('vllm','vllm_ascend','betterscale','ie_betterscale_bundle'):
   protected[member.name[len(PREFIX):]]=sha(build/member.name)
  if count%10000==0:print('SDK_MEMBERS_RECOVERED',count,flush=True)
print('SDK_EXTRACTED',count,'protected',len(protected),flush=True)
probe='import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ("torch","torch-npu","transformers","vllm","vllm-ascend")}))'
r=subprocess.run([str(venv/'bin/python'),'-c',probe],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,check=True,timeout=60)
metadata=json.loads(r.stdout);expected={'torch':'2.13.0+cpu','torch-npu':'2.13.0rc1','transformers':'5.14.1','vllm':'0.28.1.post1.dev143+gf18cf803c.empty','vllm-ascend':'0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27'}
assert metadata==expected,(metadata,expected)
(build/'source-manifest.json').write_text(json.dumps({'sdk_sha256':EXPECTED,'files':protected},indent=2)+'\n')
d={'completed':True,'sdk_sha256':EXPECTED,'sdk_archive':str(SOURCE),'python':str(venv/'bin/python'),'metadata':metadata,'recovered_members':count,'protected_files':len(protected),'ended':time.time(),'scope':'Exact same pinned2.13 base SDK; original Engine sources/overlay are restored and verified separately; not a native qualification'}
receipt.write_text(json.dumps(d,indent=2)+'\n');print('SDK_RECOVERY_COMPLETE',json.dumps(d),flush=True)

from shutil import copyfile
copyfile(receipt,ROOT/"runtime-recovery-20261113-r2/sdk-local-recovery.json")
