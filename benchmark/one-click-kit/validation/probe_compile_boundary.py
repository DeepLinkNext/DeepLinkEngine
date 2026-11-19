from pathlib import Path
import os,json,tempfile
os.environ['PYTORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import torch
from compile_boundary import Boundary
class Scope:
 def check(self): pass
class Worker:
 rank=0
 _compilation_scope=Scope()
with tempfile.TemporaryDirectory() as tmp:
 os.environ['VLLM_HUST_TEST_OUTPUT']=tmp
 boundary=Boundary(Worker())
 def fun(x):return x.sin()+1
 compiled=torch.compile(fun,backend='eager',dynamic=False)
 result=compiled(torch.ones(3));assert torch.allclose(result,torch.ones(3).sin()+1)
 before=boundary.status();assert before['observed_compilations']>0
 boundary.sealed=True
 assert torch.allclose(compiled(torch.ones(3)),result)
 blocked=False
 try:compiled(torch.ones(5))
 except Exception as error:
  blocked=boundary.failure is not None
 assert blocked,'new shape compiled after boundary'
 print('BOUNDARY_PROBE '+json.dumps({'torch':torch.__version__,'warmup_compile_observed':True,'cached_execution_after_seal':True,'new_shape_blocked':True,'no_eager_fallback':True,'events':[json.loads(l) for l in (Path(tmp)/('compilation-events-'+str(os.getpid())+'.jsonl')).read_text().splitlines()]}))
