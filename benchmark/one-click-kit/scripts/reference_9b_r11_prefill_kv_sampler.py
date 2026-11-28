"""Calibrate exact missing r11 cold target KV operations on the admitted TP2 devices."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys

from calibrate_r11_first_target_fia import CONTROLS as FIA_CONTROLS, sample_case as existing_fia
from calibrate_r11_native_kv_task import (
    BASE, DONOR, NAMESPACE, NAMESPACE_SHA256, ORIGINAL, PYTHON, WRITER, WRITER_SHA256,
    load_writer, read, require, save, sha,
)


DEMAND="cccea458cafe9f8f4e211c3d2b7810427256645e5f4eaafb5c9e912bb21a6979"
OPERANDS="r11-cold-prefill-target-kv-operands.json"
VERSIONS={"torch":"2.13.0+cpu","torch-npu":"2.13.0rc1"}
ROWS=1369
REPETITIONS=10
WARMUPS=3
CALLS=20


def selection(root):
    file=root/OPERANDS
    required=read(root/"request.json")
    require(sha(file)==required["operands_sha256"]==DEMAND,"The exact cold target proof changed")
    plan=read(file)
    allowed={c["id"]:c for c in plan["missing_independent_clocks"]}
    names=required["case_ids"]
    require(isinstance(names,list) and names and len(set(names))==len(names) and set(names)<=set(allowed),
            "The device sampler was offered a foreign cold target KV operation")
    return plan,[allowed[key] for key in names]


def scatter(torch,npu,root,plan,case,pool):
    device=pool[0].device
    slots=torch.tensor(case["slot_mapping"],dtype=torch.int64,device=device)
    writer=load_writer(npu)
    reader=torch.npu.Stream(device=device)
    cpu=(torch.arange(ROWS)%97).to(torch.bfloat16).reshape(-1,1,1).expand(-1,2,256).contiguous()
    for tag in (24,0):
        want=cpu+tag
        key,value=want.to(device),(-want-3).to(device)
        def call():
            writer(None,key=key,value=value,key_cache=pool[0],value_cache=pool[1],slot_mapping=slots)
        call()
        ready=torch.npu.Event()
        ready.record(torch.npu.current_stream(device))
        with torch.npu.stream(reader):
            reader.wait_event(ready)
            for plane,expected in zip(pool,(want,-want-3),strict=True):
                found=plane.view(-1,2,256).index_select(0,slots).cpu()
                torch.testing.assert_close(found,expected,atol=0,rtol=0)
    torch.npu.synchronize(device)
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    graph.replay()
    torch.npu.synchronize(device)
    exp=npu.profiler._ExperimentalConfig(profiler_level=npu.profiler.ProfilerLevel.Level0)
    with npu.profiler.profile(
      activities=[npu.profiler.ProfilerActivity.NPU],
      schedule=npu.profiler.schedule(wait=0,warmup=0,active=1,repeat=1),
      on_trace_ready=npu.profiler.tensorboard_trace_handler(str(root/"scatter-profile")),
      experimental_config=exp) as recorder:
        for _ in range(REPETITIONS):
            for _ in range(WARMUPS):
                graph.replay()
            torch.npu.synchronize(device)
            for _ in range(CALLS):
                graph.replay()
            torch.npu.synchronize(device)
        recorder.step()
    return dict(numerical_passed=True,fenced_generations=2,key_shape=list(key.shape),
                cache_plane_shape=list(pool[0].shape),cache_plane_stride=list(pool[0].stride()),
                slot_mapping=case["slot_mapping"])


def fia(torch,npu,root,plan,case,pool):
    device=pool[0].device
    pages=case["page_prefix"]
    random=torch.Generator().manual_seed(FIA_CONTROLS["seed"])
    def normal(shape):
        return (torch.randn(shape,generator=random)*0.15).bfloat16()
    query=normal((ROWS,8,256))
    key,value=[normal((len(pages),128,512)) for _ in range(2)]
    full_k,full_v=[x.reshape(-1,2,256)[:ROWS].transpose(0,1).float().unsqueeze(0).repeat_interleave(4,1)
                   for x in (key,value)]
    result=torch.nn.functional.scaled_dot_product_attention(query.transpose(0,1).float().unsqueeze(0),
        full_k,full_v,dropout_p=0.0,scale=plan["scale"],is_causal=True)[0].transpose(0,1).contiguous()
    mask=torch.triu(torch.ones(plan["mask_shape"]),diagonal=1).to(torch.int8).to(device)
    inputs=(query.to(device),pool[0].view(plan["cache_plane_shape"]),
            pool[1].view(plan["cache_plane_shape"]),key,value,result,mask)
    recorded=existing_fia(torch,npu,root,plan,case["rank"],dict(repetition=case["repetition"],
        cached_blocks_observed_prefix=pages,page_operand_read_prefix=pages),inputs)
    for name in ("cpu_fp32_samples","device_bf16_samples"):
        recorded[name]=recorded[name][::29]
    return recorded


def sample(root,rank):
    import torch
    import torch_npu

    plan,selected=selection(root)
    require({name:version(name) for name in VERSIONS}==VERSIONS and
            os.environ["ASCEND_HOME_PATH"].endswith("/cann-9.1.0") and sha(WRITER)==WRITER_SHA256,
            "Torch, CANN or the canonical cold target writer is outside r11")
    torch.set_num_threads(4)
    torch.npu.set_device(rank)
    device=torch.device("npu",rank)
    cases=[case for case in selected if case["rank"]==rank]
    require(cases,"A device task rank has no actual cache miss")
    with torch.inference_mode():
        pool=[torch.empty(plan["rank_local_scatter"]["cache_shape"],dtype=torch.bfloat16,device=device) for _ in range(2)]
        for case in cases:
            current=case|plan["rounds"][str(case["repetition"])]
            path=root/case["id"].replace(":","-")
            path.mkdir()
            if case["operator"]=="fia":
                witness=fia(torch,torch_npu,path,plan["rank_local_fia"],current,pool)
            else:
                witness=scatter(torch,torch_npu,path,plan["rank_local_scatter"],current,pool)
            save(path/"numeric.json",dict(operands_sha256=DEMAND,case=case,runtime=VERSIONS,
                sampler_sha256=sha(Path(__file__)),model_inference=False,**witness))


def guard(root):
    sys.path.insert(0,str(ORIGINAL))
    from run_revised_context import prepare
    from candidate_campaign import resource_guard, screen_campaign

    _,missing=selection(root)
    config,*_=prepare()
    allocation=root/"allocation"
    allocation.mkdir()
    with screen_campaign.reserve(config,allocation,resource_guard,expected_pair=[4,5]) as (quota,fds):
        env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=",".join(map(str,quota["logical"])),
                 TORCH_DEVICE_BACKEND_AUTOLOAD="0",PYTHONDONTWRITEBYTECODE="1")
        for rank in sorted({row["rank"] for row in missing}):
            subprocess.run([sys.executable,"-B",__file__,str(root),"rank",str(rank)],cwd=root,env=env,
                           pass_fds=fds,timeout=720,check=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("mode",choices=("launch","guard","rank"))
    parser.add_argument("rank",nargs="?",type=int,choices=(0,1))
    args=parser.parse_args()
    root=args.root.resolve()
    if args.mode=="launch":
        selection(root)
        require(root.parent==BASE and sha(NAMESPACE)==NAMESPACE_SHA256,"The frozen r11 operator namespace changed")
        command=["unshare","-m","--propagation","private","bash",str(NAMESPACE),str(root),str(DONOR),
                 str(PYTHON),"-B",__file__,str(root),"guard"]
        os.execvp(command[0],command)
    elif args.mode=="guard":
        guard(root)
    else:
        sample(root,args.rank)
