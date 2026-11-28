"""Resolve measured device clocks for the exact original r11 first target KV graph."""
import csv
from decimal import Decimal
from pathlib import Path
import shutil
from statistics import median
import subprocess
from tempfile import mkdtemp

from child_runner import run_stage
from dataset_contract import save, sha
from reference_9b_archive import PINS, member, read
from reference_9b_target_fia import BUNDLE, release, require, task_clock


OPERANDS="r11-cold-prefill-target-kv-operands.json"
DEMAND="cccea458cafe9f8f4e211c3d2b7810427256645e5f4eaafb5c9e912bb21a6979"
SAMPLER="reference_9b_r11_prefill_kv_sampler.py"
SAMPLER_SHA256="6bdd2813029a2b883de0cf700f6736f3dca13b321dba5e938fd7b3937f8f90f3"
SUPPORT=("calibrate_r11_first_target_fia.py","calibrate_r11_native_kv_task.py","audit_r11_first_target_fia.py")
INDEX="r11-prefill-kv-cache-index.json"
VERSIONS={"torch":"2.13.0+cpu","torch-npu":"2.13.0rc1"}


def scatter_clock(path,card):
    with path.open(newline="",encoding="utf-8-sig") as stream:
        rows=list(csv.DictReader(stream))
    kernel="aclnnScatterPaKvCache_ScatterPaKvCache_ScatterPaKvCache"
    require(len(rows)==230 and all(row["Name"]==kernel and int(row["Device_id"])==card for row in rows)
            and len({row["Block Num"] for row in rows})==1
            and int(rows[0]["Block Num"])>0,"Cold scatter capture has an unexpected kernel or tiling")
    tasks=sorted((Decimal(r["Start Time(us)"]),Decimal(r["Duration(us)"])) for r in rows)
    require(all(start.is_finite() and amount.is_finite() and amount>0 for start,amount in tasks)
            and all(a+d<=b for (a,d),(b,_) in zip(tasks,tasks[1:])),
            "The independent cold scatter capture overlaps or has invalid timing")
    samples=[median(d for _,d in tasks[i+3:i+23]) for i in range(0,230,23)]
    return str(median(samples)/10**6)


def cases_for(plan,clocks):
    require(clocks["B1"]["checksums_sha256"]==plan["original_native_sha256"]==PINS["native-clock/B1/checksums.json"]
            and plan["schema"]=="m01-9b-r11-original-cold-target-kv-and-fia-operands/v1"
            and plan["physical_cards"]==[4,5],"Cold KV operations came from a different deployment")
    for clock in clocks["B1"]["rounds"]:
        round_plan=plan["rounds"][str(clock["repetition"])]
        require(round_plan["original_timing_sha256"]==clock["source_sha256"],
                "Cold KV and native wall refer to different repetitions")
    cases=plan["missing_independent_clocks"]
    allowed={f"prefill-{kind}:{rank}:{rep}" for rank in (0,1) for rep in (1,2,3) for kind in ("scatter","fia")}
    require(len(cases)==len(allowed) and {i["id"] for i in cases}==allowed,
            "Cold target KV case domain changed")
    return {case["id"]:case for case in cases}


def one_clock(job, manifest, plan, case):
    request=read(member(job,"request.json",manifest["request.json"]))
    require(request["operands_sha256"]==DEMAND and case["id"] in request["case_ids"] and
            manifest[OPERANDS]==DEMAND and manifest[SAMPLER]==SAMPLER_SHA256,
            "The sealed device job does not bind the required original cold KV operand")
    for name in (*SUPPORT,OPERANDS,SAMPLER):
        member(job,name,manifest[name])
        if name in SUPPORT:
            require(manifest[name]==BUNDLE[name],"The shared r11 calibration helper is outside its sealed ABI")
    for name in ("quota.json","npu-after.txt","locks-after.txt"):
        key="allocation/"+name
        member(job,key,manifest[key])
    released=release(job)
    slug=case["id"].replace(":","-")
    numeric=slug+"/numeric.json"
    result=read(member(job,numeric,manifest[numeric]))
    require(result["case"]==case and result["operands_sha256"]==DEMAND and result["sampler_sha256"]==SAMPLER_SHA256
            and result["runtime"]==VERSIONS and result["numerical_passed"] and not result["model_inference"],
            f"Numerical, origin or r11 runtime witness changed for {case['id']}")
    prefix=slug+"/"
    csvs=[name for name in manifest if name.startswith(prefix) and name.endswith("/kernel_details.csv")]
    require(len(csvs)==1,"Cold KV device task trace is absent or ambiguous")
    path=member(job,csvs[0],manifest[csvs[0]])
    current=plan["rounds"][str(case["repetition"])]
    if case["operator"]=="fia":
        route=plan["rank_local_fia"]
        require(result["pages"]==current["page_prefix"] and result["query_shape"]==route["query_shape"]
                and result["cache_shape"]==route["cache_plane_shape"] and result["table_shape"]==route["block_table_shape"]
                and result["kv_length"]==1369, "The cold FIA task used a different page or query")
        seconds=task_clock(path,case["physical_card"])
    else:
        route=plan["rank_local_scatter"]
        require(result["slot_mapping"]==current["slot_mapping"] and result["key_shape"]==route["key_shape"]
                and result["cache_plane_shape"]==route["cache_shape"] and result["cache_plane_stride"]==route["cache_stride"]
                and result["fenced_generations"]==2, "The cold scatter task used another slot or tensor layout")
        seconds=scatter_clock(path,case["physical_card"])
    return dict(seconds=seconds,task_sha256=manifest[csvs[0]],numeric_sha256=manifest[numeric],
                original_wall_sha256=current["original_timing_sha256"],operand=case,
                source_job=str(job),source_trace=str(path),release_sha256=released)


def calibrate(archive, output, source_proof, target_cache, needed, plan):
    job=Path(mkdtemp(prefix="r11-prefill-kv-auto.",dir=archive.parent))
    helpers={name:member(target_cache,"sampler/"+name,BUNDLE[name]) for name in SUPPORT}
    helpers[SAMPLER]=member(Path(__file__).parent,SAMPLER,SAMPLER_SHA256)
    helpers[OPERANDS]=source_proof
    for name,path in helpers.items():
        shutil.copyfile(path,job/name)
    save(job/"request.json",dict(operands_sha256=DEMAND,case_ids=list(needed)))
    donor=archive.parent/("r10-kernel-native-10204-20261124.6ke9QK/runtime/clean-cache-v2/"
              "environments/dc75dd91be2a521c-9ebc8e5cb831/venv/bin/python")
    command=[str(donor),"-B",str(job/SAMPLER),str(job),"launch"]
    save(output/"r11-prefill-kv-dispatch.json",dict(command=command,case_ids=list(needed),job=str(job),inference=False))
    with (output/"r11-prefill-kv-sampler.log").open("x") as log:
        child=run_stage(command,stdout=log,stderr=subprocess.STDOUT)
    require(child.returncode==0,f"Independent r11 cold target KV sampler failed at {job}, exit {child.returncode}")
    require({p.parent.name for p in job.glob("prefill-*/numeric.json")}=={name.replace(":","-") for name in needed},
            "Cold target KV sampler touched cases outside its exact scope")
    names=[*helpers,"request.json"]+["allocation/"+name for name in ("quota.json","npu-after.txt","locks-after.txt")]
    names+=sorted(path.relative_to(job).as_posix() for pattern in ("prefill-*/numeric.json","prefill-*/**/kernel_details.csv")
                  for path in job.glob(pattern))
    manifest={name:sha(job/name) for name in names}
    save(job/"readback-checksums.json",manifest)
    clocks={name:one_clock(job,manifest,plan,case) for name,case in needed.items()}
    return clocks,dict(source_root=str(job),raw_manifest_sha256=sha(job/"readback-checksums.json"))


def resolve(archive, output, cache, cache_pin, target_cache, clocks):
    sealed=read(member(cache,"checksums.json",cache_pin))
    require(sealed[OPERANDS]==DEMAND,"Cold KV cache did not retain its original operand proof")
    proof=member(cache,OPERANDS,DEMAND)
    plan=read(proof)
    cases=cases_for(plan,clocks)
    index=read(member(cache,INDEX,sealed[INDEX]))
    entries=index["available"]
    require(index["schema"]=="m01-r11-prefill-kv-cache/v1" and index["operands_sha256"]==DEMAND
            and set(entries)<=set(cases),"An isolated cold KV clock cache offers a foreign operand")
    retained={}
    for name,entry in entries.items():
        job=Path(entry["source_root"])
        manifest=read(member(job,"readback-checksums.json",entry["raw_manifest_sha256"]))
        retained[name]=one_clock(job,manifest,plan,cases[name])
    needed={name:case for name,case in cases.items() if name not in retained}
    result=dict(operands_sha256=DEMAND,cache_sha256=cache_pin,retained=len(retained),
                sampled=len(needed),sampler_invocations=int(bool(needed)),exact_missing_cases=list(needed))
    if needed:
        fresh,admission=calibrate(archive,output,proof,target_cache,needed,plan)
        result["device_run"]=admission
        retained.update(fresh)
    save(output/"r11-prefill-kv-clocks.json",dict(**result,clocks=retained))
    return retained,result
