"""Resolve the exact r11 first-prefill BF16 TP2 all-reduce device clock."""

import csv
from decimal import Decimal
from pathlib import Path
import re
from statistics import median
import subprocess
from tempfile import mkdtemp

from child_runner import run_stage
from dataset_contract import save, sha
from reference_9b_archive import CARDS, PINS, member, read
from reference_9b_dense import NAMESPACE_PIN
from reference_9b_ffn_branch import R11_PREFILL


CLOCK="r11:first-prefill:1369x4096:bf16:tp2:hccl"
ROWS=1369
CASE=dict(kind="all_reduce",dtype="BFP16",elements=ROWS*4096,rank_size=2,input_bytes_per_call=ROWS*4096*2)
SETUP="6003004df8298b5fb8ca0122a414ae536d3205223be8409ed8e1d26442950665"
OLD_PANEL="qwen9b-hccl-task.QctTn6BZ"
LOG="4cfe378876cd0ea25ac0e7a412731693af91d07621b8573b02fefb39904f616c"
PLAN="recovery-logits-collective-plan-r1.json"
RESULT="recovery-logits-collective-r1"
CONTROL="recovery-logits-collective-control-r1"
PROFILE="device-task-profiles"
RAW_MANIFEST="readback-checksums.json"
PROBE="recovery_logits_collective_probe_r1.py"
TAG=re.compile(r"hcom_allReduce__(\d+)_(\d+)_(\d+)")
ORDER=[(replay,i) for replay in range(1,33) for i in range(16)]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def prove(archive, prefills, clocks):
    require(len(prefills)==3 and all(row["minimum_necessary_rows"]==row["maximum_compatible_rows"]==ROWS
                                  and row["clock_branch_proof_sha256"]==R11_PREFILL for row in prefills.values()),
            "The three native r11 first prefill operands must be proved before calibrating")
    native=archive/"native-clock/B1"
    seal=PINS["native-clock/B1/checksums.json"]
    require(clocks["B1"]["checksums_sha256"]==seal, "The original r11 walls changed")
    files=read(member(native,"checksums.json",seal))
    log=member(native,"service.log",LOG).read_text(encoding="utf8")
    require(files["service.log"]==LOG and "backend=hccl" in log and "FlashComm1 is disabled" in log
            and all(f"'{flag}': False" in log for flag in ("fuse_gemm_comms","fuse_allreduce_rms","enable_sp")),
            "Resolved r11 target reduction can use a different collective/fusion")
    return files


def cut_job(job, archive):
    prior=archive.parent/OLD_PANEL
    catalog=read(member(prior,RAW_MANIFEST,SETUP))
    plan=read(member(prior,PLAN,catalog[PLAN]))
    for name, pin in plan["source_sha256"].items():
        require(pin==catalog[name],f"Previous r11 collective source was not independently admitted: {name}")
        original=member(prior,name,pin).read_bytes()
        if name==PROBE:
            text=original.decode("utf8")
            anchor="assert [x[\"elements\"] for x in grid] == [3 * 4096, 108 * 4096]"
            require(text.count(anchor)==1,"The previous sampler changed its device boundary")
            changed=text.replace(anchor,"assert [x[\"elements\"] for x in grid] == [1369 * 4096]")
            original=changed.encode("utf8")
            compile(original,str(job/name),"exec")
        (job/name).write_bytes(original)
    plan.update(status="exact_missing_r11_first_prefill_hccl_only",shape_rows=[ROWS],calibration_grid=[CASE],
                source_sha256={name:sha(job/name) for name in plan["source_sha256"]},
                operand_proof_sha256=R11_PREFILL)
    plan["device_task_requirement"]["exact_peer_rank_rows"]=[ROWS]
    save(job/PLAN,plan)
    sdk=archive/"installation/profiles/qwen9b/sdk"
    for rel,pin in plan["native_9b_static_source_sha256"].items():
        member(sdk,rel,pin)
    return sdk


def tasks(path, rank, phase):
    records={}
    with path.open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            require(row["Device_id"]==str(CARDS[rank]),"Collective attributed to an unexpected physical device")
            if row["Type"]!="hcom_allReduce_":
                require(row["Name"]=="aclnnInplaceFillScalar_FillAiCore_Fill" and row["Accelerator Core"]!="COMMUNICATION",
                        "Unexpected non-collective kernel in the calibration profile")
                continue
            match=TAG.fullmatch(row["Name"])
            require(match is not None and row["Accelerator Core"]=="COMMUNICATION", "Invalid CANN collective tag")
            phase_id, ordinal, replay=map(int,match.groups())
            key=(replay, ordinal)
            require(key not in records,"Duplicate CANN collective tag")
            begin=Decimal(row["Start Time(us)"])
            elapsed=Decimal(row["Duration(us)"])
            require(begin.is_finite() and elapsed.is_finite() and elapsed>0,"Invalid CANN duration")
            records[key]=(begin,elapsed,phase_id)
    require(sorted(records)==ORDER and len({row[2] for row in records.values()})==1,
            "The complete 512-task collective phase was not retained")
    values=[records[key] for key in ORDER]
    require(Decimal(str(phase["start_wall"]))*10**6-200<=values[0][0]
            <sum(values[-1][:2])<=Decimal(str(phase["end_wall"]))*10**6+200
            and all(a[0]+a[1]<=b[0]+Decimal("0.01") for a,b in zip(values,values[1:])),
            "A CANN task left the qualified collective interval")
    return [row[1] for row in values]


def audit(job, raw_pin):
    catalog=read(member(job,RAW_MANIFEST,raw_pin))
    def raw(name):
        return member(job,name,catalog[name])

    raw_plan=raw(PLAN)
    plan=read(raw_plan)
    request, profile_layers=plan["device_task_requirement"], plan["device_timing_boundary"]
    require(plan["calibration_grid"]==[CASE] and request["exact_peer_rank_rows"]==[ROWS]
            and request["clock"]=="CANN task duration" and not request["use_graph_submission_gaps_as_operator_weight"]
            and plan["operand_proof_sha256"]==R11_PREFILL and profile_layers["operations_per_graph"]==16,
            "Independent r11 first prefill task plan changed")
    plan_release=read(raw("allocation/quota.json"))
    locks=raw("allocation/locks-after.txt").read_text(encoding="utf8")
    npu=raw("allocation/npu-after.txt").read_text(encoding="utf8")
    held=plan_release["allocation"]+plan_release["admission"]["exclusive_leases"]
    ctrl=read(raw(CONTROL+"/receipt.json"))
    peer=read(raw(CONTROL+"/summary.json"))
    require(plan_release["physical"]==plan_release["logical"]==CARDS and
            plan_release["admission"]["admitted"] and len(held)==18 and all(lock not in locks for lock in held)
            and all(f"No running processes found in NPU {i}" in npu for i in CARDS) and
            ctrl["physical"]==ctrl["logical"]==CARDS and ctrl["all_devices_released"]
            and ctrl["probe_exit_code"]==ctrl["cleanup_exit_code"]==0
            and all(ctrl["input_hashes"].get(name)==value==catalog[name] for name,value in plan["source_sha256"].items())
            and ctrl["input_hashes"].get(PLAN)==catalog[PLAN]
            and peer["all_devices_released"] and peer["counter_resets"]=={"4":0,"5":0},
            "The exact r11 devices, topology, numerics or release evidence failed")
    require(len(peer["shapes"])==1 and peer["shapes"][0]["counter_qualified_repetitions"]==3
            and all(peer["shapes"][0][name]==v for name,v in CASE.items()),
            "The aligned HCCS counter envelope differs for the only admitted collective")
    for name,pin in peer["source_sha256"].items():
        require(Path(name).resolve().is_relative_to(job.resolve()) and
                sha(raw(Path(name).resolve().relative_to(job.resolve()).as_posix()))==pin,
                "A source HCCS counter or phase receipt changed")
    ranks=[read(raw(f"{RESULT}/rank-{rank}.json")) for rank in (0,1)]
    compare=[]
    for rank,r in enumerate(ranks):
        require(r["rank"]==rank and r["torch_version"]=="2.13.0+cpu" and
                r["torch_npu_version"]=="2.13.0.rc1" and r["grid_sha256"]==catalog[PLAN] and len(r["phases"])==3,
                "Numerics or r11 software version changed")
    for rep in range(3):
        durations=[]
        source={}
        for rank,r in enumerate(ranks):
            phase=r["phases"][rep]
            require(phase["repetition"]==rep and phase["shape_index"]==0 and phase["rank"]==rank and
                    phase["calls"]==512 and phase["numerics_passed"] and
                    all(phase[name]==value for name,value in CASE.items()),"R11 timing phase changes an operand")
            prefix=f"{PROFILE}/rank-{rank}/rows-{ROWS}/repeat-{rep}/"
            matched=[name for name in catalog if name.startswith(prefix) and name.endswith("/kernel_details.csv")]
            require(len(matched)==1,"R11 device task profile is missing")
            durations.append(tasks(raw(matched[0]),rank,phase))
            source[str(rank)]=catalog[matched[0]]
        pair=median(min(a,b) for a,b in zip(*durations,strict=True))/10**6
        compare.append(dict(repetition=rep,paired_tasks=512,seconds_median_lower=str(pair),task_trace_sha256=source))
    return dict(case_id=CLOCK,source_root=str(job),raw_manifest_sha256=raw_pin,operator=CASE,
        device_task_seconds_lower=str(median(Decimal(row["seconds_median_lower"]) for row in compare)),
        paired_clock_results=compare,physical_cards=CARDS,lease_paths_verified_released=len(held),
        allocation_sha256={name:catalog["allocation/"+name] for name in ("quota.json","locks-after.txt","npu-after.txt")})


def measure(archive, baseline, folder, clocks, files):
    owner=Path(clocks["B0"]["archive"]).parents[1]
    namespace=member(owner,"namespace-9b.sh",NAMESPACE_PIN)
    native=archive/"native-clock/B1"
    python=read(member(native,"launch-command.json",files["launch-command.json"]))["command"][0]
    job=Path(mkdtemp(prefix="r11-prefill-hccl-auto.",dir=archive.parent))
    sdk=cut_job(job,archive)
    command=["unshare","-m","--propagation","private","bash",str(namespace),str(job),str(baseline.parents[1]),
             python,"-B",str(job/"run_9b_collective_panel.py"),str(job),str(owner),str(sdk)]
    save(folder/"r11-prefill-tp2-dispatch.json",dict(case_id=CLOCK,command=command,known_native_proof_sha256=R11_PREFILL,
        new_model_or_clean_wall_requested=False))
    with (folder/"r11-prefill-tp2-sampler.log").open("x",encoding="utf8") as log:
        process=run_stage(command,stdout=log,stderr=subprocess.STDOUT)
    require(process.returncode==0,f"The single r11 HCCL device-clock job failed at {job}, exit {process.returncode}")
    plan=read(job/PLAN)
    files=sorted(set([job/PLAN]+[job/name for name in plan["source_sha256"]]+list((job/"allocation").glob("*.json"))
        +list((job/"allocation").glob("*.txt"))+list((job/PROFILE).rglob("kernel_details.csv"))
        +[file for root in (job/RESULT,job/CONTROL) for file in root.iterdir() if file.is_file()]))
    save(job/RAW_MANIFEST,{path.relative_to(job).as_posix():sha(path) for path in files})
    return audit(job,sha(job/RAW_MANIFEST))


def resolve(archive, baseline, folder, clocks, prefills, cache, cache_pin):
    files=prove(archive,prefills,clocks)
    index=read(member(cache,"r11-prefill-tp2-index.json",cache_pin))
    require(index["schema"]=="m01-9b-r11-first-prefill-hccl-cache/v1"
            and index["operand_proof_sha256"]==R11_PREFILL and set(index["entries"])<={CLOCK},
            "The isolated first-prefill collective clock cache changed")
    selected=index["entries"].get(CLOCK)
    if selected is None:
        clock=measure(archive,baseline,folder,clocks,files)
    else:
        clock=audit(Path(selected["source_root"]),selected["raw_manifest_sha256"])
    save(folder/"r11-prefill-tp2-clock.json",clock)
    return clock,dict(required_case=CLOCK,case_total=1,cache_seal_sha256=cache_pin,
        retained=int(selected is not None),sampled=int(selected is None),sampler_invocations=int(selected is None),
        raw_manifest_sha256=clock["raw_manifest_sha256"],archive_receipt=clock)
