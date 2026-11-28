"""Qualify the 9B reference-time lower bound separately from an exact completion time."""

from collections import Counter
from decimal import Decimal, ROUND_FLOOR, localcontext
from pathlib import Path
from statistics import median

from dataset_contract import save, sha
from reference_9b_archive import CARDS, CHECKPOINT_SEAL, MODEL, RELEASE, REVISION, WORKLOAD_SEAL, member, read
from reference_9b_projection import DIMS, DRAFT_PROJECTION_SHA256, FFN, GDN_ARITHMETIC, collect_ffn_times, required


METHOD_SHA = "615e1dd9cdab681a83775736d2d5ae2eff4bb3aacde445370f23826df8a21db4"
PANELS = {
    "b0-host-row-clocks.json": "50f2cceecd9409c1b518190181f939626d2c616e4c80cffcc2d780cdd71b1348",
    "r11-host-row-clocks.json": "f4747eba79870393af6aecbe5f54c416469d35f58c62da0bee7c181ddeac6bb9",
    "b0-slot-mapping-h2d-clocks-readback.json": "c0e5f3ef6f10683cf786a3ce46511bf17d60880444500577044d09273c0258c8",
    "b0-first-prefill-tp2-clocks-readback.json": "6a73ae000be5178041b3b0b1131b530bb43b601b02b9efc6606ecfd0c8bdb033",
    "b0-tp2-task-clocks-readback.json": "b0326357257da036201612f360d07cdf0b9487a8edb63f7b939aea7cc25b5499",
    "r11-hccl-task-clocks-readback.json": "140b3fe3ff360bcb0755cb010bd02d5038d5f9eaaf88586315946e99236fcb0d",
    "r11-native-kv-task-clocks-readback.json": "555ebfbf55a2a678ce14b15a34bc47fae4d261a3d806f72c7ec577c1a672ebd9",
}
ACTIVE = {
    "target-attention-direct-prefill-fia": ("qualified-b0-direct-fia-clocks.json", "cases", "B0"),
    "target-attention-fia-read": ("qualified-target-fia-clocks.json", "cases", "B1"),
    "original018-cold-kv-scatter": ("b0-cold-kv-clocks.json", "clocks", "B0"),
    "r11-cold-gdn-convolution": ("r11-cold-gdn-conv-clocks.json", "clocks", "B1"),
    "target-attention-cold-prefill-scatter": ("r11-prefill-kv-clocks.json", "clocks", "B1"),
    "target-attention-cold-prefill-fia": ("r11-prefill-kv-clocks.json", "clocks", "B1"),
    "r11-retained-draft-phase0-fia": ("qualified-first-draft-clocks.json", "cases", "B1"),
    "r11-retained-draft-phase1-kv-write": ("qualified-first-draft-clocks.json", "cases", "B1"),
    "r11-retained-draft-phase1-fia": ("qualified-first-draft-clocks.json", "cases", "B1"),
}
COMPUTE = FFN | {
    "target-attention-direct-prefill-fia", "target-attention-fia-read", "r11-cold-gdn-convolution",
    "target-attention-cold-prefill-fia", "r11-retained-draft-phase0-fia", "r11-retained-draft-phase1-fia",
}
KV = {
    "original018-cold-kv-scatter", "target-attention-cold-prefill-scatter", "target-attention-kv-write",
    "r11-retained-draft-phase0-kv-write", "r11-retained-draft-phase1-kv-write",
}


def evidence_file(path, digest):
    required(sha(path) == digest, f"Reference evidence SHA-256 differs for {path.name}")
    return dict(path=str(path), sha256=digest)


def original_binding(archived, role):
    root = Path(archived["archive"])
    seal = read(member(root, "checksums.json", archived["checksums_sha256"]))
    plan = read(member(root, "native-plan.json", seal["native-plan.json"]))
    extra = "frozen-original018-argv.json" if role == "B0" else "r11-context-source.json"
    files = {name:evidence_file(root / name, seal[name]) for name in
             ("native-plan.json", "client/effective-engine-args.json", "launch-command.json",
              "service.log", "cleanup.json", extra)}
    if role == "B0":
        origin = Path(plan["baseline_source_admission"])
        source = evidence_file(origin, "5533d8610d69e74311a5db5459ff665cdb56b54824d2059fc7ef6c91ad0b4f71")
    else:
        origin = Path(plan["deployment"]) / "provenance/installed-files.json"
        source = evidence_file(origin, "c42219d14540035cd57fb553b183e7e1689e0e89ad4952b71783551f21bc13b8")
        source["commit"] = plan["candidate_source_commit"]
    return dict(archive_manifest=evidence_file(root / "checksums.json", archived["checksums_sha256"]),
                native_files=files, source=source, versions=plan["expected_versions"], capture_profile=plan["reference_capture_profile"],
                checkpoint=evidence_file(Path(plan["checkpoint_receipt"]), CHECKPOINT_SEAL),
                ordered_input=evidence_file(Path(plan["workload"]), WORKLOAD_SEAL),
                physical_devices=CARDS, runtime="original018" if role == "B0" else RELEASE,
                runtime_geometry=plan["engine_geometry"]), seal


def host_lookup(panel, role):
    controls = panel["controls"]
    required({k:controls[k] for k in ("samples", "retained_prefix_samples", "calls_per_sample", "numpy")}
             == dict(samples=10, retained_prefix_samples=7, calls_per_sample=32, numpy="1.26.4")
             and panel["no_model_inference_or_npu_lease"], f"{role} host clock qualification differs")
    admitted = {}
    for case in panel["cases"]:
        samples = case["samples_ns_per_32"]
        required(len(samples) == 10 and all(type(v) is int and v > 0 for v in samples), "Invalid independent host samples")
        per_call = Decimal(median(samples[:7])) / 32
        required(per_call == Decimal(str(case["median_ns_per_call"])), "Host median differs from the retained sample panel")
        for row in case["native_batches" if role == "B0" else "exact_native_frames"]:
            key = row["repetition"], row["public_packet"]
            required(key not in admitted, "Ambiguous native host case")
            admitted[key] = (row["raw_frames_sha256"], per_call / 10**9)
    return admitted


def collective_lookup(panel, role):
    available = {}
    for case in panel["cases"]:
        elements = case["bf16_elements_per_rank"] if role == "B1" else case["size_by_rank"]
        values = [Decimal(row["seconds_median_lower" if role == "B1" else "qualified_device_seconds_lower"])
                  for row in case["repetitions"]]
        wanted = Decimal(case["qualified_device_seconds_lower"])
        required(median(values) == wanted and wanted > 0, "TP2 lower statistic differs from the paired calibration")
        available[elements] = wanted
    packets = {}
    source = panel["native_qualified_packets"] if role == "B0" else [
        {"repetition":row["repetition"], "frame":case["frame"],
         "bf16_elements_per_rank":case["candidate_payload_elements_per_rank"]}
        for row in panel["original_repetition_shape_obligations"] for case in row["exact_row_frames"]
    ]
    for record in source:
        key = record["repetition"], record["frame"]
        size = record["bf16_elements_per_rank"]
        if "actual_retained_rows" in record and record["actual_retained_rows"] * 4096 != size:
            continue
        packets[key] = (size, available[size], record.get("raw_frames_sha256"))
    return packets


def child_case(kind, node, repetition):
    rank = node["rank"]
    if kind == "target-attention-direct-prefill-fia":
        return f"rank-{rank}"
    if kind == "target-attention-fia-read":
        return f"rank-{rank}/round-{repetition}"
    if kind == "original018-cold-kv-scatter":
        return f"b0-cold-kv:{rank}"
    if kind == "r11-cold-gdn-convolution":
        return f"r11-cold-gdn-conv:{rank}"
    return node["case_id"]


def verify_nodes(role, repetition, window, row, nodes, graph_row, ffn, witnesses):
    host, shared, dma, legacy_kv, active, cold_tp2 = witnesses
    calibration = set()
    inventory = Counter()
    critical = Counter()
    finishes = []
    seen_hccl = set()
    chain = graph_row["critical_path"]
    chain_set = set(chain)
    for index,node in enumerate(nodes):
        predecessors = node["predecessors"]
        kind = node["kind"]
        rank, packet = node["rank"], node["packet"]
        sample = Decimal(node["seconds"])
        required(sample.is_finite() and sample > 0 and len(predecessors) == len(set(predecessors))
                 and all(type(parent) is int and 0 <= parent < index for parent in predecessors)
                 and rank in (0,1,None), "Reference duration or topological order differs")
        if kind in FFN:
            candidates = node["measured_candidate_rows"]
            required(candidates == list(range(min(candidates),max(candidates)+1)), "Uncalibrated FFN branch inside the admissible width range")
            value, selected = min((sum(ffn[rank,m,n,k] for n,k in DIMS),m) for m in candidates)
            required(sample == node.get("layers",1)*value and node["lowest_qualified_canonical_rows"] == selected,
                     "FFN duration is not the minimum over its independently measured compatible cases")
        elif kind == "host-index-preparation":
            digest,expected = host[role][repetition,packet]
            required(sample == expected and digest == window["public_frames_sha256"], "Host clock has another native input width")
        elif kind == "tp2-allreduce":
            require_key = packet,node["layer"]
            required(require_key not in seen_hccl and len(predecessors) == 2
                     and {nodes[p]["rank"] for p in predecessors} == {0,1}
                     and all(nodes[p]["kind"] == "target-ffn-matmuls" and nodes[p]["layer"] == node["layer"]
                             and nodes[p]["packet"] == packet for p in predecessors),
                     "TP2 operation must join the real two device paths once")
            seen_hccl.add(require_key)
            if role == "B1" and packet == 0:
                size, expected, digest = cold_tp2["operator"]["elements"], Decimal(cold_tp2["device_task_seconds_lower"]), None
                pin = cold_tp2["raw_manifest_sha256"]
            else:
                (size, expected, digest), pin = shared[role][repetition,packet]
            required(size == node["bf16_elements_per_rank"] and sample == expected
                     and node["source_clock_sha256"] == pin and (digest is None or digest == window["public_frames_sha256"]),
                     "TP2 clock is outside its precise native payload")
            calibration.add(pin)
        elif kind == "full-attention-kv-slot-index-h2d":
            source, timing, pin = dma[repetition,packet,rank]
            required(role == "B0" and source["source_sha256"] == window["public_frames_sha256"]
                     and node["bytes_per_rank"] == source["bytes_per_rank"]
                     and sample == timing and node["source_task_sha256"] == pin and len(predecessors) == 1
                     and nodes[predecessors[0]]["kind"] == "host-index-preparation"
                     and nodes[predecessors[0]]["packet"] == packet, "Slot-index H2D clock or parent changed")
            calibration.add(node["source_task_sha256"])
        elif kind in {"target-attention-kv-write", "r11-retained-draft-phase0-kv-write"}:
            expected = legacy_kv[repetition,rank]
            required(role == "B1" and packet == 1 and node["physical_slots"] == expected["slot_mapping"]
                     and sample == Decimal(expected["seconds"])
                     and node["source_device_task_sha256"] == expected["task_sha256"], "A retained scatter was mispaired")
            calibration.add(expected["task_sha256"])
        elif kind in ACTIVE:
            _,_,owner = ACTIVE[kind]
            source = active[kind][child_case(kind,node,repetition)]
            required(role == owner and sample == Decimal(source["seconds"])
                     and node["calibration_clock_sha256"] == source["task_sha256"],
                     "Independent exact operator clock is bound to a different rank, role or operand")
            calibration.add(source["task_sha256"])
        else:
            raise ValueError(f"Unqualified reference task kind: {kind}")

        family = ("compute" if kind in COMPUTE else "kv_tensor_movement" if kind in KV else
                  {"host-index-preparation":"host_processing", "tp2-allreduce":"tp2_communication",
                   "full-attention-kv-slot-index-h2d":"kv_metadata_movement"}[kind])
        inventory[family] += 1
        if index in chain_set:
            critical[family] += 1
        finishes.append(sample + max((finishes[parent] for parent in predecessors),default=Decimal(0)))

    required({"compute","kv_tensor_movement","host_processing","tp2_communication"} <= inventory.keys(),
             f"{role} round {repetition} lower proof is missing a required reference family")
    required(chain and len(chain) == len(chain_set) and all(0 <= idx < len(nodes) for idx in chain)
             and all(a in nodes[b]["predecessors"] for a,b in zip(chain,chain[1:]))
             and max(finishes) == Decimal(graph_row["seconds_lower"]) == Decimal(row["projected_seconds_lower"])
             == sum(Decimal(nodes[i]["seconds"]) for i in chain),
             "The independent longest-path recomputation differs")
    return dict(node_families=dict(inventory), critical_path_families=dict(critical),
                independently_admitted_device_clock_sha256=sorted(calibration), critical_path=chain)


def qualify(folder, evidence, reference):
    projection = reference["component_projection"]
    required(projection["source_sha256"] == DRAFT_PROJECTION_SHA256
             and projection["target_gdn_arithmetic_proof_sha256"] == GDN_ARITHMETIC,
             "The original compatibility proof set is required for the formal lower bound")
    graph = read(member(folder, projection["replay_file"], projection["replay_sha256"]))
    required(set(graph["arms"]) == set(reference["native_target_kv_logical_evidence"]) == {"B0","B1"},
             "Both original arms and all logical dependency ledgers are required")
    method = Path(__file__).resolve().parents[1] / "protocols/m01-confirmed-reference-method.md"
    files = {name:member(evidence,name,pin) for name,pin in PANELS.items()}
    data = {name:read(path) for name,path in files.items()}
    host = {role:host_lookup(data[name],role) for role,name in
            (("B0","b0-host-row-clocks.json"),("B1","r11-host-row-clocks.json"))}
    shared = {"B0":{}, "B1":{}}
    for name,role in (("b0-first-prefill-tp2-clocks-readback.json","B0"),
                      ("b0-tp2-task-clocks-readback.json","B0"),("r11-hccl-task-clocks-readback.json","B1")):
        for key,value in collective_lookup(data[name],role).items():
            required(key not in shared[role], "Two collective calibrations for one original packet")
            shared[role][key] = value,PANELS[name]
    panel = data["b0-slot-mapping-h2d-clocks-readback.json"]
    dma = {}
    for copy in panel["copies"]:
        expected = median(Decimal(x) for x in copy["ten_device_task_medians_us"])/10**6
        required(expected == Decimal(copy["duration_seconds"]), "H2D timing statistic differs")
        for row in panel["native_packets"]:
            if copy["bytes"] == row["bytes_per_rank"]:
                dma[row["repetition"],row["packet"],copy["rank"]] = row,expected,copy["source_task_sha256"]
    legacy_kv = {(row["repetition"],row["rank"]):row for row in data["r11-native-kv-task-clocks-readback.json"]["cases"]}
    raw = folder / "compute-reference"
    active = {kind:read(raw/name)[key] for kind,(name,key,_) in ACTIVE.items()}
    witnesses = host,shared,dma,legacy_kv,active,read(raw/"r11-prefill-tp2-clock.json")
    clock = reference["dense_calibration"]
    ffn = collect_ffn_times(folder / clock["index"],clock["index_sha256"])
    result = {}
    for role, originals in reference["native_clocks"].items():
        config,manifest = original_binding(originals,role)
        root = Path(originals["archive"])
        admitted = reference["native_target_kv_logical_evidence"][role]["original_windows"]
        rounds = []
        for raw_graph,summary,wall,work in zip(graph["arms"][role],projection["arms"][role],originals["rounds"],admitted,strict=True):
            repetition = wall["repetition"]
            timing_file = f"client/formal-{repetition}-timing.json"
            public_file = f"client/formal-{repetition}/raw-public-output-frames.json"
            required(repetition == raw_graph["repetition"] == summary["repetition"] == work["repetition"]
                     and manifest[timing_file] == summary["source_wall_sha256"] == work["wall_sha256"] == wall["source_sha256"]
                     and manifest[public_file] == work["public_frames_sha256"]
                     and int(summary["arithmetic_basic_lower"]) > 0,
                     f"Original model, arithmetic or elapsed boundaries differ for {role} round {repetition}")
            receipt = verify_nodes(role,repetition,work,summary,raw_graph["nodes"],raw_graph,ffn,witnesses)
            rounds.append(dict(**summary, **receipt, wall_evidence=evidence_file(root/timing_file,manifest[timing_file]),
                               output_evidence=evidence_file(root/public_file,manifest[public_file])))
        required(len(rounds) == 3, "Reference results must retain three original rounds")
        aggregate = projection["aggregate_lower_bounds"][role]
        t = sum(Decimal(row["original_wall_seconds"]) for row in rounds)
        ref = sum(Decimal(row["projected_seconds_lower"]) for row in rounds)
        f = sum(row["arithmetic_basic_lower"] for row in rounds)
        with localcontext() as ctx:
            ctx.prec,ctx.rounding = 32,ROUND_FLOOR
            required(Decimal(aggregate["wall_seconds"]) == t and Decimal(aggregate["independent_reference_seconds_lower"]) == ref
                     and aggregate["arithmetic_basic_lower"] == f
                     and Decimal(aggregate["efficiency_percent_lower"]) == 100*ref/t
                     and Decimal(aggregate["effective_tflops_per_card_lower"]) == f/(2*10**12*t),
                     "Aggregate reference and measured arithmetic do not reproduce")
        result[role] = dict(original_configuration=config, rounds=rounds, aggregate=aggregate)
    payload = dict(schema="m01-9b-qualified-conservative-reference/v1", qualification="qualified-lower-bound",
                   method=evidence_file(method,METHOD_SHA), model=MODEL, checkpoint_revision=REVISION,
                   dtype="bfloat16", tensor_parallel_size=2, output_token_cap=256, input_requests=200,
                   graph=dict(path=projection["replay_file"], sha256=projection["replay_sha256"]),
                   canonical_matmul_index=dict(path=clock["index"], sha256=clock["index_sha256"]),
                   source_projection_sha256=projection["source_sha256"],
                   historical_clocks={name:evidence_file(path,PANELS[name]) for name,path in files.items()},
                   verified_cache_clocks={path.name:sha(path) for path in
                      {raw/name for name,_,_ in ACTIVE.values()} | {raw/"r11-prefill-tp2-clock.json"}},
                   arms=result, scope=projection["omissions"],
                   applies_to="reference under the fixed canonical operator and resource contract across source-compatible branches",
                   lower_bound_theorem="Dropping unqualified nonnegative work or unproved edges only decreases the longest path. "
                                       "Retained branch-varying widths take the minimum of their entire qualified range. "
                                       "A canonical address relabeling preserves the required layout; no physical-address speed guarantee is made.",
                   exact_reference_time_known=False, new_native_walls=False)
    receipt = raw / "conservative-reference.json"
    save(receipt,payload)
    return dict(path=receipt.relative_to(folder).as_posix(), sha256=sha(receipt),
                method_sha256=METHOD_SHA, arithmetic_direction="lower", reference_direction="lower",
                efficiency_direction="lower", exact_reference_time_known=False,
                arms={role:section["aggregate"] for role,section in result.items()})
