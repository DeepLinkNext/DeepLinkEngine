"""Replay the verified 9B component graph using the current cache resolution."""

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import gzip
import json

from dataset_contract import save, sha
from reference_9b_archive import B0_SEAL, PINS, read
from reference_9b_b0_fia import PROJECTION as DUAL_FIA_PROJECTION, SCOPE as B0_FIA_SCOPE
from reference_9b_target_fia import PROJECTION as FIA_PROJECTION, SCOPE as FIA_SCOPE

DRAFT_PROJECTION_SHA256 = "f2b4bf38c01dfdd5ac2166c266d0924123843bf6ffaf5c4a0c546881bb0ac2d8"
RETAINED_QUERY_WORK = "41fcf4462ddfee49e42c93c87780ef76a282e6040b2947c970bc0c091f9b2806"
GDN_ARITHMETIC = "63d0315accf3b7837ae34e1d490b9ba38420fdda7ae06e0010bc458492342751"
DRAFT_PROOF_SHA256 = "496225a11e8948b8741d4b69564c5813a7c9c87731fb032886ed396962ec8e1e"
PROJECTION_SHA256 = "9d90e73573bfc9f44eaa3b6ed6621dbfaea76bc0e080c8249182c61dcc6836b8"
SCHEMA = "vllm-hust-9b-cann-task-two-arm-collective-kv-dma-lower-dag/v1"
INVARIANT_SHA256 = "9b4e89b239d415ad2b59c7d28f130885dd3679c60995c77da396753410cf0247"
INVARIANT_SCHEMA = "vllm-hust-9b-necessary-invariant-reference-projection/v1"
FFN = {"target-ffn-matmuls", "serial-target-ffn-matmuls"}
DIMS = ((4096, 6144), (12288, 4096))


def required(condition, detail):
    if not condition:
        raise ValueError(detail)


def collect_ffn_times(index, checksum):
    required(sha(index) == checksum, "The resolved 9B component clock index changed")
    clocks = read(index)
    required(clocks["timing_domain"] == "cann-device-task-duration/v1", "The 9B matmul clock domain differs")
    times = {}
    for record in clocks["cases"]:
        shape = record["case"]
        require_key = shape["n"], shape["k"]
        required(require_key in DIMS and len(record["rank_clock_source"]) == 2,
                 "A selected matmul is outside the original 9B component domain")
        for rank, item in enumerate(record["rank_clock_source"]):
            key = rank, shape["m"], *require_key
            seconds = Decimal(item["seconds"])
            required(key not in times and seconds.is_finite() and seconds > 0,
                     "A duplicate or invalid device task clock reached the component projection")
            times[key] = seconds
    return times


def replay(source, times, fia=None, repetition=None, b0_fia=None):
    finishes, previous, nodes = [], [], []
    for index, original in enumerate(source):
        node = dict(original)
        preds = node["predecessors"]
        required(type(preds) is list and all(type(p) is int and 0 <= p < index for p in preds),
                 "The original 9B component projection is not a directed acyclic graph")
        duration = Decimal(node["seconds"])
        if node["kind"] in FFN:
            rank = node["rank"]
            counts = node["measured_candidate_rows"]
            elapsed, chosen = min((sum(times[rank, m, n, k] for n, k in DIMS), m) for m in counts)
            duration = node.get("layers", 1) * elapsed
            node["lowest_qualified_canonical_rows"] = chosen
            node["seconds"] = str(duration)
            # The selected rank clock is checked in the active cache receipt.
            node.pop("chosen_device_task_clocks", None)
        if node["kind"] == "target-attention-fia-read":
            required(fia is not None and node["packet"] == 1 and len(preds) == 1,
                     "The exact R11 FIA kernel lost its KV scatter dependency")
            writer = nodes[preds[0]]
            rank = node["rank"]
            required(writer["kind"] == "target-attention-kv-write" and writer["rank"] == rank
                     and writer["layer"] == node["layer"] and writer["packet"] == node["packet"],
                     "FIA consumer does not follow its proven rank-local scatter")
            clock = fia[rank, repetition]
            duration = Decimal(clock["seconds"])
            node["seconds"] = str(duration)
            node["calibration_clock_sha256"] = clock["task_sha256"]
        if node["kind"] == "target-attention-direct-prefill-fia":
            rank = node["rank"]
            required(b0_fia is not None and node["packet"] == 0 and node["reads_direct_kv"] is True
                     and node["qualified_case_id"] == f"rank-{rank}" and len(preds) == 1,
                     "Exact original018 first direct FIA scope changed")
            parent = nodes[preds[0]]
            if parent["kind"] == "original018-cold-kv-scatter":
                required(parent["rank"] == rank and parent["layer"] == node["layer"] and parent["packet"] == 0
                         and len(parent["predecessors"]) == 1, "Original018 FIA lost its rank-local stream predecessor")
                parent = nodes[parent["predecessors"][0]]
            required(parent["kind"] == "tp2-allreduce" and parent["layer"] == node["layer"] - 1,
                     "Exact original018 first direct FIA lost its upstream layer dependency")
            clock = b0_fia[rank]
            duration = Decimal(clock["seconds"])
            node["seconds"] = str(duration)
            node["calibration_clock_sha256"] = clock["task_sha256"]
        if node["kind"] == "r11-retained-draft-phase0-kv-write":
            required(len(preds) == 1 and node["packet"] == 1 and node["layer"] == 0,
                     "First retained R11 draft scatter must consume same-wave target output")
            target_read = nodes[preds[0]]
            required(target_read["kind"] == "target-attention-fia-read" and target_read["layer"] == 31
                     and target_read["packet"] == 1 and target_read["rank"] == node["rank"]
                     and len(target_read["predecessors"]) == 1,
                     "R11 draft scatter lost its original target layer output")
            original_scatter = nodes[target_read["predecessors"][0]]
            required(original_scatter["physical_slots"] == node["physical_slots"]
                     and original_scatter["source_device_task_sha256"] == node["source_device_task_sha256"]
                     and Decimal(original_scatter["seconds"]) == duration,
                     "R11 draft scatter does not reproduce its source-equivalent independent clock")
        required(duration.is_finite() and duration > 0, "The component graph has a nonpositive duration")
        predecessor = max(preds, key=finishes.__getitem__) if preds else None
        finishes.append(duration + (finishes[predecessor] if predecessor is not None else 0))
        previous.append(predecessor)
        nodes.append(node)
    leaf = max(range(len(nodes)), key=finishes.__getitem__)
    longest = str(finishes[leaf])
    chain = []
    while leaf is not None:
        chain.append(leaf)
        leaf = previous[leaf]
    return dict(seconds_lower=longest, critical_path=chain[::-1], nodes=nodes)


def attach_original018_cold_kv(source, clocks):
    output, index_map, inserted = [], {}, set()

    def append_write(rank, layer, predecessors):
        proof = clocks[rank]
        required(proof["case"]["rank"] == rank and proof["page_normalized"],
                 "Original018 canonical cold KV clock belongs to another rank or address domain")
        location = len(output)
        output.append(dict(kind="original018-cold-kv-scatter", packet=0, rank=rank, layer=layer,
                           live_rows=1369, page_scope="canonical-aligned-two-page", predecessors=predecessors,
                           seconds=proof["seconds"], calibration_clock_sha256=proof["task_sha256"]))
        inserted.add((rank, layer))
        return location

    for index, node in enumerate(source):
        preds = [index_map[pred] for pred in node["predecessors"]]
        if node["kind"] == "target-attention-direct-prefill-fia":
            preds = [append_write(node["rank"], node["layer"], preds)]
        index_map[index] = len(output)
        output.append(dict(node, predecessors=preds))
        if node["packet"] == 0 and node["kind"] == "tp2-allreduce" and node["layer"] == 30:
            for rank in (0, 1):
                append_write(rank, 31, [index_map[index]])
    required(inserted == {(rank, layer) for rank in (0, 1) for layer in range(3, 32, 4)},
             "The Original018 cold reference graph lost necessary first-packet KV writes")
    return output


def attach_first_retained_draft(source, clocks, repetition, native_wall):
    nodes, moved, ranks = [], [], set()
    for node in source:
        preds = [moved[i] for i in node["predecessors"]]
        nodes.append(dict(node, predecessors=preds))
        if node["kind"] == "r11-retained-draft-phase0-kv-write":
            rank = node["rank"]
            required(rank not in ranks and rank in (0, 1), "Duplicate first-retained draft rank")
            ranks.add(rank)
            for label, phase, kind in (
                ("draft-fia", 0, "r11-retained-draft-phase0-fia"),
                ("draft-kv-scatter", 1, "r11-retained-draft-phase1-kv-write"),
                ("draft-fia", 1, "r11-retained-draft-phase1-fia"),
            ):
                name = f"{label}:{rank}:{repetition}:{phase}"
                clock = clocks[name]
                scope = clock["operand"]
                pages = scope["actual_page_prefix"]
                first_slots = [pages[pos // 128] * 128 + pos % 128 for pos in (1369, 1370, 1371)]
                required(scope["id"] == name and scope["original_timing_sha256"] == native_wall
                         and scope["rank"] == rank and scope["phase"] == phase
                         and node["physical_slots"] == first_slots,
                         f"First draft operation is outside the original graph for {name}")
                nodes.append(dict(kind=kind, packet=1, rank=rank, layer=0, predecessors=[len(nodes)-1],
                                  seconds=clock["seconds"], calibration_clock_sha256=clock["task_sha256"],
                                  case_id=name, operand=scope))
        moved.append(len(nodes)-1)
    required(ranks == {0, 1}, "The first-retained draft graph is missing a rank-local scatter")
    return nodes

def read_retained_work(path, clocks):
    from reference_9b_archive import member

    proof = read(member(path.parent, path.name, RETAINED_QUERY_WORK))
    required(proof["schema"] == "m01-9b-six-window-retained-query-arithmetic-floor/v1"
             and set(proof["arms"]) == set(clocks),
             "Retained-query arithmetic differs from the original pair")
    for role, rows in proof["arms"].items():
        for row, native in zip(rows, clocks[role]["rounds"], strict=True):
            rep = row["repetition"]
            required(row["native_seal_sha256"] == clocks[role]["checksums_sha256"]
                     and rep == native["repetition"]
                     and row["sources_sha256"][f"client/formal-{rep}-timing.json"] == native["source_sha256"]
                     and row["arithmetic_basic_lower"] == sum(row["arithmetic_terms"].values()),
                     "Retained-query arithmetic and original wall differ")
    return proof


def read_gdn_work(path, clocks, inherited):
    from reference_9b_archive import member

    proof = read(member(path.parent, path.name, GDN_ARITHMETIC))
    required(proof["schema"] == "m01-9b-retained-target-gdn-arithmetic-floor/v1"
             and proof["inherited_floor_sha256"] == RETAINED_QUERY_WORK and set(proof["arms"]) == {"B0", "B1"}
             and proof["native_seals"] == {role: arm["checksums_sha256"] for role, arm in clocks.items()},
             "The GDN arithmetic scope does not match both original archives and lower-bound terms")
    for role, arm in clocks.items():
        for row, old, now in zip(proof["arms"][role], inherited["arms"][role], arm["rounds"], strict=True):
            before = old["arithmetic_terms"]
            terms = row["arithmetic_disjoint_terms"]
            required(row["repetition"] == old["repetition"] == now["repetition"]
                     and row["original_wall_sha256"] == now["source_sha256"]
                     and all(terms[name] == value for name, value in before.items())
                     and row["arithmetic_basic_lower"] == sum(terms.values())
                     and row["arithmetic_basic_lower"] > old["arithmetic_basic_lower"],
                     "GDN arithmetic failed to preserve its bound original request and wall")
            upper = row.get("in_window_catalogue_upper")
            if role == "B1":
                required(upper["upper"] == sum(upper["components"].values())
                         and row["arithmetic_basic_lower"] < upper["upper"] < upper["previous"],
                         "B1 in-window necessary arithmetic is outside its verified upper catalogue")
            else:
                required(upper is None, "A B0 ceiling must not be inherited from the r11 catalogue")
    return proof


def compose(source, out, draft, clocks, fia=None, b0_fia=None, r11_draft=None,
            ffn_branches=None, b0_order=None, retained_work=None, r11_prefill=None, r11_prefill_hccl=None, cold_kv=None, gdn_work=None, *, cold_gdn=None, b0_cold_kv=None):
    seals = {role: arm["checksums_sha256"] for role, arm in clocks.items()}
    source_sha256 = sha(source)
    required(seals == {"B0": B0_SEAL, "B1": PINS["native-clock/B1/checksums.json"]}
             and source_sha256 in {PROJECTION_SHA256, INVARIANT_SHA256, FIA_PROJECTION, DUAL_FIA_PROJECTION, DRAFT_PROJECTION_SHA256},
             "The 9B component projection or native seals changed")
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        graph = json.load(stream)
    invariant = source_sha256 != PROJECTION_SHA256
    if source_sha256 in {FIA_PROJECTION, DUAL_FIA_PROJECTION, DRAFT_PROJECTION_SHA256}:
        required(graph["typed_fia_clock_scope_sha256"] == FIA_SCOPE and fia is not None,
                 "The source-bound target attention graph has no qualified FIA clock panel")
    if source_sha256 in {DUAL_FIA_PROJECTION, DRAFT_PROJECTION_SHA256}:
        required(graph["typed_b0_direct_fia_clock_scope_sha256"] == B0_FIA_SCOPE and b0_fia is not None,
                 "The source-bound original018 direct FIA graph has no qualified device clock")
    if source_sha256 == DRAFT_PROJECTION_SHA256:
        required(graph["typed_r11_initial_retained_draft_proof_sha256"] == DRAFT_PROOF_SHA256,
                 "The first retained R11 draft scope changed")
    expected_schema = INVARIANT_SCHEMA if invariant else SCHEMA
    unresolved = graph.get("components_unresolved_in_complete_graph") if invariant else not graph["eligible_as_full_reference"]
    required(graph["schema"] == expected_schema and set(graph["arms"]) == set(seals) and unresolved is True,
             "Unexpected 9B component projection")

    current = collect_ffn_times(out / draft["index"], draft["index_sha256"])
    arms = {}
    compact = {}
    for role, rows in graph["arms"].items():
        qualified = []
        summaries = []
        required([item["repetition"] for item in rows] == [1, 2, 3], "Projection lost an original timing round")
        for row, window in zip(rows, clocks[role]["rounds"], strict=True):
            repetition = row["repetition"]
            nodes = row["nodes"]
            if r11_draft is not None and role == "B1":
                nodes = attach_first_retained_draft(nodes, r11_draft, repetition, window["source_sha256"])
            if ffn_branches is not None:
                from reference_9b_ffn_branch import compose as compose_ffns

                order = b0_order[repetition] if role == "B0" and b0_order is not None else None
                prefill = r11_prefill[repetition] if role == "B1" and r11_prefill is not None else None
                collective = r11_prefill_hccl if role == "B1" else None
                kv = None
                if role == "B1" and cold_kv is not None:
                    kv = {(kind,rank):cold_kv[f"prefill-{kind}:{rank}:{repetition}"]
                          for kind in ("scatter","fia") for rank in (0,1)}
                    required(all(item["original_wall_sha256"] == window["source_sha256"] for item in kv.values()),
                             "R11 cold KV operand is paired to a different original wall")
                conv = cold_gdn if role == "B1" else None
                nodes = compose_ffns(role, nodes, ffn_branches["arms"][role][repetition-1], current, order, prefill, collective, kv, cold_gdn=conv)
            if role == "B0" and b0_cold_kv is not None:
                nodes = attach_original018_cold_kv(nodes, b0_cold_kv)
            measured = replay(nodes, current, fia, repetition, b0_fia)
            qualified.append(dict(repetition=repetition, **measured))
            values = dict(repetition=repetition, projected_seconds_lower=measured["seconds_lower"],
                          critical_path_nodes=len(measured["critical_path"]))
            if invariant:
                required(row["source_window_sha256"] == window["source_sha256"] and repetition == window["repetition"],
                         "The invariant graph is paired to a different native timing window")
                floor = row["retained_target_arithmetic_basic_lower"]
                if retained_work is not None:
                    changed = retained_work["arms"][role][repetition-1]["arithmetic_basic_lower"]
                    required(changed >= floor, "Retained-query proof weakened the existing arithmetic floor")
                    floor = changed
                gdn = gdn_work["arms"][role][repetition-1] if gdn_work is not None else None
                if gdn is not None:
                    required(gdn["arithmetic_basic_lower"] > floor,"The enhanced GDN work weakened the arithmetic lower bound")
                    floor = gdn["arithmetic_basic_lower"]
                    values["guaranteed_mtp_recurrent_target_rows"] = gdn["guaranteed_mtp_recurrent_target_rows"]
                    if role == "B1":
                        values["in_window_catalogue_arithmetic_basic_upper"] = gdn["in_window_catalogue_upper"]["upper"]
                        values["in_window_catalogue_tflops_per_card_upper"] = gdn["measured_tflops_per_card_upper"]
                projected = Decimal(measured["seconds_lower"])
                wall = Decimal(str(window["wall_seconds"]))
                with localcontext() as context:
                    context.prec = 32
                    context.rounding = ROUND_FLOOR
                    values.update(arithmetic_basic_lower=floor, source_wall_sha256=window["source_sha256"],
                                  original_wall_seconds=str(wall), measured_tflops_per_card_lower=str(floor / (2 * 10**12 * wall)),
                                  efficiency_percent_lower=str(100 * projected / wall))
            summaries.append(values)
        arms[role] = qualified
        compact[role] = summaries
    omissions = list(graph["omissions"])
    if b0_cold_kv is not None:
        omissions[0] = ("Original018 retains the canonical first cold target KV writes on both ranks, "
                        "and measured host-to-device slot metadata. Later physical tensor scatters remain unresolved.")
    if r11_draft is not None:
        omissions[-1] = ("The first two retained draft FIA rows and their KV writes have qualified clocks. "
                         "Other draft waves and the intervening unmeasured compute remain outside this lower bound.")
    if ffn_branches is not None:
        omissions[2] = ("FFN widths inside fully calibrated original branch intervals are retained. "
                        "Other widths and unmeasured compute, host work and synchronization remain outside the bound.")
    if cold_kv is not None:
        omissions[1] = ("The r11 target KV clocks cover the first cold prefill and the first exact unpadded "
                        "target decode wave as recorded in the graph. Later target KV remains outside this bound.")
    if cold_gdn is not None:
        omissions.append("The first r11 cold convolution is retained at each of the 24 GDN target layers. "
                         "Later GDN shapes, WY/recurrent clocks and gated normalization remain outside this bound.")
    elif gdn_work is not None:
        omissions.append("The new GDN state, convolution and gated normalization terms are in the arithmetic "
                         "bound. Their independent graph clocks remain outside this projection.")
    omissions.append("Nodes serialize only along their verified edges. Rank-local operations without a "
                     "dependency and independent host, movement or collective operations may overlap. "
                     "No uncovered cost is assigned a zero-duration clock.")
    annotations = {}
    if invariant:
        aggregates = {}
        for role, rows in compact.items():
            total_wall = sum(Decimal(row["original_wall_seconds"]) for row in rows)
            total_lower = sum(Decimal(row["projected_seconds_lower"]) for row in rows)
            total_arithmetic = sum(row["arithmetic_basic_lower"] for row in rows)
            with localcontext() as context:
                context.prec = 32
                context.rounding = ROUND_FLOOR
                aggregates[role] = dict(wall_seconds=str(total_wall), independent_reference_seconds_lower=str(total_lower),
                    arithmetic_basic_lower=total_arithmetic,
                    effective_tflops_per_card_lower=str(total_arithmetic / (2 * 10**12 * total_wall)),
                    efficiency_percent_lower=str(100 * total_lower / total_wall))
        if gdn_work is not None:
            r11_rows = compact["B1"]
            total_wall = sum(Decimal(row["original_wall_seconds"]) for row in r11_rows)
            high = sum(row["in_window_catalogue_arithmetic_basic_upper"] for row in r11_rows)
            with localcontext() as context:
                context.prec = 32
                context.rounding = ROUND_CEILING
                aggregates["B1"]["in_window_catalogue_arithmetic_basic_upper"] = high
                aggregates["B1"]["in_window_catalogue_tflops_per_card_upper"] = str(high / (2 * 10**12 * total_wall))
        annotations["aggregate_lower_bounds"] = aggregates
    if retained_work is not None:
        annotations["retained_query_work_sha256"] = RETAINED_QUERY_WORK
    if b0_order is not None:
        from reference_9b_ffn_branch import B0_REQUEST_ORDER
        annotations["original018_request_order_sha256"] = B0_REQUEST_ORDER
    if r11_prefill is not None:
        from reference_9b_ffn_branch import R11_PREFILL
        annotations["r11_initial_prefill_sha256"] = R11_PREFILL
    if r11_prefill_hccl is not None:
        annotations["r11_first_prefill_tp2_clock_sha256"] = r11_prefill_hccl["raw_manifest_sha256"]
    if cold_kv is not None:
        from reference_9b_r11_prefill_kv import DEMAND
        annotations["r11_cold_target_kv_operands_sha256"] = DEMAND
    if gdn_work is not None:
        annotations["target_gdn_arithmetic_proof_sha256"] = GDN_ARITHMETIC
    target = out / "compute-reference/component-projection.json"
    save(target, dict(schema=expected_schema, original_projection_sha256=source_sha256,
                      current_clock_index_sha256=draft["index_sha256"], native_seals=seals,
                      omissions=omissions, full_reference_denominator_qualified=False, arms=arms, **annotations))
    return dict(source_sha256=source_sha256, replay_sha256=sha(target),
                replay_file=target.relative_to(out).as_posix(), omissions=omissions, arms=compact,
                full_reference_denominator_qualified=False, **annotations)
