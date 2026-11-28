"""Expand the reference FFN clock domain on source-bound original packet intervals."""

from decimal import Decimal

from reference_9b_archive import B0_SEAL, PINS, member, read


WITNESS = "d0d80268e551395865f6849c97064e2e8cd9d1675af16f605721ec3413ce416c"
B0_REQUEST_ORDER = "d0d0a333a1ac69acddd8dad93b8cf06de128decd90b82b3dc93927cd6ada1447"
R11_PREFILL = "cf6e97d02e0083b8d1e7fb88a29879e0a771992bf2f31b3df1501febf5d8913d"
BASE_PROJECTION = "f2b4bf38c01dfdd5ac2166c266d0924123843bf6ffaf5c4a0c546881bb0ac2d8"
DIMS = ((4096, 6144), (12288, 4096))
FFN = {"target-ffn-matmuls", "serial-target-ffn-matmuls"}


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def admit(path, clocks):
    value = read(member(path.parent, path.name, WITNESS))
    expected = {"B0": B0_SEAL, "B1": PINS["native-clock/B1/checksums.json"]}
    require(value["schema"] == "m01-9b-native-required-ffn-branch-clock-demand/v1"
            and value["physical_cards"] == [4, 5] and value["precision"] == "bfloat16"
            and value["target_early_layers"] == 31
            and value["source_reports_sha256"]["admitted_graph"] == BASE_PROJECTION
            and value["native_seals"] == {role: arm["checksums_sha256"] for role,arm in clocks.items()} == expected,
            "The FFN row proof is outside the frozen original 9B windows")
    by_role = value["arms"]
    for role, cases in by_role.items():
        require([r["repetition"] for r in cases] == [1, 2, 3], "The FFN row proof lost native rounds")
        for proof, runtime in zip(cases,clocks[role]["rounds"],strict=True):
            require(proof["original_timing_sha256"] == runtime["source_sha256"],
                    "The FFN row proof and formal wall belong to different repetitions")
            cap = 72 if role == "B0" else 108
            packets = proof["packets"]
            ids = [row["packet"] for row in packets]
            require(ids == sorted(set(ids)) and len(packets) == proof["candidate_packet_count"]
                    and all(1 <= r["minimum_necessary_rows"] <= r["maximum_compatible_rows"] <= cap for r in packets),
                    "The FFN row proof included an unknown or noncausal packet")
    return value


def admit_b0_order(path, ffn, clocks):
    proof = read(member(path.parent, path.name, B0_REQUEST_ORDER))
    require(proof["schema"] == "m01-original018-tail-ffn-required-request-order/v1"
            and proof["native_seal_sha256"] == clocks["B0"]["checksums_sha256"] == B0_SEAL
            and proof["ffn_branch_demand_sha256"] == WITNESS,
            "Original018 order proof belongs to a different branch panel")
    output = {}
    for row, branch, clock in zip(proof["rounds"], ffn["arms"]["B0"], clocks["B0"]["rounds"], strict=True):
        rep = row["repetition"]
        require(rep == branch["repetition"] == clock["repetition"]
                and row["wall_sha256"] == branch["original_timing_sha256"] == clock["source_sha256"],
                "Original018 cross-packet proof belongs to different elapsed boundaries")
        seen = set()
        prev = {}
        for edges, packet in zip(row["graph"], branch["packets"], strict=True):
            index = packet["packet"]
            parents = [link["predecessor_packet"] for link in edges["predecessor_witnesses"]]
            require(edges["packet"] == index and set(parents) <= seen,
                    "Original018 cross-packet source order is invalid")
            prev[index] = parents
            seen.add(index)
        output[rep] = prev
    return output


def admit_r11_prefill(path, clocks):
    proof = read(member(path.parent, path.name, R11_PREFILL))
    require(proof["schema"] == "m01-r11-native-first-prefill-exact-width-and-slots/v1"
            and proof["original_seal_sha256"] == clocks["B1"]["checksums_sha256"]
            and proof["admitted_live_widths"] == [1369],
            "R11 first prefill does not bind to the original windows")
    result = {}
    for row, clock in zip(proof["rounds"], clocks["B1"]["rounds"], strict=True):
        rep = row["repetition"]
        require(rep == clock["repetition"] and row["wall_sha256"] == clock["source_sha256"]
                and row["first_public_packet"] == 0 and row["necessary_prefill_rows"] == 1369,
                "R11 first prefill and original wall differ")
        result[rep] = dict(packet=0, minimum_necessary_rows=1369, maximum_compatible_rows=1369,
                           clock_branch_proof_sha256=R11_PREFILL)
    return result


def duration(rank, candidates, times):
    return 31 * min(sum(times[rank, m, n, k] for n,k in DIMS) for m in candidates)


def packet_nodes(output, row, times, tails, parents):
    bounds = row["minimum_necessary_rows"], row["maximum_compatible_rows"]
    widths = list(range(bounds[0], bounds[1]+1))
    for rank, tail in enumerate(parents):
        node = dict(kind="serial-target-ffn-matmuls", packet=row["packet"], rank=rank,
                    layers=31, measured_candidate_rows=widths, seconds=str(duration(rank,widths,times)),
                    predecessors=tail, clock_branch_proof_sha256=row.get("clock_branch_proof_sha256", WITNESS))
        tails[rank] = len(output)
        output.append(node)


def cold_kv_io(output, parents, rank, layer, kind, clocks):
    evidence = clocks[kind,rank]
    index = len(output)
    output.append(dict(kind=f"target-attention-cold-prefill-{kind}", packet=0, rank=rank, layer=layer,
        live_rows=1369, predecessors=parents, seconds=evidence["seconds"],
        calibration_clock_sha256=evidence["task_sha256"], case_id=evidence["operand"]["id"]))
    return [index]


def first_prefill_with_collective(output, tails, parents, times, clock, cold_kv, cold_gdn):
    from reference_9b_r11_prefill_hccl import CASE, CLOCK

    require(clock["operator"] == CASE and clock["case_id"] == CLOCK,
            "R11 initial-packet tensor collective does not match the canonical operands")
    starts = parents
    for layer in range(31):
        pair = []
        for rank in (0, 1):
            upstream = starts[rank]
            if cold_kv is not None and layer % 4 == 3:
                upstream = cold_kv_io(output, upstream, rank, layer, "scatter", cold_kv)
                upstream = cold_kv_io(output, upstream, rank, layer, "fia", cold_kv)
            if cold_gdn is not None and layer % 4 != 3:
                clock_case = cold_gdn[rank]
                require(clock_case["case"]["rank"] == rank, "GDN convolution clock refers to another rank")
                output.append(dict(kind="r11-cold-gdn-convolution", packet=0, rank=rank, layer=layer,
                    seconds=clock_case["seconds"], predecessors=upstream,
                    calibration_clock_sha256=clock_case["task_sha256"], input_rows=1369))
                upstream = [len(output) - 1]
            pair.append(len(output))
            output.append(dict(kind="target-ffn-matmuls", packet=0, rank=rank, layer=layer,
                measured_candidate_rows=[1369], seconds=str(sum(times[rank,1369,n,k] for n,k in DIMS)),
                predecessors=upstream, clock_branch_proof_sha256=R11_PREFILL))
        tail = len(output)
        output.append(dict(kind="tp2-allreduce", packet=0, rank=None, layer=layer,
            seconds=clock["device_task_seconds_lower"], predecessors=pair, bf16_elements_per_rank=CASE["elements"],
            source_clock_sha256=clock["raw_manifest_sha256"],
            operator_scope="required r11 first-prefill down projection on 1369 live rows"))
        starts = [[tail], [tail]]
    tails[:] = [tail, tail]
    if cold_kv is not None:
        for rank in (0, 1):
            # All prompt keys remain needed for future retained decode.
            tails[rank] = cold_kv_io(output, [tail], rank, 31, "scatter", cold_kv)[0]


def compose(role, source, scope, times, b0_order=None, r11_prefill=None, r11_prefill_hccl=None, cold_kv=None, *, cold_gdn=None):
    selected = {r["packet"]:r for r in scope["packets"]}
    if r11_prefill is not None:
        require(role == "B1" and 0 not in selected, "An original r11 prefill already belongs to the branch clock scope")
        selected[0] = r11_prefill
    existing = {}
    active_ffn = set()
    for index, node in enumerate(source):
        packet = node["packet"]
        existing.setdefault(packet, []).append((index,node))
        if node["kind"] in FFN:
            active_ffn.add(packet)
    output, moved = [], {}
    tails = [None, None]
    past = {}
    for packet in sorted(set(existing) | set(selected)):
        if role == "B0":
            tails = [None,None]
        parent_tails = [list() if node is None else [node] for node in tails]
        if role == "B0" and b0_order is not None:
            predecessors = b0_order.get(packet, [])
            parent_tails = [[past[p][rank] for p in predecessors] for rank in (0,1)]
            require(all(node is not None for rank in parent_tails for node in rank),
                    "Original018 required packet lost its earlier measured FFN")
        first = set()
        for index,node in existing.get(packet,[]):
            rank = node["rank"]
            parents = [moved[p] for p in node["predecessors"]]
            if rank in (0,1) and node["kind"] in FFN and rank not in first:
                first.add(rank)
                parents.extend(parent_tails[rank])
            moved[index] = len(output)
            output.append(dict(node, predecessors=sorted(set(parents))))
            if node["kind"] == "tp2-allreduce":
                tails[:] = [moved[index]]*2
            elif rank in (0,1) and (
                node["kind"] in FFN or node["kind"].startswith(("target-attention-", "r11-retained-draft-"))
            ):
                tails[rank] = moved[index]
        if packet in selected and packet not in active_ffn:
            if packet == 0 and role == "B1" and r11_prefill_hccl is not None:
                first_prefill_with_collective(output, tails, parent_tails, times, r11_prefill_hccl, cold_kv, cold_gdn)
            else:
                packet_nodes(output,selected[packet],times,tails,parent_tails)
        elif packet in active_ffn:
            require(first == {0,1}, "A projected native packet lost a rank")
        past[packet] = tails[:]
    return output
