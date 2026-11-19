"""Explicit projection of a canonical resource DAG onto qualified dense leaves.

Zero-duration records are logical joins with no arithmetic. All omitted source
families remain listed as omissions. This module alone does not qualify the
source/resource model or a measured calibration's applicability.
"""

import math

from public_decode_tail import dense_shapes, projection_shapes


def compile_graph(tail, model, cases, calibration):
    expected_shapes = {(c["m"], c["n"], c["k"]) for c in dense_shapes(tail, model)}
    index, identities = {}, set()
    for c in cases:
        shape = c["m"], c["n"], c["k"]
        if (set(c) != {"case", "m", "n", "k"} or shape in index
                or any(type(c[k]) is not int or c[k] < (0 if k == "case" else 1) for k in c)
                or c["case"] in identities):
            raise ValueError("Duplicate or invalid canonical dense case")
        index[shape] = c["case"]
        identities.add(c["case"])
    if not expected_shapes <= index.keys():
        raise ValueError("Own tail has missing exact canonical dense shapes")
    if (calibration["family"] != "dense" or calibration["complete"] is not True
            or len(calibration["ranks"]) != 2
            or {r["rank"] for r in calibration["ranks"]} != {0, 1}):
        raise ValueError("Both complete independently audited dense ranks required")
    durations = {}
    for rank in calibration["ranks"]:
        for timing in rank["timings"]:
            key = rank["rank"], timing["case"]
            seconds = timing["seconds"]
            if (key in durations or type(seconds) not in (int, float)
                    or not math.isfinite(seconds) or seconds <= 0
                    or timing["case"] not in index.values()):
                raise ValueError("Foreign, missing or invalid calibrated dense duration")
            durations[key] = seconds
    nodes, previous_cube = [], {0: None, 1: None}

    def append(kind, parents, **fields):
        identity = len(nodes)
        nodes.append(dict(id=identity, kind=kind,
            parents=sorted({p for p in parents if p is not None}), **fields))
        return identity

    for wave in tail["waves"]:
        # Each public delivery corresponds to one canonical necessary target
        # batch. A later wave retains the same rank's Cube submission order;
        # no cross-rank barrier is added at the wave boundary.
        frontier = dict(previous_cube)
        m = wave["necessary_target_rows"]
        for layer, family in enumerate(model.layer_types[:-1]):
            end = {}
            for rank in (0, 1):
                current = frontier[rank]
                for operation, (n, k) in enumerate(projection_shapes(model, family)):
                    case = index[m, n, k]
                    if (rank, case) not in durations:
                        raise ValueError("Own exact shape has no qualified rank duration")
                    current = append("calibrated_dense", [current, previous_cube[rank]],
                        public_step=wave["public_step"], layer=layer, rank=rank,
                        operation=operation, shape=dict(m=m, n=n, k=k),
                        case=case, seconds=durations[rank, case],
                        resources=[f"rank{rank}:cube"],
                        duration_binding=dict(plan_sha256=calibration["plan_sha256"], rank=rank, case=case))
                    previous_cube[rank] = current
                end[rank] = current
            # Every retained next-layer input follows the paired TP reductions
            # in the original layer. Their unknown costs are omitted; their
            # source dependencies remain a zero-work logical join.
            joined = append("dependency_join", list(end.values()), seconds=0,
                public_step=wave["public_step"], layer=layer, rank=None, resources=[],
                source_dependency="TP attention/MoE reduction ancestors of the next layer")
            frontier = {0: joined, 1: joined}
    return dict(schema="canonical-dense-tail-reference-lower-graph/v1", nodes=nodes,
        geometry_sha256=tail["geometry_sha256"], calibration_plan_sha256=calibration["plan_sha256"],
        omitted_groups=["All deliveries up to and including the final request's first token",
            "Entire final target decoder and target vocabulary",
            "Entire MTP producer/dependency subgraph", "All norm/scalar/convolution/GDN core leaves",
            "All MoE leaves and all TP collective durations", "All host metadata, copies and output handoff"],
        lower_bound_argument=[
            "Delete whole omitted leaves of nonnegative duration from the source-qualified canonical DAG.",
            "Retain their transitive data dependencies; same-rank Cube submission order is preserved.",
            "Paired-rank reductions contribute dependency joins with no assigned operation duration.",
            "Every remaining weighted leaf has its own exact shape/rank calibration; none is partially extrapolated.",
            "Deletion with dependency splicing cannot increase a DAG's longest-path duration."],
        missing_duration_assigned_zero=False, elapsed_native_time_used=False,
        source_runtime_applicability_certified=False, full_model_certificate=False,
        reference_seconds=None, efficiency_percent=None)
