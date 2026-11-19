"""Independent shape/resource inventory and Kahn/Decimal graph recomputation."""

from collections import Counter, deque
from decimal import Decimal, localcontext
import math


def audit(graph, tail, model, cases, calibration):
    if graph["geometry_sha256"] != tail["geometry_sha256"]:
        raise ValueError("Graph belongs to another native public output population")
    if graph["calibration_plan_sha256"] != calibration["plan_sha256"]:
        raise ValueError("Graph belongs to another component calibration")
    shapes = {c["case"]: (c["m"], c["n"], c["k"]) for c in cases}
    timings = {(r["rank"], t["case"]): Decimal(str(t["seconds"]))
               for r in calibration["ranks"] for t in r["timings"]}
    nodes = graph["nodes"]
    by_id = {}
    for n in nodes:
        if type(n["id"]) is not int or n["id"] < 0 or n["id"] in by_id:
            raise ValueError("Duplicate or invalid reference node identity")
        if (type(n["seconds"]) not in (int, float) or not math.isfinite(n["seconds"])
                or n["seconds"] < 0 or len(n["parents"]) != len(set(n["parents"]))):
            raise ValueError("Invalid node duration or duplicate dependency")
        by_id[n["id"]] = n
    expected = Counter()
    h, q = model.hidden_size, model.num_attention_heads * model.head_dim
    kv = model.num_key_value_heads * model.head_dim
    lk = model.linear_num_key_heads * model.linear_key_head_dim
    lv = model.linear_num_value_heads * model.linear_value_head_dim
    if h != 2048 or len(model.layer_types) != 40 or model.layer_types[-1] != "full_attention":
        raise ValueError("Unreviewed projection architecture")
    for wave in tail["waves"]:
        for layer, family in enumerate(model.layer_types[:-1]):
            nk = ([((2 * lk + 2 * lv) // 2, h), (model.linear_num_value_heads, h), (h, lv // 2)]
                  if family == "linear_attention" else [((2 * q + 2 * kv) // 2, h), (h, q // 2)])
            for rank in (0, 1):
                for operation, (n, k) in enumerate(nk):
                    expected[(wave["public_step"], layer, rank, operation,
                              wave["necessary_target_rows"], n, k)] += 1
    actual, last_cube, layer_ends, joins = Counter(), {}, {}, {}
    for node in sorted(nodes, key=lambda n: n["id"]):
        step, layer, rank = node["public_step"], node["layer"], node["rank"]
        parents = set(node["parents"])
        if any(type(p) is not int or p not in by_id or p == node["id"] for p in parents):
            raise ValueError("Missing, foreign or self-dependent reference ancestor")
        if node["kind"] == "calibrated_dense":
            shape = node["shape"]
            coordinates = step, layer, rank, node["operation"], shape["m"], shape["n"], shape["k"]
            actual[coordinates] += 1
            binding = node["duration_binding"]
            if (type(rank) is not int or rank not in (0, 1)
                    or node["resources"] != [f"rank{rank}:cube"]
                    or shapes.get(node["case"]) != (shape["m"], shape["n"], shape["k"])
                    or binding != dict(plan_sha256=calibration["plan_sha256"], rank=rank, case=node["case"])
                    or Decimal(str(node["seconds"])) != timings.get((rank, node["case"]))
                    or node["seconds"] <= 0):
                raise ValueError("Dense shape, rank, resource or qualified duration differs")
            if rank in last_cube and last_cube[rank] not in parents:
                raise ValueError("Required rank Cube submission dependency is missing")
            if layer > 0 and node["operation"] == 0 and joins.get((step, layer - 1)) not in parents:
                raise ValueError("Required paired-rank layer dependency is missing")
            last_cube[rank] = node["id"]
            layer_ends[step, layer, rank] = node["id"]
        elif node["kind"] == "dependency_join":
            if ((step, layer) in joins or rank is not None or node["resources"]
                    or node["seconds"] != 0 or parents != {
                        layer_ends.get((step, layer, 0)), layer_ends.get((step, layer, 1))}):
                raise ValueError("Paired-rank zero-work dependency join differs")
            joins[step, layer] = node["id"]
        else:
            raise ValueError("Unknown retained reference node")
    if actual != expected or len(joins) != 39 * len(tail["waves"]):
        raise ValueError("Retained source operation multiplicity differs")
    successors = {identity: [] for identity in by_id}
    remaining = {identity: len(n["parents"]) for identity, n in by_id.items()}
    for identity, node in by_id.items():
        for parent in node["parents"]:
            successors[parent].append(identity)
    ready = deque(identity for identity, count in remaining.items() if not count)
    end, parent_choice = {}, {}
    with localcontext() as context:
        context.prec = 40
        while ready:
            identity = ready.popleft()
            node = by_id[identity]
            parent = max(node["parents"], key=lambda p: end[p], default=None)
            start = end[parent] if parent is not None else Decimal(0)
            end[identity] = start + Decimal(str(node["seconds"]))
            parent_choice[identity] = parent
            for child in successors[identity]:
                remaining[child] -= 1
                if remaining[child] == 0:
                    ready.append(child)
        if len(end) != len(nodes):
            raise ValueError("Retained graph has a cycle")
        terminal = max(end, key=end.get, default=None)
        lower = end[terminal] if terminal is not None else Decimal(0)
        path = []
        while terminal is not None:
            path.append(terminal)
            terminal = parent_choice[terminal]
    return dict(schema="independent-dense-tail-graph-audit/v1",
        graph_integrity_passed=True, nodes=len(nodes), dense_leaves=sum(actual.values()),
        dependency_joins=len(joins), reference_model_lower_seconds=str(lower),
        critical_path=list(reversed(path)),
        arithmetic="Decimal40 Kahn topological longest path; all qualified per-rank medians retained",
        source_applicability_certified=False, full_model_certificate=False,
        efficiency_percent=None, pending="Qualify complete source-domain embedding and leaf timing applicability before using this conditional graph value.")
