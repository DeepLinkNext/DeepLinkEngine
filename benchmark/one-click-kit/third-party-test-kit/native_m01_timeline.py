"""Preserve device stream order and trace-reported dependencies for live M01."""

from bisect import bisect_right
from collections import defaultdict
import json

from reference_capture_worker import write


def export(directory):
    nodes = {}
    edges = set()
    missing = []
    for rank in (0, 1):
        trace = json.loads((directory / f"rank-{rank}-device-trace.json").read_bytes())
        events = trace["traceEvents"] if isinstance(trace, dict) else trace
        metadata = {}
        streams = defaultdict(list)
        flows = defaultdict(list)
        for event in events:
            if event.get("ph") == "M" and event.get("name") == "process_name":
                metadata[event["pid"]] = str(event["args"]["name"]).lower()
        for event in events:
            phase = event.get("ph")
            pid, tid = event.get("pid"), event.get("tid")
            if phase == "X" and event.get("dur", 0) > 0 and "ts" in event:
                name = metadata.get(pid, "")
                args = event.get("args") or {}
                is_device = ("ascend hardware" in name or "npu" in name or
                             ("Stream Id" in args and "Task Id" in args))
                if is_device:
                    streams[(pid, tid)].append(event)
            elif phase in ("s", "t", "f") and "id" in event:
                flows[(event.get("cat"), str(event["id"]))].append(event)
        if not streams:
            raise ValueError(f"Rank {rank} device stream events are missing from the Ascend trace")
        stream_index = {}
        for (pid, tid), members in streams.items():
            order = sorted(members, key=lambda event: event["ts"])
            identifiers = []
            for index, event in enumerate(order):
                node = f"{rank}:{pid}:{tid}:{index}"
                nodes[node] = dict(id=node, rank=rank, stream=[pid, tid], name=event["name"],
                                   start_us=event["ts"], duration_us=event["dur"])
                if identifiers:
                    previous = nodes[identifiers[-1]]
                    if previous["start_us"] + previous["duration_us"] <= event["ts"]:
                        edges.add((previous["id"], node, "stream_order"))
                    else:
                        missing.append(dict(rank=rank, stream=[pid, tid], reason="overlapping_spans",
                                            nodes=[previous["id"], node]))
                identifiers.append(node)
            stream_index[(pid, tid)] = ([item["ts"] for item in order], identifiers)
        for events_in_flow in flows.values():
            endpoints = []
            for event in sorted(events_in_flow, key=lambda item: item["ts"]):
                starts, ids = stream_index.get((event.get("pid"), event.get("tid")), ([], []))
                index = bisect_right(starts, event["ts"]) - 1
                if index >= 0:
                    item = nodes[ids[index]]
                    if event["ts"] <= item["start_us"] + item["duration_us"]:
                        endpoints.append(ids[index])
            for origin, destination in zip(endpoints, endpoints[1:]):
                if origin != destination:
                    edges.add((origin, destination, "trace_flow"))
    result = dict(nodes=list(nodes.values()), edges=sorted(edges), unresolved=missing,
                  includes_inferred_cross_rank_edges=False)
    write(directory / "device-dependencies.json", result)
