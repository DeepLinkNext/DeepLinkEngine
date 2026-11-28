"""Resolve exact dense calibration cases independently of archive-local IDs."""

from collections import namedtuple
import math

Case = namedtuple("Case", ("m", "n", "k"))


def shapes(cases):
    result = {}
    for ordinal, case in enumerate(cases):
        if (type(case.get("case")) is not int or case["case"] != ordinal
                or set(case) != {"case", "m", "n", "k"}
                or any(type(case[name]) is not int or case[name] <= 0 for name in Case._fields)):
            raise ValueError("Invalid dense calibration shape or local case ID")
        key = Case(**{name: case[name] for name in Case._fields})
        if key in result:
            raise ValueError(f"Duplicate dense calibration shape {tuple(key)}")
        result[key] = ordinal
    return result


def missing_cases(required, available, visible=None):
    present = {shape for shape, case in shapes(available).items() if visible is None or case in visible}
    return [dict(case=index, **key._asdict())
            for index, key in enumerate(key for key in shapes(required) if key not in present)]


def merge(required, sources, physical_cards):
    """Join already re-audited raw collections to the complete current request.

    Each source tuple has its label, original cases, raw-derived result and visible
    case IDs. Returns current case provenance and the per-rank numerical catalog.
    """
    wanted = shapes(required)
    origins = {}
    by_rank = [{}, {}]
    for label, cases, summary, visible in sources:
        if (summary["family"] != "dense" or summary["complete"] is not True
                or [row["rank"] for row in summary["ranks"]] != [0, 1]
                or [row["physical_card"] for row in summary["ranks"]] != physical_cards):
            raise ValueError("Audited dense result belongs to another calibration pair")
        catalog = shapes(cases)
        timings = [row["timings"] for row in summary["ranks"]]
        if any([row["case"] for row in rank] != list(range(len(catalog))) for rank in timings):
            raise ValueError("Dense result has an incomplete or foreign case population")
        for shape in wanted.keys() & catalog.keys():
            case = wanted[shape]
            source_case = catalog[shape]
            if visible is not None and source_case not in visible:
                continue
            if case in origins:
                raise ValueError(f"More than one qualified clock source for dense shape {tuple(shape)}")
            for rank, clock in enumerate(timings):
                sample = clock[source_case]
                if (type(sample["seconds"]) not in (int, float)
                        or not math.isfinite(sample["seconds"]) or sample["seconds"] <= 0):
                    raise ValueError(f"Invalid dense timing for shape {tuple(shape)}, rank {rank}")
                by_rank[rank][case] = dict(sample, case=case)
            origins[case] = dict(case=case, source=label,
                source_plan_sha256=summary["plan_sha256"], source_case=source_case)
    if len(origins) != len(wanted):
        raise ValueError("Incomplete independent calibration after exact dense case resolution")
    return (
        [origins[case] for case in range(len(wanted))],
        [dict(rank=rank, physical_card=card, timings=[by_rank[rank][case] for case in range(len(wanted))])
         for rank, card in enumerate(physical_cards)],
    )
