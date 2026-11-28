"""Read device-task clocks gated to the frozen original018/r11 RC4 pair."""

from decimal import Decimal
from hashlib import sha256
import re

from canonical_dense_cache import missing_cases
from dataset_contract import save, sha
from reference_9b_archive import B0_SEAL, CARDS, PINS, RELEASE, REVISION, member, read
from reference_9b_dense import PLAN_PIN
from reference_9b_ffn_branch import WITNESS as FFN_BRANCH_WITNESS
from reference_9b_task_clock_results import audit

DOMAIN = "cann-device-task-duration/v1"
DEMAND_PINS = frozenset({
    "260e3bf1c51893df790c3b10e6f9e14038e7ec67b8810b35860c77ee119278c0",
    "d5816f4909c5abd88e9a43f99f3d36eab07116d61e12daffd13a51596f768091",
})
COMBINED_WITNESS = sha256("".join(sorted(DEMAND_PINS)).encode("ascii")).hexdigest()
PREFILL_PROOF = "30f7921a4e28ba9ea5b76c6a5b3473811e948bc6c4a0c85a8a6b089569190d0f"
PREFILL_WITNESS = sha256("".join(sorted(DEMAND_PINS | {PREFILL_PROOF})).encode("ascii")).hexdigest()
ROWS = (3, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 108)
DIMS = ((4096, 6144), (12288, 4096))
PREFILL_SHAPES = tuple((1369, n, k) for n, k in DIMS)


def shape(case):
    return case["m"], case["n"], case["k"]


def recovered_prefill(root, envelope, terminal, cases):
    if terminal["status"] != "requalified_components" or tuple(cases) != PREFILL_SHAPES:
        return False
    original = read(member(root, "original-execution.json", terminal["supersedes_execution_sha256"]))
    release = read(member(root, "posthoc-release-readback.json", terminal["observed_release_sha256"]))
    return (original["status"] == "failed" and original["component_cases"] == len(cases)
            and original["error"].startswith("ValueError: Unexpected card or device task in ")
            and "/device-task-profiles/rank-0/case-000/" in original["error"]
            and release["execution.json"] == terminal["supersedes_execution_sha256"] == envelope["original-execution.json"]
            and release["quota.json"] == envelope["allocation/quota.json"]
            and release["locks-after.txt"] == envelope["allocation/locks-after.txt"]
            and release["npu-after.txt"] == envelope["allocation/npu-after.txt"]
            and release["physical_cards"] == CARDS and release["verified_free_leases"] == 18)


def request(clocks, prefill_proof=None, ffn_branches=None):
    expected = {"B0": B0_SEAL, "B1": PINS["native-clock/B1/checksums.json"]}
    if {key: arm["checksums_sha256"] for key, arm in clocks.items()} != expected:
        raise ValueError("9B device task request belongs to different native timing windows")
    shapes = [(m, n, k) for m in ROWS for n, k in DIMS]
    pins = DEMAND_PINS
    if prefill_proof is not None:
        proof = read(member(prefill_proof.parent, prefill_proof.name, PREFILL_PROOF))
        if (proof["native_seal_sha256"] != expected["B0"]
                or [shape(row) for row in proof["ffn_bf16_cases"]] != list(PREFILL_SHAPES)
                or [row["repetition"] for row in proof["rounds"]] != [1, 2, 3]
                or any(row["packet"] != 0 or row["rows"] != 1369 for row in proof["rounds"])):
            raise ValueError("The extra original018 first-prefill operands are outside the archived native proof")
        shapes.extend(PREFILL_SHAPES)
        pins = pins | {PREFILL_PROOF}
    if ffn_branches is not None:
        additional = [shape(row) for row in ffn_branches["kernel_demand"]]
        candidates = {(m, n, k) for m in range(1, 109) for n, k in DIMS}
        expected_width_gaps = sorted(candidates - set(shapes))
        if additional != expected_width_gaps:
            raise ValueError("The full FFN branch plan does not match the absent exact calibrated shapes")
        shapes.extend(additional)
        pins = pins | {FFN_BRANCH_WITNESS}
    cases = [dict(case=index, m=m, n=n, k=k) for index, (m, n, k) in
             enumerate(shapes)]
    return dict(schema="vllm-hust-9b-device-task-clock-cache/v1", timing_domain=DOMAIN,
        checkpoint_revision=REVISION, release=RELEASE, native_seals=expected,
        bf16_tp2_physical_cards=CARDS, cases=cases, observed_demand_sha256=sorted(pins))


def audit_panel(root, manifest_sha256, authority, wanted):
    envelope = read(member(root, "readback-checksums.json", manifest_sha256))
    for name, checksum in envelope.items():
        member(root, name, checksum)
    terminal = read(member(root, "execution.json", envelope["execution.json"]))
    admission = read(member(root, "source-admission.json", envelope["source-admission.json"]))
    plan_path = member(root, "dense-reference-plan-r2.json", envelope["dense-reference-plan-r2.json"])
    plan = read(plan_path)
    case_shapes = [shape(item) for item in plan["cases"]]
    witnesses = DEMAND_PINS | {COMBINED_WITNESS}
    if PREFILL_PROOF in wanted["observed_demand_sha256"]:
        witnesses = witnesses | {PREFILL_WITNESS}
    if FFN_BRANCH_WITNESS in wanted["observed_demand_sha256"]:
        witnesses = witnesses | {FFN_BRANCH_WITNESS}
    if ((terminal["status"] != "completed_components" and not recovered_prefill(root, envelope, terminal, case_shapes))
            or terminal["component_cases"] != len(case_shapes)
            or admission["physical"] != admission["logical"] or admission["physical"] != CARDS
            or admission["versions"] != {"torch": "2.13.0+cpu", "torch-npu": "2.13.0rc1"}
            or admission["source_bindings_verified"] != len(authority["current_source_sha256"])
            or admission["sampler_sha256"] != envelope["dense_reference_repeat.py"]
            or admission["plan_sha256"] != envelope[plan_path.name]
            or plan["witnessed_operands_sha256"] not in witnesses
            or plan["current_source_sha256"] != authority["current_source_sha256"]
            or any(plan[name] != authority[name] for name in ("warmups", "seed", "atol", "rtol"))
            or not set(case_shapes) <= {shape(row) for row in wanted["cases"]}):
        raise ValueError("Cached 9B task clock has a foreign runtime, shape, sample boundary, or demand")
    quota = read(member(root, "allocation/quota.json", envelope["allocation/quota.json"]))
    npu = member(root, "allocation/npu-after.txt", envelope["allocation/npu-after.txt"]).read_text()
    locks = member(root, "allocation/locks-after.txt", envelope["allocation/locks-after.txt"]).read_text()
    held = quota["allocation"] + quota["admission"]["exclusive_leases"]
    if (quota["physical"] != quota["logical"] or quota["physical"] != CARDS
            or len(held) != 18 or any(name in locks for name in held)
            or any(not re.search(rf"No running processes found in NPU {card}\s", npu) for card in CARDS)):
        raise ValueError("Cached 9B task clock did not release the exact physical card pair")
    for rank in (0, 1):
        name = f"dense-reference-r2/rank-{rank}.json"
        member(root, name, envelope[name])
    result = audit(root / "dense-reference-r2", plan_path, CARDS)
    if result != read(member(root, "component-timings.json", envelope["component-timings.json"])):
        raise ValueError("Raw device task panel does not reproduce the cached medians")
    for rank in result["ranks"]:
        for sample in rank["timings"]:
            if sample["task_trace_sha256"] != envelope.get(sample["task_trace"]):
                raise ValueError(f"Device tracing source is outside the sealed panel: {sample['task_trace']}")
    return plan["cases"], result


def resolve(archive, independent, cache, cache_seal, folder, clocks, baseline_sdk, candidate_sdk, prefill_proof=None, ffn_branches=None):
    manifest = read(member(cache, "checksums.json", cache_seal))
    index = read(member(cache, "task-clock-cache.json", manifest["task-clock-cache.json"]))
    wanted = request(clocks, prefill_proof, ffn_branches)
    if index["request"] != wanted:
        raise ValueError("The 9B clock cache request or exact native measurement seal differs")
    authority = read(member(independent, "dense-reference-plan-r2.json", PLAN_PIN))
    roots = {"B0": baseline_sdk, "B1": candidate_sdk}
    for name, expected in authority["current_source_sha256"].items():
        role, relative = name.split("/", 1)
        member(roots[role], relative, expected)
    by_shape = {}
    for panel in index["panels"]:
        name = panel["directory"] + "/readback-checksums.json"
        member(cache, name, manifest[name])
        cases, measured = audit_panel(cache / panel["directory"], manifest[name], authority, wanted)
        for row in cases:
            key = shape(row)
            if key in by_shape:
                raise ValueError(f"More than one clock source claims the 9B device task {key}")
            by_shape[key] = dict(source="retained", archive=panel["directory"], source_plan_sha256=measured["plan_sha256"],
                rank_clock_source=[rank["timings"][row["case"]] for rank in measured["ranks"]])
    visible = index.get("visible_cases", list(range(len(wanted["cases"]))))
    if (not isinstance(visible, list) or any(type(item) is not int or item not in range(len(wanted["cases"]))
            for item in visible) or len(set(visible)) != len(visible)):
        raise ValueError("Isolated device task clock cache visibility differs from the frozen request")
    qualified = [row for row in wanted["cases"] if row["case"] in visible and shape(row) in by_shape]
    present = [dict(row, case=case) for case, row in enumerate(qualified)]
    needed = missing_cases(wanted["cases"], present)
    selection = folder / "device-task-clock-selection.json"
    selection_doc = dict(request=wanted, cache_sha256=cache_seal, retained_cases=len(present),
        missing_cases=needed, sampler_invocations=int(bool(needed)))
    save(selection, selection_doc)
    if needed:
        from reference_9b_task_sampling import calibrate

        witness = COMBINED_WITNESS if prefill_proof is None else PREFILL_WITNESS
        if ffn_branches is not None:
            # The existing operator catalogue admits every decode row size.
            # Preserve the separate native proof for a removed cold-prefill clock.
            if any(shape(row) in PREFILL_SHAPES for row in needed):
                raise ValueError("Restore the separate original018 cold-prefill clocks before the FFN branch panel")
            witness = FFN_BRANCH_WITNESS
        job, seal, dispatch = calibrate(archive, independent, baseline_sdk, candidate_sdk,
            folder, clocks, needed, witness)
        cases, measured = audit_panel(job, seal, authority, wanted)
        if [shape(row) for row in cases] != [shape(row) for row in needed]:
            raise ValueError("9B device task sampler supplied a different set from the actual missing operations")
        for row in cases:
            by_shape[shape(row)] = dict(source="sampled", archive=str(job), archive_manifest_sha256=seal,
                source_plan_sha256=measured["plan_sha256"],
                rank_clock_source=[rank["timings"][row["case"]] for rank in measured["ranks"]])
        selection_doc["dispatch_sha256"] = sha(dispatch)
        save(selection, selection_doc)
    rows = []
    for case in wanted["cases"]:
        record = by_shape[shape(case)]
        if any(Decimal(rank["seconds"]) <= 0 for rank in record["rank_clock_source"]):
            raise ValueError(f"Invalid independent CANN device task duration for {shape(case)}")
        rows.append(dict(case=case, **record))
    output = folder / "qualified-device-task-clocks.json"
    save(output, dict(timing_domain=DOMAIN, selection_sha256=sha(selection), cases=rows))
    return dict(index=output.relative_to(folder.parent).as_posix(), index_sha256=sha(output),
        timing_domain=DOMAIN, retained=len(present), sampled=len(needed), new_sampler_invocations=int(bool(needed)))
