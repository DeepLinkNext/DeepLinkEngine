"""Execute the pinned HLE/Frontier clients with audited streaming transport."""

import argparse
import json
from pathlib import Path

from dataset_contract import client, sha, save, DATASETS
from streaming_transport import call, streamed_body


def checked(spec):
    spec = Path(spec).resolve()
    data = json.loads(spec.read_bytes())
    if data["dataset"] in ("swe", "terminal"):
        import agent_trace

        return agent_trace.checked(spec), agent_trace
    module = client(data["dataset"])
    plan = Path(data["plan"])
    if sha(plan / "contract.json") != data["contract_sha256"]:
        raise ValueError("Frozen dataset contract changed")
    contract, tasks = module.load_plan(plan)
    if contract["engine_transport"]["adapter_sha256"] != sha(
        Path(__file__).with_name("streaming_transport.py")
    ):
        raise ValueError("Stream adapter changed after preparation")
    if data["dataset"] == "hle":
        from hle_scope import verify_selected

        verify_selected(plan, contract, tasks)
    elif len(tasks) != DATASETS[data["dataset"]]["tasks"]:
        raise ValueError("Full dataset denominator changed")
    ids = data["case_ids"]
    if len(ids) != len(set(ids)) or set(ids) - {t["case_id"] for t in tasks}:
        raise ValueError("Unknown or duplicate dataset case selection")
    if not data["diagnostic"] and ids != [t["case_id"] for t in tasks]:
        raise ValueError("Formal dataset execution cannot select a smaller subset")
    if data["stage"] == "judge" and data["service_role"] != "B1":
        raise ValueError(
            "Both solver arms must use the same pinned B1 substitute judge"
        )
    return data, module


def run(spec, out):
    data, module = checked(spec)
    if data["dataset"] in ("swe", "terminal"):
        return module.run(spec, out)
    if data["dataset"] == "hle" and data["stage"] == "solver":
        from image_qualification import qualify

        contract, _ = module.load_plan(data["plan"])
        save(
            out / "image-interface-qualification.json",
            qualify(contract["solver"], out / "image-qualification"),
        )
    original = module.request_body
    module.request_body = lambda config, messages: streamed_body(
        original, config, messages
    )
    module.call = lambda config, body: call(config, body, out / "transport")
    args = argparse.Namespace(
        plan=data["plan"],
        output=str(out / "evaluation"),
        responses=data.get("responses"),
        resume=False,
        limit=None,
        track=None,
        case_ids=data["case_ids"],
    )
    start = __import__("time").monotonic()
    original_plan = module.load_plan
    if data["dataset"] == "hle" and data["stage"] == "judge":
        from judge_grammar import effective_format

        def grading_plan(path):
            contract, tasks = original_plan(path)
            contract = dict(contract)
            contract["judge_response_format"] = effective_format(
                data, contract["judge_response_format"]
            )
            return contract, tasks

        module.load_plan = grading_plan
    try:
        result = module.execute(args, data["stage"])
    finally:
        module.load_plan = original_plan
    end = __import__("time").monotonic()
    meta, records = module.verify_run(
        out / "evaluation", data["contract_sha256"], data["stage"]
    )
    if set(meta["case_ids"]) != set(data["case_ids"]) or len(records) != len(
        data["case_ids"]
    ):
        raise ValueError("Dataset execution omitted planned cases")
    statuses = {}
    for row in records.values():
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    save(
        out / "dataset-phase.json",
        {
            "dataset": data["dataset"],
            "stage": data["stage"],
            "service_role": data["service_role"],
            "solver_role": data["solver_role"],
            "spec_sha256": sha(Path(spec)),
            "start_monotonic": start,
            "end_monotonic": end,
            "planned": len(data["case_ids"]),
            "full_dataset_tasks": DATASETS[data["dataset"]]["tasks"],
            "statuses": statuses,
            "diagnostic": data["diagnostic"],
            "formal_complete": False,
            "original_client_result": result,
        },
    )
    # Invalid judge answers remain unresolved; transport/solver errors are
    # execution failures and must not become a successful shell exit.
    return (
        2
        if any(
            name not in ("ok", "invalid_judge", "solver_failed") for name in statuses
        )
        else 0
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return run(args.spec, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
