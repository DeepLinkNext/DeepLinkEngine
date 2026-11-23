"""Read and watch a run without changing its plan or evidence."""

import argparse
import json
from pathlib import Path
import sys
import time


COMPLETED = {"completed", "reused_verified"}
TERMINAL = {"prepared", "completed", "completed_command_failed", "reference_incomplete", "failed"}
STATE_TEXT = {
    "preparing": "准备中", "running": "运行中", "prepared": "准备完成",
    "completed": "评测阶段已完成", "completed_command_failed": "评测阶段已完成，命令执行失败",
    "reference_incomplete": "性能评测已完成，参考认证未完成", "failed": "评测失败",
}
STAGE_TEXT = {
    "planned": "待执行", "running": "运行中", "completed": "完成",
    "reused_verified": "复用已校验结果", "failed": "失败", "not_run": "未执行",
}


class SnapshotUpdating(ValueError):
    """The running controller is updating an observation record."""


def read(path, default=None):
    try:
        return json.loads(path.read_bytes())
    except FileNotFoundError:
        return {} if default is None else default
    except json.JSONDecodeError as error:
        raise SnapshotUpdating(f"正在更新运行记录：{path}") from error


def stage_rows(plan, records, state, current, failure):
    by_name = {row["name"]: row for row in records}
    rows = []
    for index, item in enumerate(plan or records, 1):
        name = item["name"]
        record = by_name.get(name, {})
        status = record.get("status", "planned")
        if status == "planned" and state == "failed":
            status = "not_run"
        row = dict(index=index, name=name, status=status)
        for key in ("started", "ended", "returncode", "failure"):
            if key in record:
                row[key] = record[key]
        if current["stage"].partition("/")[0] == name:
            row["phase"] = current["phase"]
            if current["stage"] != name:
                row["active_arm"] = current["stage"].partition("/")[2]
            if status == "failed" and failure and not row.get("failure"):
                row["failure"] = failure
        rows.append(row)
    return rows


def current_item(run, current, rows, reference):
    from live_console import active

    name = current["stage"]
    parent, _, child = name.partition("/")
    if parent == "compute-reference":
        plan = read(run / "compute-reference/automatic-plan.json") or read(run / "plans/compute-reference/automatic-plan.json")
        recorded = reference.get("stages", [])
        declared = plan.get("stages", recorded)
        position = next((index for index, item in enumerate(declared, 1) if item["id"] == child), None)
        row = next((item for item in recorded if item["id"] == child), {})
        total = len(declared)
    else:
        row = next((item for item in rows if item["name"] == parent), {})
        position = row.get("index")
        total = len(rows)
    if position is None:
        return None

    _, leaf, active_name, _ = active(run)
    receipt = {}
    first_sample_at = None
    if leaf and active_name == name:
        try:
            receipt = read(leaf / "receipt.json")
        except SnapshotUpdating:
            pass
        if parent == "compute-reference" and child in {"baseline-native-synchronized-timing", "current-native-synchronized-timing"}:
            try:
                with (leaf / "device-samples.jsonl").open("rb") as stream:
                    sample = json.loads(stream.readline(65536))
                first_sample_at = sample.get("wall") if isinstance(sample, dict) else None
            except (OSError, ValueError):
                # Device sampling can start after the reference task is selected.
                pass
    # The parent MMLU timer covers both arms, so it cannot time the active arm.
    timing = {} if parent == "mmlu-pair" and child else row
    return dict(index=position, total=total, status=row.get("status"),
                started=timing.get("started", receipt.get("started")),
                ended=timing.get("ended", receipt.get("ended")), first_sample_at=first_sample_at)


def snapshot(run, *, include_item=False):
    from compute_reference import target as reference_target
    from live_console import progress

    run = run.expanduser().resolve()
    logs = run.with_name(run.name + "-logs")
    paths = {
        "campaign": run / "campaign.json", "checklist": run / "execution-checklist.json",
        "supplement": run / "execution.json", "reference": run / "compute-reference/status.json",
        "exit": logs / "exit.json",
    }
    if not run.is_dir() and not logs.is_dir():
        raise ValueError(f"run directory not found: {run}")
    plan = read(run / "frozen-stages.json", [])
    records = {name: read(path) for name, path in paths.items()}
    if not (plan or any(records.values()) or (run / "output-volume-admission.json").is_file() or logs.is_dir()):
        raise ValueError(f"no evaluation record under {run}")
    current = progress(run)
    if any(read(path) != records[name] for name, path in paths.items()):
        raise SnapshotUpdating("运行进度正在更新")

    campaign = records["campaign"]
    checklist = records["checklist"]
    supplement = records["supplement"]
    reference = records["reference"] or campaign.get("compute_reference") or {}
    campaign_status = campaign.get("status")
    returncode = records["exit"].get("returncode")
    if type(returncode) is not int:
        returncode = None
    report = run.with_name(run.name + "-report") / "report.html"
    if not report.is_file():
        report = run / "report.html"

    stages = checklist.get("stages") or campaign.get("stages") or []
    if not plan and not stages and supplement.get("schema") == "vllm-hust-acceptance-supplement-execution/v1":
        plan = [dict(name=role) for role in supplement.get("planned_roles", ["B0", "B1"])]
        stages = [dict(row, name=row["role"]) for row in supplement.get("stages", [])]
    if current["stage"] == "prepare" and not checklist.get("stages") and stages:
        latest = next((row for row in reversed(stages) if row["status"] != "planned"), None)
        if latest:
            current.update(stage=latest["name"], phase=latest["status"])

    reference_status = reference.get("status")
    reference_failure = None
    if reference_status not in (None, "not_requested"):
        qualified = reference_target(reference)
        if reference_status in ("blocked", "incomplete") or (
            reference_status == "measured" and qualified["status"] != "current_workload_measured"
        ):
            reference_failure = qualified["reason"]
    overall = campaign_status or supplement.get("status")
    if overall == "prepared_only":
        state = "prepared"
    elif overall == "failed":
        state = "failed"
    elif overall == "completed":
        state = "completed" if returncode in (None, 0) else "completed_command_failed"
        if reference_failure:
            state = "reference_incomplete"
    elif returncode is not None:
        state = "completed" if returncode == 0 else "failed"
    elif (run / "source-campaign.json").is_file() and report.is_file():
        state = "reference_incomplete" if reference_failure else "completed"
    elif any(row["status"] != "planned" for row in stages) or current["stage"] != "prepare":
        state = "running"
    else:
        state = "preparing"

    failure = current["failure"] or campaign.get("failure") or reference_failure
    rows = stage_rows(plan, stages, state, current, failure)
    reference_progress = None
    if reference_status is not None:
        reference_progress = dict(status=reference_status, failure=reference_failure)
        if current["stage"].startswith("compute-reference"):
            reference_progress.update(stage=current["stage"], phase=current["phase"])
            if reference.get("stages") or read(run / "compute-reference/automatic-plan.json").get("stages") \
                    or read(run / "plans/compute-reference/automatic-plan.json").get("stages"):
                reference_progress.update(stages_done=current["stages_done"], stages_total=current["stages_total"])

    summary = dict(current, state=state, campaign_status=campaign_status,
                   failure=failure, returncode=returncode, reference_status=reference_status,
                   report_path=str(report) if report.is_file() else None,
                   stages=rows, reference=reference_progress)
    if rows:
        summary.update(stages_done=sum(row["status"] in COMPLETED for row in rows), stages_total=len(rows))
    if include_item:
        summary["item"] = current_item(run, current, rows, reference)
    return summary


def emit(value, as_json, *, details):
    if as_json:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
        return
    fields = [f"状态 {STATE_TEXT[value['state']]}",
              f"阶段 {value['stages_done']}/{value['stages_total']}",
              f"{value['stage']} {value['phase']}"]
    if value["planned"] is not None:
        fields.append(f"请求 {value['responses']}/{value['planned']}")
    elif value["responses"]:
        fields.append(f"请求 {value['responses']}")
    if value["returncode"] is not None:
        fields.append(f"退出码 {value['returncode']}")
    print(" | ".join(fields), flush=True)
    if details:
        for row in value["stages"]:
            print(f"  [{row['index']:02}/{len(value['stages'])}] {row['name']}  "
                  + STAGE_TEXT.get(row["status"], row["status"]), flush=True)
    if reference := value["reference"]:
        count = f" {reference['stages_done']}/{reference['stages_total']}" if "stages_total" in reference else ""
        print(f"参考认证 {reference['status']}{count}", flush=True)
    if value["failure"]:
        print("失败原因 " + value["failure"], flush=True)
    if value["report_path"]:
        print("报告 " + value["report_path"], flush=True)


def main(argv):
    parser = argparse.ArgumentParser(prog="benchmark/one_click.py status", description="只读查看或持续观察评测进度。")
    parser.add_argument("--run", required=True, help="运行结果目录")
    parser.add_argument("--json", action="store_true", help="输出 JSON 快照")
    parser.add_argument("--watch", action="store_true", help="持续显示状态变化，直到运行结束")
    args = parser.parse_args(argv)
    if not args.run.strip():
        parser.error("--run requires a non-empty value")
    run = Path(args.run)
    previous = None
    try:
        while True:
            try:
                value = snapshot(run)
            except SnapshotUpdating:
                if not args.watch:
                    time.sleep(0.1)
                    value = snapshot(run)
                else:
                    time.sleep(1)
                    continue
            if value != previous:
                emit(value, args.json, details=previous is None or value["stages"] != previous["stages"])
            if not args.watch or value["state"] in TERMINAL:
                return 0
            previous = value
            time.sleep(1)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as error:
        print("STATUS_ERROR " + str(error), file=sys.stderr, flush=True)
        return 2
