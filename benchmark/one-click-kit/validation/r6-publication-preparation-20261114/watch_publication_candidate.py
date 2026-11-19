"""Prepare local review artifacts after the bound native campaign exits; no inference or network."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import traceback
import zipfile

sys.dont_write_bytecode = True


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def controller_alive(pid, ticks):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return stat[0] != "Z" and stat[19] == str(ticks)
    except (OSError, IndexError):
        return False


def main(config_path):
    config = json.loads(config_path.read_bytes())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.json"
    source = Path(config["source"])
    run = Path(config["run"])
    for name, expected in config["source_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError("Publication source changed: " + name)
    while controller_alive(config["controller_pid"], config["controller_start_ticks"]):
        save(
            status,
            dict(
                state="waiting_for_native_campaign_exit",
                observed=time.time(),
                run_id=run.name,
                interval_seconds=300,
                inference_started_by_this_watcher=False,
                native_controller_pid=config["controller_pid"],
            ),
        )
        time.sleep(300)
    if not (run / "campaign.json").is_file():
        raise RuntimeError(
            "Native controller exited without a completed campaign receipt"
        )
    receipt = json.loads((run / "campaign.json").read_bytes())
    if (
        receipt.get("status") != "completed"
        or receipt.get("selected_execution_complete") is not True
    ):
        save(
            status,
            dict(
                state="native_campaign_not_completed",
                observed=time.time(),
                receipt_status=receipt.get("status"),
                failed_stages=[
                    r["name"]
                    for r in receipt.get("stages", [])
                    if r.get("status") == "failed"
                ],
                inference_retries=0,
                review_complete=False,
            ),
        )
        return 1
    # Recheck source bindings immediately before executing the frozen postprocessor.
    for name, expected in config["source_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError("Publication source changed while waiting: " + name)
    save(
        status,
        dict(
            state="verifying_and_plotting", observed=time.time(), review_complete=False
        ),
    )
    metrics = output / "metrics"
    metrics.mkdir(exist_ok=False)
    summary = metrics / "summary-public-v1.json"
    python = config["report_python"]
    commands = [
        [
            python,
            str(source / "scripts/export_five_dataset_results.py"),
            "--run",
            str(run),
            "--engine-tools",
            config["engine_tools"],
            "--aggregation-commit",
            config["engine_commit"],
            "--output",
            str(summary),
        ],
        [
            python,
            str(source / "scripts/plot_five_dataset_results.py"),
            "--summary",
            str(summary),
            "--out",
            str(metrics),
        ],
        [
            python,
            str(source / "scripts/plot_project_final_goals.py"),
            "--summary",
            str(summary),
            "--output",
            str(metrics),
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    data = json.loads(summary.read_bytes())
    (metrics / "project-final-goals-v1.json").write_text(
        json.dumps(data["project_final_assessment"], ensure_ascii=False, indent=2)
        + "\n"
    )
    text = [
        f"本轮运行：{data['run_id']}",
        f"汇总执行提交：{data['aggregation_engine_commit']}",
        f"复用阶段：{data['reused_stages']}；新执行阶段：{data['new_inference_stages']}",
        "以下为脚本产出的数值与证据状态，尚待最终复核；未发布到网站。",
        "",
        "里程碑：",
    ]
    for name, goal in data["formal_targets"].items():
        text += [
            f"{name} {'达标' if goal['accepted'] else '未通过'}：{goal.get('display_value', '')}",
            goal["reason"],
        ]
    text += ["", "项目最终目标："]
    for name, goal in data["project_final_assessment"]["targets"].items():
        text += [f"{name} {goal['attainment']}：{goal['requirement']}", goal["reason"]]
    (metrics / "REVIEW.txt").write_text("\n".join(text) + "\n")
    checksums = {p.name: digest(p) for p in sorted(metrics.iterdir()) if p.is_file()}
    save(metrics / "checksums.json", checksums)
    archive = output / "candidate-results.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(metrics.iterdir()):
            bundle.write(path, path.name)
    save(
        status,
        dict(
            state="local_candidate_ready_for_review",
            observed=time.time(),
            run_id=run.name,
            archive_sha256=digest(archive),
            publication_source_commit=config["publication_commit"],
            execution_complete=True,
            review_complete=False,
            websites_updated=False,
            inference_retries=0,
        ),
    )
    return 0


if __name__ == "__main__":
    config_path = Path(sys.argv[1])
    try:
        raise SystemExit(main(config_path))
    except Exception as error:
        config = json.loads(config_path.read_bytes())
        output = Path(config["output"])
        output.mkdir(parents=True, exist_ok=True)
        save(
            output / "status.json",
            dict(
                state="candidate_preparation_failed",
                observed=time.time(),
                error=type(error).__name__ + ": " + str(error),
                review_complete=False,
                inference_retries=0,
            ),
        )
        traceback.print_exc()
        raise SystemExit(2)
