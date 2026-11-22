"""Record startup progress and read the full evaluation without running it."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import time
import uuid

from benchmark.recovery_source import alive, process_identity
from benchmark.run_status import SnapshotUpdating, read

ROOT = Path(__file__).resolve().parents[1]
RUN_ID_ENV = "DEEPLINKENGINE_RUN_PROGRESS_ID"
TERMINAL = {"prepared", "completed", "failed", "interrupted"}
LABELS = {
    "read-config": "读取配置",
    "prepare-official-sdk": "核验官方 SDK",
    "select-engine": "选择引擎版本和来源",
    "prepare-assessment-tools": "准备评测工具",
    "verify-recovery": "核验恢复与复用记录",
    "prepare-environment": "准备 CANN 和 Python 环境",
    "check-cann": "检查 CANN",
    "prepare-python": "准备 Python 环境",
    "python-handoff": "切换到已准备的 Python",
    "preflight": "检查运行环境",
    "initialize-run": "初始化验收输出",
    "verify-assessment-kit": "校验评测工具",
    "prepare-datasets": "准备固定数据",
    "install-package": "安装 B1 软件包和运行依赖",
    "prepare-local-python": "准备本机 Python 运行目录",
    "verify-engine": "校验 B1 软件包",
    "install-engine": "准备 B1 SDK",
    "prepare-fixed-five": "准备 B0 SDK 与五项评测输入",
    "run-fixed-five": "运行固定五项与有效算力参考",
    "verify-reference": "核验有效算力参考",
    "verify-installed-tools": "校验已安装的评测工具",
    "run-context-tenant": "运行长上下文和四租户测试",
    "verify-evidence": "核验验收证据",
    "render-full-report": "生成完整验收报告",
    "verify-report": "校验完整报告",
}
STATE_TEXT = {
    "not_started": "尚未启动",
    "running": "运行中",
    "prepared": "准备完成",
    "completed": "完成",
    "failed": "失败",
    "interrupted": "已中断",
}


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def announce(stage, detail=None):
    fields = [f"{LABELS.get(stage, stage)} ({stage})"]
    if detail:
        fields.append(LABELS.get(detail, detail))
    print("[PROGRESS] " + " | ".join(fields), flush=True)


def progress_path(out):
    return out.with_name(out.name + ".progress.json")


def write_snapshot(path, value):
    with NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def owner_alive(owner):
    if owner:
        try:
            return alive(owner)
        except (OSError, ValueError):
            pass
    return None


@contextmanager
def claim(path):
    if sys.platform != "linux":
        yield
        return
    import fcntl

    with path.with_name("." + path.name + ".lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


class BootstrapProgress:
    def __init__(self, out, started_at, rerun_b0):
        self.path = progress_path(out)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            owner = process_identity()
        except OSError:
            # Live ownership can only be verified where /proc is available.
            owner = {"pid": os.getpid()}
        with claim(self.path):
            previous = read(self.path)
            if previous.get("state") == "running" and previous.get("owner") == owner:
                self.data = previous
                return
            results = read(out / "results.json")
            if results.get("status") == "running" and owner_alive(results.get("controller_owner")):
                raise ValueError("A run is already executing this output: " + str(out))
            if previous.get("state") == "running" and owner_alive(previous.get("owner")):
                raise ValueError("A run is already preparing this output: " + str(out))
            self.data = dict(
                schema="deeplinkengine-run-progress/v1", run_id=uuid.uuid4().hex, owner=owner,
                state="running", started_at=started_at, rerun_b0=rerun_b0,
                stages=[dict(name="read-config", status="running", started_at=started_at)],
            )
            self.save()

    @property
    def run_id(self):
        return self.data["run_id"]

    def save(self):
        self.data["updated_at"] = timestamp()
        write_snapshot(self.path, self.data)

    def advance(self, stage):
        stages = self.data["stages"]
        if any(row["name"] == stage for row in stages):
            # An exec handoff revisits some already-recorded setup steps.
            return
        now = timestamp()
        stages[-1].update(status="completed", ended_at=now)
        stages.append(dict(name=stage, status="running", started_at=now))
        self.data.pop("detail", None)
        self.save()
        announce(stage)

    def detail(self, step, *, receipt=None):
        self.data["detail"] = dict(step=step)
        if receipt is not None:
            self.data["detail"]["receipt"] = str(receipt)
        self.save()
        announce(self.data["stages"][-1]["name"], step)

    def finish(self, state, error=None):
        last = self.data["stages"][-1]
        last.update(status=state if state in {"failed", "interrupted"} else "completed",
                    ended_at=timestamp())
        self.data["state"] = state
        if error:
            self.data["error"] = str(error) or type(error).__name__
        self.save()
        if state == "prepared":
            announce("prepare-environment", STATE_TEXT[state])


def configured_output(config):
    config = Path(config).resolve()
    data = json.loads(config.read_bytes())
    out = data.get("out") if isinstance(data, dict) else None
    if not isinstance(out, str) or not out.strip():
        raise ValueError("Expected an output path in " + str(config))
    return Path(os.path.abspath(config.parent / Path(out).expanduser()))


def record_config_error(config, started_at, error):
    try:
        out = configured_output(config)
        if not os.path.lexists(out):
            BootstrapProgress(out, started_at, None).finish("failed", error)
    except (OSError, ValueError):
        # An unreadable configuration cannot supply a writable observation path.
        pass


def environment_detail(detail, run_id):
    if not detail:
        return None
    current = {"step": detail["step"]}
    if "receipt" not in detail:
        return current
    path = Path(detail["receipt"])
    try:
        receipt = read(path)
    except (OSError, SnapshotUpdating):
        return current
    if receipt.get("progress_id") == run_id:
        latest = (receipt.get("steps") or [{}])[-1]
        if latest.get("status") in {"running", "launch_failed", "failed", "interrupted"}:
            current["step"] = latest["name"]
        current["receipt"] = str(path)
    return current


def evaluation_detail(out, stage):
    directory = {"run-fixed-five": "fixed-five", "run-context-tenant": "context-tenant"}.get(stage)
    if directory is None or not (out / directory).is_dir():
        return None
    scripts = str(ROOT / "benchmark/one-click-kit/scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from benchmark.run_status import snapshot as assessment_snapshot

    try:
        return assessment_snapshot(out / directory)
    except (OSError, ValueError):
        # A missing or partially written child receipt cannot hide the outer state.
        return None


def elapsed(started, ended=None):
    if started is None:
        return None
    end = datetime.fromisoformat(ended) if ended else datetime.now(timezone.utc)
    return max(0, int((end - datetime.fromisoformat(started)).total_seconds()))


def stage_summary(row):
    fields = ("name", "status", "mode", "started_at", "ended_at", "returncode", "error")
    return {field: row[field] for field in fields if field in row}


def snapshot(out):
    bootstrap = read(progress_path(out))
    results_path = out / "results.json"
    results = read(results_path)
    if bootstrap and results.get("progress_id") != bootstrap["run_id"]:
        # During in-place recovery, results.json still belongs to the prior attempt.
        results = {}

    stages = bootstrap.get("stages", []) + results.get("stages", [])
    row = results.get("activity") or (stages[-1] if stages else {})
    state = results.get("status", bootstrap.get("state", "not_started"))
    if state == "handed_off":
        state = "running"
    owner = results.get("controller_owner") or bootstrap.get("owner")
    controller_alive = owner_alive(owner) if state == "running" else None
    failure = results.get("error") or bootstrap.get("error") or row.get("error")
    if state == "failed" and row.get("status") == "interrupted":
        state = "interrupted"
    if state == "running" and controller_alive is False:
        state = "interrupted"
        failure = failure or "控制进程已退出，保留最后记录的阶段"

    stage = row.get("name")
    detail = (evaluation_detail(out, stage) if results
              else environment_detail(bootstrap.get("detail"), bootstrap.get("run_id")))
    if results and detail and detail.get("failure"):
        failure = detail["failure"]
    report = results.get("reports", {}).get("overview")
    return dict(
        state=state, run_id=bootstrap.get("run_id") or results.get("progress_id"), out=str(out),
        stage=stage, stage_label=LABELS.get(stage, stage), stage_status=row.get("status"), detail=detail,
        stages=[stage_summary(item) for item in stages], failure=failure, returncode=row.get("returncode"),
        started_at=bootstrap.get("started_at") or results.get("started_at"),
        updated_at=results.get("updated_at") or bootstrap.get("updated_at"),
        elapsed_seconds=elapsed(row.get("started_at"), row.get("ended_at")),
        rerun_b0=results.get("baseline_policy", {}).get("rerun_b0", bootstrap.get("rerun_b0")),
        controller_alive=controller_alive,
        results_path=str(results_path) if results else None,
        report_path=str(out / report) if report and (out / report).is_file() else None,
    )


def emit(value, as_json):
    if as_json:
        print(json.dumps(value, ensure_ascii=False), flush=True)
        return
    fields = [STATE_TEXT.get(value["state"], value["state"])]
    if value["stage_label"]:
        fields.append(value["stage_label"])
    if detail := value["detail"]:
        if "step" in detail:
            fields.append(LABELS.get(detail["step"], detail["step"]))
        else:
            fields.append(f"{detail['stage']} {detail['phase']}")
            fields.append(f"评测阶段 {detail['stages_done']}/{detail['stages_total']}")
            if reference := detail.get("reference"):
                if "stages_total" in reference:
                    fields.append(f"参考认证 {reference['stages_done']}/{reference['stages_total']}")
            if detail["planned"] is not None:
                fields.append(f"请求 {detail['responses']}/{detail['planned']}")
    if value["rerun_b0"] is not None:
        fields.append("B0 计划重跑" if value["rerun_b0"] else "B0 计划复用")
    if value["elapsed_seconds"] is not None:
        fields.append(f"本阶段 {value['elapsed_seconds']}s")
    if value["failure"]:
        fields.append(value["failure"])
    if value["report_path"]:
        fields.append(value["report_path"])
    print("[PROGRESS] " + " | ".join(fields), flush=True)


def observe(config, *, watch=False, as_json=False):
    try:
        out = configured_output(config)
        previous = None
        last_emit = 0
        while True:
            try:
                value = snapshot(out)
            except SnapshotUpdating:
                time.sleep(1 if watch else 0.1)
                if watch:
                    continue
                value = snapshot(out)
            change = {key: item for key, item in value.items() if key != "elapsed_seconds"}
            now = time.monotonic()
            if change != previous or now - last_emit >= 30:
                emit(value, as_json)
                previous = change
                last_emit = now
            if not watch or value["state"] in TERMINAL:
                return 0
            time.sleep(1)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as error:
        print("STATUS_ERROR " + str(error), file=sys.stderr, flush=True)
        return 2
