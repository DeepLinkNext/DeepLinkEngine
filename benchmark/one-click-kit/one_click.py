#!/usr/bin/env python3
"""vLLM-HUST Engine one-click third-party inference, scoring and plots.

Two entry styles share the same controller:

    python3 one_click.py prepare --config evaluation.json
    python3 one_click.py run     --config evaluation.json
    python3 one_click.py --engine-profile r6 --model ... --out ...   (direct arguments)

The dashboard subcommand reads existing reports without invoking the controller.
The reference-9b subcommand resumes the sealed original018/r11 RC4 submission.

MMLU-Pro uses the frozen200 subset of source test12032, not the official full
test. Generation seed0 is not a sampling seed. See protocols/mmlu-fixed200.json.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from judge_disclosure import NOTICE_ZH  # noqa: E402 - bundled scripts

SUBCOMMANDS = ("prepare", "run", "status")
PROFILES = ("r4", "r5", "r6", "r7", "r8")
TEST_MODES = {"full": "mmlu,hle,frontier,swe,terminal", "quick": "mmlu"}
CONFIG_FIELDS = (
    "engine_profile",
    "test_mode",
    "reference_profile",
    "model",
    "b1_python",
    "deployment",
    "physical_devices",
    "cache",
    "out",
    "baseline_deployment",
    "dataset_assets",
    "devices",
    "env_file",
)
PATH_FIELDS = (
    "model",
    "b1_python",
    "deployment",
    "cache",
    "out",
    "baseline_deployment",
    "dataset_assets",
    "env_file",
)
SUBCOMMAND_HELP = """用法：
  python3 one_click.py prepare --config evaluation.json [--out DIR]
  python3 one_click.py run     --config evaluation.json [--out DIR]

prepare 只检查准备情况并冻结计划，默认写入 <out>-prepare；run 执行完整评测。
默认 full 为固定五项，quick 只跑 MMLU200；两种模式均自动准备并执行本轮参考认证，共用同一 cache。

配置字段（相对路径以配置文件所在目录为基准）：
  test_mode             full|quick，默认 full；固定五项或仅 MMLU200
  reference_profile     submission-v2（默认）：MMLU200/out256，每臂三轮；full-answer-v1 复现旧开发采集
  engine_profile        r4|r5|r6|r7|r8，默认 r6
  model                 模型目录（必填）
  deployment            已有 B1 部署目录；首次恢复 B1 时改填 b1_python（两者至少填一个）
  b1_python             固定 B1 SDK 的 python
  physical_devices      两个不同的非负整数，例如 [0, 1]（必填）
  cache                 缓存目录；未填时使用控制器默认用户缓存
  out                   结果目录（必填）；prepare 默认使用 <out>-prepare
  devices               逻辑设备，两个不同的非负整数，例如 [0, 1]
  baseline_deployment   已有 B0 original018 SDK
  dataset_assets        已有 hle/ 与 frontier/ 数据根目录
  env_file              GitHub 凭据 .env

示例见 examples/evaluation.json。报告写入 <out>-report/report.html。
状态查询：python3 one_click.py status --run <目录> [--json]。
查看多个批次的汇总用法见 python3 one_click.py dashboard --help。
原有直接传参方式继续可用：python3 one_click.py --help 查看控制器参数。"""
STATUS_HELP = """用法：
  python3 one_click.py status --run <目录> [--json]

一次性只读查询，不写入结果目录、不启动任何进程；环境准备期间即可查询，此时显示 prepare 阶段。
  --run    评测结果目录，例如 <out> 或 <out>-prepare（必填）
  --json   只输出一个 JSON 对象，未知值为 null

state 取值：
  prepared                 准备完成（campaign status=prepared_only）
  completed                评测阶段已完成且命令成功
  completed_command_failed 评测阶段已完成，但退出码非零
  failed                   评测失败，failure 字段给出原因
  unknown                  缺少终态回执；显示最后记录的阶段，进程状态未知

退出码：查询成功为0，即使被查询的评测失败；目录无效或无评测记录为2。"""


class ConfigError(ValueError):
    """Entry-level configuration problem; always names the offending field."""


class StatusError(ValueError):
    """The run directory cannot be read as an evaluation record."""


def resolve_path(base, value, field):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            field + " must be a non-empty path string, got " + repr(value)
        )
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


def device_pair(value, field):
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(
            field
            + " must contain two distinct non-negative integers, for example [0, 1], got "
            + repr(value)
        )
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ConfigError(
                field + " must contain only non-negative integers, got " + repr(value)
            )
    if value[0] == value[1]:
        raise ConfigError(
            field + " must contain two distinct devices, got " + repr(value)
        )
    return str(value[0]) + "," + str(value[1])


def device_value(value, field):
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
        if any(not item.isdigit() for item in parts):
            raise ConfigError(
                field
                + " must contain two distinct non-negative integers, got "
                + repr(value)
            )
        value = [int(item) for item in parts]
    return device_pair(value, field)


def parse_config(subcommand, arguments):
    """Parse and validate the subcommand arguments; None means help was shown."""
    if any(item in ("--help", "-h") for item in arguments):
        print(SUBCOMMAND_HELP, flush=True)
        return None
    options = {"--config": None, "--out": None}
    remaining = list(arguments)
    while remaining:
        token = remaining.pop(0)
        if token in options:
            if not remaining:
                raise ConfigError(token + " requires a path")
            value = remaining.pop(0)
            if not value.strip():
                raise ConfigError(token + " requires a non-empty path")
            options[token] = value
        elif token.startswith("--config=") or token.startswith("--out="):
            name, value = token.split("=", 1)
            if not value.strip():
                raise ConfigError(name + " requires a non-empty path")
            options[name] = value
        else:
            raise ConfigError(
                "unknown argument for "
                + subcommand
                + ": "
                + token
                + "; run `python3 one_click.py "
                + subcommand
                + " --help`"
            )
    if not options["--config"]:
        raise ConfigError(
            "--config is required; run `python3 one_click.py "
            + subcommand
            + " --help` for the field list"
        )
    return options["--config"], options["--out"]


def configuration(subcommand, arguments):
    """Turn one evaluation.json into controller arguments."""
    parsed = parse_config(subcommand, arguments)
    if parsed is None:
        return None
    config, override = parsed
    config = Path(config).expanduser()
    if not config.is_file():
        raise ConfigError("configuration file not found: " + str(config))
    try:
        data = json.loads(config.read_bytes())
    except json.JSONDecodeError as error:
        raise ConfigError(str(config) + " is not valid JSON: " + str(error))
    if not isinstance(data, dict):
        raise ConfigError(str(config) + " must contain a JSON object")
    unknown = sorted(set(data) - set(CONFIG_FIELDS))
    if unknown:
        raise ConfigError(
            "unknown configuration field(s) in "
            + str(config)
            + ": "
            + ", ".join(unknown)
            + "; allowed fields: "
            + ", ".join(CONFIG_FIELDS)
        )
    base = config.resolve().parent
    for field in ("model", "out", "physical_devices"):
        if field not in data:
            raise ConfigError(field + " is required in " + str(config))
    if not data.get("b1_python") and not data.get("deployment"):
        raise ConfigError(
            "b1_python or deployment is required in "
            + str(config)
            + "; use b1_python for a first restore and deployment for an existing B1 tree"
        )
    profile = data.get("engine_profile", "r6")
    if profile not in PROFILES:
        raise ConfigError(
            "engine_profile must be one of "
            + "|".join(PROFILES)
            + ", got "
            + repr(profile)
        )
    mode = data.get("test_mode", "full")
    if not isinstance(mode, str) or mode not in TEST_MODES:
        raise ConfigError("test_mode must be full|quick, got " + repr(mode))
    reference_profile = data.get("reference_profile", "submission-v2")
    if reference_profile not in ("submission-v2", "full-answer-v1"):
        raise ConfigError("reference_profile must be submission-v2|full-answer-v1")
    paths = {}
    for field in PATH_FIELDS:
        if field in data:
            paths[field] = resolve_path(base, data[field], field)
    if override is not None:
        out = resolve_path(Path.cwd(), override, "--out")
    else:
        out = paths["out"]
        if subcommand == "prepare":
            out = out.with_name(out.name + "-prepare")
    # A dangling symlink still occupies the path: the launcher or report
    # generator would fail on mkdir(exist_ok=False) after the campaign started.
    conflicts = [
        path
        for path in (
            out,
            out.with_name(out.name + "-logs"),
            out.with_name(out.name + "-report"),
        )
        if os.path.lexists(path)
    ]
    if conflicts:
        raise ConfigError(
            "these output paths are already occupied: "
            + ", ".join(str(path) for path in conflicts)
            + "; choose a new directory with --out or change out in "
            + str(config)
        )
    args = [
        "--engine-profile",
        profile,
        "--model",
        str(paths["model"]),
        "--out",
        str(out),
        "--physical-devices",
        device_pair(data["physical_devices"], "physical_devices"),
        "--datasets",
        TEST_MODES[mode],
        "--reference-profile",
        reference_profile,
    ]
    for field, flag in (
        ("cache", "--cache"),
        ("deployment", "--deployment"),
        ("b1_python", "--b1-python"),
        ("baseline_deployment", "--baseline-deployment"),
        ("dataset_assets", "--dataset-assets"),
        ("env_file", "--env-file"),
    ):
        if field in paths:
            args += [flag, str(paths[field])]
    if "devices" in data:
        args += ["--devices", device_value(data["devices"], "devices")]
    if subcommand == "prepare":
        args.append("--prepare-only")
    return args


def read_record(path):
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def status_value(run):
    """Compose one read-only status object from the same files the console reads."""
    from live_console import progress

    # The launcher resolves the directory before it derives -logs and -report;
    # a query through a link must land on the same records.
    run = Path(run).expanduser().resolve()
    if not run.is_dir():
        raise StatusError("run directory not found: " + str(run))
    campaign_path = run / "campaign.json"
    checklist_path = run / "execution-checklist.json"
    # SDK restoration and data preparation start before the first checklist is
    # written, so those runs are queryable instead of looking absent.
    records = (
        campaign_path,
        checklist_path,
        run / "output-volume-admission.json",
        run / "execution.json",
        run / "compute-reference/status.json",
    )
    if not any(path.is_file() for path in records):
        raise StatusError(
            "no evaluation record under "
            + str(run)
            + "; expected campaign.json, execution-checklist.json"
            " or an initial preparation record"
        )
    campaign = read_record(campaign_path)
    campaign_status = campaign.get("status")
    try:
        current = progress(run)
    except (OSError, ValueError) as error:
        raise StatusError("cannot read progress for " + str(run) + ": " + str(error))
    exit_record = read_record(run.with_name(run.name + "-logs") / "exit.json")
    returncode = exit_record.get("returncode")
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        returncode = None
    if campaign_status == "prepared_only":
        state = "prepared"
    elif campaign_status == "completed":
        state = (
            "completed_command_failed" if returncode not in (None, 0) else "completed"
        )
    elif campaign_status == "failed":
        state = "failed"
    else:
        state = "unknown"
    reference = read_record(run / "compute-reference/status.json") or campaign.get("compute_reference") or {}
    from compute_reference import target as reference_target

    reference_status = reference.get("status")
    reference_failure = None
    if reference_status and reference_status != "not_requested":
        qualified = reference_target(reference)
        if reference_status in ("blocked", "incomplete") or (
            reference_status == "measured" and qualified["status"] != "current_workload_measured"
        ):
            if campaign_status == "completed":
                state = "reference_incomplete"
            reference_failure = qualified["reason"]
    report = run.with_name(run.name + "-report") / "report.html"
    return {
        "state": state,
        "campaign_status": campaign_status,
        "stage": current.get("stage"),
        "phase": current.get("phase"),
        "stages_done": current.get("stages_done"),
        "stages_total": current.get("stages_total"),
        "responses": current.get("responses"),
        "planned": current.get("planned"),
        "returncode": returncode,
        "failure": campaign.get("failure") or current.get("failure") or reference_failure,
        "reference_status": reference_status,
        "report_path": str(report) if report.is_file() else None,
    }


STATE_TEXT = {
    "prepared": "准备完成",
    "completed": "评测阶段已完成",
    "completed_command_failed": "评测阶段已完成；命令执行失败",
    "reference_incomplete": "性能评测已完成；完整参考认证未完成",
    "failed": "评测失败",
    "unknown": "进程状态未知（缺少终态回执）",
}


def status_lines(value):
    stage = value["stage"] or "未知"
    stages = "%s/%s 阶段" % (value["stages_done"], value["stages_total"])
    if value["planned"] is not None:
        responses = "%s/%s 请求" % (value["responses"], value["planned"])
    elif value["responses"]:
        responses = "%s 个请求" % value["responses"]
    else:
        responses = "无已返回请求"
    yield "状态: " + STATE_TEXT[value["state"]]
    yield "campaign: " + (value["campaign_status"] or "未知")
    yield "阶段: " + stage + "（" + stages + "，" + responses + "）"
    yield "退出码: " + (
        "未知" if value["returncode"] is None else str(value["returncode"])
    )
    yield "失败原因: " + (value["failure"] or "无")
    yield "报告: " + (value["report_path"] or "未生成")


def status_arguments(arguments):
    """Parse the status arguments; None means help was shown."""
    if any(item in ("--help", "-h") for item in arguments):
        print(STATUS_HELP, flush=True)
        return None
    run = None
    as_json = False
    remaining = list(arguments)
    while remaining:
        token = remaining.pop(0)
        if token == "--run":
            if not remaining:
                raise StatusError("--run requires a path")
            run = remaining.pop(0)
            if not run.strip():
                raise StatusError("--run requires a non-empty path")
        elif token.startswith("--run="):
            run = token.split("=", 1)[1]
            if not run.strip():
                raise StatusError("--run requires a non-empty path")
        elif token == "--json":
            as_json = True
        else:
            raise StatusError(
                "unknown argument for status: "
                + token
                + "; run `python3 one_click.py status --help`"
            )
    if run is None:
        raise StatusError(
            "--run is required; run `python3 one_click.py status --help`"
        )
    return run, as_json


def status_command(arguments):
    try:
        parsed = status_arguments(arguments)
        if parsed is None:
            return 0
        value = status_value(parsed[0])
    except StatusError as error:
        print("STATUS_ERROR " + str(error), file=sys.stderr, flush=True)
        return 2
    if parsed[1]:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    else:
        for line in status_lines(value):
            print(line, flush=True)
    return 0


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "reference-9b":
        from reference_9b import main as reference_9b

        return reference_9b(args[1:])
    if args and args[0] == "dashboard":
        from acceptance_dashboard import main as dashboard_main

        return dashboard_main(args[1:])
    if args and args[0] in SUBCOMMANDS:
        if args[0] == "status":
            return status_command(args[1:])
        try:
            configured = configuration(args[0], args[1:])
        except ConfigError as error:
            print("CONFIG_ERROR " + str(error), file=sys.stderr, flush=True)
            return 2
        if configured is None:
            return 0
        return launch(configured)
    return launch(args)


def launch(args):
    args = list(args)
    supplement = "--context-tenant-only" in args
    args = [x for x in args if x != "--context-tenant-only"]
    follow = "--no-follow" not in args
    args = [x for x in args if x != "--no-follow"]
    if not any(x == "--model" or x.startswith("--model=") for x in args):
        args += ["--model", "/models/Qwen3.5-35B-A3B"]
    if not any(x == "--out" or x.startswith("--out=") for x in args):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args += ["--out", str(Path.cwd() / "runs" / stamp)]
    if not any(x in ("--help", "-h") for x in args):
        print("ENTRYPOINT vLLM-HUST/vllm-hust-engine/one_click.py", flush=True)
        if supplement:
            print(
                "专项补测：真实4K/16K/32K与四租户B0/B1配对；不重跑已封存五数据集，不包含有效算力参考标定。",
                flush=True,
            )
        else:
            print(
                "MMLU-Pro: 固定200题子集，源test12,032题、14学科；"
                "复用原题号及顺序，生成seed0不是抽样seed。"
                "原始清单见protocols/mmlu-fixed200.json。",
                flush=True,
            )
            print(
                "HLE默认固定分层200题（Gold文本64/图像10，Revision文本113/图像13）；"
                "原完整回答及裁判预算不变。显式--hle-profile full1811可执行全集。",
                flush=True,
            )
            print(
                "HLE裁判默认bounded-v2：限制结构空白，confidence生成为既有严格评分器要求的0–100整数；"
                "字符串正常空格、原提示与预算保持不变。bounded-v1/unrestricted显式复现历史grammar。",
                flush=True,
            )
            print(
                "SWE/Terminal默认使用公开真实轨迹：每项固定32任务、每任务最多2请求；"
                "max2048/并发8，只测推理性能，不计任务解决率。来源与抽样清单见protocols/。",
                flush=True,
            )
            print(NOTICE_ZH, flush=True)
            print(
                "有效算力参考：一键流程自动生成本轮绑定的认证计划，在推理服务退出后执行参考阶段并出报告，"
                "无需手工准备参考计划。完整算量、参考DAG或同工况同步计时缺失时返回非零，"
                "列出具体缺项，不将单算子通过或历史59.7573%作为本轮效率。"
                "原有--compute-reference-plan仅保留为显式专家入口。",
                flush=True,
            )
            print(
                "验收口径：性能或成本目标由一个固定数据集满足该项全部条件即可，"
                "不要求五项均达标；同一项指标不跨数据集拼接。五项结果完整披露，专项要求分别判定。",
                flush=True,
            )
    command = [
        sys.executable,
        str(
            ROOT
            / (
                "scripts/acceptance_suite.py"
                if supplement
                else "scripts/third_party.py"
            )
        ),
        *args,
    ]
    if follow and not any(x in ("--help", "-h") for x in args):
        # Keep operational logs outside the campaign's immutable evidence tree.
        output = next(
            (x.split("=", 1)[1] if x.startswith("--out=") else args[i + 1])
            for i, x in reversed(list(enumerate(args)))
            if x == "--out" or x.startswith("--out=")
        )
        command = [
            sys.executable,
            str(ROOT / "scripts/live_console.py"),
            "--run",
            output,
            "--",
            *command,
        ]
    elif any(x in ("--help", "-h") for x in args):
        print(
            "默认实时跟随日志；--no-follow 关闭自动跟随。--context-tenant-only 仅补长上下文/多租户，不重跑五数据集。",
            flush=True,
        )
    # The console forwards cancellation and waits for controller cleanup.
    env = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="", PYTHONUNBUFFERED="1"
    )
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    raise SystemExit(main())
