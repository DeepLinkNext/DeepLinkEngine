"""只读地暴露 run_all.py / one_click.py 的产物，并做尽可能细的进度解析：
  - <out>/results.json                逐阶段状态
  - <out>/engine-selection.json       B1 版本 / 档位 / 发行标识
  - 主日志 + <out>/fixed-five-logs/*   子阶段、下载/展开/写入等细粒度进度
  - <out>/full-report/report.html     报告

不改动任何评测逻辑，UI 只是外壳。
"""
import glob
from hashlib import sha256
import json
import mimetypes
import os
import re
import subprocess
import threading
import sys
from datetime import datetime
from pathlib import Path
from monitor import Monitor
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, quote, unquote

UI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(UI_DIR))


def _load_config():
    candidates = []
    override = os.environ.get("ACCEPTANCE_UI_CONFIG")
    if override:
        candidates.append(override)
    candidates += [os.path.join(UI_DIR, "config.json")]
    for path in candidates:
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    return {}


CFG = _load_config()
ROOT = CFG.get("repo_root") or REPO_ROOT
RUNS = CFG.get("runs_dir") or os.path.join(ROOT, "acceptance-runs")
LOGS = CFG.get("logs_dir") or os.path.join(ROOT, "acceptance-logs")
CACHE = CFG.get("cache_dir") or os.path.join(ROOT, ".cache")
UISTATE = UI_DIR
B1_CACHE = CFG.get("b1_cache") or ""
B0_CACHE = CFG.get("b0_cache") or ""
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = CFG.get("model") or ""
PYTHON = CFG.get("python") or sys.executable
ENGINE_VERSION = CFG.get("engine_version") or ""
ENGINE_PROFILE = CFG.get("engine_profile") or "r10"
SELECTION_ENV = "DEEPLINKENGINE_RELEASE_SELECTION"
METADATA_CLIENT_ENV = "DEEPLINKENGINE_METADATA_ASSESSMENT_CLIENT"
PHYSICAL = CFG.get("physical_devices") or "6,7"
DEVICES = CFG.get("devices") or "0,1"

PROGRESS_RE = re.compile(
    r"\[PROGRESS ([^\]]+)\] stages=(\d+)/(\d+) stage=(\S*) phase=(\S*) log_quiet_seconds=(\d+)"
)

STAGE_ORDER = [
    "prepare-datasets",
    "install-package",
    "verify-engine",
    "install-engine",
    "prepare-fixed-five",
    "run-fixed-five",
    "verify-installed-tools",
    "run-context-tenant",
    "render-full-report",
]

STAGE_LABELS = {
    "prepare-datasets": "固定数据包",
    "install-package": "依赖安装",
    "verify-engine": "引擎校验",
    "install-engine": "B1 引擎安装",
    "prepare-fixed-five": "B0 基线与输入",
    "run-fixed-five": "五项配对测量",
    "run-context-tenant": "长上下文 / 四租户",
    "verify-installed-tools": "校验已安装评测工具",
    "render-full-report": "报告渲染",
}

DATASET_LABELS = {
    "mmlu": "MMLU-Pro",
    "hle": "HLE",
    "frontier": "FrontierScience",
    "swe": "SWE-bench",
    "terminal": "Terminal-Bench",
}

# 五项测量的 15 个子阶段（顺序与 third_party.py 一致）
SUB_STAGE_ORDER = [
    "continuous-B0", "continuous-B1", "mmlu-pair",
    "frontier-solver-B0", "frontier-solver-B1", "frontier-judge-B0", "frontier-judge-B1",
    "hle-solver-B0", "hle-solver-B1", "hle-judge-B0", "hle-judge-B1",
    "swe-trace-B0", "swe-trace-B1",
    "terminal-trace-B0", "terminal-trace-B1",
]

def sub_label(name):
    """把 third_party 的子阶段名翻译成中文标签。"""
    if name in ("B0", "B1"):
        return "基线B0" if name == "B0" else "优化B1"
    if name in ("continuous-B0", "continuous-B1"):
        return "连续吞吐 · " + name.split("-")[1]
    if name == "mmlu-pair":
        return "MMLU-Pro · 双塔配对"
    parts = name.split("-")
    if len(parts) < 2:
        return name
    dataset = DATASET_LABELS.get(parts[0], parts[0])
    kind = parts[1]
    role = parts[2] if len(parts) > 2 else ""
    kind_cn = {"solver": "解算", "judge": "判分", "trace": "回放"}.get(kind, kind)
    return "%s · %s%s" % (dataset, kind_cn, (" " + role) if role else "")


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def bound_json(path, expected):
    """Read a JSON object only when it matches the hash in the run record."""
    if not expected:
        return {}
    try:
        raw = Path(path).read_bytes()
        if sha256(raw).hexdigest() == expected:
            document = json.loads(raw)
            return document if isinstance(document, dict) else {}
    except (OSError, ValueError):
        pass
    return {}


_CACHE = {}


def _ttl(key, seconds, producer):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < seconds:
        return hit[1]
    value = producer()
    _CACHE[key] = (now, value)
    if len(_CACHE) > 200:
        for k in sorted(_CACHE, key=lambda x: _CACHE[x][0])[:100]:
            _CACHE.pop(k, None)
    return value


def read_text_tail(path, max_bytes=400000):
    if not path or not os.path.isfile(path):
        return ""
    try:
        stat = os.stat(path)
    except OSError:
        return ""
    memo_key = ("tail", path, max_bytes)
    memo = _CACHE.get(memo_key)
    if memo and memo[0] == stat.st_size and memo[1] == stat.st_mtime:
        return memo[2]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            text = fh.read()
        _CACHE[memo_key] = (stat.st_size, stat.st_mtime, text)
        return text
    except Exception:
        return ""


def tail_log(log_path, lines=200):
    text = read_text_tail(log_path, 200000)
    if not text:
        return []
    return [line.rstrip("\n") for line in text.splitlines()[-lines:]]


# ---------------- 进程与运行发现 ----------------
def process_start(name):
    for proc in live_processes():
        if proc["name"] != name:
            continue
        try:
            with open("/proc/%s/stat" % proc["pid"], "r") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
            start_ticks = float(fields[19])
            hz = os.sysconf("SC_CLK_TCK")
            with open("/proc/uptime", "r") as fh:
                uptime = float(fh.read().split()[0])
            return time.time() - (uptime - start_ticks / hz)
        except Exception:
            return None
    return None


def find_log(run_name):
    return _ttl(("log", run_name), 2.0, lambda: _find_log_uncached(run_name))


def _find_log_uncached(run_name):
    if not os.path.isdir(LOGS):
        return None
    candidates = []
    for name in os.listdir(LOGS):
        path = os.path.join(LOGS, name)
        if not os.path.isfile(path):
            continue
        if run_name in name or run_name in read_text_tail(path, 262144):
            candidates.append(path)
    if candidates:
        return max(candidates, key=os.path.getmtime)
    return None


def parse_progress_any(texts):
    best = None
    for text in texts:
        for match in PROGRESS_RE.finditer(text):
            stamp = match.group(1)
            row = {
                "at": stamp,
                "done": int(match.group(2)),
                "total": int(match.group(3)),
                "stage": match.group(4),
                "phase": match.group(5),
                "quiet": int(match.group(6)),
            }
            if best is None or row["at"] >= best["at"]:
                best = row
    return best


# 4.1 算力标定的 10 个步骤，顺序取自 plans/compute-reference/automatic-plan.json
COMPUTE_ORDER = [
    "gdn-decode-independent-reference",
    "gdn-prefill-independent-reference",
    "raw-route-recorder-qualification",
    "shape-matched-independent-reference-dag",
    "current-native-synchronized-timing",
    "baseline-native-synchronized-timing",
    "current-http-work-observation",
    "complete-current-necessary-work",
    "independent-reference-dag-audit",
    "matched-current-numerator-and-recomputation",
]

COMPUTE_LABELS = {
    "gdn-decode-independent-reference": "GDN 解码独立参考",
    "gdn-prefill-independent-reference": "GDN 预填充独立参考",
    "raw-route-recorder-qualification": "原始路由记录器资格校验",
    "shape-matched-independent-reference-dag": "形状匹配独立参考 DAG",
    "current-native-synchronized-timing": "原生同步计时（当前版本）",
    "baseline-native-synchronized-timing": "原生同步计时（基线）",
    "current-http-work-observation": "当前 HTTP 工作观测",
    "complete-current-necessary-work": "补全当前必要算量",
    "independent-reference-dag-audit": "独立参考 DAG 审计",
    "matched-current-numerator-and-recomputation": "分子匹配与复算",
}


def _aggregate_groups(groups):
    """连续类报告的 groups 结构：按正式（非 warmup）组汇总。"""
    tokens = wall = reqs = ok = failed = 0
    ttft_sum = tpot_sum = ttft_w = tpot_w = 0.0
    for key, group in (groups or {}).items():
        if not isinstance(group, dict) or "warmup" in key:
            continue
        reqs += int(group.get("recorded") or 0)
        ok += int(group.get("successful") or 0)
        failed += int(group.get("failed") or 0)
        tokens += int(group.get("successful_output_tokens") or 0)
        wall += float(group.get("cohort_wall_seconds") or 0)
        n = int(group.get("successful") or 0)
        ttft = group.get("mean_ttft_ms")
        tpot = group.get("mean_tpot_ms")
        if isinstance(ttft, (int, float)) and n:
            ttft_sum += ttft * n; ttft_w += n
        if isinstance(tpot, (int, float)) and n:
            tpot_sum += tpot * n; tpot_w += n
    return {
        "throughput": (tokens / wall) if wall > 0 else None,
        "ttft_ms": (ttft_sum / ttft_w) if ttft_w else None,
        "tpot_ms": (tpot_sum / tpot_w) if tpot_w else None,
        "requests": reqs, "ok": ok, "failed": failed,
    }


def _context_metrics(report):
    """长上下文/四租户报表：汇总 formal 阶段的吞吐与首token/每token时延。"""
    phases = [p for p in (report.get("phases") or [])
              if isinstance(p, dict) and str(p.get("phase") or "").endswith("formal")]
    if not phases:
        return None, None
    tokens = wall = reqs = succ = failed = 0.0
    ttft_num = tpot_num = weight = 0.0
    for phase in phases:
        m = phase.get("metrics") or {}
        tokens += float(m.get("output_tokens") or 0)
        wall += float(m.get("duration_seconds") or 0)
        reqs += float(m.get("requests") or 0)
        succ += float(m.get("successful") or 0)
        failed += float(m.get("failed") or 0)
        n = float(m.get("successful") or m.get("requests") or 0)
        for key, bucket in (("ttft_ms", "ttft"), ("tpot_ms", "tpot")):
            value = m.get(key)
            mean = value.get("mean") if isinstance(value, dict) else value
            if isinstance(mean, (int, float)) and n:
                if bucket == "ttft":
                    ttft_num += float(mean) * n
                else:
                    tpot_num += float(mean) * n
                if bucket == "ttft":
                    weight += n
    metrics = {
        "requests": int(reqs), "successful": int(succ), "failed": int(failed),
        "output_tokens": int(tokens), "wall_seconds": wall, "duration_s": wall,
        "throughput": (tokens / wall) if wall else None,
        "ttft_ms": (ttft_num / weight) if weight else None,
        "tpot_ms": (tpot_num / weight) if weight else None,
        "ok": int(succ),
    }
    note = "%d 个正式阶段（4K/16K/32K + 四租户）" % len(phases)
    return metrics, note


def _flat_metrics(report):
    """解算 / 判分 / 回放类报告：指标直接在顶层。"""
    def num(*keys):
        for key in keys:
            value = report.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None
    return {
        "throughput": num("output_tokens_per_second"),
        "ttft_ms": num("mean_ttft_ms"),
        "tpot_ms": num("mean_tpot_ms"),
        "requests": int(report.get("recorded") or 0),
        "ok": int(report.get("successful") or 0),
        "failed": int(report.get("failed") or 0),
        "duration": num("duration_seconds"),
    }


def campaign_root(out_dir):
    """run_all 布局是 <out>/fixed-five；one_click 直接写在 <out>。"""
    nested = os.path.join(out_dir, "fixed-five")
    return nested if os.path.isdir(nested) else out_dir


def stage_results(out_dir, current=""):
    return _ttl(("stages", out_dir, current), 3.0, lambda: _stage_results_uncached(out_dir, current))


def reused_stage_roots(root):
    root = Path(root).resolve()
    frozen = read_json(root / "frozen-execution-plan.json") or {}
    manifest = bound_json(root / "reused-stages.json", frozen.get("reuse_manifest_sha256"))
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        return {}
    runs_root = Path(RUNS).resolve()
    if root == runs_root or not root.is_relative_to(runs_root):
        return {}
    current_name = root.relative_to(runs_root).parts[0]
    sources = {}
    for name in SUB_STAGE_ORDER:
        entry = stages.get(name)
        if not isinstance(entry, dict) or entry.get("stage") != name:
            continue
        location = entry.get("source_root")
        if not isinstance(location, str) or not os.path.isabs(location):
            continue
        try:
            source = Path(location).resolve(strict=True)
            relative = source.relative_to(runs_root)
        except (OSError, RuntimeError, ValueError):
            continue
        if (relative.parts and RUN_NAME.fullmatch(relative.parts[0])
                and relative.parts[0] != current_name
                and source == Path(campaign_root(runs_root / relative.parts[0])).resolve()):
            sources[name] = source
    return sources


def _stage_results_uncached(out_dir, current=""):
    """三种报告结构统一成一张表：continuous(groups) / solver-trace(flat) / mmlu-pair(arms)。"""
    rows = []
    root = campaign_root(out_dir)
    reused = reused_stage_roots(root)
    names = [n for n in SUB_STAGE_ORDER if n in reused or os.path.isdir(os.path.join(root, n))]
    for path in glob.glob(os.path.join(root, "*-report", "summary.json")):
        inferred = os.path.basename(os.path.dirname(path))[:-len("-report")]
        if inferred and inferred not in names:
            names.append(inferred)
    for name in names:
        source = reused.get(name, root)
        base = os.path.join(source, name)
        receipt = read_json(os.path.join(base, "receipt.json")) or {}
        report = read_json(os.path.join(source, name + "-report", "summary.json")) or {}
        if isinstance(report.get("arms"), dict) and not isinstance(report["arms"].get("B1"), dict):
            continue   # 指针型汇总报告，不是测量行

        metrics, note = None, None
        if isinstance(report.get("groups"), dict):
            metrics = _aggregate_groups(report["groups"])
        elif isinstance(report.get("arms"), dict) and isinstance(report["arms"].get("B1"), dict):
            arm = report["arms"]["B1"]
            metrics = _aggregate_groups(arm["groups"]) if arm.get("groups") else _flat_metrics(arm)
            gains = report.get("gains") or {}
            parts = []
            for key, label in (("throughput_gain_pct", "吞吐"),
                               ("mean_ttft_improvement_pct", "TTFT"),
                               ("mean_tpot_improvement_pct", "TPOT")):
                value = gains.get(key)
                if isinstance(value, (int, float)):
                    parts.append("%s %+.1f%%" % (label, value))
            if parts:
                note = "B1 相对 B0：" + " · ".join(parts) + "；TTFT 为 joint200 批式口径（含排队）"
        elif "output_tokens_per_second" in report or "mean_ttft_ms" in report:
            metrics = _flat_metrics(report)
        elif isinstance(report.get("phases"), list):
            metrics, note = _context_metrics(report)

        if "-judge-" in name:
            # 判分阶段测的是「判分模型」的推理，且始终由 B1 提供，
            # 性能数字既不属于该行标注的臂，也不在指标 4.2 的口径内。
            planned = report.get("planned")
            ok_n = report.get("successful")
            unresolved = report.get("unresolved_grades")
            bits = []
            if ok_n is not None and planned:
                bits.append("评分 %s/%s 条" % (ok_n, planned))
            if unresolved:
                bits.append("未解析 %s 条" % unresolved)
            bits.append("判分服务由 B1 提供，不计性能指标")
            note = " · ".join(bits)
            metrics = {
                "throughput": None, "ttft_ms": None, "tpot_ms": None,
                "requests": int(planned or 0),
                "ok": int(ok_n or 0),
                "failed": 0,
            }

        started, ended = receipt.get("started"), receipt.get("ended")
        duration = (ended - started) if isinstance(started, (int, float)) and isinstance(ended, (int, float)) else None
        if duration is None:
            # 配对类阶段自身没有 receipt，回退到两臂的 receipt 跨度
            spans = []
            for role in ("B0", "B1"):
                arm_receipt = read_json(os.path.join(base, role, "receipt.json")) or {}
                s, e = arm_receipt.get("started"), arm_receipt.get("ended")
                if isinstance(s, (int, float)) and isinstance(e, (int, float)) and e > s:
                    spans.append((s, e))
            if spans:
                duration = max(e for _, e in spans) - min(s for s, _ in spans)
        if duration is None and metrics and metrics.get("duration"):
            duration = metrics["duration"]

        raw_status = receipt.get("status")
        if raw_status == "completed":
            status = "completed"
        elif raw_status == "failed":
            status = "failed"
        elif raw_status:
            status = "running"
        else:
            status = "running" if name == current else ("completed" if report else "pending")

        rows.append({
            "name": name,
            "label": sub_label(name),
            "status": status,
            "duration_s": duration if duration is not None else (metrics or {}).get("duration_s"),
            "throughput": (metrics or {}).get("throughput"),
            "ttft_ms": (metrics or {}).get("ttft_ms"),
            "tpot_ms": (metrics or {}).get("tpot_ms"),
            "requests": (metrics or {}).get("requests") or 0,
            "ok": (metrics or {}).get("ok") or 0,
            "failed": (metrics or {}).get("failed") or 0,
            "note": note,
        })
    return rows

def engine_info(run_dir, log_text):
    sel = read_json(os.path.join(run_dir, "engine-selection.json")) or {}
    version = sel.get("version")
    if not version:
        match = re.search(r"vllm_hust_engine-([0-9][\w.]*?)\.tar\.gz", log_text or "")
        if match:
            version = match.group(1)
    return {
        "version": version,
        "profile": sel.get("engine_profile"),
        "release": sel.get("release"),
    }



PREFLIGHT_STEPS = tuple(STAGE_ORDER[:5])
CONTEXT_STEPS = ("context-inputs", "B0", "B1", "context-report")
REPORT_LABELS = {
    "context-inputs": "准备真实输入与设备准入",
    "context-report": "生成长上下文 / 四租户配对报告",
    "dataset-inputs": "准备数据、核验部署并冻结执行计划",
    "dataset-report": "封存并生成单项测量报告",
    "compute-reference": "4.1 算力标定",
    "reference-report": "封存并生成 4.1 标定报告",
}
FINISHED = {"completed", "reused", "skipped"}


def task_groups(key):
    if key == "preflight":
        return [("环境预检与输入准备", PREFLIGHT_STEPS)]
    if key == "full":
        return [
            ("环境预检与输入准备", PREFLIGHT_STEPS),
            ("五项配对测量", SUB_STAGE_ORDER),
            ("4.1 算力标定", COMPUTE_ORDER),
            ("长上下文 / 四租户", ("verify-installed-tools", *CONTEXT_STEPS)),
            ("验收报告", ("render-full-report",)),
        ]
    if key == "context-tenant":
        return [("长上下文 / 四租户", CONTEXT_STEPS)]
    if key == "compute":
        return [("4.1 算力标定", (*COMPUTE_ORDER, "reference-report"))]
    return [
        ("准备与连续吞吐", ("dataset-inputs", "continuous-B0", "continuous-B1")),
        (DATASET_LABELS[key], tuple(name for name in SUB_STAGE_ORDER if name.startswith(key + "-"))),
        ("单项报告", ("dataset-report",)),
    ]


def step_label(name):
    return STAGE_LABELS.get(name) or COMPUTE_LABELS.get(name) or REPORT_LABELS.get(name) or sub_label(name)


def status_for(value):
    return {
        "reused_verified": "reused", "measured": "completed", "passed": "completed",
        "preparing": "running", "blocked": "failed", "stopped": "failed",
    }.get(value, value if value in ("completed", "running", "failed", "reused") else "pending")


def artifact_rows(document, name_key="name"):
    rows = {}
    for entry in document.get("stages") or []:
        name = entry.get(name_key)
        if not name:
            continue
        start, end = entry.get("started"), entry.get("ended")
        rows[name] = {
            "status": "reused" if entry.get("reused") else status_for(entry.get("status")),
            "duration_s": end - start if isinstance(start, (int, float))
            and isinstance(end, (int, float)) and end >= start else None,
        }
    return rows


def make_plan(key, states=None, results=None):
    states, results = states or {}, results or {}
    groups = []
    for label, names in task_groups(key):
        steps = []
        for name in names:
            row = {"name": name, "label": step_label(name), "status": "pending", **states.get(name, {})}
            steps.append(row)
        groups.append({"label": label, "steps": steps})

    names = {
        "preflight": ("prepare-fixed-five",),
        "full": (*SUB_STAGE_ORDER, "compute-reference", "B0", "B1", "render-full-report"),
        "context-tenant": ("B0", "B1", "context-report"),
        "compute": ("compute-reference", "reference-report"),
    }.get(key)
    if names is None:
        names = ("continuous-B0", "continuous-B1", *task_groups(key)[1][1], "dataset-report")
    step_lookup = {step["name"]: step for group in groups for step in group["steps"]}
    output = []
    for name in names:
        row = {"name": name, "label": step_label(name), "status": "pending", **results.get(name, {})}
        if name in step_lookup:
            step = step_lookup[name]
            row["status"] = step["status"]
            if row.get("duration_s") is None:
                row["duration_s"] = step.get("duration_s")
            if step.get("note"):
                row["note"] = step["note"]
        else:
            row.update(states.get(name, {}))
        output.append(row)
    return {"groups": groups, "results": output}


def run_documents(name):
    out = os.path.join(RUNS, name)
    root = campaign_root(out)
    documents = {
        "outer": read_json(os.path.join(out, "results.json")) or read_json(os.path.join(out, "receipt.json")) or {},
        "campaign": read_json(os.path.join(root, "campaign.json")) or {},
        "checklist": read_json(os.path.join(root, "execution-checklist.json")) or {},
        "frozen": read_json(os.path.join(root, "frozen-execution-plan.json")) or {},
        "reference": read_json(os.path.join(root, "compute-reference", "status.json")) or {},
        "context": read_json(os.path.join(out, "context-tenant", "execution.json")) or
                   read_json(os.path.join(out, "execution.json")) or {},
        "job": read_json(job_path(name)) or {},
    }
    return out, root, documents


def identify_task(name, out, docs):
    task = docs["job"].get("task")
    if task in TASKS:
        return task
    for key in sorted(TASKS, key=len, reverse=True):
        if name.startswith("ui-" + key + "-"):
            return key
    if name.startswith("ui-resume-") or os.path.isfile(os.path.join(out, "source-campaign.json")):
        return "compute"
    if (docs["outer"].get("status") == "prepared" or
            docs["outer"].get("protocol_preparation_complete") or name.startswith("preflight")):
        return "preflight"
    if docs["outer"].get("configuration") or os.path.isdir(os.path.join(out, "fixed-five")):
        return "full"
    if docs["context"] or os.path.isfile(os.path.join(out, "acceptance-admission.json")):
        return "context-tenant"
    selected = docs["frozen"].get("datasets") or []
    if len(selected) == 1 and selected[0] in DATASET_LABELS:
        return selected[0]
    return "full"


def run_status(docs, key, is_active):
    job = docs["job"]
    outer = docs["outer"]
    compute = docs["reference"]
    primary = (
        docs["context"] if key == "context-tenant" else
        compute if key == "compute" else
        outer if key in ("full", "preflight") else
        (docs["campaign"] or docs["checklist"] or outer)
    )
    raw = primary.get("status")
    if is_active and job.get("returncode") is None:
        if raw == "failed":
            return "failed"
        return "running" if raw or primary else "starting"
    if job.get("returncode"):
        return "failed"
    normalized = {
        "measured": "completed", "blocked": "failed", "prepared_only": "prepared",
    }.get(raw, raw)
    if normalized in ("prepared", "completed", "failed"):
        return normalized
    if job or raw in ("running", "preparing"):
        return "failed"
    return "unknown"


def report_path(out, key):
    choices = {
        "preflight": ("fixed-five-prepare-report",),
        "full": ("full-report", "fixed-five-report", "fixed-five-prepare-report"),
        "context-tenant": ("paired-report",),
        "compute": (".", "reference-report"),
    }.get(key, ())
    for path in [*(os.path.join(out, rel) for rel in choices), out, out + "-report"]:
        if os.path.isfile(os.path.join(path, "report.html")):
            return path
    return None


def summarize_run(name, process=None):
    out, root, docs = run_documents(name)
    key = identify_task(name, out, docs)
    status = run_status(docs, key, bool(process))
    outer, campaign, checklist, reference, context, job = (
        docs[k] for k in ("outer", "campaign", "checklist", "reference", "context", "job")
    )
    states = artifact_rows(outer)
    states.update(artifact_rows(checklist or campaign))
    states.update(artifact_rows(reference, "id"))
    states.update(artifact_rows(context, "role"))
    if os.path.isfile(os.path.join(root, "reused-local-continuous.json")):
        states["continuous-B0"] = {"status": "skipped", "note": "本轮使用已封存的 B1 准入凭证"}
        states["continuous-B1"] = {"status": "reused", "note": "已通过连续吞吐凭证准入"}
    context_dir = os.path.join(out, "context-tenant") if key == "full" else out
    if context:
        if context.get("input_admission") or context.get("stages"):
            states["context-inputs"] = {"status": "completed"}
        if os.path.isfile(os.path.join(context_dir, "paired-report", "report.html")):
            states["context-report"] = {"status": "completed"}
    if docs["frozen"] and (checklist.get("stages") or campaign.get("stages")):
        states["dataset-inputs"] = {"status": "completed"}
    if not reference and campaign.get("compute_reference"):
        candidate = campaign["compute_reference"]
        if candidate.get("status") in ("blocked", "failed", "measured", "completed"):
            reference = candidate
    if reference:
        value = status_for(reference.get("status"))
        states["compute-reference"] = {"status": value}
        if value == "completed":
            for step in COMPUTE_ORDER:
                states.setdefault(step, {"status": "completed"})
    report = report_path(out, key)
    if key == "compute" and report:
        states["reference-report"] = {"status": "completed"}
    if key in DATASET_LABELS and report:
        states["dataset-report"] = {"status": "completed"}

    log_path = find_log(name)
    main_text = read_text_tail(log_path, 180000)
    texts = [main_text]
    for directory in (out + "-logs", os.path.join(out, "fixed-five-logs"),
                      context_dir + "-logs"):
        for filename in ("controller.log", "console.log"):
            texts.append(read_text_tail(os.path.join(directory, filename), 100000))
    progress = parse_progress_any(texts)
    raw_step = (progress or {}).get("stage", "").split("/")[-1]
    alive = status in ("running", "starting")
    rows = [step for _, steps in task_groups(key) for step in steps]
    active = next((step for step in reversed(rows) if states.get(step, {}).get("status") == "running"), None)
    if alive and raw_step in rows and not active and raw_step != "prepare":
        if states.get(raw_step, {}).get("status", "pending") == "pending":
            active = raw_step
    if not active and (alive or status == "failed"):
        first_pending = next((step for step in rows if states.get(step, {}).get("status", "pending") == "pending"), None)
        failed = any(states.get(step, {}).get("status") == "failed" for step in rows)
        if alive or not failed:
            active = first_pending
    if active:
        states[active] = {**states.get(active, {}), "status": "running" if alive else "failed"}
        if not alive:
            active = None
    for step in rows:
        if not alive and states.get(step, {}).get("status") == "running":
            states[step]["status"] = "failed" if status == "failed" else "pending"

    measurements = {}
    if key in DATASET_LABELS or key == "full":
        measurements.update({row["name"]: row for row in stage_results(out, raw_step)})
    if key in ("context-tenant", "full") and os.path.isdir(context_dir):
        measurements.update({row["name"]: row for row in stage_results(context_dir) if row["name"] in ("B0", "B1")})
    efficiency = reference.get("efficiency_exact_percent")
    lower_bound = reference.get("efficiency_lower_observed_percent")
    if isinstance(efficiency, (int, float)):
        measurements.setdefault("compute-reference", {})["note"] = "有效算力效率 %.2f%%" % efficiency
    elif isinstance(lower_bound, (int, float)):
        measurements.setdefault("compute-reference", {})["note"] = "有效算力效率下界 %.2f%%" % lower_bound
    plan = make_plan(key, states, measurements)
    started = primary_time(job, outer, campaign, context, "started")
    ended = primary_time(job, outer, campaign, context, "ended")
    if started is None and process:
        started = process_start(name)
    if started is None and log_path:
        started = os.path.getctime(log_path)
    if ended is None and not alive and log_path:
        ended = os.path.getmtime(log_path)
    if ended is None and not alive and os.path.isdir(out):
        ended = os.path.getmtime(out)
    engine = engine_info(out, main_text)
    if not engine.get("version") and job.get("engine"):
        engine.update(job["engine"])
    return {
        "name": name, "task": key, "status": status, "plan": plan, "engine": engine,
        "step": step_label(active) if active else None,
        "elapsed": max(0, (time.time() if alive else (ended or started)) - started) if started else None,
        "log_tail": tail_log(log_path, 220), "report": bool(report),
        "error": outer.get("error") or campaign.get("failure") or context.get("failure")
                 or ("命令退出码 %s，详见日志。" % job["returncode"] if job.get("returncode") else None),
    }


def primary_time(*documents_and_key):
    *documents, key = documents_and_key
    return next((doc[key] for doc in documents if isinstance(doc.get(key), (int, float))), None)


TASKS = {
    "preflight": {
        "group": "环境准备",
        "label": "预检准备",
        "hint": "数据 + B0/B1，不占卡测量",
        "kind": "run_all",
        "eta": "热缓存约 5 分钟",
    },
    "full": {
        "group": "完整验收",
        "label": "完整五项 + 专项",
        "hint": f"{ENGINE_PROFILE.upper()} 正式口径：15 个子阶段 + 长上下文 + 四租户 + 4.1 算力",
        "kind": "run_all_full",
        "eta": "约 10.5 小时",
    },
    "mmlu": {
        "group": "单项验证",
        "label": "MMLU-Pro 配对",
        "hint": "B0+B1 成对 · MMLU-Pro 固定 200 题 · 双塔配对",
        "kind": "dataset",
        "datasets": "mmlu",
        "eta": "约 71 分钟",
    },
    "frontier": {
        "group": "单项验证",
        "label": "FrontierScience",
        "hint": "B0+B1 成对 · FrontierScience 160 题 · 解算 + 判分",
        "kind": "dataset",
        "datasets": "frontier",
        "eta": "约 2.8 小时",
    },
    "hle": {
        "group": "单项验证",
        "label": "HLE-Verified",
        "hint": "B0+B1 成对 · HLE-Verified 200 题 · 解算 + 判分",
        "kind": "dataset",
        "datasets": "hle",
        "eta": "约 3.2 小时",
    },
    "swe": {
        "group": "单项验证",
        "label": "SWE-bench 回放",
        "hint": "B0+B1 成对 · SWE-bench Pro 32 条轨迹回放 · 仅性能",
        "kind": "dataset",
        "datasets": "swe",
        "eta": "约 43 分钟",
    },
    "terminal": {
        "group": "单项验证",
        "label": "Terminal-Bench 回放",
        "hint": "B0+B1 成对 · Terminal-Bench 32 条轨迹回放 · 仅性能",
        "kind": "dataset",
        "datasets": "terminal",
        "eta": "约 44 分钟",
    },
    "context-tenant": {
        "group": "专项验证",
        "label": "长上下文 / 四租户",
        "hint": "真实 4K / 16K / 32K 与四租户混合请求，B0/B1 配对。",
        "kind": "context",
    },
    "compute": {
        "group": "专项验证",
        "label": "4.1 算力标定补跑",
        "hint": "从已封存五项性能批次补跑 4.1，生成独立补跑报告。",
        "kind": "compute",
    },
}

def one_click_entry():
    """one_click 入口在 2026-11-23 之后从仓库根目录移到了 benchmark/。"""
    for rel in ("benchmark/one_click.py", "one_click.py"):
        path = os.path.join(ROOT, rel)
        if os.path.isfile(path):
            return path
    raise ValueError("找不到 one_click.py（仓库根目录与 benchmark/ 均缺失）")



ACTION_LOCK = threading.Lock()
TRACKED = {}
JOBS = Path(UISTATE) / ".jobs"
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CONTROLLERS = ("run_all.py", "one_click.py", "live_console.py", "acceptance_suite.py", "third_party.py")
RUN_MARKERS = (
    "results.json", "receipt.json", "campaign.json", "execution-checklist.json",
    "source-campaign.json", "execution.json", "acceptance-admission.json", "fixed-five/campaign.json",
)


def job_path(name):
    return JOBS / (name + ".json")


def save_job(name, job):
    JOBS.mkdir(exist_ok=True)
    target = job_path(name)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)


def live_processes():
    external = _ttl("live-controllers", 2.0, _live_processes_uncached)
    return merge_active(external)


def merge_active(external):
    active = {proc["name"]: proc for proc in external}
    for name, process in TRACKED.copy().items():
        if process.poll() is None:
            active[name] = {"name": name, "pid": process.pid}
    return list(active.values())


def option(args, name):
    for index, arg in enumerate(args):
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
        if arg == name and index + 1 < len(args):
            return args[index + 1]
    return None


def _live_processes_uncached():
    roots = Path(RUNS).resolve()
    candidates = []
    try:
        with os.scandir("/proc") as proc:
            pids = [entry.name for entry in proc if entry.name.isdecimal()]
    except OSError:
        return []
    for pid in pids:
        proc = Path("/proc") / pid
        try:
            args = [os.fsdecode(arg) for arg in (proc / "cmdline").read_bytes().split(b"\0") if arg]
            controller = next((Path(arg).name for arg in args[1:5] if Path(arg).name in CONTROLLERS), None)
            if not controller:
                continue
            out = option(args, "--out")
            if controller == "run_all.py":
                config = option(args, "--config")
                out = (read_json(proc / "cwd" / config) or {}).get("out") if config else None
            if not out:
                continue
            relative = (proc / "cwd" / out).resolve().relative_to(roots)
        except (OSError, ValueError):
            continue
        if relative.parts and RUN_NAME.fullmatch(relative.parts[0]):
            candidates.append((len(relative.parts), CONTROLLERS.index(controller), int(pid), relative.parts[0]))
    found = {}
    for _, _, pid, name in sorted(candidates):
        found.setdefault(name, {"name": name, "pid": pid})
    return list(found.values())


def stop_run(name, active):
    proc = next((proc for proc in active if proc["name"] == name), None)
    if not proc:
        raise ValueError("该轮次当前没有运行")
    os.kill(proc["pid"], 15)
    return proc


def list_runs(active):
    names = {proc["name"] for proc in active}
    active_names = names.copy()
    names.update(path.stem for path in JOBS.glob("*.json"))
    if os.path.isdir(RUNS):
        for entry in os.scandir(RUNS):
            if (entry.is_dir() and not entry.name.endswith(("-logs", "-report", "-prepare"))
                    and any(os.path.isfile(os.path.join(entry.path, marker)) for marker in RUN_MARKERS)):
                names.add(entry.name)
    rows = []
    recovery_sources = {}
    runs_root = Path(RUNS).resolve()
    for name in names:
        if not RUN_NAME.fullmatch(name):
            continue
        out, _, docs = run_documents(name)
        key = identify_task(name, out, docs)
        reuse = docs["outer"].get("reused_run")
        if key == "full" and isinstance(reuse, dict) and isinstance(reuse.get("root"), str):
            try:
                original = Path(reuse["root"]).resolve(strict=True)
            except (OSError, RuntimeError):
                original = None
            if (original is not None and original.parent == runs_root
                    and original.name != name and RUN_NAME.fullmatch(original.name)
                    and bound_json(original / "results.json", reuse.get("results_sha256"))):
                recovery_sources[name] = original.name
        job = docs["job"]
        mtime = job.get("started") or (os.path.getmtime(out) if os.path.isdir(out) else 0)
        rows.append({
            "name": name, "task": key, "status": run_status(docs, key, name in active_names),
            "mtime": mtime, "archived": "FAILED" in name, "active": name in active_names,
        })
    rows.sort(key=lambda row: (row["active"], row["mtime"], row["name"]), reverse=True)
    merged = {}
    for row in rows:
        name = row["name"]
        original = recovery_sources.get(name)
        if (original and not row["active"] and row["status"] == "completed"
                and os.path.isfile(os.path.join(RUNS, name, "full-report", "report.html"))):
            merged.setdefault(original, name)
    for row in rows:
        if not row["active"] and row["status"] == "failed":
            parent = merged.get(recovery_sources.get(row["name"], row["name"]))
            if parent:
                row["merged_into"] = parent
    return rows


def write_config(name):
    cfg = {
        "model": MODEL,
        "physical_devices": [int(card) for card in PHYSICAL.split(",")],
        "devices": [int(device) for device in DEVICES.split(",")],
        "cache": CACHE,
        "out": os.path.join(RUNS, name),
        "rerun_b0": True,
        "environment": {"python": PYTHON, "cann_home": CFG.get("cann_home") or "/usr/local/Ascend/cann-9.1.0"},
    }
    if ENGINE_VERSION:
        cfg["engine_version"] = ENGINE_VERSION
    if CFG.get("wheelhouse"):
        cfg["environment"]["wheelhouse"] = CFG["wheelhouse"]
    path = JOBS / (name + ".config")
    JOBS.mkdir(exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def selection_matches(record, version=None):
    saved_version = record.get("version")
    return (record.get("schema") == "deeplinkengine-stable-selection/v1"
            and isinstance(saved_version, str) and re.fullmatch(r"\d{4}\.\d{1,2}\.\d{1,2}\.\d+(?:\.post\d+)?", saved_version)
            and (not version or saved_version == version)
            and record.get("engine_profile") == ENGINE_PROFILE
            and bool(record.get("release")))


def sdk_python(deployment, selection):
    if not deployment:
        return None
    sdk = Path(deployment)
    installed = read_json(sdk.parent / "installation.json") or {}
    profile = read_json(sdk / "deployment.json") or {}
    release = installed.get("release") or ""
    python = profile.get("python")
    if (not python or not os.path.isfile(python) or installed.get("python") != python
            or release != selection.get("release") or profile.get("release") != release
            or installed.get("package_version") != selection.get("version")):
        return None
    return python


def find_sdk():
    for path in (os.environ.get("DEEPLINKENGINE_UI_SDK"), CFG.get("sdk_dir")):
        if path:
            sdk = Path(path).resolve()
            receipt = sdk.parent / "engine-selection.json"
            selected = read_json(receipt) or {}
            if selection_matches(selected, ENGINE_VERSION) and sdk_python(sdk, selected):
                return str(sdk), receipt
    if not os.path.isdir(RUNS):
        return None
    candidates = []
    for entry in os.scandir(RUNS):
        path = os.path.join(entry.path, "installation", "sdk")
        receipt = Path(entry.path) / "engine-selection.json"
        selected = read_json(receipt) or {}
        if not selection_matches(selected, ENGINE_VERSION) or not sdk_python(path, selected):
            continue
        run = read_json(os.path.join(entry.path, "results.json")) or {}
        installed = any(row.get("name") == "install-engine" and row.get("status") == "completed"
                        for row in run.get("stages", []))
        if installed:
            candidates.append((os.path.getmtime(path), path, receipt))
    return max(candidates)[1:] if candidates else None


EVIDENCE_FILES = ("checksums.json", "receipt.json", "deployment.json",
                  "resource-admission.json", "service/request-plan.json",
                  "service/tokenize-preflight.json", "service/continuous.json")


def continuous_sources():
    sources = []
    if not os.path.isdir(RUNS):
        return sources
    expected_cards = sorted(part.strip() for part in PHYSICAL.split(","))
    for name in os.listdir(RUNS):
        base = os.path.join(RUNS, name, "fixed-five", "continuous-B1")
        sdk = os.path.join(RUNS, name, "installation", "sdk")
        if not all(os.path.isfile(os.path.join(base, rel)) for rel in EVIDENCE_FILES):
            continue
        selection = read_json(os.path.join(RUNS, name, "engine-selection.json")) or {}
        admission = read_json(os.path.join(base, "resource-admission.json")) or {}
        cards = admission.get("physical_cards") or []
        if sorted(str(card) for card in cards) != expected_cards:
            continue
        if not selection_matches(selection, ENGINE_VERSION) or not sdk_python(sdk, selection):
            continue
        sources.append({"run": name, "path": base, "sdk": sdk, "version": selection.get("version"),
                        "mtime": os.path.getmtime(base)})
    return sorted(sources, key=lambda item: item["mtime"], reverse=True)


def resume_targets(name):
    run = Path(RUNS) / name
    if not run.resolve().is_relative_to(Path(RUNS).resolve()):
        return None
    receipt = run / "engine-selection.json"
    selection = read_json(receipt) or {}
    if receipt.is_symlink() or not selection_matches(selection):
        return None
    campaign = str(run / "fixed-five")
    required = ("campaign.json", "campaign-checksums.json", "frozen-execution-plan.json")
    if not all(os.path.isfile(os.path.join(campaign, filename)) for filename in required):
        return None
    ledger = read_json(os.path.join(campaign, "campaign.json")) or {}
    plan = read_json(os.path.join(campaign, "frozen-execution-plan.json")) or {}
    reference = read_json(os.path.join(campaign, "compute-reference", "status.json")) or {}
    if ledger.get("status") != "completed" or "mmlu" not in (plan.get("datasets") or []):
        return None
    if not (ledger.get("stages") and all(status_for(row.get("status")) in FINISHED for row in ledger["stages"])):
        return None
    if (reference.get("status") in ("measured", "completed")
            or plan.get("engine_release") != selection["release"]
            or not sdk_python(plan.get("optimized_deployment"), selection)
            or not os.path.isdir(plan.get("baseline_deployment") or "")):
        return None
    physical = plan.get("physical_devices") or plan.get("physical_cards")
    if isinstance(physical, str):
        physical = physical.split(",")
    if physical and sorted(map(str, physical)) != sorted(PHYSICAL.split(",")):
        return None
    return {"campaign": campaign, "plan": plan, "receipt": receipt, "version": selection["version"]}


def task_list(runs):
    sdk = find_sdk()
    sources = {}
    for row in runs:
        if row["task"] == "full" and row["status"] == "failed" and not row["active"]:
            target = resume_targets(row["name"])
            if target:
                sources[row["name"]] = target["version"]
    missing = f"没有可用的 {ENGINE_VERSION or ENGINE_PROFILE.upper()} SDK，请先完成预检。"
    tasks = []
    for key, spec in TASKS.items():
        reason = ""
        if spec["kind"] in ("dataset", "context") and not sdk:
            reason = missing
        if key == "compute" and not sources:
            reason = f"需要含 MMLU 的 {ENGINE_PROFILE.upper()} 已封存性能批次及其对应 SDK，4.1 须尚未完成。"
        tasks.append({
            "key": key, "group": spec["group"], "label": spec["label"], "hint": spec["hint"],
            "eta": spec.get("eta"), "needs_npu": key != "preflight",
            "available": not reason, "reason": reason, "reusable": spec["kind"] == "dataset",
            "plan": make_plan(key), "sources": list(sources) if key == "compute" else [],
            "source_versions": sources if key == "compute" else {},
        })
    return tasks


def build_command(task, name, evidence=None, source=None):
    spec = TASKS.get(task)
    if not spec:
        raise ValueError("请选择一个测试模块")
    if not MODEL or not os.path.isdir(MODEL):
        raise ValueError("找不到配置的模型目录")
    if spec["kind"] in ("run_all", "run_all_full"):
        cmd = [PYTHON, os.path.join(ROOT, "run_all.py"), "--config", write_config(name)]
        if task == "preflight":
            cmd.append("--prepare-protocol-only")
        return cmd, None

    cache = CACHE
    selected_evidence = None
    if task == "compute":
        target = resume_targets(source) if source and RUN_NAME.fullmatch(source) else None
        if not target:
            raise ValueError("请选择满足准入条件的已封存性能批次")
        deployment = target["plan"]["optimized_deployment"]
        receipt = target["receipt"]
        if ENGINE_VERSION and target["version"] != ENGINE_VERSION:
            cache = str(Path(UI_DIR) / ".runtime-cache" / f"{ENGINE_PROFILE}-{target['version']}")
    elif evidence:
        if spec["kind"] != "dataset":
            raise ValueError("该任务不接受连续吞吐凭证")
        selected_evidence = next((item for item in continuous_sources() if item["path"] == evidence), None)
        if not selected_evidence:
            raise ValueError("连续吞吐凭证或其对应部署不可用")
        deployment = selected_evidence["sdk"]
        receipt = Path(RUNS) / selected_evidence["run"] / "engine-selection.json"
    else:
        sdk = find_sdk()
        if not sdk:
            raise ValueError("找不到可复用的已安装 SDK，请先完成预检")
        deployment, receipt = sdk
    b1_python = sdk_python(deployment, read_json(receipt) or {})
    if not b1_python:
        raise ValueError("B1 SDK 或原安装的 Python 与所选轮次的版本不匹配")
    cmd = [PYTHON, "-B", one_click_entry(), "--model", MODEL, "--out", os.path.join(RUNS, name),
           "--cache", cache, "--physical-devices", PHYSICAL, "--devices", DEVICES,
           "--deployment", deployment, "--rerun-b0"]
    if task == "compute":
        cmd.insert(2, os.path.join(UI_DIR, "compute_one_click.py"))
        cmd += ["--compute-reference-only", "--reference-campaign", target["campaign"],
                "--baseline-deployment", target["plan"]["baseline_deployment"]]
    elif task == "context-tenant":
        cmd += ["--context-tenant-only", "--b1-python", b1_python]
        run = Path(deployment).parent.parent
        frozen = read_json(run / "fixed-five" / "frozen-execution-plan.json") or {}
        baseline = frozen.get("baseline_deployment")
        if baseline and os.path.isdir(baseline):
            cmd += ["--baseline-deployment", baseline]
    else:
        cmd += ["--datasets", spec["datasets"], "--performance-only"]
        if selected_evidence:
            cmd += ["--continuous-evidence", selected_evidence["path"]]
    return cmd, receipt


def record_exit(name, process, job):
    code = process.wait()
    with ACTION_LOCK:
        save_job(name, {**job, "returncode": code, "ended": time.time()})
        TRACKED.pop(name, None)
        _CACHE.pop("live-controllers", None)


def start_run(task, evidence=None, source=None):
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    name = "ui-%s-%s" % (task, stamp)
    cmd, receipt = build_command(task, name, evidence, source)
    env = dict(os.environ, DEEPLINKENGINE_RUNTIME_CACHE=B1_CACHE,
               DEEPLINKENGINE_B0_RUNTIME_CACHE=B0_CACHE, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    selection_hash = SELECTION_ENV + "_SHA256"
    for key in (SELECTION_ENV, selection_hash, METADATA_CLIENT_ENV):
        env.pop(key, None)
    version = ENGINE_VERSION
    if receipt:
        raw = receipt.read_bytes()
        selection = json.loads(raw)
        if receipt.is_symlink() or not selection_matches(selection, None if task == "compute" else ENGINE_VERSION):
            raise ValueError("找不到对应的引擎版本记录")
        env[SELECTION_ENV] = str(receipt.absolute())
        env[selection_hash] = sha256(raw).hexdigest()
        version = selection["version"]
    os.makedirs(LOGS, exist_ok=True)
    log_path = os.path.join(LOGS, name + ".log")
    job = {"task": task, "evidence": evidence, "source": source, "started": time.time(),
           "engine": {"version": version, "profile": ENGINE_PROFILE}}
    save_job(name, job)
    with open(log_path, "ab", buffering=0) as log:
        try:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log, env=env,
                                       start_new_session=True)
        except OSError:
            save_job(name, {**job, "returncode": 1, "ended": time.time()})
            raise
    TRACKED[name] = process
    threading.Thread(target=record_exit, args=(name, process, job), daemon=True).start()
    return {"run": name, "pid": process.pid}



MONITOR = Monitor(PHYSICAL)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def send_body(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload, code=200):
        self.send_body(code, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def serve_asset(self, name, rest):
        if not RUN_NAME.fullmatch(name):
            self.send_json({"error": "轮次不存在"}, 404)
            return
        out, _, docs = run_documents(name)
        folder = report_path(out, identify_task(name, out, docs))
        if folder:
            base = Path(folder).resolve()
            path = (base / rest).resolve()
            if path.is_relative_to(base) and path.is_file():
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                if content_type.startswith("text/") or content_type == "application/javascript":
                    content_type += "; charset=utf-8"
                self.send_body(200, path.read_bytes(), content_type)
                return
        self.send_json({"error": "报告文件尚未生成"}, 404)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_body(200, (Path(HERE) / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/health":
            self.send_json({"service": "deeplink-acceptance-ui", "status": "ok"})
            return
        if parsed.path == "/api/live":
            self.send_json(MONITOR.snapshot())
            return
        query = parse_qs(parsed.query)
        name = (query.get("run") or [None])[0]
        if parsed.path == "/api/state":
            active = live_processes()
            processes = {proc["name"]: proc for proc in active}
            runs = list_runs(active)
            if name and not any(row["name"] == name for row in runs):
                self.send_json({"error": "轮次不存在"}, 404)
                return
            self.send_json({
                "runs": runs, "tasks": task_list(runs), "active": [proc["name"] for proc in active],
                "evidence": continuous_sources(), "cards": MONITOR.cards,
                "engine": {"version": ENGINE_VERSION, "profile": ENGINE_PROFILE},
                "run": summarize_run(name, processes.get(name)) if name else None,
            })
            return
        if parsed.path == "/api/report" and name and RUN_NAME.fullmatch(name):
            out, _, docs = run_documents(name)
            if report_path(out, identify_task(name, out, docs)):
                self.send_response(302)
                self.send_header("Location", "/report/%s/report.html" % quote(name, safe=""))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        if parsed.path.startswith("/report/"):
            name, _, rest = parsed.path.removeprefix("/report/").partition("/")
            self.serve_asset(unquote(name), unquote(rest))
            return
        self.send_json({"error": "未找到页面"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/start", "/api/stop", "/api/resume"):
            self.close_connection = True
            self.send_json({"error": "未找到操作"}, 404)
            return
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            self.close_connection = True
            self.send_json({"error": "请求来源与实验台地址不符"}, 403)
            return
        try:
            size = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= 16384:
                self.close_connection = True
                raise ValueError("请求格式无效")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict) or any(
                value is not None and not isinstance(value, str)
                for key, value in data.items() if key in ("task", "run", "source", "evidence")
            ):
                raise ValueError("请求参数无效")
            with ACTION_LOCK:
                active = merge_active(_live_processes_uncached())
                if path == "/api/stop":
                    result = stop_run(data.get("run"), active)
                elif active:
                    name = active[0]["name"]
                    self.send_json({"ok": False, "busy": name, "error": "仍有测试在运行：" + name}, 409)
                    return
                elif path == "/api/resume":
                    result = start_run("compute", source=data.get("run"))
                else:
                    result = start_run(data.get("task"), evidence=data.get("evidence"), source=data.get("source"))
            self.send_json({"ok": True, "result": result})
        except (ValueError, OSError) as exc:
            self.send_json({"ok": False, "error": str(exc)}, 400)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        MONITOR.start()
        print("dashboard on http://127.0.0.1:%d" % port, flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
