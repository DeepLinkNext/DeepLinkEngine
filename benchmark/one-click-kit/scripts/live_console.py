"""Run once, follow live logs and show progress without changing sealed evidence."""

import argparse
import codecs
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from child_runner import cancellation_handlers, run_stage
from monitor_stage import locate, read

REFERENCE_OUTPUTS = {
    "gdn-decode-independent-reference": "gdn-decode",
    "gdn-prefill-independent-reference": "gdn-prefill",
    "raw-route-recorder-qualification": "route-recorder",
    "current-native-synchronized-timing": "native-current-timing",
    "baseline-native-synchronized-timing": "native-baseline-timing",
    "current-http-work-observation": "http-current-work",
}


def reference_collectors(reference):
    # Expert reference runs retain their original fixed workload/round/arm
    # layout. Do not recursively enter per-attempt compiler cache directories.
    return [reference / "collector.log", reference / "canonical-dense/collector.log",
        *reference.glob("native/*/r*/B?/*/collector.log")]


def active(run):
    supplement = read(run / "execution.json")
    if supplement.get("schema") == "vllm-hust-acceptance-supplement-execution/v1":
        rows = supplement.get("stages", [])
        latest = rows[-1] if rows else None
        checklist = dict(
            status=supplement["status"],
            failure=supplement.get("failure"),
            stages=[
                dict(
                    name=role,
                    status=next(
                        (
                            row.get("status", "preparing")
                            for row in rows
                            if row["role"] == role
                        ),
                        "pending",
                    ),
                )
                for role in supplement.get("planned_roles", ["B0", "B1"])
            ],
        )
        leaf = Path(latest["run"]) if latest else None
        return (
            checklist,
            leaf,
            latest["role"] if latest else None,
            read(leaf / "receipt.json") if leaf else {},
        )
    checklist = read(run / "execution-checklist.json")
    reference = run / "compute-reference"
    reference_status = read(reference / "status.json")
    if reference_status:
        plan = read(reference / "automatic-plan.json") or read(run / "plans/compute-reference/automatic-plan.json")
        rows = reference_status.get("stages", [])
        states = {row["id"]: row for row in rows}
        declared = plan.get("stages", rows)
        if declared:
            checklist = dict(status=reference_status.get("status", "preparing"),
                failure=reference_status.get("failure") or "; ".join(reference_status.get("blockers", [])),
                stages=[dict(name=row["id"], status=states.get(row["id"], {}).get("status", "pending"))
                    for row in declared])
        unfinished = [row for row in rows if row.get("status") in ("running", "failed")]
        latest = unfinished[-1] if unfinished else rows[-1] if rows else None
        if latest:
            identifier = latest["id"]
            leaf = reference / REFERENCE_OUTPUTS[identifier] if identifier in REFERENCE_OUTPUTS else reference
            receipt = read(leaf / "receipt.json") if leaf != reference else {}
            return checklist, leaf, "compute-reference/" + identifier, receipt or reference_status
        return checklist, reference, "compute-reference", reference_status
    stages = checklist.get("stages", [])
    attempted = [
        s for s in stages if s.get("status") in ("running", "failed", "completed")
    ]
    stage = attempted[-1]["name"] if attempted else None
    leaf, name = locate(run, stage, checklist)
    receipt = read(leaf / "receipt.json") if leaf else {}
    return checklist, leaf, name, receipt


def progress(run):
    checklist, leaf, name, receipt = active(run)
    stages = checklist.get("stages", [])
    done = sum(s.get("status") in ("completed", "reused_verified") for s in stages)
    phase = receipt.get("status", checklist.get("status", "preparing"))
    completed = 0
    expected = None
    if leaf:
        if name == "compute-reference":
            logs = [path for path in reference_collectors(leaf) if path.exists()]
            if logs and phase == "running":
                phase += ":" + max(logs, key=lambda p: p.stat().st_mtime_ns).parent.name
        acceptance = leaf / "service/acceptance"
        if acceptance.is_dir():
            phases = [
                p for p in acceptance.iterdir() if p.is_dir() and p.name != "tokenize"
            ]
            if phases:
                latest = max(phases, key=lambda p: p.stat().st_mtime_ns)
                phase += ":" + latest.name
            completed = len(list(acceptance.rglob("request.json")))
        spec = read(leaf / "dataset-spec.json")
        if "case_ids" in spec:
            expected = len(spec["case_ids"])
            completed = len(list((leaf / "service/evaluation/requests").glob("*.json")))
        elif name and name.startswith("mmlu-pair/"):
            expected = 200
            completed = len(list((leaf / "service").glob("case-*.json")))
        elif name == "compute-reference/current-http-work-observation":
            expected = 200
            completed = len(list((leaf / "service").glob("case-*.json")))
    failure = receipt.get("failure") or checklist.get("failure")
    return {
        "stages_done": done,
        "stages_total": len(stages),
        "stage": name or "prepare",
        "phase": phase,
        "responses": completed,
        "planned": expected,
        "failure": failure,
    }


class Follow:
    def __init__(self, run, logs, stream=None, persist=True):
        self.run = run
        self.logs = logs
        self.terminal = stream if stream is not None else sys.stdout
        self.console = (
            (logs / "console.log").open("a", encoding="utf-8", buffering=1)
            if persist
            else None
        )
        self.positions = {}
        self.stop = threading.Event()
        self.previous = None
        self.last_progress = 0
        self.last_change = time.monotonic()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def emit(self, text):
        if self.console is not None:
            self.console.write(text)
        if self.terminal is not None:
            try:
                self.terminal.write(text)
                self.terminal.flush()
            except (BrokenPipeError, OSError):
                # A closed display must not abandon controller cleanup or disk logs.
                self.terminal = None

    def tail(self, path):
        try:
            with path.open("rb") as source:
                stat = os.fstat(source.fileno())
                identity = (stat.st_dev, stat.st_ino)
                old = self.positions.get(path)
                if old is None or old[0] != identity or stat.st_size < old[1]:
                    old = (
                        identity,
                        0,
                        codecs.getincrementaldecoder("utf-8")("replace"),
                    )
                source.seek(old[1])
                raw = source.read(65536)
                self.positions[path] = (identity, source.tell(), old[2])
        except OSError:
            return False
        if raw:
            label = (
                path.name
                if path.parent == self.logs
                else str(path.relative_to(self.run))
            )
            self.emit("\n[LOG " + label + "]\n" + old[2].decode(raw))
            self.last_change = time.monotonic()
        return bool(raw)

    def tick(self, final=False):
        _, leaf, name, _ = active(self.run)
        paths = {self.logs / "controller.log", *self.positions}
        if leaf:
            paths.update(
                leaf / name for name in ("service.log", "client.log", "warmup.log", "collector.log")
            )
            if name and name.startswith("compute-reference"):
                reference = self.run / "compute-reference"
                paths.update(reference.glob("*.log"))
                paths.update(reference_collectors(reference))
        pending = False
        for path in sorted(paths):
            pending = self.tail(path) or pending
        state = progress(self.run)
        # Phase transitions print immediately, counts/heartbeat at most every30s.
        key = (state["stage"], state["phase"], state["failure"])
        now = time.monotonic()
        if final or key != self.previous or now - self.last_progress >= 30:
            count = (
                (
                    f" responses={state['responses']} (time-window workload)"
                    if state["responses"]
                    else ""
                )
                if state["planned"] is None
                else f" responses={state['responses']}/{state['planned']}"
            )
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            quiet = int(now - self.last_change)
            line = (
                f"\n[PROGRESS {stamp}] stages={state['stages_done']}/{state['stages_total']} "
                f"stage={state['stage']} phase={state['phase']}{count} "
                f"log_quiet_seconds={quiet}"
            )
            if state["failure"]:
                line += " failure=" + str(state["failure"])
            self.emit(line + "\n")
            self.previous, self.last_progress = key, now
        return pending

    def loop(self):
        while not self.stop.is_set():
            try:
                self.tick()
            except Exception as error:
                self.emit(
                    "\n[MONITOR_ERROR] "
                    + type(error).__name__
                    + ": "
                    + str(error)
                    + "\n"
                )
            self.stop.wait(1)

    def finish(self, code):
        self.stop.set()
        self.thread.join()
        # Drain buffered output after the controller has reaped all services.
        while self.tick():
            pass
        self.tick(final=True)
        self.emit(
            f"\n[EXIT] code={code}; logs={self.logs}; reports={self.run}-report\n"
        )
        if self.console is not None:
            self.console.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--follow-only",
        action="store_true",
        help="Read an existing run; never start inference",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    run = args.run.expanduser().resolve()
    logs = run.with_name(run.name + "-logs")
    if args.follow_only:
        if not logs.is_dir():
            parser.error("Run log directory does not exist: " + str(logs))
    else:
        if run.exists():
            parser.error(
                "Use a new output directory; refusing to rerun or overwrite: "
                + str(run)
            )
        if not args.command or args.command[0] != "--" or len(args.command) < 2:
            parser.error("Expected -- followed by the controller command")
        logs.mkdir(parents=True, exist_ok=False)
    follow = Follow(run, logs, persist=not args.follow_only)
    print("LIVE_LOG " + str(logs / "console.log"), flush=True)
    follow.thread.start()
    code = 2
    try:
        with cancellation_handlers():
            if args.follow_only:
                while not (logs / "exit.json").exists():
                    time.sleep(1)
                code = read(logs / "exit.json").get("returncode", 2)
            else:
                env = dict(os.environ, PYTHONUNBUFFERED="1")
                with (logs / "controller.log").open("xb", buffering=0) as output:
                    result = run_stage(
                        args.command[1:],
                        env=env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                    )
                code = (
                    result.returncode
                    if result.returncode >= 0
                    else 128 - result.returncode
                )
    except KeyboardInterrupt:
        code = 130
    finally:
        if not args.follow_only:
            (logs / "exit.json").write_text(
                json.dumps({"returncode": code, "controller_reaped": True}) + "\n"
            )
        follow.finish(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
