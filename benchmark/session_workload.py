"""Execute one frozen multi-turn workload through a site-installed session client."""
import csv
import hashlib
import io
import subprocess
import uuid
from pathlib import Path

from benchmark.benchmark_io import load, save


def run(config, site, folder, plan, nodes, control):
    from benchmark.engine_pool import running_pool

    raw = Path(site["requests"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["request_sha256"]:
        raise ValueError("Session workload digest differs")
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    programs = {}
    for row in rows:
        programs.setdefault(row["session_id"], []).append(int(row["round_idx"]))
    if (not rows or len(rows) != config["requests"] or len(programs) != config["programs"]
            or any(sorted(indices) != list(range(len(indices))) for indices in programs.values())):
        raise ValueError("Unexpected session or round population")
    client = site["session_client"]
    if not isinstance(client, list) or not client or not all(isinstance(v, str) for v in client):
        raise ValueError("Expected session_client command as a list")
    concurrency = config["concurrency"]
    if type(concurrency) is not int or concurrency < 0:
        raise ValueError("Expected nonnegative program concurrency")
    folder.mkdir(parents=True, exist_ok=False)
    workload = folder / "workload"
    workload.mkdir()
    (workload / "sessions.csv").write_bytes(raw)
    replay = folder / "replay"
    urls = [node["endpoint"].rstrip("/") + "/v1" for node in nodes]
    argv = [*client, "--workload", str(workload), "--out", str(replay),
            "--model", config["model"], "--api", "completions", "--timeout", "-1",
            "--seed", str(config["seed"]), "--concurrency", str(concurrency),
            "--cache-salt", "myc-" + uuid.uuid4().hex, "--cache-salt-scope", "program",
            "--base-url", control.endpoints[0].rstrip("/") + "/v1"]
    for url in urls:
        argv.extend(["--engine-url", url])
    argv.extend(["--completion-program-ids"] if control.program_release else ["--direct-engine-pool", *urls])
    save(folder / "plan.json", dict(plan, client_argv=argv))
    print("BENCHMARK_OUTPUT", folder, flush=True)
    try:
        with running_pool(nodes, folder, config["model"]), control.running(folder):
            with (folder / "replay.log").open("x") as log:
                subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = load(replay / "summary.json")
            result = summary["result"]
            if (summary["sessions_sha256"] != config["request_sha256"]
                    or summary["replayer_sha256"] != config["replayer_sha256"]
                    or summary["program_errors"] or result["rounds_failed"]
                    or result["sessions_completed"] != len(programs)
                    or result["rounds_succeeded"] != len(rows)
                    or any(item["count"] for item in result["mismatches"].values())):
                raise ValueError("Session replay failed the frozen workload acceptance checks")
    except BaseException as error:
        save(folder / "status.json", dict(status="failed", error=str(error)))
        raise
    save(folder / "status.json", dict(status="completed"))
    return 0
