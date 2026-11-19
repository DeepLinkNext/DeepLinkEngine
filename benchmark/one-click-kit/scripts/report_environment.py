"""Provision the pinned CPU-only plotting environment after releasing NPUs."""

import os
from pathlib import Path
import subprocess
import sys


def report_python(explicit=None):
    env = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    selected = explicit or os.environ.get("VLLM_HUST_REPORT_PYTHON")
    if selected:
        python = Path(selected).expanduser().absolute()
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        python = cache / "vllm-hust-engine" / "report-mpl3108" / "bin/python"
        if not python.exists():
            python.parent.parent.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, "-m", "venv", str(python.parent.parent)],
                env=env,
                check=True,
            )
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import matplotlib;assert matplotlib.__version__=='3.10.8'",
        ],
        env=env,
        capture_output=True,
    )
    if probe.returncode:
        if selected:
            raise RuntimeError("Selected report Python must provide matplotlib==3.10.8")
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "-r",
                str(Path(__file__).with_name("requirements-report.txt")),
            ],
            env=env,
            check=True,
        )
    return python, env


def finish_report(args, root, result):
    # Library callers can opt out; the command-line entry always renders.
    if not hasattr(args, "report_python"):
        return result
    try:
        python, env = report_python(args.report_python)
        output = root.parent / (root.name + "-report")
        subprocess.run(
            [
                str(python),
                str(
                    Path(__file__).with_name(
                        "acceptance_report.py"
                        if getattr(args, "acceptance_contract", None)
                        else "report_execution.py"
                    )
                ),
                "--run",
                str(root),
                "--out",
                str(output),
            ],
            env=env,
            check=True,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("REPORT_FAILED", type(error).__name__ + ": " + str(error), flush=True)
        return 2
    return result
