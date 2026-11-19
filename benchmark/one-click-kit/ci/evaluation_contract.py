#!/usr/bin/env python3
"""CPU-only normative entrypoint acceptance; never restore or start a model.

Only the final controller's run() is intercepted. The real configuration parser,
launcher, standalone cache verification and controller argparse all execute.
Expectations come from the published evaluation contract, not CLI constants.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
POLICY = "protocols/evaluation-entry-contract-v1.json"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def policy(root):
    result = json.loads((root / POLICY).read_bytes())
    # Independent business expectations: changing launcher constants AND this
    # manifest together must not silently turn three datasets into 'full'.
    require(
        result["modes"]
        == {"full": ["mmlu", "hle", "frontier", "swe", "terminal"], "quick": ["mmlu"]},
        "Frozen full/quick standard changed",
    )
    require(
        result["default_mode"] == "full"
        and result["reference_required_by_default"] is True,
        "Default evaluation/reference policy changed",
    )
    return result


def controller(root, command):
    if "--" in command and command[1].endswith("live_console.py"):
        command = command[command.index("--") + 1 :]
    require(
        Path(command[1]) == root / "scripts/third_party.py",
        "Wrong controller dispatched",
    )
    saved = list(sys.path)
    try:
        sys.path.insert(0, str(root / "scripts"))
        module = load(root / "scripts/third_party.py", "contract_controller")
        with patch.object(sys, "argv", command[1:]), patch.object(module, "run") as run:
            module.main()
        require(run.call_count == 1, "Controller was not reached exactly once")
        return run.call_args.args[0]
    finally:
        sys.path[:] = saved


def probe(root, args, bootstrap=None):
    entry = load(root / "one_click.py", "contract_entry")
    with contextlib.redirect_stdout(io.StringIO()):
        if bootstrap:
            wrapper = load(bootstrap, "contract_download")
            # Use the real fixed-manifest verification against extracted tools;
            # no network/credentials or patched ensure() on this acceptance path.
            with patch.object(wrapper.os, "execve") as execute:
                code = wrapper.main(["--engine-tools", str(root), *args])
            require(
                execute.call_count == 1,
                "Standalone entry rejected configuration: " + str(code),
            )
            command = execute.call_args.args[1]
            require(Path(command[1]) == root / "one_click.py", "Wrong bundled entry")
            args = command[2:]
        with patch.object(entry.os, "execve") as execute:
            code = entry.main(args)
        require(execute.call_count == 1, "Entry rejected configuration: " + str(code))
        return controller(root, execute.call_args.args[1])


def verify(root, bootstrap=None, runner_profile=None):
    root = root.resolve()
    contract = policy(ROOT)
    # Frontier has no separate sampling manifest: validate its actual frozen
    # source cardinalities independently of the entrypoint's dataset labels.
    contract_module = load(
        root / "scripts/dataset_contract.py", "contract_dataset_sources"
    )
    frontier = contract_module.DATASETS["frontier"]
    require(
        frontier["tasks"] == 160
        and {name: part[0] for name, part in frontier["parts"].items()}
        == {"olympiad": 100, "research": 60},
        "Frozen Frontier160 scope changed",
    )
    for name, expected in contract["frozen_policy_sha256"].items():
        require(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
            "Frozen sample or generation budget changed: " + name,
        )
    records = []
    before = inventory(root)
    profile = runner_profile or contract["default_engine_profile"]
    with tempfile.TemporaryDirectory(prefix="engine-entry-contract-") as tmp:
        tmp = Path(tmp)
        for command in ("run", "prepare"):
            for mode in (None, "full", "quick"):
                config = {
                    "engine_profile": profile,
                    "model": "weights",
                    "deployment": "sdk",
                    "physical_devices": [0, 1],
                    "out": "results",
                }
                if mode is not None:
                    config["test_mode"] = mode
                cfg = tmp / "evaluation.json"
                cfg.write_text(json.dumps(config))
                args = probe(root, [command, "--config", str(cfg)], bootstrap)
                expected = contract["modes"][mode or contract["default_mode"]]
                require(
                    args.datasets.split(",") == expected,
                    f"{command}/{mode}: expected {expected}, got {args.datasets}",
                )
                require(
                    args.performance_only is False,
                    f"{command}/{mode}: reference disabled",
                )
                require(
                    args.hle_profile == "fixed200-v1", "HLE sampling default changed"
                )
                require(
                    args.smoke_evaluators is False,
                    "Smoke silently substituted for full",
                )
                require(
                    args.prepare_only == (command == "prepare"), "Prepare/run mismatch"
                )
                require(args.engine_profile == profile, "Engine profile changed")
                expected_out = tmp / (
                    "results-prepare" if command == "prepare" else "results"
                )
                require(
                    args.out == expected_out, "Configuration output isolation changed"
                )
                records.append(
                    {
                        "entry": "standalone" if bootstrap else "source",
                        "command": command,
                        "mode": mode or "default",
                        "datasets": expected,
                        "reference_required": True,
                    }
                )
        # Direct argument path must agree with config defaults. An explicitly
        # selected quick/custom path is intentionally still allowed.
        for mode in ("full", "quick"):
            flags = ["--model", str(tmp / "weights"), "--out", str(tmp / "direct")]
            if runner_profile:
                flags += ["--engine-profile", runner_profile]
            if mode == "quick":
                flags += ["--datasets", "mmlu"]
            args = probe(root, flags, bootstrap)
            require(
                args.datasets.split(",") == contract["modes"][mode],
                "Direct/config scope mismatch",
            )
            require(args.performance_only is False, "Direct default reference disabled")
            require(args.engine_profile == profile, "Direct/config profile mismatch")
            records.append(
                {
                    "entry": "standalone" if bootstrap else "source",
                    "command": "direct",
                    "mode": mode,
                    "datasets": args.datasets.split(","),
                    "reference_required": True,
                }
            )
    require(
        before == inventory(root),
        "Entrypoint mutated the checksum-sealed tools inventory",
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--runner-profile", choices=("r6", "r8"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "vllm-hust-entry-contract-check/v1",
        "model_loaded": False,
        "policy_sha256": hashlib.sha256((ROOT / POLICY).read_bytes()).hexdigest(),
    }
    try:
        report["cases"] = verify(args.root, args.bootstrap, args.runner_profile)
        report["passed"] = True
    except Exception as error:
        report.update(passed=False, error=str(error))
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"ENTRY_CONTRACT_OK: {len(report['cases'])} routes; no model loaded")


if __name__ == "__main__":
    main()
