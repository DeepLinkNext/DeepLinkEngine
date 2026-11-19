#!/usr/bin/env python3
"""Run one prospective original-protocol B1 arm and retain all failures."""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import shutil
import re
import subprocess
import sys
import time
import urllib.request

from owned_processes import Owned
from dataset_contract import sha
from resource_guard import acquire, release, cards, verify_mapping
from restore_b1_r4 import verify_installed_sources
from checkpoint_cache import release_loaded_checkpoint_cache
from report_environment import finish_report
from dependency_admission import admit_dependencies


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish_client_window(service_out, model, *, observation, observed_warmup):
    partial = None
    if observation:
        from reference_observation import finish

        partial = finish(service_out, model)
    if observed_warmup:
        from serving_warmup import verify_after

        verify_after(service_out)
    return partial


def http_client_context(command, deployment, profile, output, cache, devices):
    """Use the parent-admitted SDK context for HTTP-only child phases.

    Both the controller and the already-running service verified the complete
    SDK. The old HTTP child path bootstrapped it again before issuing any HTTP
    request. Preserve the exact final interpreter/environment and every CLI
    argument; neither the model service nor its source admission is changed.
    """
    from engine_entry import environment

    child = list(command)
    child[0] = profile["python"]
    env = environment(deployment, profile, output, cache, devices)
    env["VLLM_HUST_TEST_OUTPUT"] = str(output)
    return child, env


def check_memory(guard, initial, root):
    for attempt in range(3):
        value = guard.sample()
        stat = value["stat"]
        invalid = [
            name
            for name in ("inactive_file", "shmem", "file_dirty", "file_writeback")
            if not 0 <= stat.get(name, 0) <= stat["file"]
        ]
        value["counter_validation"] = {
            "valid": not invalid,
            "attempt": attempt + 1,
            "invalid_fields": invalid,
        }
        # Preserve every raw counter, including rejected observations.
        save(root / "memory-latest.json", value)
        with (root / "memory-samples.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        if not invalid:
            guard.check(value, initial)
            return value
        if attempt < 2:
            time.sleep(0.02)
    # Some kernels expose unsigned file_writeback underflow for longer than
    # one stat update. No cache can safely be credited from that observation.
    # Check the entire raw allocation instead: this is a stricter upper bound
    # than current minus any reclaimable cache, with the same reserve/OOM gate.
    value["sampled_working_estimate"] = value["working_estimate"]
    value["working_estimate"] = value["current"]
    value["counter_validation"]["fallback"] = "raw_current_no_cache_credit"
    save(root / "memory-latest.json", value)
    with (root / "memory-samples.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
    guard.check(value, initial)
    return value


def admit_startup_memory(guard, model, weights, root):
    """Reclaim this checkpoint's verified cache before testing allocation reserve."""
    initial = guard.sample()
    save(root / "memory-initial.json", initial)
    # Hashing the checkpoint can fill the active file cache. Do not fail the
    # working-set gate before the existing read-only checkpoint hint can run.
    # Credit only actual raw headroom after the hint, retaining the12GiB gate.
    release_loaded_checkpoint_cache(
        model, weights, root / "checkpoint-cache-startup.json", guard.sample
    )
    check_memory(guard, initial, root)
    return initial


def listener_owned(owned, port):
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        for row in Path(table).read_text().splitlines()[1:]:
            fields = row.split()
            if int(fields[1].split(":")[1], 16) == port and fields[3] == "0A":
                inodes.add(fields[9])
    if not inodes:
        return False
    for item in owned.scan():
        try:
            for path in Path(f"/proc/{item['pid']}/fd").iterdir():
                try:
                    if os.readlink(path) in {
                        "socket:[" + inode + "]" for inode in inodes
                    }:
                        return True
                except OSError:
                    pass
        except OSError:
            pass
    raise RuntimeError("The fixed port is owned by another task; no requests sent")


def wait_before_service(root, timeout):
    """Recheck devices after lengthy source/weight checks, retaining our leases.

    Other tools may not use these advisory locks. Wait before spawning our
    service if a foreign owner appeared during preparation; never stop it.
    The existing post-start ownership monitor remains authoritative afterward.
    """
    deadline = time.monotonic() + timeout
    while True:
        sample = subprocess.run(["npu-smi", "info"], capture_output=True,
                                text=True, check=True, timeout=20)
        clear = all(f"No running processes found in NPU {card}" in sample.stdout
                    for card in cards())
        save(root / "pre-service-device-admission.json", dict(
            checked_at=time.time(), physical_cards=cards(), npu_smi=sample.stdout,
            clear=clear, model_service_started=False, foreign_processes_stopped=False))
        if clear:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Selected devices remained occupied after preparation; no model service spawned")
        print("RESOURCE_WAIT before model spawn: selected devices have another owner", flush=True)
        time.sleep(5)


def admitted_run(args, root, cleanup):
    role = getattr(args, "role", "B1")
    deployment = args.deployment.resolve()
    profile = json.loads((deployment / "deployment.json").read_text())
    bindings = verify_installed_sources(deployment)
    save(root / "dependency-admission.json", admit_dependencies(deployment))
    if role == "B0":
        if args.baseline_deployment is None:
            raise ValueError(
                "B0 requires --baseline-deployment restored from the pinned018/CANN8.5.1 image"
            )
        from baseline_admission import admit

        variant = "text"
        if args.kind == "dataset" and args.dataset_spec:
            from dataset_client import checked

            spec, _ = checked(args.dataset_spec)
            if spec["dataset"] == "hle" and spec["stage"] == "solver":
                variant = "multimodal"
        admit(
            args.baseline_deployment, root / "baseline-source-admission.json", variant
        )
    for path in (deployment / "inputs/model-metadata").iterdir():
        if (
            path.is_file()
            and hashlib.sha256((args.model / path.name).read_bytes()).hexdigest()
            != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("Checkpoint metadata differs: " + path.name)
    save(
        root / "source-admission.json",
        {
            "installed_files_verified": len(bindings),
            "deployment": str(deployment),
            "release": profile["release"],
            "model_metadata_verified": True,
            "historical_dependency_byte_equivalence": False,
        },
    )
    if args.kind == "dataset":
        from dataset_client import checked

        if args.dataset_spec is None:
            raise ValueError(
                "Dataset execution requires a prospectively frozen --dataset-spec"
            )
        dataset, _ = checked(args.dataset_spec)
        if dataset["service_role"] != role:
            raise ValueError("Actual service role differs from dataset specification")
        shutil.copyfile(args.dataset_spec, root / "dataset-spec.json")
    qualification = None
    if args.kind in ("joint", "dataset"):
        if args.qualified_continuous is None:
            raise ValueError(
                "Joint inference requires --qualified-continuous; historical qualification is not reusable"
            )
        from qualification import qualify

        qualification = qualify(args.qualified_continuous, deployment)
        if qualification["physical_cards"] != cards():
            raise ValueError(
                "Continuous qualification belongs to another physical NPU pair"
            )
        save(root / "continuous-qualification.json", qualification)
    for name in (
        "deployment.json",
        "provenance/installed-files.json",
        "provenance/deployment-changes.json",
    ):
        shutil.copyfile(deployment / name, root / Path(name).name)
    save(
        root / "controller-source.json",
        {
            str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in Path(__file__).resolve().parent.iterdir()
            if p.is_file()
        },
    )
    save(root / "device-mapping.json", verify_mapping(args.devices))
    leases = acquire(root, args.lease_timeout)
    cleanup.callback(release, leases)
    descriptors = tuple(entry["fd"] for entry in leases)
    weights = {}
    for path in sorted(args.model.glob("*.safetensors")):
        with path.open("rb") as stream:
            weights[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                "mtime_ns": path.stat().st_mtime_ns,
                "inode": path.stat().st_ino,
            }
    expected_names = {
        f"model.safetensors-{index:05d}-of-00014.safetensors" for index in range(1, 15)
    }
    if (
        set(weights) != expected_names
        or sum(row["bytes"] for row in weights.values()) != 71903878016
    ):
        raise ValueError(
            "Expected the fixed BF16 Qwen3.5-35B-A3B14-shard checkpoint and original total size"
        )
    save(root / "model-weights.json", weights)
    if qualification and qualification["weights"] != weights:
        raise ValueError("Current weights differ from the qualified continuous run")
    spec = importlib.util.spec_from_file_location(
        "engine_memory_guard", deployment / "mods/packed-host/memory_guard.py"
    )
    memory_guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory_guard)
    initial_memory = admit_startup_memory(memory_guard, args.model, weights, root)
    if shutil.disk_usage(root).free < 4 * 2**30:
        raise RuntimeError("4GiB startup disk reserve reached")
    acceptance = None
    if getattr(args, "acceptance_contract", None) is not None:
        if args.kind != "continuous":
            raise ValueError(
                "Acceptance supplements require their separate continuous service"
            )
        from acceptance_admission import prepare as prepare_acceptance

        acceptance = prepare_acceptance(args.acceptance_contract, root)
    inputs = None
    if args.kind == "continuous":
        result_file = root / "input-preparation-result.json"
        env = dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).parent),
            PYTHONDONTWRITEBYTECODE="1",
        )
        result = subprocess.run(
            [
                profile["python"],
                str(Path(__file__).with_name("continuous_inputs.py")),
                "--deployment", str(deployment),
                "--out", str(root / "inputs"),
                "--result", str(result_file),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        (root / "input-preparation.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(
                "Original continuous input preparation failed; see input-preparation.log"
            )
        inputs = json.loads(result_file.read_bytes())
        if acceptance:
            from acceptance_admission import extend_warmup

            inputs = extend_warmup(inputs, acceptance["contract"])
        save(root / "input-preparation.json", inputs)
    with socket.socket() as check:
        check.bind(("127.0.0.1", 18390))
    service_out = root / "service"
    cache = (getattr(args, "cache_root", None) or root.parent) / (
        root.name + "-compiler-cache"
    )
    command = [
        sys.executable,
        str(
            Path(__file__).with_name(
                "baseline_entry.py" if role == "B0" else "engine_entry.py"
            )
        ),
        "--deployment",
        str(deployment),
        "--out",
        str(service_out),
        "--cache",
        str(cache),
        "--devices",
        args.devices,
        "--model",
        str(args.model.resolve()),
        "--kind",
        args.kind,
    ]
    if role == "B0":
        command += ["--baseline-deployment", str(args.baseline_deployment.resolve())]
    if acceptance:
        command += ["--acceptance-contract", acceptance["contract"]]
        if getattr(args, "acceptance_baseline", None):
            command += [
                "--acceptance-baseline",
                str(args.acceptance_baseline.resolve()),
            ]
    if args.kind == "dataset":
        command += ["--dataset-spec", str(args.dataset_spec.resolve())]
    if qualification:
        command += ["--qualified-continuous", str(args.qualified_continuous.resolve())]
    observation = getattr(args, "reference_observation", False)
    if observation:
        if role != "B1" or args.kind != "joint":
            raise ValueError("Raw reference observation requires a separate B1 joint run")
        command += ["--reference-observation"]
        command += ["--reference-profile", getattr(args, "reference_profile", "full-answer-v1")]
    if inputs:
        command += ["--continuous-inputs", inputs["path"]]
    command += ["--phase", "serve", "--lease-timeout", str(args.lease_timeout)]
    owned = Owned(service_out)
    children = []
    receipt = {
        "status": "admitting",
        "role": role,
        "kind": args.kind,
        "started": time.time(),
        "model_service_started": False,
        "client_started": False,
        "completed": False,
        "planned_joint_requests": 200 if args.kind == "joint" else None,
        "original_input_recipe": inputs,
        "failure": None,
        "acceptance_suite": acceptance,
    }
    env = dict(
        os.environ, VLLM_HUST_TEST_OUTPUT=str(service_out), PYTHONDONTWRITEBYTECODE="1"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    health = None
    client = None
    warmup = None
    from serving_warmup import selected as uses_warmup

    observed_warmup = uses_warmup(profile)
    admitted = None
    checkpoint_cache_admitted = False
    queue_start = time.monotonic()
    handles = []
    try:
        wait_before_service(root, args.lease_timeout)
        log = (root / "service.log").open("xb")
        handles.append(log)
        service = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=descriptors,
        )
        children.append(service)
        owned.add(service.pid)
        save(root / "service-process.json", owned.known[service.pid])
        samples = (root / "device-samples.jsonl").open("x")
        handles.append(samples)
        last_sample = 0
        profiler_cache, last_cache_sweep = None, 0
        if getattr(args, "reference_observation", False):
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third-party-test-kit/proc_capture"))
            from m01_output_cache import ProfilerCache
            profiler_cache = ProfilerCache(service_out / "raw-work")
        while True:
            now = time.monotonic()
            current_owners = owned.scan()
            if profiler_cache is not None and now - last_cache_sweep >= 5:
                hints = profiler_cache.sweep([row["pid"] for row in current_owners])
                if hints:
                    with (root / "observation-output-cache-hints.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(wall=time.time(), files=hints,
                            actual_headroom_gate_unchanged=True)) + "\n")
                last_cache_sweep = now
            memory = check_memory(memory_guard, initial_memory, root)
            if not checkpoint_cache_admitted:
                ranks = set(
                    re.findall(
                        r"Worker_TP([01]) pid=\d+.*(?:Model runner load_model total time:|Loading model weights took)",
                        (root / "service.log").read_text(errors="replace"),
                    )
                )
                if ranks == {"0", "1"}:
                    release_loaded_checkpoint_cache(
                        args.model,
                        weights,
                        root / "checkpoint-cache-after-load.json",
                        lambda: check_memory(memory_guard, initial_memory, root),
                    )
                    checkpoint_cache_admitted = True
            elif memory["limit"] - memory["current"] < 4 * 2**30:
                raise RuntimeError(
                    "Raw cgroup allocation reserve below4GiB after checkpoint cache admission"
                )
            if shutil.disk_usage(root).free < 2 * 2**30:
                raise RuntimeError("2GiB disk reserve reached")
            if service.poll() is not None:
                raise RuntimeError(
                    "Model service exited with " + str(service.returncode)
                )
            if client is None:
                from startup_failure import inspect as inspect_startup

                fatal = inspect_startup(root / "service.log")
                if fatal:
                    save(root / "startup-failure.json", fatal)
                    raise RuntimeError(
                        "Model worker failed before formal requests: "
                        + (fatal["root_error"] or fatal["fatal"])
                        + "; see startup-failure.json and original service.log"
                    )
            if admitted is None and (service_out / "resource-admission.json").exists():
                admitted = now
                receipt.update(status="starting-model", model_service_started=True)
            if admitted is None and now - queue_start > args.lease_timeout + 30:
                raise TimeoutError("Device admission wait expired")
            if args.kind == "dataset":
                if admitted is not None and client is None and now - admitted > 3600:
                    raise TimeoutError(
                        "Dataset service startup did not become healthy within3600s; no task submitted"
                    )
                # Full HLE may legitimately run beyond12h. Retain the original
                # per-request timeout and all guards, without a new overall
                # batch deadline that silently changes the task protocol.
            elif admitted is not None and now - admitted > (
                7200 if acceptance else 3000 if args.kind == "continuous" else 10800
            ):
                raise TimeoutError("Original arm deadline reached")
            if now - last_sample >= 5:
                sample_start, wall_start = time.monotonic(), time.time()
                sample = subprocess.run(
                    ["npu-smi", "info"], capture_output=True, text=True, timeout=20
                )
                sample_end, wall_end = time.monotonic(), time.time()
                samples.write(
                    json.dumps(
                        {
                            "monotonic": now,
                            "wall": wall_end,
                            "monotonic_start": sample_start,
                            "monotonic_end": sample_end,
                            "wall_start": wall_start,
                            "wall_end": wall_end,
                            "phase": receipt["status"],
                            "returncode": sample.returncode,
                            "raw": sample.stdout,
                        }
                    )
                    + "\n"
                )
                samples.flush()
                if sample.returncode:
                    raise RuntimeError("Device sampler failed; raw sample retained")
                known = {value["pid"] for value in owned.scan()}
                for line in sample.stdout.splitlines():
                    cols = [v.strip() for v in line.strip("|").split("|")]
                    if (
                        cols
                        and re.fullmatch(
                            r"(?:" + "|".join(map(str, cards())) + r")\s+0", cols[0]
                        )
                        and len(cols) >= 4
                        and cols[1].isdigit()
                    ):
                        pid = int(cols[-1] if cols[-1].isdigit() else cols[1])
                        if pid not in known:
                            raise RuntimeError(
                                "Unknown reserved-device owner; stop only our processes: "
                                + str(pid)
                            )
                last_sample = now
            if admitted is not None and client is None and listener_owned(owned, 18390):
                try:
                    with opener.open(
                        "http://127.0.0.1:18390/health", timeout=2
                    ) as response:
                        health = response.status == 200
                except OSError:
                    health = False
                if health:
                    identity = json.loads(
                        (service_out / "runtime-identity.json").read_text()
                    )
                    if role == "B0":
                        baseline = json.loads(
                            (root / "baseline-source-admission.json").read_bytes()
                        )
                        from baseline_admission import validate_identity

                        validate_identity(identity, baseline)
                    elif (
                        identity.get("worker")
                        != "corrected_serving_runtime.CorrectedWorker"
                    ):
                        raise ValueError("Actual serving worker differs")
                    client_command, client_env = http_client_context(
                        command, deployment, profile, service_out, cache, args.devices)
                    receipt["http_clients_reuse_admitted_sdk_environment"] = True
                    index = client_command.index("--phase")
                    if observed_warmup and warmup is None:
                        warmup_command = list(client_command)
                        warmup_command[index + 1] = "warmup"
                        warmup_log = (root / "warmup.log").open("xb")
                        handles.append(warmup_log)
                        warmup = subprocess.Popen(
                            warmup_command,
                            env=client_env,
                            stdout=warmup_log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            pass_fds=descriptors,
                        )
                        children.append(warmup)
                        owned.add(warmup.pid)
                        receipt.update(
                            status="warming-service",
                            warmup_started=True,
                            warmup_pid=warmup.pid,
                        )
                    if observed_warmup:
                        if warmup.poll() is None:
                            save(root / "receipt.json", receipt)
                            time.sleep(2)
                            continue
                        if warmup.returncode:
                            raise RuntimeError(
                                "Shared pre-measurement warmup failed; no formal tasks submitted"
                            )
                        receipt["measurement_ready"] = time.time()
                    client_command[index + 1] = "client"
                    if observation:
                        from reference_observation import boundary

                        receipt["raw_observation_begin"] = boundary("begin")
                        receipt["timing_eligible_for_performance_comparison"] = False
                    client_log = (root / "client.log").open("xb")
                    handles.append(client_log)
                    client = subprocess.Popen(
                        client_command,
                        env=client_env,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=descriptors,
                    )
                    children.append(client)
                    owned.add(client.pid)
                    receipt.update(
                        status="running-client",
                        client_started=True,
                        client_pid=client.pid,
                    )
                    print("ORIGINAL_CLIENT_STARTED", args.kind, client.pid, flush=True)
            if client is not None and client.poll() is not None:
                if client.returncode:
                    raise RuntimeError(
                        "Original client failed with " + str(client.returncode)
                    )
                partial = finish_client_window(service_out, args.model,
                    observation=observation, observed_warmup=observed_warmup)
                if observation:
                    receipt["raw_observation"] = dict(completed=True,
                        effective_flops=partial["effective_flops"], full_model_certificate=False)
                receipt.update(status="completed", completed=True)
                break
            save(root / "receipt.json", receipt)
            time.sleep(2)
    except BaseException as error:
        receipt.update(
            status="failed", failure=type(error).__name__ + ": " + str(error)
        )
        print("ARM_FAILED", receipt["failure"], flush=True)
    finally:
        actions = owned.stop()
        for child in children:
            child.wait(timeout=10)
        for handle in handles:
            handle.close()
        receipt.update(ended=time.time(), quiescent=True)
        if role == "B0":
            from baseline_profile import attach as attach_b0_profile

            attach_b0_profile(receipt, service_out)
        save(root / "cleanup.json", {"quiescent": True, "actions": actions})
        save(root / "receipt.json", receipt)
        hashes = {
            str(p.relative_to(root)): sha(p)
            for p in root.rglob("*")
            if p.is_file() and p.name != "checksums.json"
        }
        save(root / "checksums.json", hashes)
    return 0 if receipt["completed"] else 2


def run(args):
    if getattr(args, "physical_devices", None):
        os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = args.physical_devices
    root = args.out.resolve()
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    if args.kind == "dataset" and args.dataset_spec and args.dataset_spec.is_file():
        shutil.copyfile(args.dataset_spec, root / "dataset-spec.json")
    try:
        with contextlib.ExitStack() as cleanup:
            result = admitted_run(args, root, cleanup)
    except BaseException as error:
        # Preparation may fail before any native/model process is started.
        # Never mark an unsuccessful process cleanup as quiescent or seal it.
        if (root / "service-process.json").exists():
            raise
        save(
            root / "receipt.json",
            {
                "status": "failed",
                "kind": args.kind,
                "role": getattr(args, "role", "B1"),
                "completed": False,
                "model_service_started": False,
                "client_started": False,
                "quiescent": True,
                "failure": type(error).__name__ + ": " + str(error),
            },
        )
        save(
            root / "cleanup.json",
            {"quiescent": True, "actions": [], "model_started": False},
        )
        save(
            root / "checksums.json",
            {
                str(p.relative_to(root)): sha(p)
                for p in root.rglob("*")
                if p.is_file() and p != root / "checksums.json"
            },
        )
        print(
            "ARM_FAILED_BEFORE_MODEL",
            type(error).__name__ + ": " + str(error),
            flush=True,
        )
        result = 2
    if getattr(args, "reference_observation", False):
        return result
    if args.kind == "dataset":
        from report_environment import report_python

        try:
            python, env = report_python(args.report_python)
            subprocess.run(
                [
                    str(python),
                    str(Path(__file__).with_name("report_dataset.py")),
                    "--run",
                    str(root),
                    "--out",
                    str(root.parent / (root.name + "-report")),
                ],
                env=env,
                check=True,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            print("REPORT_FAILED", type(error).__name__ + ": " + str(error), flush=True)
            return 2
        return result
    return finish_report(args, root, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--role", choices=["B0", "B1"], default="B1")
    parser.add_argument("--baseline-deployment", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--physical-devices", default="5,6")
    parser.add_argument("--model", type=Path, default=Path("/models/Qwen3.5-35B-A3B"))
    parser.add_argument(
        "--kind", choices=["continuous", "joint", "dataset"], required=True
    )
    parser.add_argument("--dataset-spec", type=Path)
    parser.add_argument("--acceptance-contract", type=Path)
    parser.add_argument("--acceptance-baseline", type=Path)
    parser.add_argument("--qualified-continuous", type=Path)
    parser.add_argument("--reference-observation", action="store_true")
    parser.add_argument("--reference-profile", choices=("submission-v2", "full-answer-v1"), default="full-answer-v1")
    parser.add_argument(
        "--report-python",
        type=Path,
        help="Existing matplotlib3.10.8 environment; otherwise provision a separate cached report venv",
    )
    parser.add_argument("--lease-timeout", type=float, default=1800)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
