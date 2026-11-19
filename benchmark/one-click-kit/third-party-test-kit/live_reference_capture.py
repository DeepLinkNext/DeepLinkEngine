#!/usr/bin/env python3
"""Run live independent calibration, whole-chain validation and inference accounting."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from uuid import uuid4

from reference_capture_worker import canonical_sha, write

NATIVE_ACCOUNTING = "native_m01_accounting.py"
NATIVE_SOURCES = ("native_m01_worker.py", "native_m01_scheduler.py", "native_m01_timeline.py", "native_m01_work.py")


def read(path):
    return json.loads(path.read_bytes())


def reference_basis(spec, files):
    return canonical_sha({
        "workload": files[spec["workload"]],
        "cards": spec["physical_cards"],
        "logical_devices": spec["logical_devices"],
        "implementation": spec["reference_runtime_identity"],
        "config": spec["reference_config"],
        "backend": files[spec["reference_backend"]],
        "accounting": files.get(spec.get("reference_accounting")),
        "certifier": files.get(spec.get("reference_certifier")),
        "operations": spec.get("operations", []),
    })


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frozen_files(root, files, spec, observe_only=False):
    roles = ["workload", "resource_guard", "target_backend"]
    users = ["target"]
    if not observe_only:
        roles.append("reference_backend")
        users.append("reference")
    result = {role: checked_file(root, files, spec[role]) for role in roles}
    for role in users:
        name = spec.get(f"{role}_accounting")
        if name:
            accounting = checked_file(root, files, name)
            result[f"{role}_accounting"] = accounting
            if accounting.name == NATIVE_ACCOUNTING:
                for dependency in NATIVE_SOURCES:
                    shipped = checked_file(root, files, dependency)
                    if shipped != Path(__file__).resolve().with_name(dependency):
                        raise ValueError(f"Freeze the actual shipped native module: {dependency}")
                certifier = spec.get(f"{role}_certifier")
                if certifier:
                    result[f"{role}_certifier"] = checked_file(root, files, certifier)
                elif not observe_only:
                    raise ValueError(f"{role}: full M01 scoring needs verified arithmetic and reference bindings")
        if result[f"{role}_backend"].name == "vllm_reference_backend.py" and not name:
            raise ValueError(f"Missing pinned native effective-work collector for {role}")
    if observe_only and (result["target_backend"].name != "vllm_reference_backend.py"
                         or "target_accounting" not in result
                         or result["target_accounting"].name != NATIVE_ACCOUNTING):
        raise ValueError("capture-live-work requires the shipped vLLM adapter and native accounting collector")
    return result


def checked_file(root, files, name):
    if not isinstance(name, str) or not name:
        raise ValueError("Freeze the required capture input path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or name not in files or sha(path) != files[name]:
        raise ValueError(f"Missing or changed frozen capture input: {name}")
    return path


def validate(spec, method, paths, target_config, observe_only=False):
    if sha(paths["workload"]) != method["workload_sha256"]:
        raise ValueError("Capture workload differs from the reference method")
    workload = read(paths["workload"])
    for name in ("warmup_token_ids", "request_token_ids"):
        population = workload[name]
        if (not isinstance(population, list) or not population
                or any(not isinstance(row, list) or not row
                       or any(type(token) is not int or token < 0 for token in row) for row in population)):
            raise ValueError(f"Freeze a nonempty token-ID population: {name}")
    if not isinstance(workload["sampling"], dict) or not workload["sampling"]:
        raise ValueError("Freeze the sampling configuration")
    configs = [("target", target_config)]
    if not observe_only:
        configs.append(("reference", spec["reference_config"]))
    for role, config in configs:
        if paths[f"{role}_backend"].name == "vllm_reference_backend.py":
            llm = config["llm"]
            if (not isinstance(llm, dict) or not llm.get("model")
                    or llm.get("tensor_parallel_size") != 2 or llm.get("enforce_eager", False)):
                raise ValueError(f"Freeze the {role} weight location and TP2 graph mode")
        accounting = paths.get(f"{role}_accounting")
        if accounting and accounting.name == NATIVE_ACCOUNTING:
            if config.get("m01_in_process_engine") is not True or not config.get("implementation_files"):
                raise ValueError(f"{role}: freeze the in-process collector and runtime source digests")
        if not Path(spec[f"{role}_python"]).is_absolute():
            raise ValueError(f"Freeze the absolute interpreter path for {role}")
        identity = spec[f"{role}_runtime_identity"]
        if observe_only and identity is None:
            continue
        if not isinstance(identity, dict):
            raise ValueError(f"Freeze the runtime identity object for {role}")
        digest = identity.get("implementation_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Freeze the effective implementation digest for {role}")
    cards = spec["physical_cards"]
    if not isinstance(cards, list) or len(cards) != 2 or any(type(card) is not int or card < 0 for card in cards) or len(set(cards)) != 2:
        raise ValueError("Exactly two distinct physical cards are required")
    logical = str(spec["logical_devices"]).split(",")
    if len(logical) != 2 or not all(index.isdecimal() for index in logical) or len({int(index) for index in logical}) != 2:
        raise ValueError("Freeze the corresponding two logical device indices")
    if not observe_only:
        if spec["target_runtime_identity"]["implementation_sha256"] == spec["reference_runtime_identity"]["implementation_sha256"]:
            raise ValueError("Target and independent reference cannot use the same implementation")
        if spec["reference_id"] != method["reference_id"]:
            raise ValueError("Capture reference implementation differs from the frozen method")
        if not isinstance(spec["review_basis"], str) or not spec["review_basis"].strip():
            raise ValueError("Freeze the independent reference applicability review basis")
    if not observe_only and method["estimator"] == "critical_path":
        groups = spec["operations"]
        if not groups or any(not row["operation"] or row["kind"] not in
                             ("compute", "memory", "communication", "control") for row in groups):
            raise ValueError("Freeze the calibrated operations and kinds")
        if len({row["operation"] for row in groups}) != len(groups):
            raise ValueError("Duplicate calibration operation")
    if type(spec["phase_timeout_s"]) is not int or spec["phase_timeout_s"] <= 0:
        raise ValueError("Freeze a positive native phase timeout")
    if "reference_inputs" in spec:
        raise ValueError("Live capture cannot import previous calibration or timing records")


def launch(request, folder, interpreter, env, lease_fds, timeout):
    folder.mkdir()
    request_path = folder / "request.json"
    write(request_path, request)
    worker = Path(__file__).with_name("reference_capture_worker.py")
    with (folder / "collector.log").open("wb") as log:
        with subprocess.Popen(
            [interpreter, "-B", str(worker), str(request_path), str(folder)],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            pass_fds=lease_fds,
        ) as process:
            try:
                code = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt, SystemExit):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                raise
            finally:
                if process.poll() is not None:
                    # A native collector must not leave any workers from its own phase active.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
    if code:
        raise ValueError(f"Native {request['phase']} collector exited {code}; see {folder / 'collector.log'}")
    result = read(folder / "observation.json")
    for key in ("capture_id", "phase", "workload_sha256"):
        if result[key] != request[key]:
            raise ValueError(f"Native {request['phase']} identity differs: {key}")
    if request["runtime_identity"] is not None and result["runtime_identity"] != request["runtime_identity"]:
        raise ValueError("Native runtime identity differs from the frozen implementation")
    return result


def capture(root, files, spec, method, config, out, export, observe_only=False):
    paths = frozen_files(root, files, spec, observe_only=observe_only)
    validate(spec, method, paths, config, observe_only)
    roles = ("target",) if observe_only else ("target", "reference")
    if any(not Path(spec[f"{role}_python"]).is_file() for role in roles):
        raise ValueError("Frozen native interpreter path is unavailable")
    guard_spec = importlib.util.spec_from_file_location("m01_resource_guard", paths["resource_guard"])
    guard = importlib.util.module_from_spec(guard_spec)
    guard_spec.loader.exec_module(guard)
    capture_id = uuid4().hex
    previous = {key: os.environ.get(key) for key in ("VLLMHUST_ENGINE_LOCKS", "VLLM_HUST_PHYSICAL_DEVICES")}
    # Inherited descriptors belong to an enclosing runner, if any; do not close those here.
    if previous["VLLMHUST_ENGINE_LOCKS"]:
        raise ValueError("Launch live reference from the outer test runner without inherited resource leases")
    os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = ",".join(str(card) for card in spec["physical_cards"])
    lease = []
    try:
        write(out / "capture-session.json", {"capture_id": capture_id, "physical_cards": spec["physical_cards"],
                                            "method": method, "capture": spec,
                                            "reference_basis": None if observe_only else reference_basis(spec, files)})
        mapping = guard.verify_mapping(spec["logical_devices"])
        if mapping["physical_cards"] != spec["physical_cards"]:
            raise ValueError("Resource guard mapped a different physical pair")
        write(out / "device-mapping.json", mapping)
        lease = guard.acquire(out, 0)
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = spec["logical_devices"]
        lease_fds = tuple(item["fd"] for item in lease)
        cards = spec["physical_cards"]
        common = {"capture_id": capture_id, "workload_sha256": method["workload_sha256"],
                  "workload": str(paths["workload"]), "cards": cards, "estimator": method.get("estimator")}
        reference_accounting = paths.get("reference_accounting")
        separate_reference_trace = (reference_accounting is not None
                                    and reference_accounting.name == NATIVE_ACCOUNTING
                                    and method.get("estimator") == "critical_path")

        def stage(role, phase, label, **options):
            request = dict(common, phase=phase, backend=str(paths[f"{role}_backend"]),
                           runtime_identity=spec[f"{role}_runtime_identity"],
                           config=config if role == "target" else spec["reference_config"], **options)
            started = time.monotonic_ns()
            stage_env = dict(env)
            stage_env.pop("VLLM_HUST_M01_ACCOUNTING", None)
            stage_env.pop("VLLM_HUST_M01_CERTIFIER", None)
            if f"{role}_accounting" in paths:
                stage_env["VLLM_HUST_M01_ACCOUNTING"] = str(paths[f"{role}_accounting"])
            if f"{role}_certifier" in paths:
                stage_env["VLLM_HUST_M01_CERTIFIER"] = str(paths[f"{role}_certifier"])
            result = launch(request, out / label, spec[f"{role}_python"], stage_env, lease_fds, spec["phase_timeout_s"])
            ended = time.monotonic_ns()
            intervals = [result["interval"]] if "interval" in result else [
                value for operation in result["operations"] for value in operation["observations"]]
            if not intervals or any(not started <= item["start_ns"] < item["end_ns"] <= ended
                                    for item in intervals):
                raise ValueError(f"Stale or foreign monotonic interval in {label}")
            return result

        if observe_only:
            stage("target", "observe", "target-observed")
            return
        if method["estimator"] == "critical_path":
            operators = stage("reference", "operators", "operator-calibration", operations=spec["operations"],
                              samples=method["min_calibration_samples"])
            calibration = {"operations": operators["operations"]}
            calibration_runs = []
        else:
            calibration_runs = [stage("reference", "calibration", f"calibration-{i:03d}")
                                for i in range(1, method["min_calibration_samples"] + 1)]
            calibration = {"whole_chain_observations": [row["interval"] for row in calibration_runs]}
        reference_work = None
        if separate_reference_trace:
            reference_work = stage("reference", "reference-work", "reference-work")
        validation = [stage("reference", "validation", f"validation-{i:03d}",
                            separate_reference_trace=separate_reference_trace)
                      for i in range(1, method["min_validation_samples"] + 1)]
        target = stage("target", "target", "target-work")
        timing = stage("target", "timing", "timed-target")
    finally:
        try:
            guard.release(lease)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    work = target["work"]
    if target["request_identity"] != timing["request_identity"] or target["work"] != timing["work"]:
        raise ValueError("Timed target and separately observed target differ in output or necessary work")
    reference_runs = [*calibration_runs, *validation]
    if reference_work is not None:
        reference_runs.append(reference_work)
    for row in reference_runs:
        if (row["work"]["work_signature_sha256"] != work["work_signature_sha256"]
                or row["work"]["components"] != work["components"]):
            raise ValueError("Live reference and target did not execute the same necessary work")
    binding = {"capture_id": capture_id, "workload_sha256": method["workload_sha256"],
               "work_signature_sha256": work["work_signature_sha256"]}
    reference = dict(binding, reference_id=method["reference_id"], cards=cards,
                     effective_flops=work["effective_flops"])
    whole = dict(reference, validation_observations=[row["interval"] for row in validation])
    if method["estimator"] == "critical_path":
        graph = target["nodes"]
        graph_runs = [reference_work] if reference_work is not None else validation
        if any(row["nodes"] != graph for row in graph_runs):
            raise ValueError("Live whole-chain dependency graph differed across validation runs")
        whole["nodes"] = graph
    artifacts = {
        "effective_config": {"capture_id": capture_id, "config": config, "runtime": target["runtime_identity"]},
        "work_ledger": dict(binding, run_id=capture_id, components=work["components"]),
        "timing": dict(binding, **timing["interval"], run_id=capture_id,
                       measurement_method=timing.get("measurement_method",
                           "Host monotonic time with native two-card fences, separate from path observation")),
        "calibration": dict(reference, **calibration),
        "whole_chain": whole,
        "reference_review": dict(reference, passed=True,
            basis=spec["review_basis"]),
    }
    names = {role: f"{role}.json" for role in artifacts}
    for role, value in artifacts.items():
        write(out / names[role], value)
    write(out / "receipt.json", dict(export, run_id=capture_id, cards=cards, evidence=names))

