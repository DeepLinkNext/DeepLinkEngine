"""Own current native timing admission, model lifetime, resource guard and seal."""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from checkpoint_cache import release_loaded_checkpoint_cache
from dataset_contract import sha
from dependency_admission import admit_dependencies
from engine_entry import environment
from owned_processes import Owned
from qualification import qualify
from resource_guard import acquire, release, cards, verify_mapping
from restore_b1_r4 import verify_installed_sources
from run_b1_r4 import admit_startup_memory, check_memory, save, wait_before_service


def run(args):
    baseline = getattr(args, "baseline_deployment", None)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    service = root / "service"
    service.mkdir()
    owned = Owned(service)
    receipt = dict(schema="baseline-native-m01-owned-run/v1" if baseline else "current-native-m01-owned-run/v1", role="B0" if baseline else "B1", started=time.time(), completed=False,
                   quiescent=False, model_service_started=False, preflight_only=args.preflight,
                   formal_repetitions=args.formal_repetitions, necessary_work_certified=False,
                   efficiency_percent=None, failure=None)
    process = None
    try:
        deployment, model = args.deployment.resolve(), args.model.resolve()
        profile = json.loads((deployment / "deployment.json").read_bytes())
        bindings = verify_installed_sources(deployment)
        save(root / "dependency-admission.json", admit_dependencies(deployment))
        qualification = qualify(args.qualified_continuous, deployment)
        if qualification["physical_cards"] != cards():
            raise ValueError("Current native qualification is bound to another physical pair")
        save(root / "continuous-qualification.json", qualification)
        for path in (deployment / "inputs/model-metadata").iterdir():
            if path.is_file() and sha(path) != sha(model / path.name):
                raise ValueError("Native checkpoint metadata differs: " + path.name)
        baseline_admission = None
        if baseline:
            from baseline_admission import admit

            baseline_admission = admit(baseline, service / "baseline-source-admission.json", "text")
        save(root / "source-admission.json", dict(deployment=str(deployment), release=profile["release"],
                                                 installed_files_verified=len(bindings), model_metadata_verified=True))
        save(root / "device-mapping.json", verify_mapping(args.devices))
        with contextlib.ExitStack() as cleanup:
            leases = acquire(root, args.lease_timeout)
            cleanup.callback(release, leases)
            weights = {}
            if not args.preflight:
                for path in sorted(model.glob("*.safetensors")):
                    with path.open("rb") as stream:
                        identity = os.fstat(stream.fileno())
                        weights[path.name] = dict(bytes=identity.st_size, sha256=hashlib.file_digest(stream, "sha256").hexdigest(),
                                                  mtime_ns=identity.st_mtime_ns, inode=identity.st_ino)
                if weights != qualification["weights"] or len(weights) != 14 or sum(row["bytes"] for row in weights.values()) != 71903878016:
                    raise ValueError("Native weights differ from the current qualified frozen model")
                save(root / "model-weights.json", weights)
            else:
                save(root / "model-admission-scope.json", dict(model_load_permitted=False,
                     weight_tensors_read=False, model_metadata_verified=True,
                     reason="Composition/tokenizer preflight returns before model construction; actual model runs retain full fresh shard admission."))
            spec = importlib.util.spec_from_file_location("native_timing_memory_guard", deployment / "mods/packed-host/memory_guard.py")
            memory_guard = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(memory_guard)
            if args.preflight:
                initial = memory_guard.sample()
                save(root / "memory-initial.json", initial)
                check_memory(memory_guard, initial, root)
            else:
                initial = admit_startup_memory(memory_guard, model, weights, root)
            if shutil.disk_usage(root).free < 4 * 2**30:
                raise RuntimeError("Native startup disk reserve reached")
            cache = root / "compiler-cache"
            cache.mkdir()
            if baseline:
                from baseline_entry import baseline_environment

                env = baseline_environment(baseline, service, cache, args.devices)
                env.pop("VLLMHUST_NATIVE_OBSERVER_PLAN", None)
                env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + ":" + env["PYTHONPATH"]
            else:
                env = environment(deployment, profile, service, cache, args.devices)
            env.update(VLLM_HUST_TEST_OUTPUT=str(service), PYTHONUNBUFFERED="1")
            executable = baseline_admission["python"] if baseline else profile["python"]
            entry = "baseline_native_prepare.py" if baseline else "native_timing_prepare.py"
            command = [executable, "-S" if baseline else "-B", str(Path(__file__).with_name(entry)),
                       "--deployment", str(deployment), "--model", str(model), "--out", str(service),
                       "--cache", str(cache), "--qualified-continuous", str(args.qualified_continuous.resolve()),
                       "--devices", args.devices, "--formal-repetitions", str(args.formal_repetitions),
                       "--reference-profile", getattr(args, "reference_profile", "full-answer-v1")]
            if baseline:
                command += ["--baseline-deployment", str(baseline.resolve())]
            if args.preflight:
                command += ["--preflight"]
            save(root / "launch-command.json", dict(command=command))
            wait_before_service(root, args.lease_timeout)
            try:
                with (root / "service.log").open("xb") as log, (root / "device-samples.jsonl").open("x") as samples:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True, pass_fds=tuple(row["fd"] for row in leases))
                    owned.add(process.pid)
                    save(root / "service-process.json", owned.known[process.pid])
                    deadline, last_sample, admitted_cache = time.monotonic() + 14400, 0, False
                    while process.poll() is None:
                        now = time.monotonic()
                        owned.scan()
                        memory = check_memory(memory_guard, initial, root)
                        if not args.preflight and not admitted_cache:
                            log_text = (root / "service.log").read_text(errors="replace")
                            loaded = set(re.findall(r"Worker_TP([01]) pid=\d+.*(?:Model runner load_model total time:|Loading model weights took)", log_text))
                            if loaded == {"0", "1"} or (service / "client/model-ready.json").exists():
                                release_loaded_checkpoint_cache(model, weights, root / "checkpoint-cache-after-load.json",
                                                                lambda: check_memory(memory_guard, initial, root))
                                # The pre-release sample can be close to the
                                # raw limit because of reclaimable shard pages.
                                # Enforce the unchanged reserve against a new
                                # sample after the admitted release completed.
                                memory = check_memory(memory_guard, initial, root)
                                admitted_cache = True
                                receipt["model_service_started"] = True
                        if admitted_cache and memory["limit"] - memory["current"] < 4 * 2**30:
                            raise RuntimeError("Native raw cgroup allocation reserve reached")
                        if now - last_sample >= 5:
                            sample = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20, check=True)
                            samples.write(json.dumps(dict(monotonic=now, wall=time.time(), raw=sample.stdout)) + "\n")
                            samples.flush()
                            known = {row["pid"] for row in owned.scan()}
                            for line in sample.stdout.splitlines():
                                cols = [v.strip() for v in line.strip("|").split("|")]
                                if cols and re.fullmatch(r"(?:" + "|".join(map(str, cards())) + r")\s+0", cols[0]) and len(cols) >= 4 and cols[1].isdigit():
                                    pid = int(cols[-1] if cols[-1].isdigit() else cols[1])
                                    if pid not in known:
                                        raise RuntimeError("Foreign selected-device owner appeared; stopping only this native attempt")
                            last_sample = now
                        if shutil.disk_usage(root).free < 2 * 2**30 or now >= deadline:
                            raise RuntimeError("Native disk reserve or prospective attempt deadline reached")
                        time.sleep(1)
                    receipt["returncode"] = process.returncode
                    if process.returncode:
                        raise RuntimeError("Native child exited with " + str(process.returncode) + "; original service.log retained")
                    result_path = service / ("client/composition.json" if args.preflight else "client/timings.json")
                    result = json.loads(result_path.read_bytes())
                    if args.preflight:
                        if result["preflight_completed"] is not True or result["model_loaded"] is not False:
                            raise ValueError("Native composition preflight is incomplete or loaded the model")
                        receipt["preflight_npu_context_initialized"] = result["npu_context_initialized"]
                    elif len(result["repetitions"]) != args.formal_repetitions or not admitted_cache:
                        raise ValueError("Native formal timing population or checkpoint admission is incomplete")
                    receipt.update(completed=True, result=str(result_path.relative_to(root)), result_sha256=sha(result_path))
            finally:
                actions = owned.stop()
                if process is not None:
                    process.wait(timeout=30)
                receipt["quiescent"] = not owned.scan()
                save(root / "cleanup.json", dict(quiescent=receipt["quiescent"], actions=actions))
    except BaseException as error:
        receipt["failure"] = f"{type(error).__name__}: {error}"
        print(receipt["failure"], file=sys.stderr, flush=True)
    finally:
        if not owned.known:
            receipt["quiescent"] = True
        receipt["ended"] = time.time()
        save(root / "receipt.json", receipt)
        # Compiler caches are local attempt artifacts, outside the evidence seal.
        manifest = {str(p.relative_to(root)): sha(p) for p in root.rglob("*")
                    if p.is_file() and "compiler-cache" not in p.relative_to(root).parts and p.name != "checksums.json"}
        save(root / "checksums.json", manifest)
    return 0 if receipt["completed"] and receipt["quiescent"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--baseline-deployment", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--qualified-continuous", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--physical-devices", required=True)
    parser.add_argument("--formal-repetitions", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--lease-timeout", type=float, default=86400)
    parser.add_argument("--reference-profile", choices=("submission-v2", "full-answer-v1"), default="full-answer-v1")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    os.environ["VLLM_HUST_PHYSICAL_DEVICES"] = args.physical_devices
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
