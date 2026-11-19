"""Prepare native original018 from its verified original serving arguments."""

import argparse
import json
import os
from pathlib import Path
import sys

from baseline_entry import model_view, serving_arguments
from baseline_profile import save as save_profile, verify as verify_profile
from dataset_contract import save, sha
from engine_entry import check_pinned_host_memory
from resource_guard import acquire


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("deployment", "baseline-deployment", "model", "out", "cache", "qualified-continuous"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--formal-repetitions", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--reference-profile", choices=("submission-v2", "full-answer-v1"), default="full-answer-v1")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    from reference_capture_profile import profile as capture_profile

    capture = capture_profile(args.reference_profile)
    if args.formal_repetitions != capture["formal_repetitions"]:
        raise ValueError("Native repetitions differ from the selected frozen reference profile")
    root, out, cache = args.deployment.resolve(), args.out.resolve(), args.cache.resolve()
    if json.loads((root / "deployment.json").read_bytes())["release"] != "full-answer-b1-20261117-r8":
        raise ValueError("This native paired workload requires the frozen r8 campaign")
    if (os.environ.get("VLLMHUST_BASELINE_BOOTSTRAPPED") != str(out) or not sys.flags.no_site
            or Path(sys.prefix).resolve() != (args.baseline_deployment / "rootfs/usr/local/python3.11.14").resolve()):
        raise ValueError("Native baseline requires the admitted original018 no-site interpreter")
    acquire(out, 0)
    if not args.preflight:
        check_pinned_host_memory(out, os.environ)
    original_args = serving_arguments(root, "joint")
    admission = verify_profile(original_args, "text", "joint", deployment=root,
         source_sha256=sha(root / "runtime/entrypoints/prepare_retrieval_b0.py"),
         entry_sha256=sha(root / "runtime/entrypoints/serve_b0_r2.py"))
    save_profile(out, admission)
    view = model_view(args.model.resolve(), cache, out, False)
    from native_timing_inputs import prepare

    workload = out / "token-workload.json"
    prepare(root, args.model, args.qualified_continuous, workload)
    log_path = out / "vllm-ascend-logs"
    log_path.mkdir()
    serving = ["serve", str(view), *original_args, "--worker-extension-cls",
               "baseline_native_worker.BaselineNativeWorker", "--additional-config",
               json.dumps(dict(ascend_log_path=str(log_path)))]
    script_root = Path(__file__).resolve().parent
    sources = [script_root / name for name in ("baseline_native_prepare.py", "baseline_native_entry.py",
        "baseline_native_worker.py", "baseline_entry.py", "baseline_profile.py", "native_timing_inputs.py",
        "native_timing_session.py", "native_public_client.py", "native_output_geometry.py", "native_state_cache.py", "reference_capture_profile.py", "native_plan_values.py",
        "runtime_identity.py", "compile_boundary.py")]
    sources += [root / "deployment.json", root / "provenance/installed-files.json", workload,
                out / "baseline-source-admission.json", out / "b0-profile-admission.json",
                root / "runtime/entrypoints/serve_b0_r2.py", root / "runtime/entrypoints/prepare_retrieval_b0.py"]
    plan = dict(schema="baseline-native-m01-timing/v1", role="B0", engine_release="original018-cann8.5.1",
        paired_engine_release=json.loads((root / "deployment.json").read_bytes())["release"],
        deployment=str(root), baseline_deployment=str(args.baseline_deployment.resolve()),
        model=str(args.model.resolve()), model_view=str(view), output=str(out / "client"),
        admission_root=str(out.parent), workload=str(workload), files={str(p): sha(p) for p in sources},
        baseline_admission=str(out / "baseline-source-admission.json"), serve_arguments=serving,
        b0_profile=admission, expected_versions=dict(vllm="0.18.0", vllm_ascend="0.18.0"),
        formal_repetitions=args.formal_repetitions, reference_capture_profile=capture["id"], max_tokens=capture["max_tokens"],
        sampling=dict(max_tokens=capture["max_tokens"], temperature=0, seed=0, ignore_eos=capture["ignore_eos"],
                      top_p=1.0, top_k=-1, presence_penalty=0.0, detokenize=False),
        statistic="Preserve every predeclared full fixed200 repetition; sum matched reference times / sum native times. No best-run selection.",
        intentional_frontend_changes=["Shared B0/B1 public token-prompt DELTA client; HTTP/chat formatting/tokenization outside T",
            "disable_log_stats=True; no B1 MODs or observation hooks; shared named fence/compilation RPCs",
            "Additional config changes only the Ascend log destination to this run directory"],
        necessary_work_certified=False, efficiency_percent=None)
    plan_path = out / "native-plan.json"
    save(plan_path, plan)
    command = [sys.executable, "-S", str(script_root / "baseline_native_entry.py"), "--config", str(plan_path)]
    if args.preflight:
        command.append("--preflight")
    save(out / "launch-command.json", dict(command=command, preserved_original_serving_arguments=original_args,
        environment={key: value for key, value in os.environ.items() if key.startswith(("VLLM", "ASCEND", "HCCL"))
                     or key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")}))
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
