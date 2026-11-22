# One-click acceptance testing

Run `run_all.py` once on an Ascend host or container to prepare the environment, run the acceptance tests, and generate the reports.

## 1. Check the environment

Use Linux aarch64, CPython 3.12, Ascend drivers and CANN compatible with the selected engine release, and readable Qwen3.5-35B-A3B model weights. Check that both assigned devices are healthy and idle. Use their physical IDs from the `NPU` column.

```bash
npu-smi info
```

## 2. Write the configuration

Replace each placeholder, including the angle brackets. Use a writable absolute path with enough free space for `<WORK_DIR>` and the model weights directory for `<MODEL_DIR>`. Enter two distinct physical NPU IDs as integers for `<NPU_A>` and `<NPU_B>`.

```bash
mkdir -p '<WORK_DIR>'
cd '<WORK_DIR>'
cat > evaluation.json <<'JSON'
{
  "model": "<MODEL_DIR>",
  "physical_devices": [<NPU_A>, <NPU_B>],
  "cache": "<WORK_DIR>/cache",
  "out": "<WORK_DIR>/run-01"
}
JSON
```

`cache` stores reusable downloads and Python environments. Set `out` to a new directory for each run. To specify environment paths, set `environment.python` or `environment.cann_home`.

## 3. Run the tests

Set `<PYTHON>` to a Python command or interpreter path and `<REPO_DIR>` to the DeepLinkEngine checkout. Use `--rerun-b0` to measure the B0 baseline in this run. The entry point selects and records the latest stable B1 release by default.

```bash
'<PYTHON>' '<REPO_DIR>/run_all.py' --config '<WORK_DIR>/evaluation.json' --rerun-b0
```

| Stage | What the command does |
| --- | --- |
| Environment, data, and deployment | Prepare an isolated Python environment, download and verify the evaluation data and engine packages, and install B1. Restore or verify the fixed B0 deployment during preflight. |
| Paired preflight | Check the model, device mapping, and test inputs, then save the execution plan. |
| Five datasets and compute reference | Test B0 and B1 on the same model and device pair. Run MMLU-Pro, HLE, FrontierScience, SWE-bench-Pro, and Terminal-Bench, then validate the compute reference. The SWE and Terminal stages replay fixed traces. |
| Context and four-tenant workloads | Run the built-in 4K, 16K, and 32K context tests and the four-tenant workload. |
| Reporting | Verify the raw records and checksums, then generate the HTML and JSON reports. |

## 4. Review the results

- `<WORK_DIR>/run-01/results.json` records stage status and report paths.
- `<WORK_DIR>/run-01/full-report/report.html` shows the overall results. The `summary.json` in the same directory contains the structured acceptance decisions.

Check `results.json` for completion and the full report for which acceptance targets were met. See the [two-NPU example](one-click-validation-0-1.md) for a complete configuration.
