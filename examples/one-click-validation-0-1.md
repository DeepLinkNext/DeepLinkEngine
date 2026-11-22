# One-click acceptance example for physical NPUs 0 and 1

This example runs in `/root/.deeplinkengine-validation`. It assumes a checkout at `/root/DeepLinkEngine`, model weights at `/data/shared_models/Qwen3.5-35B-A3B`, and a Linux aarch64 environment with compatible Ascend drivers, CANN, and CPython 3.12.

## 1. Check the devices

Check the `NPU` column to confirm physical NPUs `0` and `1` are assigned to this environment, healthy, and idle.

```bash
npu-smi info
```

## 2. Write the configuration

Both engines use the same model weights. The configuration stores the reusable cache under the working directory and writes this run to `run-01`. Use a new `out` directory for a fresh run.

```bash
mkdir -p /root/.deeplinkengine-validation
cd /root/.deeplinkengine-validation
cat > evaluation.json <<'JSON'
{
  "model": "/data/shared_models/Qwen3.5-35B-A3B",
  "physical_devices": [0, 1],
  "cache": "/root/.deeplinkengine-validation/cache",
  "out": "/root/.deeplinkengine-validation/run-01"
}
JSON
```

## 3. Run the tests

```bash
python3 /root/DeepLinkEngine/run_all.py \
  --config /root/.deeplinkengine-validation/evaluation.json --rerun-b0
```

The command prepares Python and the datasets, installs and verifies B1, and prepares the fixed B0 baseline. After preflight, it runs the paired tests and compute reference, then runs the built-in context and four-tenant tests and generates the reports. `--rerun-b0` measures B0 as part of this run. See the [one-click acceptance guide](one-click-validation.md) for the full stage breakdown.

## 4. Review the results

- Stage status in `/root/.deeplinkengine-validation/run-01/results.json`
- HTML report in `/root/.deeplinkengine-validation/run-01/full-report/report.html`
- JSON summary in `/root/.deeplinkengine-validation/run-01/full-report/summary.json`
