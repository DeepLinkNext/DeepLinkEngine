# 一键验收示例 · 物理 NPU 0、1

本例在 `/root/.deeplinkengine-validation` 执行。仓库位于 `/root/DeepLinkEngine`，模型权重位于 `/data/shared_models/Qwen3.5-35B-A3B`，已准备兼容的 Linux aarch64、昇腾驱动、CANN 和 CPython 3.12。

## 1. 核对卡号

查看 `NPU` 列，确认物理卡 `0`、`1` 都分配给当前环境，且健康、空闲。

```bash
npu-smi info
```

## 2. 写入验收配置

配置使用同一份模型，缓存留在工作目录下，本轮结果写入新的 `run-01`。再次从零测试时，先把 `out` 改为尚未创建的目录。

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

## 3. 启动本轮验收

```bash
python3 /root/DeepLinkEngine/run_all.py \
  --config /root/.deeplinkengine-validation/evaluation.json --rerun-b0
```

脚本自动准备 Python 和数据，安装并校验 B1、准备固定 B0，预检通过后从零运行两臂配对测试及算力参考，再运行内置上下文和四租户测试并生成报告。`--rerun-b0` 显式要求本轮重新测 B0。各阶段和测试范围见[一键验收说明](one-click-validation.md)。

## 4. 查看结果

- 阶段状态 `/root/.deeplinkengine-validation/run-01/results.json`
- HTML 总报告 `/root/.deeplinkengine-validation/run-01/full-report/report.html`
- JSON 汇总 `/root/.deeplinkengine-validation/run-01/full-report/summary.json`
