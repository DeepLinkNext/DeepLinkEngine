# 一键验收示例 · 物理 NPU 0、1

本例在 `/root/.deeplinkengine-validation` 执行。仓库位于 `/root/DeepLinkEngine`，模型权重位于 `/data/shared_models/Qwen3.5-35B-A3B`。宿主机或容器需提前具备 Linux aarch64、昇腾驱动、CANN 9.1.0 和可用的 CPython 3.12。

## 1. 核对卡号

查看 `NPU` 列，确认物理卡 `0`、`1` 都分配给当前环境，且健康、空闲。

```bash
npu-smi info
```

## 2. 写入验收配置

两套引擎使用同一份模型权重。缓存留在工作目录下，本轮结果写入 `run-01`。再次从零测试时，先把 `out` 改为尚未创建的目录。

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

这条命令为 B1 准备独立 Python 环境、校验依赖并安装 SDK，也会在预检时从固定 OCI 镜像恢复或校验 B0 运行环境。预检通过后，脚本运行固定五项、算力参考、内置上下文和四租户测试，再生成报告。`--rerun-b0` 要求本轮重新测 B0。各阶段的内容见[一键验收说明](one-click-validation.md)。

## 4. 查看结果

- 阶段状态 `/root/.deeplinkengine-validation/run-01/results.json`
- HTML 总报告 `/root/.deeplinkengine-validation/run-01/full-report/report.html`
- JSON 汇总 `/root/.deeplinkengine-validation/run-01/full-report/summary.json`
