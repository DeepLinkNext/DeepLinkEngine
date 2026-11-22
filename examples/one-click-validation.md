# 一键验收说明

在昇腾主机或容器中，填写本机配置后执行一次 `run_all.py`，即可从环境准备跑到验收报告生成。

## 1. 核对环境

提前准备与所选引擎发行版兼容的 Linux aarch64、昇腾驱动、CANN、CPython 3.12，以及可读的 Qwen3.5-35B-A3B 权重。运行下面的命令，确认实际分配的两张物理 NPU 健康且空闲。卡号取 `npu-smi info` 的 `NPU` 列。

```bash
npu-smi info
```

## 2. 填写本机配置

将尖括号占位符连同括号替换为实际值。`<WORK_DIR>` 是有写权限和足够空间的工作目录，`<MODEL_DIR>` 是模型权重目录，`<NPU_A>`、`<NPU_B>` 填两张不同的物理卡号，使用整数。

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

`cache` 用于保存可复用的下载和环境缓存。每轮测试的 `out` 使用尚未创建的目录。如需指定环境路径，可在配置的 `environment` 中填写 `python` 或 `cann_home`。

## 3. 一键运行

`<PYTHON>` 填启动用的 Python 命令或解释器路径，`<REPO_DIR>` 填 DeepLinkEngine 仓库根目录。首次验收用 `--rerun-b0` 实测 B0，与本轮安装的 B1 配对。入口默认选择并记录最新稳定版 B1。

```bash
'<PYTHON>' '<REPO_DIR>/run_all.py' --config '<WORK_DIR>/evaluation.json' --rerun-b0
```

| 自动阶段 | 做什么 |
| --- | --- |
| 环境、数据和部署 | 准备隔离的 Python 环境，下载并校验评测数据与引擎，安装 B1，并在预检时恢复或校验固定 B0。 |
| 配对预检 | 校验模型、设备映射和测试输入，冻结后续执行计划。 |
| 固定五项与算力参考 | 用同一模型和同一对卡测 B0/B1，运行 MMLU-Pro、HLE、FrontierScience、SWE-bench-Pro 和 Terminal-Bench，并完成算力参考认证。SWE 和 Terminal 按固定轨迹回放。 |
| 上下文与四租户 | 运行仓库内置的 4K、16K、32K 上下文和四租户负载。 |
| 汇总 | 核对原始记录及校验值，生成 HTML 和 JSON 报告。 |

## 4. 查看结果

- `<WORK_DIR>/run-01/results.json` 记录各阶段执行状态和报告路径。
- `<WORK_DIR>/run-01/full-report/report.html` 是可在浏览器打开的总报告，同目录的 `summary.json` 保存结构化结论。

执行是否完成查 `results.json`，各项验收指标是否达标查总报告。固定路径示例见[双卡验收示例](one-click-validation-0-1.md)。
