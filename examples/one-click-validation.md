# 一键验收说明

在昇腾主机或容器中填好配置后，执行一次 `run_all.py`，即可准备 B0 和 B1 的运行环境、完成配对测试并生成报告。

## 1. 核对宿主环境

宿主机或容器需提前具备 Linux aarch64、昇腾驱动、CANN 9.1.0、可用的 CPython 3.12 和可读的 Qwen3.5-35B-A3B 权重。运行下面的命令，确认实际分配的两张 910B2 健康且空闲。物理卡号取 `NPU` 列。

```bash
npu-smi info
```

## 2. 填写本机配置

将尖括号占位符连同括号替换为实际值。用有写权限且有足够空间的绝对路径填写 `<WORK_DIR>`，用模型权重目录填写 `<MODEL_DIR>`，用两个不同的整数填写物理卡号 `<NPU_A>`、`<NPU_B>`。

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

`cache` 保存可复用的下载和环境缓存。每轮测试的 `out` 使用尚未创建的目录。如需指定宿主环境路径，可填写 `environment.python` 或 `environment.cann_home`。

## 3. 一键运行

`<PYTHON>` 填启动用的 Python 命令或解释器路径，`<REPO_DIR>` 填 DeepLinkEngine 仓库根目录。`--rerun-b0` 要求本轮实测 B0，与本轮 B1 配对。入口默认选择并记录最新稳定版 B1。

```bash
'<PYTHON>' '<REPO_DIR>/run_all.py' --config '<WORK_DIR>/evaluation.json' --rerun-b0
```

| 自动阶段 | 做什么 |
| --- | --- |
| B1 环境和数据准备 | 加载已安装的 CANN 9.1.0，创建独立 Python 虚拟环境，补齐并校验依赖。下载并校验固定评测数据。 |
| B1 安装 | 安装并校验选定版本的 B1 SDK，记录到本轮 `out/installation`。 |
| B0 环境和配对预检 | 从校验过的固定 `original018` OCI 镜像恢复 B0 SDK，包括镜像内的 Python、依赖和 CANN 8.5.1 运行库。已提供 `baseline_deployment` 时校验该 B0 SDK。随后核对模型、设备映射和测试输入，冻结执行计划。 |
| 固定五项与算力参考 | 用同一模型和同一对卡测 B0/B1，运行 MMLU-Pro、HLE、FrontierScience、SWE-bench-Pro 和 Terminal-Bench，并完成算力参考认证。SWE 和 Terminal 按固定轨迹回放。 |
| 上下文与四租户 | 运行仓库内置的 4K、16K、32K 上下文和四租户负载。 |
| 汇总 | 核对原始记录和校验值，生成 HTML 和 JSON 报告。 |

## 4. 查看结果

- `<WORK_DIR>/run-01/results.json` 记录各阶段执行状态和报告路径。
- `<WORK_DIR>/run-01/full-report/report.html` 是 HTML 总报告。同目录的 `summary.json` 保存结构化验收结论。

执行是否完成查 `results.json`，各项指标是否达标查总报告。固定路径配置见[双卡验收示例](one-click-validation-0-1.md)。
