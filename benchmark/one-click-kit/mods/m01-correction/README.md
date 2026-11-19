# M01 correction

CorrectedWorker 的进度源快照、独占乐观操作数、因果 draft 长度和主机 prefill 工作量统计修正。

## 冻结来源与实现

本条目对应 **r6 冻结 B1 默认启用组件**。完整机器清单见 [mod.json](mod.json)，组合与原件哈希见 [r6 已启用栈](../active-stack-r6.json)。

- 部署目录：`mods/correction`、`mods/correction/correction-plugin-installed-r1`。
- 基座与固定配置：[r4](../../engines/b1-r4.json) → [r6](../../engines/b1-r6.json)，精确变更见 [r6 overlay](../../engines/b1-r6-overlay.json)。
- 原件由 [下载入口](../../scripts/fetch_b1_r4.py)和[恢复入口](../../scripts/restore_b1_r4.py)从 [r4 Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/full-answer-b1-20261030-r4)取回并校验。完整文件集合由原件中的 source/component manifest 与显式 overlay 共同约束；mod.json 另列部署入口 SHA256。

目录统一提供模块说明和清单；执行源码由上述冻结原件恢复到 SDK。覆盖文件继续使用仓库已有 overlays，避免多份源码漂移。

## 启用与依赖

general plugin hust_m01_correction 通过 corrected_serving_runtime.register 安装 CorrectedWorker；必须显式 HUST_M01_CORRECTED_SERVING=1 与 HUST_M01_HOST_PREFILL_AGGREGATE=1。

直接依赖：[betterscale](../betterscale/README.md)、[microflow](../microflow/README.md)。本组合所有已启用模块都保留为共同基线；修改组合应建立新 profile 并重新验收。恢复则使用同版、同哈希原件。

## 适用条件与资源

适用：Ascend 910B2、Qwen3.5-35B-A3B、BF16、TP2/PP1；固定 E24/R36、MTP2、FULL。状态快照、拥有权和因果检查及主机计数开销计入正式请求；冻结路径不启用诊断 journals 或 route recording。

CorrectedWorker 要求 TP2/PP1，并校验先由 Manager 启用 Microflow Worker。目录 ID 为 m01-correction，部署路径沿用 mods/correction，避免改变 PYTHONPATH。ABI 签名路径重定位由 restore 脚本单独校验。

## 证据

固定组合及部署状态见 [r6 执行回执](../../validation/r6-self-run-status-20261115.json)。该证据属于组合；本次整理没有增加独立消融数据，不为该组件单列加速比例。历史实验与失败记录仍从 [MOD 总目录](../README.md)进入。
