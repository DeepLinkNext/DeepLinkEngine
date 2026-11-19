# BetterScale

Qwen3.5 的执行与状态管理底座；提供 LiveStateScheduler、执行席位/常驻席位、MTP2 和 FULL 图执行。

## 冻结来源与实现

本条目对应 **r6 冻结 B1 默认启用组件**。完整机器清单见 [mod.json](mod.json)，组合与原件哈希见 [r6 已启用栈](../active-stack-r6.json)。

- 部署目录：`mods/betterscale`。
- 基座与固定配置：[r4](../../engines/b1-r4.json) → [r6](../../engines/b1-r6.json)，精确变更见 [r6 overlay](../../engines/b1-r6-overlay.json)。
- 原件由 [下载入口](../../scripts/fetch_b1_r4.py)和[恢复入口](../../scripts/restore_b1_r4.py)从 [r4 Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/full-answer-b1-20261030-r4)取回并校验。完整文件集合由原件中的 source/component manifest 与显式 overlay 共同约束；mod.json 另列部署入口 SHA256。

目录统一提供模块说明和清单；执行源码由上述冻结原件恢复到 SDK。覆盖文件继续使用仓库已有 overlays，避免多份源码漂移。

## 启用与依赖

冻结启动命令使用 betterscale.qwen35_worker.Worker 与 betterscale.models.qwen35.seat_scheduler.LiveStateScheduler；Manager/M01 注册后实际 Worker 为 corrected_serving_runtime.CorrectedWorker。

直接依赖：固定引擎基座，见组合清单。本组合所有已启用模块都保留为共同基线；修改组合应建立新 profile 并重新验收。恢复则使用同版、同哈希原件。

## 适用条件与资源

适用：Ascend 910B2、Qwen3.5-35B-A3B、BF16、TP2/PP1；固定 E24/R36、MTP2、FULL。执行图与多档形状预热占用启动时间和设备缓存；状态常驻与主机分层预算由固定组合统一限制。

这里登记的是 r4 恢复快照叠加 r6 精确覆盖文件。不能用 BetterScale 上游最新版本或单一提交代替整套冻结来源。

## 证据

固定组合及部署状态见 [r6 执行回执](../../validation/r6-self-run-status-20261115.json)。该证据属于组合；本次整理没有增加独立消融数据，不为该组件单列加速比例。历史实验与失败记录仍从 [MOD 总目录](../README.md)进入。
