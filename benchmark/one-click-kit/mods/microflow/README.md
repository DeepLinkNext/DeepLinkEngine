# Microflow

school028 FA-only 的 Prefill Microflow 执行接入，冻结配置使用 on 臂。

## 冻结来源与实现

本条目对应 **r6 冻结 B1 默认启用组件**。完整机器清单见 [mod.json](mod.json)，组合与原件哈希见 [r6 已启用栈](../active-stack-r6.json)。

- 部署目录：`mods/microflow`。
- 基座与固定配置：[r4](../../engines/b1-r4.json) → [r6](../../engines/b1-r6.json)，精确变更见 [r6 overlay](../../engines/b1-r6-overlay.json)。
- 原件由 [下载入口](../../scripts/fetch_b1_r4.py)和[恢复入口](../../scripts/restore_b1_r4.py)从 [r4 Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/full-answer-b1-20261030-r4)取回并校验。完整文件集合由原件中的 source/component manifest 与显式 overlay 共同约束；mod.json 另列部署入口 SHA256。

目录统一提供模块说明和清单；执行源码由上述冻结原件恢复到 SDK。覆盖文件继续使用仓库已有 overlays，避免多份源码漂移。

## 启用与依赖

config/manager.json 启用 org.vllm-hust.prefill-microflow-school028-fa；school028 注册要求 HUST_MICROFLOW_SCHOOL028_FA=1、HUST_MICROFLOW_ARM=on，原内置 BETTERSCALE_PREFILL_MICROFLOW 保持关闭以免重复接入。

直接依赖：[betterscale](../betterscale/README.md)、[extension-manager](../extension-manager/README.md)。本组合所有已启用模块都保留为共同基线；修改组合应建立新 profile 并重新验收。恢复则使用同版、同哈希原件。

## 适用条件与资源

适用：Ascend 910B2、Qwen3.5-35B-A3B、BF16、TP2/PP1；固定 E24/R36、MTP2、FULL。需要独立缓存/事件目录、预热与执行流资源；host-contract 检查固定宿主版本和源码。

运行时实际用的是 school028 FA 路径；不能仅根据包名推断所有 Microflow 算子、模式均已启用。r6 刷新 host-contract 的精确覆盖见 overlay。

## 证据

固定组合及部署状态见 [r6 执行回执](../../validation/r6-self-run-status-20261115.json)。该证据属于组合；本次整理没有增加独立消融数据，不为该组件单列加速比例。历史实验与失败记录仍从 [MOD 总目录](../README.md)进入。
