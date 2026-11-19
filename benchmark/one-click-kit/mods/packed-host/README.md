# Packed host state

将同一状态对象中相同 dtype 的主机状态合并为锁页分配，保留逐对象的拥有权和生命周期。

## 冻结来源与实现

本条目对应 **r6 冻结 B1 默认启用组件**。完整机器清单见 [mod.json](mod.json)，组合与原件哈希见 [r6 已启用栈](../active-stack-r6.json)。

- 部署目录：`mods/packed-host`。
- 基座与固定配置：[r4](../../engines/b1-r4.json) → [r6](../../engines/b1-r6.json)，精确变更见 [r6 overlay](../../engines/b1-r6-overlay.json)。
- 原件由 [下载入口](../../scripts/fetch_b1_r4.py)和[恢复入口](../../scripts/restore_b1_r4.py)从 [r4 Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/full-answer-b1-20261030-r4)取回并校验。完整文件集合由原件中的 source/component manifest 与显式 overlay 共同约束；mod.json 另列部署入口 SHA256。

目录统一提供模块说明和清单；执行源码由上述冻结原件恢复到 SDK。覆盖文件继续使用仓库已有 overlays，避免多份源码漂移。

## 启用与依赖

general plugin hust_joint_packed = joint_packed_plugin:register；显式 HUST_JOINT_PACKED_HOST=1 后调用 packed_host_state.install。

直接依赖：[betterscale](../betterscale/README.md)。本组合所有已启用模块都保留为共同基线；修改组合应建立新 profile 并重新验收。恢复则使用同版、同哈希原件。

## 适用条件与资源

适用：Ascend 910B2、Qwen3.5-35B-A3B、BF16、TP2/PP1；固定 E24/R36、MTP2、FULL。主机锁页状态预算每 rank 4 GiB；状态布局、申请粒度和生命周期属于固定组合的一部分。

合并范围限定在同一状态对象/同一 dtype；不能写成跨请求全局合并，也不能省略主机预算或将其当作独立已测加速百分比。

## 证据

固定组合及部署状态见 [r6 执行回执](../../validation/r6-self-run-status-20261115.json)。该证据属于组合；本次整理没有增加独立消融数据，不为该组件单列加速比例。历史实验与失败记录仍从 [MOD 总目录](../README.md)进入。
