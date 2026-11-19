# 预热与编译边界 MOD

本机制已经进入默认r6：启动期间编译实际元数据内核，HTTP计时前发送合成预热，随后同时封住两rank的Torch/Dynamo与Triton JIT。测量过程中发生新的编译就终止并留下原始错误，不允许把现场编译秒数混进正常TTFT。

| 项目 | 当前记录 |
|---|---|
| 角色 | 已被引擎吸收的稳定性修复；后续对照共同使用 |
| 运行时实现 | `overlays/b1-r6/mods/betterscale/betterscale/compilation_scope.py`、`worker.py`、`models/qwen35/{__init__,serving_warmup}.py`；r7保留相同机制 |
| 服务实现 | `scripts/{compile_boundary,serving_warmup}.py`、`engine_entry.py`、`run_b1_r4.py` |
| 激活 | 默认r6冻结入口自动使用；不是搜索时开关 |
| 适用 | 当前固定Qwen3.5-35B-A3B BF16、Ascend910B2 TP2、MTP2、FULL图和对应State后端 |
| 运行时分档 | 从实际max_num_seqs覆盖二次幂分档和最后的ragged档；使用实际张量地址和bank对齐，仅compile-only，不执行修改State |
| 服务预热 | 原C8为71次；默认任务入口为126次。容量联合实验C10双方为194次，是独立版本的任务协议，不改写历史预热 |
| 资源代价 | 增加启动编译、缓存和合成请求耗时；使用fresh缓存，耗时从日志单列。计时中不做补预热、不重新打开编译 |
| 失败条件 | 未覆盖形状/地址特化会被拦截；保留原始请求和编译失败，不回退eager |
| 覆盖边界 | Torch/Dynamo和Triton；CANN内部编译尚不在该边界覆盖范围 |
| 数学/质量 | 不更改模型数学、正式输入、EOS和回答预算；编译边界通过不等于任务质量等价 |
| 独立收益 | 未做新的单因素吞吐收益消融，不能把整个r6/r7相对r4的收益归到此MOD |
| 回退 | 使用历史完整冻结版本与对应资格，不在当前对照关闭该共同能力制造弱基线 |

r4的首批HLE出现token_progress.correct现场编译，后续r6冻结包含BetterScale `263ee2d3` / `a7a43ed3`对应机制及固定适配。r6/r7保留独立资格和原件；r7 E24的完整200正确138未达到144门槛，不能因编译边界通过就把整个r7升级为默认。

已封存的r7 C8验证71次预热与两rank的编译封闭，正式测量后0次观测到的Torch/Triton编译。该次预热耗时43.345秒，是这一轮的实测，并非所有机器的固定开销。两rank原始回执、SSE、安装源哈希和复算入口在[r7 Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/full-answer-b1-20261116-r7)；对应[验收回执](../../experiments/optimization-20261115/B1-r7-async-promotion-20261116-r1/continuous-B1-review-v1.json)。

所有实现文件哈希、上游修复和合成预热协议由[mod.json](mod.json)索引。已进入默认的机制保留为文档模块，后续调优的共同基线必须包含它；收益有负载依赖的容量与复制方案继续用各自MOD进行组合测试。
