# restore＋gated RMS融合 MOD 0.1.0

保存已重复观察到的局部收益及完整负例。它是组件候选，未安装进引擎，完整组件筛选未通过。

双卡Ascend910B2、每rank16heads×128，原生MixedCore含至少一个prefill时，将激活布局恢复和gated RMS从两次launch融合成一次；保留原BF16写回舍入和真实P转置view寻址。源码、参数、启停入口、适用条件、资源与原始附件见[mod.json](mod.json)。

| 范围 | fused16 | fused64 |
|---|---|---|
| T80/108 mixed，各2seed×2卡×正反，共16对/候选 | wall每对降低34.72%–41.89% | wall每对降低27.17%–39.69% |
| T1536/4096 prefill/mixed，共32对/候选 | 全部未过性能门槛 | 全部未过性能门槛 |
| 全部32个数值配置/候选 | 通过原误差判据 | 通过原误差判据 |
| 原定全部48对wall/device各降低≥5% | 未通过 | 未通过 |

小形状16对的输出与原生BF16逐位一致；大形状存在低位差异，两卡共120单元/候选，仍通过原数值阈值。各轮完整结果见[evidence-v1.json](evidence-v1.json)，完整图和数组见[原实验](../../experiments/optimization-20261115/C11-restore-gated-RMS-component-20261116-r3)。

收益来自组件实测，不能写成完整回答吞吐提高约40%，也不覆盖纯verification/decode。原全部形状门槛不因局部正例改变。下一版可研究小mixed分支的显式dispatch和大prefill布局，再在独立版本跑完整原池及模型验收。

使用`implementation/fused_restore_gated_rms.py`中的原始Triton入口显式调用，BLOCK_M仅取预声明16/64；关闭时仍调用原restore R16＋gated RMS M64。未提供生产开关或自动路由，默认r6不变。原始正例、负例、输入输出、图复放、编译边界、CPU阶段栅栏和canary全部保存在同一[Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/performance-exploration-20261115-r1)。
