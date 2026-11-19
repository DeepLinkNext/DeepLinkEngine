## 完整同轮上下文复核（2026-11-17）

4K两对中1对通过，16K两对通过。反向4K平均TTFT+11.38%、P95+20.23%，超过原+5%上限；吞吐+11.48%、TPOT−13.21%的收益同时保留。组合未通过全部上下文发布筛选，默认仍为r6。[全部原件与复算](../../experiments/optimization-20261115/C15-complete-matched-context-20261117-r1/README.md)。

# 短 prefill 尾段优先调度

先预留 decode，随后优先处理已经进入执行环、满足原 eligibility 条件且剩余需求为 1–256 token 的 prefill；其余请求沿原环顺序分配。新请求准入、最小不可分割额度、环轮转、ResidentLeases.offer 和算子保持原样。

原生 M11 诊断发现：4K/16K 各8条闭环请求中分别有6条的短尾曾因本轮预算先耗尽而拿不到额度。r8 将此策略与原生异步复制、E36/R54 组合，首组同轮4K/16K吞吐分别提升12.11%/4.90%，原时延筛选通过。组合收益不能归因于单个调度策略，完整反向和任务验收按 C15 原计划执行。

额外开销是一次有界的 host 请求分组。适用条件是多个 prefill 竞争4096 token预算；其他并发、长短比例和模型的收益边界尚未建立。没有改动状态预算或数值计算。

[源码](../../overlays/b1-r8/mods/betterscale/betterscale/models/qwen35/prefill_round_robin.py) · [固定清单](mod.json) · [r8组合](../../engines/b1-r8.json) · [全部实机回执](../../experiments/optimization-20261115/C15-r7-tail-combination-model-20261117-r1)
