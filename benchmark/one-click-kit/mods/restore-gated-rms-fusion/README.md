# restore＋gated RMS融合 MOD 0.2.1

连续token/head行布局已通过完整组件池：两个候选各32个数值配置和48对性能比较全部通过。尚未集成模型，默认r6保持。

| 候选 | 全部48对wall耗时降低 | 全部48对device中位数降低 | 原完整门槛 |
|---|---|---|---|
| packed32 | 35.52%–67.91% | 37.14%–68.19% | 达成 |
| packed64 | 18.39%–68.07% | 36.52%–68.43% | 达成 |

全部原6形状×2seed×2卡×正反均保留，要求每一对wall和device各降低≥5%；没有缩减大prefill负例。旧C11两候选的完整失败见[0.1.0历史](history/0.1.0/README.md)，原Release不变。每候选与原生有120个BF16低位不同单元，全部通过预先固定的数值误差；这不代替任务质量。

![全部组件对照](../../experiments/optimization-20261115/C12-packed-restore-RMS-component-20261117-r1/component-pairs-v1.svg)

实现将大形状token×16heads展平，BLOCK_ROWS32/64，每行128列，使Y/Z/rstd连续访问；P仍按真实head-major转置view和原MAP寻址。T≤108沿用原R16小形状实现。源码、哈希、启停、资源、全部对照及已下载验证的双卡数组附件见[mod.json](mod.json)和[evidence-v0.2.0.json](evidence-v0.2.0.json)。冻结可复跑入口是C12原控制器/协议，不能回写已有结果目录。

C12的纯prefill大形状使用编译时PREFILL_ONLY=True。真实模型若复用同一token桶的纯prefill/mixed图，就不能按捕获时角色固定这个值；C14已对原完整池完成统一False路径验证：两个候选各32数值配置及48对性能比较全部通过，原始数组已下载回读。实际接入还需显式传递Z和norm参数、正确声明custom-op mutation，每分支只归一化/门控一次；见[7份原生源码接口核验](../../experiments/optimization-20261115/M12-restore-RMS-integration-audit-20261117-r1/README.md)。

这些是组件耗时收益，不是完整回答吞吐。通过图复用接口与真实模型的原生、上下文和任务验收后，才升级默认B1；纯verification目前走另一条原生路径。

服务形状核验：80/108在线上是纯verification桶，不走restore。原组件结果保留；实际prefill桶16/32/64/128/256/512/1024/1536/2048/4096的完整映射尚需覆盖。见[源码范围补充](../../experiments/optimization-20261115/M12-restore-RMS-integration-audit-20261117-r1/README-serving-domain-v2.md)。
