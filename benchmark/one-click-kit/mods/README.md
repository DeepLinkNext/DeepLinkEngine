# MOD目录

## r8 已包含的 MOD

统一标记 **`r8 included / r8 已包含`**，机器清单为 [active-stack-r8.json](active-stack-r8.json)，与引擎页和 Release 使用相同 ID。MTP2、FULL、E36/R54 与资源预算属于执行配置，不另外计作 MOD。

| ID | MOD | r8 来源 |
|---|---|---|
| `extension-manager` | [Extension Manager](extension-manager/README.md) · **r8 已包含** | 继承 |
| `betterscale` | [BetterScale](betterscale/README.md) · **r8 已包含** | 继承 |
| `microflow` | [Microflow](microflow/README.md) · **r8 已包含** | 继承 |
| `kv-materialization` | [KV materialization](kv-materialization/README.md) · **r8 已包含** | 继承 |
| `m01-correction` | [M01 correction](m01-correction/README.md) · **r8 已包含** | 继承 |
| `packed-host` | [Packed host state](packed-host/README.md) · **r8 已包含** | 继承 |
| `sealed-serving-warmup` | [Sealed serving warmup](sealed-serving-warmup/README.md) · **r8 已包含** | 继承 |
| `guarded-progress-async` | [Guarded progress async](guarded-progress-async/README.md) · **r8 已包含** | r8 新增 |
| `capacity-tuning` | [Capacity tuning](capacity-tuning/README.md) · **r8 已包含** | r8 新增 |
| `short-prefill-tail` | [Short prefill tail](short-prefill-tail/README.md) · **r8 已包含** | r8 新增 |

r8 保持冻结的10个运行时覆盖文件及E36/R54、MTP2、FULL、State4GiB/rank、KV20GiB/rank、Tail256。逐MOD文档中的旧r6/r7结果保留原身份；新组合收益不相加，不据此推断每个MOD的独立贡献。restore/RMS融合尚不在r8中。


## 冻结 B1 已启用组件（r6）

所有已使用的 MOD 都在这里登记。默认冻结组合为 **E24/R36、MTP2、FULL、逐请求 KV 重算、Microflow、同对象同 dtype 锁页合并**；主机状态 **4 GiB/rank**、KV **20 GiB/rank**。机器可读版本见 [active-stack-r6.json](active-stack-r6.json)，每项提供独立 README 和 mod.json。

| 组件 | 作用 | 恢复后的 SDK 目录 |
|---|---|---|
| [BetterScale](betterscale/README.md) | Qwen3.5 的执行与状态管理底座；提供 LiveStateScheduler、执行席位/常驻席位、MTP2 和 FULL 图执行。 | `mods/betterscale` |
| [Microflow](microflow/README.md) | school028 FA-only 的 Prefill Microflow 执行接入，冻结配置使用 on 臂。 | `mods/microflow` |
| [KV materialization](kv-materialization/README.md) | 冻结 school028 路径的逐请求 KV 重算控制与请求处理接入。 | `mods/kv-materialization` |
| [M01 correction](m01-correction/README.md) | CorrectedWorker 的进度源快照、独占乐观操作数、因果 draft 长度和主机 prefill 工作量统计修正。 | `mods/correction`、`mods/correction/correction-plugin-installed-r1` |
| [Packed host state](packed-host/README.md) | 将同一状态对象中相同 dtype 的主机状态合并为锁页分配，保留逐对象的拥有权和生命周期。 | `mods/packed-host` |
| [Extension Manager](extension-manager/README.md) | 负责扩展发现、配置、依赖/能力接入与启动环境编排，统一启用冻结的 Microflow 和 KV 扩展。 | `manager` |
| [预热与编译封锁](sealed-serving-warmup/README.md) | 服务 ready 前预热并封锁观测到的 Torch/Triton 编译路径 | r6 overlays 与启动脚本 |

源码按冻结 Release 与 overlay 恢复，各模块在本目录统一索引；vLLM/Ascend 基座、支持库和加载顺序见机器清单。MTP2、FULL 和席位参数作为 BetterScale/引擎执行模式记录。后续实验候选的状态独立列在下方。

## 探索与候选 MOD

| MOD | 当前结论 |
|---|---|
| [进度复制异步化](guarded-progress-async/README.md) | 两对完整模型性能通过；r7原生/C8通过，完整任务正确数筛选未通过，默认未替换 |
| [容量调优](capacity-tuning/README.md) | 多组重复吞吐收益；TPOT、质量与资源预算一并选择 |
| [Tradeoff调优](tradeoff-tuning/README.md) | 统一证据目录、绘图与预算筛选 |
| NZ与router布局系列 | 保留兼容修复、单轮小收益和负结果；具体状态见registry |

全部模块的入口与状态见[registry.json](registry.json)。稳定更好的实现进入新版B1默认；有条件的配置保留为可组合模块。原始结果、失败和历史版本继续保留。

模块交付内容、取舍与默认B1升级流程见[MOD整理规范](STANDARD.md)。


[预热与编译边界](sealed-serving-warmup/README.md)已作为默认r6吸收的机制登记，保留实现哈希、适用域、启动成本、失败策略和CANN覆盖边界；没有独立吞吐消融就不列独立加速百分比。原r7/E24正确138未过144门槛，默认仍为r6；E36/R54＋原生异步的新组合进入独立C10验收，不能把各自历史收益相乘。

[C10 联合配置验收](../experiments/optimization-20261115/C10-joint-capacity-async-model-20261116-r1/README.md)：E36/R54＋原生异步的首个C8已完成。4K吞吐提高10.52%，但平均TTFT与P95分别增加9.63%和29.69%，未通过原5%上限；16K通过原定比较。完整200题两对已结束，吞吐+42.64%/+42.08%，两对性能门槛均通过；首对正确143低于144，反向对照截断25高于22，原质量门槛未通过，默认r6不变。

历史布局模块的实现与原件直接保留在实验目录：[NZ标量兼容修复](../experiments/optimization-20261115/mods/nz-scalar-compat/README.md)、[NZ线性兼容](../experiments/optimization-20261115/mods/nz-linear-compat/README.md)、[router预转换](../experiments/optimization-20261115/mods/router-precast/README.md)、[gate布局](../experiments/optimization-20261115/mods/nz-gates-nd/README.md)、[expert线性布局](../experiments/optimization-20261115/mods/expert-nz-linear-nd/README.md)。registry已逐项指向这些实际文档及mod.json；[目录核验](catalog-integrity-20261116-v1.json)记录9项文档和9份清单的提交、路径与Git哈希。负收益与兼容修复继续分别标记，不计作稳定性能增益。

[restore＋gated RMS融合](restore-gated-rms-fusion/README.md)已保存为组件MOD：T80/108 mixed每候选16对均获益，T1536/4096的32对均未过；完整组件门槛失败，未集成模型。局部收益与完整负例一并展示。

2026-11-17更新：[restore＋gated RMS 0.2.0](restore-gated-rms-fusion/README.md)的packed32/64各通过全部32数值/48性能对；原大形状退化已在新版本消除，旧0.1.0负例保留。组件wall降低35.52%–67.91%和18.39%–68.07%，不作完整模型吞吐归因。图角色复用及Z/norm接口接入继续独立验收，默认r6不变。
