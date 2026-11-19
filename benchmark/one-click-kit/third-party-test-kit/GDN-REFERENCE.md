# GDN 独立递推参考

`gdn_reference.py` 提供独立的纯 Python 单 head 校验器和未融合的 FP32 Tensor 参考算子。输入是卷积、Q/K 归一化和门控预处理之后的值；状态布局为 `[..., value_heads, key_dim, value_dim]`。Tensor 路径支持 grouped key heads，不导入或调用生产 GDN 内核，不修改传入状态，也不在计算路径内读取设备标量。

一个 value head、一个有效递推位置的基本运算为：

| 分项 | 基本运算 |
| --- | ---: |
| Q 乘 scale | K |
| 状态衰减 | KV |
| K 与状态点积 | (2K−1)V |
| value 减预测值 | V |
| 残差乘 beta | V |
| outer-product 更新状态 | 2KV |
| Q 与更新状态点积 | (2K−1)V |
| 合计 | 7KV+K |

EXP 单列为特殊函数；FMA 按两次基本运算，归约只含 `K−1` 次加法。K=V=128 时每 head 为 114816，历史固定200/out256、30个 GDN 层、32个全局 value heads 的解码分项为 5621391360000，与原认证分项精确一致。该历史复算不为新输出轨迹提供 draft 或预填充工作证明。

[实卡回执](../validation/gdn-independent-reference-20261119.json)绑定冻结 r8 的实际 `betterscale/models/qwen35/decode_kv.py` 源哈希及本参考源哈希。物理卡1、6分别覆盖 batch1/8/36、单 token、3列 speculative 状态和变长行，共12组，核对所有输出及整个状态池，包括未写入的槽位。最大 BF16 输出绝对差 3.814697265625e−6，最大 FP32 状态绝对差 1.4901161193847656e−8。纯 Python oracle 另核对每个实卡用例的首个 head 状态转换。

独立参考另记录每卡42个标定样本及18个留出计时样本，范围仅为 **batch1、单次转换、每rank8个 key heads/16个 value heads、K=V=128**。它们是带该卡设备同步的 host 计时，不能用来代表 batch8/36 的算子耗时，不能拼成 TP2 整链参考时间或直接套到 MMLU200/HTTP。

这完成了一个解码参考算子的数值校验，尚未完成 M01 全模型认证。分段 GDN prefill、其他标量、必要 draft 生产者闭包、与当前轨迹匹配的算子形状及跨流/跨卡参考 DAG、独立整链验证和同域同步分子仍需完成。`complete_effective_flops`、`reference_tflops_per_card` 和 `efficiency_percent` 保持 null；不能据此宣称50%或90%目标通过。

CPU 检查：`python -B -m pytest -q tests/test_gdn_reference.py`。没有 Torch 的 CPU runner 仍执行纯 Python 与运算数检查，Tensor 与纯 Python 的多步 grouped-head 比较在安装 Torch 时执行。

一键入口默认自动生成绑定当前 SDK、工况及源码哈希的计划，并在性能服务退出后调用 `qualify_gdn_reference.py`。无需用户手工提供计划；已有封存结果可用 `one_click.py --compute-reference-only --reference-campaign ...` 接续。两张卡的12组实测和每卡标定均进入回执及报告；完整认证缺项保留为 `incomplete`，返回非零，效率不填值。冻结 runtime 与旧实验保持原身份。

## 分段 prefill

`gdn_prefill_reference.py` 增加独立的三角系统正向代入和分段 WY/H/O 参考；默认 FP32 路径由纯 Python 多步递推校验，数值资格比较另显式模拟生产源码的 BF16 inverse、W/U、H、Vnew 和 score 存储。H 衰减使用 BF16 工作区，最终状态池才写 FP32；契约源码及 Git blob 核验记录见 `gdn-contract-sources/storage-contract.json`。不调用生产三角逆或 H/O 内核作为参考。

[本次双卡回执](../validation/gdn-prefill-independent-reference-20261119.json)覆盖真实1/17/64/65/129 token、连续分段、初始零/非零状态、强/弱衰减、混合三请求与空容量尾行，共18组；所有状态池槽位均核对。输出及状态使用预先固定的 `atol=0.0003, rtol=0.025`，未放宽容差。先前导入、初始化和数值失败均保留；完整算力数值仍未产生。

`gdn_work_geometry.py` 从真实 host metadata 的 request、层、位置、cu 与 initial/verify 信息复原必要分段；不把最终 token 总数重新切块。新增 HTTP 观察阶段沿用当前固定200/max8192/EOS请求，原始 SSE ID/usage 与两 rank/scheduler逐波核对。观察服务的时间含插桩，不能用作干净性能分子。
