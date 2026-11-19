# Capacity tuning MOD（实验版 0.2.0）

在固定 r6 组合上按负载选择执行容量，改变 E 席位和由其推导的图捕获集合。它是控制器参数 MOD；保留 E24 基线，提供 E26/E32
候选，不修改运行时算子或生产默认值。MTP2、CP1、R36、prefill4096、每 rank 4 GiB State、FULL 图及原 State 策略均固定。当前证据只覆盖双卡 Ascend
910B2、Qwen3.5-35B-A3B BF16、完整 MMLU200 burst200/8192；不能外推成其他模型或全部五数据集的收益。

E32 两次配对吞吐提高约21.0%/18.7%，平均 TPOT 增加9.2%/8.1%，平均 TTFT 降低约24.6%/24.7%。这是吞吐与单 token 时延的取舍。E26
两对吞吐提高9.83%/7.73%，平均 TPOT
增加1.07%/1.96%，但正确数分别148→138、149→144；两对性能初筛通过，质量仍未验收。所有重复、负收益及失败保留，不选择最好一轮。性能曲线不是质量合格的 Pareto
前沿，质量等价与容量稳定性仍未验收，不自动推荐生产档位。

## 用法

只生成命令提案，不启动服务。输入是已经完成版本核验的基线启动
JSON（`command`、`env`、`engine_release`）；`fixtures/baseline-launch.json` 仅供主机边界测试，不能替代真实部署身份。

```bash
python mods/capacity-tuning/capacity_tuning.py configure \
  --baseline /absolute/path/to/verified-baseline.json --seats 26 \
  --out /absolute/path/to/new-launch-proposal.json
```

`capacity_tuning.apply(command, env, plan, config_path)` 提供与实验控制器 `optimization_config.apply`
相同的显式接口。新建、版本化的实验 runner 可导入该函数；只有声明共同 live32 预热协议的任务才允许接入。不得覆盖正在运行或已冻结 runner
的文件。此函数校验命令参数与预热计划，**完整运行时身份/M01 准入、实际预热成功、双 rank 编译封口、计时、封存和服务清理仍由原生控制器负责**。声明预热计划不等于已完成预热。

已有配对执行器与全部源包：`experiments/optimization-20261115/E02-capacity-v2/`、`E04-padding-v4/`。原件有严格源码哈希，恢复后按对应协议复制到新的运行目录，保留顺序与全部结果；不要将本
MOD 回写历史原件。统一 live32 合成预热后才开始正式请求，计时中发现 Torch/Triton 编译则失败；CANN 编译不在现有 guard 覆盖范围。MTP3 不属于当前完整 M01
组合的受支持路径。

从原生分析器已核验封存清单的 JSON 自动出图，保留输入文件 SHA256、全部已执行结果和计划总数。绘图依赖 matplotlib；配置提案不需要该依赖。

```bash
python mods/capacity-tuning/capacity_tuning.py plot \
  --analysis experiments/optimization-20261115/E02-capacity-v2/analyses/capacity-sealed4-v2.json \
  --analysis experiments/optimization-20261115/E04-padding-v4/analyses/padding-sealed4-v5.json \
  --out /absolute/path/to/new-tradeoff-output
```

输出 PNG、SVG、来源与指标 JSON；目录必须是新目录。左图吞吐—平均 TPOT，右图吞吐—平均 TTFT（含 burst
排队），点编号对应逐轮正确数/200、截断数。连线只连接预先规定的配对，不拟合趋势、不把重复题当作独立样本。TPOT 沿用客户端
`(响应结束−首 token)/(输出 token−1)`，吞吐只用正式客户端区间；启动和预热另计。

后续可在事先确定的 TPOT/TTFT 预算及独立质量门槛下选择档位。当前工具只生成显式提案和证据图，不自动选择或修改部署。

## 执行容量与驻留预算配对

0.2.0补齐C04已完成的E24/R36与E36/R54四次冷服务对照。两对吞吐分别+25.07%/+29.10%，平均TTFT−26.48%/−32.13%，平均TPOT+14.17%/+12.87%；正确139→144、149→137。全部四轮指标、逐题翻转统计及原始Release索引随包保留在`evidence/`，不挑最好一轮。E与R共同变化，不能把收益单独归因于某个参数。

生成显式提案及原生控制器配置，不启动推理：

```bash
python mods/capacity-tuning/execution_residency.py \
  --baseline /absolute/path/to/verified-r6-baseline.json \
  --execution-seats 36 --resident-seats 54 \
  --out /absolute/path/to/new-E36-R54-proposal.json
```

E24/R36共同基线保留，预算保持4GiB/rank。提案绑定MTP2/CP1/FULL/prefill4096以及原生96/108捕获闭包；配置中给出原C04执行入口和共同capacity36预热、双rank原生快照及编译封口要求。声明不代替实际准入；输入提案不得重复应用到已改过的命令。

新B1 r7吸收异步复制后，E36/R54组合需新的原生容量检查与完整对照；本入口拒绝将r6证据直接用于r7。条件收益可以选择，r6重复结果继续作为历史证据保留。

![四次完整模型容量取舍](../../experiments/optimization-20261115/C04-full-answer-capacity36-20261116-r1/full-answer-tradeoff-v2.png)


## C10 联合组合完整实测（2026-11-17登记）

E36/R54＋原生异步与r6共同使用194条不计分预热；4次独立冷服务按r6/joint/joint/r6执行同一完整200题，8192上限、EOS和原解析不变。两对吞吐+42.6438%/+42.0798%，TTFT−34.4360%/−36.5852%，TPOT+1.2467%/+0.6755%，原定性能门槛全部通过。首对正确143→143，候选低于144；反对照149→146，候选截断25高于22。不能将四轮当800道独立题或据此宣称质量等价。

另保留原C8：4K平均TTFT+9.6261%、P95+29.6950%未过，16K通过原定屏幕。默认r6不变。完整逐轮/逐题/正反对照及全部原件见[C10 complete-review](../../experiments/optimization-20261115/C10-joint-capacity-async-model-20261116-r1/complete-review-v1.json)。这是联合组合实测，不将独立MOD收益相乘。
