# Tradeoff tuning MOD（实验版 0.1.2）

把真实推理试验整理成吞吐、TTFT、TPOT、正确数与截断数的取舍图，并按使用者给定的预算列出候选。默认关闭，不启动模型或修改部署。容量执行接口复用 [capacity-tuning](../capacity-tuning/README.md)；本模块增加跨配置的证据目录、图和约束筛选，未来执行新组合仍须经过原生 runner 的版本、组合准入、预热、编译封口与封存。

当前 v3 固定 12 份原生分析，包含 **24 次不同的模型试验：22 次完整200题、2 次启动失败**。重复参考按 trial ID、指标与原始 seal 一致性去重；真正的重复运行全部保留。覆盖 E24/E26/E32、State 策略、原生 shared-expert 多流、decode padding、权重布局/router 以及 E26/布局/多流组合。仅覆盖双卡 Ascend 910B2、Qwen3.5-35B-A3B BF16、固定 MMLU200 burst200/max8192；算子微测和 profiler 时长不混入模型吞吐。E19及E20完整结果已加入v3；v1/v2及0.1.0/0.1.1源包保留，不将复用参考当新测量。

![吞吐、时延与质量观测](../../experiments/optimization-20261115/tradeoff-catalog-v3/tradeoff.svg)

图中每个编号对应 [完整试验表](../../experiments/optimization-20261115/tradeoff-catalog-v3/all-trials.svg)。高吞吐与较低 TPOT 存在取舍，正确数和截断数也会变化。菱形只表示本次历史观测在吞吐/平均TPOT/平均TTFT三项上未被其他点同时支配，**没有把任务质量纳入此标记**。不拟合跨历史运行的连线，不把图当成质量合格的 Pareto 前沿。

## 从下载的源包复现

解压 Release 的 `tradeoff-tuning-mod-v0.1.2.zip`。包内保留与仓库相同的 `mods/` 和 `experiments/` 目录，无需重新运行模型、访问私有数据集或下载权重。绘图依赖 matplotlib；筛选仅用 Python 标准库。使用已装有 matplotlib 的 Python：

```bash
python mods/tradeoff-tuning/tradeoff_tuning.py plot \
  --catalog mods/tradeoff-tuning/catalog-v3.json \
  --out /absolute/path/to/new-tradeoff-output
```

若原生实验资料单独放在另一个目录，增加 `--source-root /absolute/path/to/optimization-20261115`。默认读取仓库的 `experiments/optimization-20261115`。输出路径须是新目录：

- `tradeoff.png/svg`：四个视图，吞吐对平均TPOT、平均TTFT、正确率、截断数；
- `all-trials.png/svg/csv`：完整编号、状态、指标、正确数/200与截断数；失败没有虚构零分或零吞吐；
- `tradeoff.json`：来源 SHA256、每次实测的原始 seal、全部指标、重复引用映射与原生比较。

输入分析的字节 SHA256 必须与目录一致，同一历史 trial 的引用发生指标/seal 冲突会报错。绘图工具验证的是**分析文件绑定**；原始运行文件的逐项验真须恢复原生 raw 附件并运行对应 analyzer，本工具没有替代它。

## 按预算筛选

明确给出平均TPOT、平均TTFT、观测正确数和截断数的预算；例如下列数值仅演示语法，不是项目验收指标：

```bash
python mods/tradeoff-tuning/tradeoff_tuning.py shortlist \
  --evidence /absolute/path/to/new-tradeoff-output/tradeoff.json \
  --max-tpot-ms 21.5 --max-ttft-ms 270000 \
  --min-correct 140 --max-truncated 30 \
  --out /absolute/path/to/new-shortlist.json
```

配置的**全部已声明重复**均满足预算才列为观测符合；一轮失败、未完成或越界会给出具体 trial 和原因。共同基线/全关选项保留，不强制选开启 MOD 的配置，也不挑最好一轮排序。不同原生协议的对照配置分组保留，不能把跨版本历史控制混称为严格重复。每个候选仍标记 `deployable=false`，因为观测分数筛选不是独立质量等价或稳定性验收。

## 证据边界与后续接入

所有试验保留原生成预算、EOS、原生客户端时钟与 scorer；TTFT 包含 burst 排队，启动/预热成本另见各试验的冷成本报告。同题重复不是新增独立题目；输出长度会随配置变化。历史运行还有源码和运行时条件差异，单项/组合收益沿用各来源的原生配对分析，不从散点图归因，也不相加单项收益。

生产默认参数、正式 B0/B1 网站和四项验收结论均不由本图改写。后续加入 HLE、其他模型、负载或独立质量集时必须新建来源目录和适用范围，不能把当前点直接外推。自动搜索策略的价值需要相同候选池与预算的独立消融。

运行主机边界检查：`python -m pytest mods/tradeoff-tuning/tests -q`。

E19补充：关闭态1127.591 tok/s、143/200、截断20；开启态历史E18为1124.222 tok/s、154/200、截断20。Off→On吞吐−0.299%，TPOT−0.490%；相对原ND的完整组合收益不能归给多流单项。两次历史运行的分数差不是质量等价或模型/引擎单方因果结论。

E20补充：E28/必需GDN96、原生ND/多流关闭，1102.424 tok/s、137/200、截断19。相对原ND吞吐+3.272%、平均TPOT+8.720%，性能初筛不通过；质量与稳定性没有晋升。
