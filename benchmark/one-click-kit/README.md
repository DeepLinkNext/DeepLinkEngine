# vLLM-HUST Engine

仓库协作、测试与发布流程统一见 [AGENTS.md](https://github.com/vLLM-HUST/vllm-hust-engine/blob/main/AGENTS.md)；CI 与自动发布详见 [ci/README.md](ci/README.md)。

## 在现有系统中替换官方包

在目标 **CPython 3.12 / Linux aarch64 / CANN 9.1.0 虚拟环境**中执行一条命令：

```sh
python -m pip install --no-deps -r https://vllm-hust.sage.org.ai/wheels/r8.txt
```

清单显式固定 Engine `2026.11.17.8.post2`、两个基础包及支持包共四项与 SHA256，pip 在一次操作中安装冻结的同名 `vllm`、`vllm-ascend` wheel 和 `vllm-hust-runtime` 支持包，替换旧官方包，保留 `import vllm`、`import vllm_ascend` 和 `vllm` 命令。无需另跑安装器、克隆仓库、访问私有源码或重编 native 产物。完整 SDK 和评测工具仍包含在 Engine 包中。`post2` 只修订安装依赖，r8 冻结引擎、MOD、原始归档及实测结果保持原身份。网页上的[安装入口](https://vllm-hust.sage.org.ai/engine.html#install-runtime)提供完整安装清单。post1 是依赖解析未通过、未上传 PyPI 的候选；保留诊断原件，正式安装用 post2。

前置依赖：CANN 9.1.0、Torch 2.13.0+cpu、Torch-NPU 2.13.0rc1、Triton-Ascend 3.6.0、Transformers 5.14.1、tokenizers 0.22.2。该命令的 `--no-deps` 保留已有固定依赖环境，仅替换清单列出的四包；上游 Triton-Ascend 与 vLLM 的部分依赖元数据存在 NumPy 版本冲突，不能递归重解供应商环境，也不删除或伪造这些元数据。不支持的 Python 或平台没有匹配 wheel。硬件与模型验收范围为双 Ascend 910B2、Qwen3.5-35B-A3B BF16 / TP2。

通用 `vllm serve` 使用替换后的基础包；**Qwen35 的10项 MOD、CorrectedWorker 和 E36/R54/MTP2/FULL 配方由下文冻结 r8 入口启用**。安装阶段不启动模型，不新增性能结论。

卸载使用目标环境的 `python -m pip uninstall vllm vllm-ascend vllm-hust-runtime`；SDK 包 `vllm-hust-engine` 可独立卸载。旧版 `.8` 的本地构建工具 `scripts/install_frozen_runtime.py` 继续保留，支持 `--build-only` 与 `--out`，不再作为新版的默认安装步骤。

## 快速开始：固定五项或快速 MMLU200

只改一份配置即可完成 B0/B1 配对评测，无需拼接脚本参数。

1. 复制[示例配置](examples/evaluation.json)，按本机填写 `model`、`b1_python`（已有部署改填 `deployment`）、`physical_devices`、`cache`、`out`：

```bash
cp examples/evaluation.json evaluation.json
```

2. 先检查准备情况，再发起正式评测；两条命令共用同一份配置和缓存：

```bash
python3 one_click.py prepare --config evaluation.json
python3 one_click.py run     --config evaluation.json
```

3. 打开配置中 `out` 对应的 `<out>-report/report.html`。

`prepare` 完成后先看 `<out>-prepare/frozen-execution-plan.json`，确认五项数据集范围、题号与预算符合预期，再执行 `run`。入口会在启动前检查 `<out>`、`<out>-logs`、`<out>-report`，任一被占用（含失效的目录链接）就拒绝并提示换目录。

关掉终端或换人接手后，可随时做一次只读查询（不写结果目录、不启动进程）：

```bash
python3 one_click.py status --run /data/vllm-hust-engine/runs/full-five-r1
python3 one_click.py status --run /data/vllm-hust-engine/runs/full-five-r1 --json
```

文本版给出整体状态、阶段与请求进度、退出码、失败原因和报告路径；`--json` 只输出一个对象供 Agent 读取，未知值为 `null`。查询成功返回 `0`，评测本身失败也返回 `0`（失败体现在字段里）；目录无效或找不到评测记录返回 `2`。

配置的 `test_mode` 默认为 **`full`：MMLU-Pro 固定200、HLE Verified 分层200、FrontierScience 160、SWE 与 Terminal 各32任务公开轨迹**。设为 `quick` 则只跑 MMLU200。两种模式均自动准备并执行本轮参考认证流程，不隐式启用 `--performance-only`。HLE/Frontier 包含单独的同模型辅助评分阶段；SWE/Terminal 测轨迹推理性能。`prepare` 默认写入 `<out>-prepare`，可用 `--out` 另选目录；目标目录已存在时会拒绝并提示改路径。字段说明见 `python3 one_click.py prepare --help`，原有直接传参方式继续可用。相对路径以配置文件所在目录为基准，GitHub 凭据仍从环境变量或 `.env` 读取。

有效算力参考属于一键流程，按 Topic4 V2 使用固定200题/max256/EOS，每臂三次正式原生测量；固定五项性能评测保持各自完整输出预算。无需使用者另写参考计划。一键流程保留已封存阶段，核验后按来源显式接续，失败原件不改写。

默认 `submission-v2` 先核验并复算既有完整M01参考原件，包括原始必要算量、独立关键路径参考时间和每卡参考算力上界，再核对本轮完整token清单、14分片权重、BF16/TP2/910B2、200题/max256/三轮及原生边界。每卡分子由本轮完整必要F与全部同步T重新计算，然后除以原参考分母；不同F不相消，不沿用旧效率。参考原件随本轮归档，来源不一致或适用域不符时返回非零并保留具体缺项。此前将dense末段子图提升为完整认证的身份已撤回；状态、报告及绘图均拒绝该旧证书。`full-answer-v1` 仍为独立8192预算诊断，不能借用out256分母。实现范围见[自动参考工作流](protocols/automatic-compute-reference-v1.json)及[同工况复用域](protocols/m01-full-reference-reuse-domain-v1.json)。

已有封存五项结果可通过同一个入口接续参考阶段，保留原性能结果，无需用户编写参考计划：

```bash
python3 one_click.py --compute-reference-only \
  --reference-campaign /absolute/path/to/sealed-results \
  --engine-profile r8 --model /absolute/path/to/model \
  --devices 0,1 --physical-devices 1,6 \
  --out /absolute/path/to/new-reference-results
```

已完成且完整封存的原生B0/B1和HTTP观测可显式接续CPU认证；入口会重新验证全部封存文件、模型/设备/工作负载、原生时钟与实际执行源码。协调器和下游CPU认证实现可以更新，计时及采集实现变化会拒绝复用。HTTP控制器中冗余局部`sys`导入的修复有单独的AST等价检查，原生计时所用的全部函数保持一致。规范标定也可显式提供同形状、同原程序与同原协议的完整封存结果：

```bash
python3 one_click.py --compute-reference-only \
  --reference-campaign /absolute/path/to/sealed-results \
  --reference-resume-source /absolute/path/to/completed-reference/compute-reference \
  --reference-canonical-source /absolute/path/to/completed-canonical-dense \
  --engine-profile r8 --model /absolute/path/to/model \
  --deployment /absolute/path/to/frozen-sdk \
  --devices 0,1 --physical-devices 1,6 \
  --out /absolute/path/to/new-reference-continuation
```

复用选项不会自动选择旧结果。两组原生计时均已完整封存、HTTP在模型和客户端启动前失败且已退出时，可以保留原失败记录，只重新执行缺少的HTTP阶段及后续认证；已启动或未退出的失败HTTP不会进入此接续路径。其他必需阶段缺失或哈希不一致均拒绝复用。编译缓存不进入科学证据清单；旧版外层清单中的`compiler-cache`项不解引用，其余记录和各阶段原始封存仍逐项验证。

## 查看多批次评测报告

在源码目录或解包后的工具目录使用 `dashboard`。`--results` 指向各批次 `out` 的上一级目录。例如评测配置的 `out` 为 `/data/vllm-hust-engine/runs/full-five-r1`，使用以下命令生成汇总网页：

```bash
python3 -B one_click.py dashboard \
  --results /data/vllm-hust-engine/runs \
  --out /data/vllm-hust-engine/dashboard/index.html
```

打开生成的 `index.html` 即可查看四项目标的证据、五个数据集的性能、评分和实验设定。批次选择框使用回执中的时间，统一显示北京时间。重评分批次在实验设定中另列原推理时间。

生成器读取结果根目录中各批次的 `*-report/summary.json` 或 `*/mmlu-pair-report/summary.json`，并核对同目录 `checksums.json`。存放在其他位置的报告可通过 `--extra /path/to/summary.json` 加入，支持多次传入。将网页输出到独立目录，保留封存报告的完整性。

命令末尾加 `--serve 8765` 可在本机浏览器打开 `http://127.0.0.1:8765/`，刷新时重新扫描结果。远程运行时可通过 SSH 端口转发访问。

## r8：异步复制、E36/R54 与短尾调度

r8 固定当前 C15 实测的10个运行时文件。继承 CorrectedWorker、BetterScale、Microflow、KV materialization、MTP2、FULL图、逐请求KV重算和 packed-host，每 rank 的主机 State 预算仍为4GiB，KV预算20GiB。新增原生 guarded async copy、E36/R54、已有短 prefill 尾段优先调度；restore/RMS 融合安排在后续版本。

C15 八阶段对照已完成。两组 MMLU200 吞吐相对 r6 提升37.94%/38.12%；反向4K平均/P95 TTFT增加11.38%/20.23%（门槛5%），反向MMLU平均TPOT增加2.28%（门槛2%）。原筛选未全通过，r8保持显式候选、默认仍r6；保留全部原件，不将局部吞吐收益称为全面稳定。[固定组合与验收状态](engines/b1-r8.json) · [MOD清单](mods/active-stack-r8.json) · [原始实验](experiments/optimization-20261115/C15-r7-tail-combination-model-20261117-r1)。

配置模式可复制 `examples/evaluation-r8.json`，其 `test_mode=full`；快速测试改为 `quick`。

显式选择 `--engine-profile r8` 即从原件恢复这套组合，再执行标准一键流程和实时日志。快速测试用 `--datasets mmlu`，全量用默认固定五项；题目、生成预算、评分、参考分母适用域和报告规则不变。新发布的 r8 单文件下载入口自动选择 r8；主入口默认 r6 的切换由完整 C15 原筛选决定。

```bash
python3 one_click.py --engine-profile r8 --datasets mmlu \
  --b1-python /absolute/path/to/fixed-b1-sdk/bin/python \
  --devices 0,1 --physical-devices 1,6 \
  --out /absolute/path/to/new-r8-results
```


[已启用 MOD 完整目录](mods/README.md) · [r6 冻结组合清单](mods/active-stack-r6.json)

<!-- r6-self-run-status-20261115:start -->
## 当前执行状态（2026-11-15）

r6 已在本机双卡 Ascend 910B2 完成模型启动与固定五项选定范围的执行：15阶段完成（10个已封存阶段核验复用、5个接续新跑），原MMLU200 B1模型身份与200/200完成回执已核验。
这更新了下文冻结时的“待实机”历史状态。**所选范围执行完成不等于项目四目标整体验收**：原活动仍记录`formal_complete=false`；SWE/Terminal是固定公开trace回放，不是工具环境解题率或官方全集质量成绩。
有效算力参考、16K/32K等专项仍按各自证据判定；这里也不宣称新环境的完整独立安装已通过第三方验收。

[核验范围与原件摘要](validation/r6-self-run-status-20261115.json) · [B0/B1报告页](https://vllm-hust.sage.org.ai/b0-b1.html) · [按日期的历史报告](https://vllm-hust.sage.org.ai/report-history.html) · [当前持续优化实验](experiments/optimization-20261115/README.md)

<!-- r6-self-run-status-20261115:end -->


## 启动前模型缓存与内存保护顺序

某轮HLE第二阶段裁判在启动模型前，因校验权重留下约60GiB文件缓存（其中约58GiB为active_file）而触发原4GiB工作集余量保护。修复仅将已有的已校验checkpoint只读fadvise提前到首次余量判定之前，要求实际回收后保留12GiB原始空间，再执行原内存/OOM检查；不把缓存估算当成可分配空间，不降低预算或保护门槛，不全局清缓存。21项CPU检查及实际无推理预检通过；当时已完成10阶段保留，剩余5阶段另建运行接续；该接续现已完成，见顶部状态说明。旧失败回执不改写，r6运行时、预热、正式预算与评分协议不变。

## 3.4.1参考标定的代表工况

按用户2026-11-14确认，3.4.1有效算力效率只选固定MMLU200作为代表工况，不要求对HLE、Frontier、SWE或Terminal分别标定。其余四项继续按冻结协议测吞吐、时延、质量或trace性能，不增加参考标定阶段。

有效算力缓存并复用既有同工况参考分母统计，包含必要算量区间、独立参考耗时、每卡参考算力上界、原始组件标定及适用域。效率按本轮同工况有效算力分子除以参考分母重新计算，不沿用旧59.7573%。out256参考不得直接套入out8192或HTTP计时；分母可用与本轮效率已测分别显示。50%与90%目标分别判定。

用户2026-11-14补充确认“只要一个数据集达成目标就行”：性能或成本目标由任一固定数据集满足该项全部条件即可，不要求五项均达标。吞吐、TTFT与TPOT须来自同一数据集的可比B0/B1配对；成本与利用率也不能跨数据集拼接。五数据集仍完整执行并披露所有结果与负收益；MMLU参考标定、16K/32K稳定性和多租户等专项按各自要求判定。该解释在部分阶段完成后确认，后续报告用`single-fixed-profile-20261114-v2`标识，旧报告保留原版本。

## r8 任务预热：沿用已冻结的容量36协议

r8 执行容量为36，任务预热在原126条基础上增加C32/C36两档短合成解码，共194条，两臂使用相同客户端请求计划；B0按原生容量调度，不改其执行配置。连续压测仍为原71条。预热不使用正式题目、不计正式性能，正式开始前封锁Torch/Dynamo与Triton编译。新任务预热会改变B0准备状态，旧126条任务B0不能直接配给r8；已有连续证据仅在来源、权重、设备、配方及协议核验一致时复用。协议见 `scripts/serving_warmup.py`。

## 任务预热补充 v3（r6-rc2）

r6 连续压测的592个正式请求及原共同预热v2已通过。MMLU B0的200个正式请求在新卷积tile变体出现时被编译守卫中止，失败原件保留。
仅非continuous任务的两臂在计时前额外使用不超过128 token的合成短提示，按并发1/2/4/8/16/24各生成32 token，再封锁编译。
短输入使24个请求能同时进入解码，覆盖旧长输入/短输出预热未触达的live-batch Triton tile档位；不使用正式题目，不修改原预算，不开放计时后编译。
任务预热协议为`observed-torch-triton-sealed-task-warmup-v3`，原连续压测协议仍为v2；新增请求和耗时单列。r6运行时overlay不变，C8来源兼容范围及源码哈希见`protocols/task-warmup-v3-c8-compatibility.json`。

## 新冻结候选 r6（M01 保护与编译预热兼容）

默认 `--engine-profile r6`。r5 在启动前的新 positions 预热中进入了真实请求的 M01 状态快照校验，因尚无活动快照而失败；B1 未提交正式请求。r6 只在
compile-only 预热中使用私有浅拷贝 runner， 沿用真实张量地址编译；不执行 kernel，不刷新真实请求状态，不修改 M01 保护。完整固定来源见
`engines/b1-r6-overlay.json`。r4/r5 原件及失败证据保留；这里是r6冻结时的历史状态；后续实机自测及剩余专项范围见顶部状态说明。

同工况且共用预热协议、B0 来源、输入配方、设备和权重均校验一致时，可显式复用本轮已封存的 `continuous-B0`，原回执继续保留其 r5 活动身份。仅 B1 启动修复不要求重跑已完成的
B0；旧 r4 预热协议的 B0 仍不能复用。

## 历史冻结候选 r5（预热与计时规则修订）

显式 `--engine-profile r5`：在 r4 原有 E24/R36、MTP2、FULL 图、Microflow、逐请求 KV 重算和每 rank 4 GiB
锁页状态预算上，显式加入启动前 serving metadata 预热与 Triton JIT 封口修复；具体源码前后哈希见 `engines/b1-r5-overlay.json`。历史 r4
原件不变，显式 `--engine-profile r4` 才使用旧部署。指定部署与所选版本不一致会在启动模型前拒绝。

两臂均采用版本化的
`observed-torch-triton-sealed-warmup-v2`：正式计时前通过正式测试同一chat/KV策略路径，使用原C8合成chat输入和服务器token哈希校验完成相同客户端预热（4K
并发 1/2/4/8/16/24，16K 并发16，每请求生成32 token），HLE 的图像接口和裁判 schema 也在计时前预热。每次服务使用新的编译缓存与请求
salt，预热内容、日志和耗时单列，预热响应不作为正式答案。随后两 rank 确认 Torch/Dynamo 与 Triton guard 已封闭；正式请求中遇到未预热编译就报错并判该轮无效，不能回退
eager，也不能混入 TTFT 后继续声称通过。此 guard 不覆盖 CANN 编译。

本轮须重新测 B0/B1，不能把旧 B0 与修复预热后的 B1 拼成改善率。固定五数据集、题号、顺序、完整回答预算和评分分母保持不变。r5 目前仍是待 NPU
验证的候选；旧报告保留历史身份，不提前替换结果。

[官网介绍 · Engine Releases](https://vllm-hust.sage.org.ai/engine.html)

面向 Ascend 的固定推理引擎组合。B1 r4 绑定 2026-10-30 双卡 Ascend 910B2 的 Qwen3.5-35B-A3B BF16
实测组合；历史实测和新机器部署验证分别记录。

## 报告范围订正

r4.1 的新运行只覆盖 B1 合成 C8 和 MMLU-Pro200；报告中将“该次单臂未测”写成项目级“缺少参考、缺少正式配对”的表述已撤回。原始请求、分数和时延保持不变。

已从 benchmark 固定提交 `53541118b3461705f1e61532710ff0a759d59af0` 取回并验证121个源文件，运行原复算脚本通过：

| 项目  | 已有证据与适用范围                                                                                                                                                               |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3.4.1 | 原固定200/out256、B0/B1各三轮的独立合格参考：B1效率下界59.7573%，达到50%门槛。不能套到新的out8192单臂。                                                                          |
| 3.4.2 | 已有原200题/max8192配对：吞吐提升90.4811%，平均TTFT/TPOT改善52.3374%/48.7757%；143/153正确，单次固定子集范围保留。                                                               |
| 3.4.3 | 五数据集尚未完整交付：SWE26/731次尝试、705未执行；Terminal4/89、85未尝试。HLE/Frontier已有记录，未决与裁判范围仍须保留。                                                         |
| 3.4.4 | 已有公开类似机型报价依据，同费率成本估算降低47.5014%，原双卡采样复算AI Core均值上升约29.39个百分点，在该估算范围内达标；保留费用适用范围，不称本机账单，也不新增必须发票的要求。 |

独立证据复算也可一键执行：`python3 scripts/project_evidence.py --out project-evidence.json`。脚本按固定SHA256下载私有Release中的121文件证据包，已有完整缓存时离线使用；不需要NPU。

详见[证据订正与固定来源](validation/project-evidence-reconciliation-20261112.json)、[M01正式配对](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/53541118b3461705f1e61532710ff0a759d59af0/reports/m01-qualified-pair-20261103)与[完整回答配对](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/53541118b3461705f1e61532710ff0a759d59af0/reports/full-answer-joint-20261030)。报告入口现已自动校验固定证据、执行原参考及配对复算，并从原双卡遥测重新计算同区间利用率。当前新跑数据单独展示；新第三方入口按本轮范围准备三项质量子集及两项公开轨迹性能；单臂补充入口另行保留，均不代表五数据集全套。

## 第三方一键入口（由本仓库维护）

主入口为本仓库根目录的`one_click.py`，版本恢复、任务冻结、推理、评分和出图实现均在Engine内；benchmark仅保留大纲、证据与下载后转交此入口的兼容引导器。`scripts/third_party.py`继续作为内部控制器，清理和非零失败退出语义保持一致。

当前入口覆盖 MMLU-Pro原固定200、HLE分层200（23图像）及Frontier160的原质量协议，并默认接入SWE-bench
Pro与Terminal-Bench2.1公开真实轨迹的推理性能抽样。两项各固定32任务、每任务最多2请求、max2048/并发8，不执行工具和任务评分；来源、版本、SHA及逐请求清单见[公开轨迹执行补充](protocols/PUBLIC-AGENT-TRACE-SCOPE.md)。报告分别标记质量子集和trace性能，不自动声称五项端到端质量通过。

控制器须Python3.11及以上；机器须有两张可独占的 Ascend910B2、同源 Qwen3.5-35B-A3B BF16 权重、CANN9.1/ATB，以及前文固定的 B1 Torch2.13
SDK。B0由脚本按原ARM64镜像digest恢复Python、Torch、Torch-NPU、Triton及CANN8.5.1，不复用宿主临时pip环境，不需要Docker服务，也不修改宿主驱动。

```bash
python3 one_click.py \
  --model /models/Qwen3.5-35B-A3B \
  --b1-python /absolute/path/to/fixed-b1-sdk/bin/python \
  --physical-devices 0,1 \
  --out /absolute/path/to/results/third-party-r1
```

启动后默认自动跟随控制器、当前阶段的模型服务、预热和客户端日志，无需另开终端执行`tail -f`。每30秒显示执行阶段完成数、当前B0/B1子阶段、已返回请求数和距最近日志的秒数；准备与模型加载期间明确显示对应状态，不将无日志自动判成失败。阶段切换立即提示。

合并日志位于输出目录旁的`third-party-r1-logs/console.log`，控制器原始输出位于同目录的`controller.log`；这些持续写入的观察日志不进入结果目录的封存清单。Ctrl+C会通知控制器，等待其子服务清理及租约释放后再退出，保留原始失败记录与日志；普通失败保留控制器非零退出码。

```bash
# 已启动的新版入口：只重新查看进度，不发起任何推理。
python3 scripts/live_console.py --run /absolute/path/to/results/third-party-r1 --follow-only

# 也可使用常规命令查看合并日志。
tail -f /absolute/path/to/results/third-party-r1-logs/console.log
```

单文件下载入口为[`download_and_run.py`](download_and_run.py)：用Python执行时，它先按固定Release附件ID、大小、源码提交和SHA256恢复完整源码包，再自动转交上述日志入口。GitHub私有附件凭据从`GH_TOKEN`、`GITHUB_TOKEN`或`.env`读取，不触发Git
Credential
Manager；传入的模型、卡号、SDK、输出目录等参数继续交给正式入口。这个文件可独立下载，不需要同时下载其他Python文件。当前下载包是带日志功能的源码候选，正在进行的实测仍保持其原冻结版本，日志候选不冒充新增实机验收。

```bash
python3 download_and_run.py \
  --model /models/Qwen3.5-35B-A3B \
  --b1-python /absolute/path/to/fixed-b1-sdk/bin/python \
  --physical-devices 0,1 \
  --out /absolute/path/to/results/third-party-r1
```

需要自行管理输出时，给`one_click.py`或下载入口加`--no-follow`，直接使用原控制器输出与进程语义。`--help`不创建结果或日志目录。单独复制仓库根目录`one_click.py`不能替代其依赖文件，应下载完整源码包或使用上述独立下载入口。

物理卡号由第三方填写，逻辑映射由`npu-smi`核验；本容器选5,6。已有B1部署可改传`--deployment`；已有B0固定SDK可传`--baseline-deployment`。原件、SDK及数据默认恢复到用户缓存，首次需要可访问固定源；可用`--cache`指定位置、`--dataset-assets`指定已有`hle/`与`frontier/`数据根目录，所有输入仍逐项校验固定SHA256。私有Release凭据读取规则与下文一致，`--env-file`同时用于原件和正式证据包。

正式执行在SDK核验和模型启动前检查`--out`所在文件系统的实际余量，至少保留原4GiB启动余量；不足时先封存失败并生成报告。不同目录可能位于不同卷，请按输出目录检查空间。

同一命令负责：恢复/校验两侧SDK，冻结任务、源码及评分配置，写出`execution-checklist.json`，按原60秒预热/300秒正式协议运行C8资格，执行原200题配对，再执行HLE/Frontier的两侧solver与共同B1替代judge，以及SWE/Terminal公开trace的两臂实际推理，独立复核原始流、评分或性能范围和分母，最后出HTML/PNG/SVG。每次使用新输出目录和新编译缓存；失败返回非零并保留所有记录及尚未执行的阶段。

中断执行时，父控制器等待本次子服务清理与租约释放后才封存记录；失败保留原始日志并停止后续步骤。SDK恢复中断的目录单独保留，只有完整准备成功的缓存才对后续执行可见。

仅准备时加`--prepare-only`：同样生成完整冻结计划与准备报告，`selected_execution_complete=false`。`--smoke-evaluators`明确只选HLE4道（两子集各一文本/一图像）和Frontier2道作诊断；如选择MMLU，仍运行原始200题，绝不静默缩短8192预算。诊断不能充当全量数据集成绩。第三方首次实测应省略`--continuous-evidence`，重新取得本机C8资格。

solver和judge保留原提示词及评分规则，生成采用显式流式传输以采集实际TTFT/TPOT与原始SSE。两侧solver均交给同一固定B1
Qwen替代judge；judge流量不计入solver性能或成本。HLE
solver在正式任务前执行相同的红/蓝合成图接口检查，单列为预热，不计任务分数或测量区间。原strict_json及mmlu_extract_first规则、长度终止和未决评分保留，不称官方排行榜复刻。

**裁判服务无需另备模型。**
内置参考Engine的HLE/Frontier评分复用同一份Qwen3.5-35B-A3B权重，在单独的固定B1服务阶段为B0/B1答案辅助评分；脚本自动启动和清理，无需第三方提供额外模型或外部裁判端点。该阶段增加整轮耗时，其流量单列，不计入solver吞吐、TTFT/TPOT和单位token成本。这是同模型辅助评分，不是独立模型验证。裁判未决表示缺少可解析的完整判决，仍保留计划分母并单列，不能等同引擎请求失败；性能、稳定性与答案质量分别报告。MMLU200使用固定答案和规则评分，不运行裁判模型。

B0文本服务使用归档原text-only补丁；HLE使用原0.18完整多模态/EP/FULL_DECODE_ONLY配置。B1
HLE保留r4代码、E24/R36、MTP2、FULL和4GiB/rank State，显式将图像上限从0改为1；这是新多模态配置，实际接口与全量推理资格须由本轮回执证明，不借历史HLE候选成绩代替。

实际通过原`self-run-kit/one_click.py`执行的新MMLU配对已完成并独立复核：两臂各200请求成功，B0/B1正确数150/149、长度终止25/18、0未解析；吞吐465.78→935.99
tokens/s（+100.95%），平均TTFT/TPOT改善51.73%/50.16%，同费率单位token成本估算降低50.24%，同区间AI
Core均值提高32.22个百分点。不推断统计质量等价，也不把同费率估算称为本机账单。

既有MMLU/Frontier和原全量HLE记录使用运行前封存的r4.3.2/e0f0824控制器及固定B1 r4运行时；r4.3.3仅新增提前磁盘检查，运行中的源代码保持原样。原B0
140正确记录及所有启动失败独立保留。两臂C8为247/247及618/618成功。本轮Frontier160两臂solver和共同judge已完成，原HLE1811在用户改定抽样后受控停止，新HLE固定分层200另行执行；SWE731/Terminal89原任务范围已冻结。CPU工作区已由平台侧准备并交接：本容器可通过`ssh hust-eval-cpu`连接，ARM64
Python3.12.13及Docker29.9.1已现场复核，先前网络隔离容器冒烟退出0；本次SSH反向转发后CPU访问当前NPU服务`/health`返回200，生成请求数0。此前“CPU工作区尚未创建、需要向NPU容器提供用户token”的状态已过时并撤回。此前r4.3.7的两项端到端任务调度、原任务镜像/资源核验和评分未接入，尚未开始执行；SSH/Docker可用不等于已完成两项测试。五数据集整体验收未完成。

详见[本轮实际配对与来源](validation/native-original200-e0f0824-20261112.json)、[下载配对报告及PNG/SVG](https://github.com/vLLM-HUST/vllm-hust-engine/releases/download/full-answer-b1-20261030-r4.3.3/native-original200-e0f0824-20261112-r2-report.zip)。完整已封存原件的2812个文件已在私有持久卷逐一读回SHA256校验；下载ZIP只含派生报告。

见[执行准备与验证状态](validation/third-party-execution-readiness-20261112.json)。

## HLE裁判grammar v2：结构空白与置信度生成域

默认 `--hle-judge-whitespace bounded-v2` 向固定B1裁判服务传入
`--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'`，并给生成Schema的`confidence`添加`minimum=0`、`maximum=100`。原严格评分器本就只接受0–100整数，因此这是生成约束对齐，不放宽评分规则。冻结原始契约不改写；题目、答案、提示、普通字符串空格、seed17、temperature0、max8192、并发16和strict_json保持原值，不裁剪空白、补JSON或缩短预算。

策略在模型启动前冻结到两份judge描述符；报告核对实际启动参数、每条实际生成请求和未修改的严格解析。改变策略只重新执行两臂judge，兼容的封存答案与性能阶段继续核验复用。新评分另存新版本，旧106/200未决等原始结果保持不变。显式`bounded-v1`可复现只禁结构空白的开发版本，`unrestricted`复现历史无约束策略，不自动回退、不混用两臂。

实际完整B0裁判200条暴露了v1的剩余缺口：190条合法，10条未决，其中7条在无上界confidence整数内重复0，3条在reasoning字符串里循环LaTeX，均达到8192
token。原始流和受控停止父回执保留；不当作修复完成。v2的实际xgrammar0.2.8
CPU探针验证正常字符串空格、0和100可通过，结构空白循环、负置信度、101和整数零循环均被拒绝；CPU探针不代替完整实机复评分。

[当前策略](protocols/hle-judge-grammar-v2.json) · [历史v1策略](protocols/hle-judge-whitespace-v1.json)

```bash
/path/to/fixed-b1-sdk/bin/python scripts/probe_judge_grammar.py \
  --contract /path/to/frozen-run/plans/hle/contract.json \
  --out /path/to/new-grammar-probe.json
```

HLE首批TTFT与裁判循环分属不同阶段。用户提供Jingyuan的独立实机JIT诊断后，源码核验确认r4没有包含BetterScale后来的263ee2d3/a7a43ed3预热与Triton封口修复；保留14个既有适配的4文件临时源码候选通过14项CPU检查，尚未通过NPU性能验收。Triton
guard不覆盖Torch/Dynamo或CANN；B0/B1的预热和正式请求编译须同口径验收。完整源码核验和补丁见[诊断Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/diagnostics-b0-b1-20261113-r1)。旧solver性能、质量与稳定性结论不据此改写。

## 恢复固定版本

机器须已有 Ascend 驱动、CANN 9.1、ATB、模型权重，以及 Torch 2.13 / torch-npu 2.13rc1 / Transformers 5.14.1 的隔离 Python
环境。源码及 MOD 从私有 Release 恢复。

```bash
python3 scripts/restore_b1_r4.py \
  --assets /opt/vllm-hust-engine/release-cache/b1-r4 \
  --out /opt/vllm-hust-engine/deployments/b1-r4 \
  --python /absolute/path/to/qualified-venv/bin/python
```

脚本自动读取 `GH_TOKEN` / `GITHUB_TOKEN`，或当前目录、父目录、用户主目录中的 `.env`；可用 `--env-file /absolute/path/to/.env`
指定。只解析两个凭据字段，不执行 `.env`，不输出凭据。完整附件缓存通过固定 SHA256 复核后可离线使用。

恢复时逐项检查 255 个冻结文件、原计划的 246 项引用及 8,975 个恢复组件。部署目录统一使用
`runtime/`、`mods/`、`inputs/`、`scorers/`、`config/` 和
`native-libs/`。历史路径只用于来源映射和校验，不作为新部署入口；输入与评分器保留原始字节。

`provenance/original-path-map.json` 记录来源；`installed-files.json` 记录实际部署哈希；`deployment-changes.json`
记录明确允许的部署适配。当前 Ascend SDK 元数据缺少历史 `.d20260909` 日期后缀，恢复器保留真实安装版本，仅在 177 项原宿主源码校验全部通过后适配 Microflow
的日期标签检查；不会伪造安装元数据。其他依赖版本变化或源码变化直接拒绝。

每次启动还会从两个 SHA256 固定的原始清单重新推导全部 9,231 个实际部署文件，复核原计划与 Engine 的 Python 搜索路径。只允许上面披露的日期标签适配和下面的 ABI
路径迁移；修改代码后同步修改本地 `installed-files.json` 也不能通过来源检查。此检查在初始化 NPU 前执行。 每次启动还会用所选 SDK 的 `python -I -S`
读取该隔离环境的精确包元数据，对照已通过来源校验的宿主契约；不导入 Torch、torch-npu、vLLM 或 Ascend 插件。依赖被升级、缺失或日期标签不符时，在取得 NPU
租约和加载权重前停止并写失败报告。恢复阶段检查不能代替本次启动检查。

冻结的 `pinned_metadata_signature.py` 还需要固定提交中的规范 ABI 源码。恢复器从本仓库的 `abi-sources/` 安装原始字节，核对
`ceabfc9d…a8e929f4`，仅将读取路径改为部署内相对路径。该文件用于解析函数签名；实际加载的 Ascend 运行时覆盖文件保持原样，不用规范源码覆盖它。

## 组件预检

```bash
python3 scripts/engine_entry.py \
  --deployment /opt/vllm-hust-engine/deployments/b1-r4 \
  --out /opt/vllm-hust-engine/results/component-check-r1 \
  --cache /opt/vllm-hust-engine/cache/component-check-r1 \
  --devices 2,3 \
  --phase preflight
```

组件注册会初始化 NPU 上下文，必须先取得原物理 NPU 5/6 及共享资源租约；繁忙时等待，不启动组件。物理 NPU 到容器逻辑设备的映射通过 `--devices` 显式指定，本次容器为
`2,3`。预检核对 Manager 实际启用的 KV materialization、Microflow、CorrectedWorker 和 packed-host
组合，不加载模型权重。通过组件预检不等于模型服务或推理验收通过。

## 真实推理复跑

部署恢复后，可用一个命令依次执行原始 C8 资格测试、原始 200 请求及各自的报告和图：

```bash
python3 scripts/reproduce_b1_r4.py \
  --deployment /opt/vllm-hust-engine/deployments/b1-r4 \
  --out /opt/vllm-hust-engine/results/campaign-b1-r1 \
  --devices 2,3
```

连续测试失败或报告生成失败时保留失败结果并返回非零，不提交联合请求，不自动重试；`campaign.json` 列出实际执行及未执行阶段。此入口覆盖 B1 r4
原协议补充测试，四目标表自动复算已有参考、历史正式配对和成本证据；未完成的五数据集任务仍逐项列出。

也可分阶段执行。先执行原始 C8 连续测试：4K 和 16K 各有 60 秒预热、300 秒正式提交窗口，8 条闭环请求流，每次输出 256
token。输入由原冻结配方生成，在服务启动前冻结；服务端须逐项复核 16 个输入的 token 哈希。

```bash
python3 scripts/run_b1_r4.py \
  --deployment /opt/vllm-hust-engine/deployments/b1-r4 \
  --out /opt/vllm-hust-engine/results/continuous-b1-r1 \
  --devices 2,3 --kind continuous
```

连续测试完成、进程全部退出且原始记录封存后，才可进入原始 200 请求测试：

```bash
python3 scripts/run_b1_r4.py \
  --deployment /opt/vllm-hust-engine/deployments/b1-r4 \
  --out /opt/vllm-hust-engine/results/joint-b1-r1 \
  --devices 2,3 --kind joint \
  --qualified-continuous /opt/vllm-hust-engine/results/continuous-b1-r1
```

入口重新核对部署文件、14 个模型权重分片、原始 SSE、全部请求、逐请求零复用、双 rank Microflow 和 packed-host
激活。历史报告中的“连续测试已通过”不能代替本次原始证据；不满足时在启动联合测试前停止。正式 200 请求保留原始同时到达、8192 输出上限、EOS、temperature=0、seed=0
和原评分器。

每次使用新结果目录与空编译缓存，保留失败，不自动重试或调整预算。输出包含真实启动命令、运行时身份、设备采样、源文件与权重哈希、原始流、退出清理记录和 `checksums.json`。vLLM RPC
与通用 `TMPDIR` 均使用独立短路径，避免两类 Unix socket 在深结果目录下超过长度限制；原 BetterScale
收件器会先实际完成绑定与消息收发检查。准备阶段失败同样生成失败记录；进程尚未清理干净时不封存为完成。

命令结束并释放 NPU 租约后，自动从封存的原始 SSE 与请求记录复算，生成相邻的
`<结果目录名>-report/report.html`、`four-targets.png`、`four-targets.svg`、实际吞吐/时延图
`observed-performance.png` / `.svg`、`summary.json`
和报告校验清单。四项正式目标分别显示是否达标与具体卡点；已有项目证据按各自冻结工况报告，不以本次单臂未测改判为项目缺失；真正缺少的记录明确列出，不绘成零值，也不将 HTTP 200
的流内错误计作成功。失败执行同样出报告，仍返回非零退出码；不能成功生成报告时也返回非零。

出图使用独立 CPU 环境和 `matplotlib==3.10.8`；默认在用户缓存中建立环境并安装 `scripts/requirements-report.txt`。离线环境可预装后传
`--report-python /absolute/path/to/report-venv/bin/python`，或设置
`VLLM_HUST_REPORT_PYTHON`。联合报告会重新核验其绑定的连续测试原始记录、不变的200项请求与服务端token哈希、全部流内内容与原评分器，以及双rank组件激活、packed-host源码和4GiB/rank原预算；报告另外记录复算工具的源码哈希。既有证据从私有Release自动恢复并校验，凭据解析与部署恢复相同；可用
`--project-evidence /absolute/path/to/verified-cache`
指定已校验缓存。参考、历史配对与本轮单臂分别标注来源，不互相代替。实际吞吐采用原提交至全部响应结束的墙钟窗口，不把重叠请求时长相加。已封存的旧结果也可单独复算：

```bash
/path/to/report-venv/bin/python scripts/report_execution.py \
  --run /absolute/path/to/sealed-run \
  --out /absolute/path/to/new-report
```

这是 B1 r4 的原协议复跑入口。项目既有M01合格参考、原B0/B1配对和费用估算须按上面的固定证据分别复算；报告已接入该复算；单臂执行成功状态仍不能代表五数据集与四项正式目标全部通过。

## 固定组合与验证边界

| 项目             | 固定身份                                                                         |
| ---------------- | -------------------------------------------------------------------------------- |
| vLLM-HUST        | `f18cf803c5f63625e2c71253ddaf8b0bad0bad1a`                                       |
| vLLM-Ascend-HUST | `74f0c0a272376412b51e1c1864803d5f3a0f1b5f`                                       |
| Worker           | `corrected_serving_runtime.CorrectedWorker`                                      |
| 组合             | E24/R36、MTP2、FULL 图、逐请求 KV 重算、Microflow、同对象内按 dtype 合并锁页分配 |
| 预算             | 主机 State 4 GiB/rank、设备 KV 20 GiB/rank                                       |

原始附件中的历史目录名不代表切换到其他项目评测。恢复组件来自后续 SDK 快照；其全部依赖与 2026-10-30 执行环境逐字节等价，尚未建立。修复后最近完成的独立 C8
复跑已通过原协议资格核验：4K 预热 21、正式 380，16K 预热 39、正式 177，共 617 个请求全部成功，每个输出 256 token；服务端逐请求 KV 复用为零，两 rank 的
Microflow 与目标 MTP 图重放均已复核。详见
[本轮连续测试记录](validation/independent-continuous-20261111.json)。这是补充连续测试资格，不代表五数据集、整体状态隔离证明或四项正式目标已通过。

独立的原始200题复跑现已完成：200请求全部成功、无未返回项，MMLU-Pro得分153/200（76.5%），23个输出达到原8192
token上限，均保留在分母中。原始流、全部任务/服务端token绑定、评分和两rank实际激活均已独立复核；主机State仍为4GiB/rank。原提交至全部响应结束窗口754.298秒，成功输出701,457
token，吞吐929.947 token/s，平均TTFT314.448秒、平均TPOT22.283毫秒；TTFT包含200请求一次突发的排队。见
[本轮联合记录](validation/independent-joint200-20261111.json)。这些是单臂描述性结果，不是B0/B1改善率、五数据集整体结论或正式四目标通过证明。

前一轮 16 成功、8 失败的原始记录继续保留在 [失败记录](validation/independent-inference-20261111.json)：当时主机 State 锁页分配返回
`207001`，cgroup 实际占用逼近 128 GiB 上限。后续通过本模型清洁页缓存提示和实际余量准入完成本轮原协议；State 仍为 4
GiB/rank，未降低并发、输出长度或正式窗口。小分配至 128 MiB 的直接 ACL 与 Torch 锁页分配此前均能成功，不能仅凭 64 KiB 进程锁页限制认定根因。

主分支提供修正后的部署工具。原 Release 的冻结源码、运行配置和恢复组件附件是来源证据；Release 自动生成的源码压缩包仍对应历史标签。实际部署请使用 Release
正文置顶的最新修正工具包，勿继续使用标记为已取代的工具附件。

```bash
python3 -m unittest discover -s tests
```

[固定组合清单](engines/b1-r4.json) ·
[固定 Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/full-answer-b1-20261030-r4)
·
[公开实测报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/0bc5f043d4707601ae15c47b77ac2649f21ad668/reports/full-answer-joint-20261030)

内存保护的后续定位见 [计数异常与修复证据](validation/independent-deployment-counter-20261111.json)：已捕获
`file_writeback=2^64−4096`，超过整个容器内存，导致原计算无法扣除任何缓存。控制器保留每次原值，先核对计数是否可能成立，最多重采样三次；连续异常仍停止。有效样本交给未修改的冻结保护器，不将溢出值改写为零，不放宽
4 GiB 余量或 OOM 停止条件。

主机分配的余量检查现在同时约束实际 cgroup 占用：在服务启动前及两 rank 权重加载完成后，使用只读描述符对已校验的本模型权重执行
`POSIX_FADV_DONTNEED`，仅提示释放其清洁页缓存；不写模型文件，不执行全局 `drop_caches`。要求为两 rank 合计 8 GiB 主机 State 加 4 GiB
余量留出至少 12 GiB 实际空间，之后持续保留 4 GiB 实际余量；原工作集和 OOM 检查继续执行。加载模型前还会在子进程中对两张卡各实际完成 32 MiB
锁页分配及往返传输，退出后再加载模型。比较臂必须采用相同的缓存与余量策略，并分别归档前后观察；不能将本环境适配称作与历史全部依赖逐字节等价。

HLE/FrontierScience正式TPOT严格采用V1.5大纲3.4.2：`(请求完成时刻−首token时刻)/(输出token数−1)`，逐请求等权平均，保留usage和HTTP收尾至请求完成的时间。首末文本块间隔仅作辅助观测，不替代正式TPOT。原始流重放核验首末缓存时间和时钟顺序。r4.3.4曾错误改用末文本块作为正式边界，该公式已撤回；运行中的r4.3.2原公式及MMLU配对报告未受影响。原报告保留，修订报告另存并绑定原封存SHA。

## MMLU-Pro 固定200题的执行范围

MMLU-Pro源test在固定revision下共有12,032题。本次复用14学科的既有固定200题及原请求顺序，B0/B1使用同一清单；不在本轮重新抽题。逐题ID、学科、源行和prompt哈希见[机器可读范围](protocols/mmlu-fixed200.json)，第三方大纲见[固定200题执行补充](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/main/reports/vllm-hust-midterm-20261019/milestone-v15/MMLU200-SCOPE.md)。不将生成seed0解释为抽样seed，不声称随机/分层抽样代表性。

一键准备阶段核验冻结输入、题号、顺序和prompt绑定，打印MMLU_SCOPE并写入mmlu-task-scope.json；总执行计划及独立配对计划保存范围与哈希。MMLU生成参数仍为max8192、temperature0、request
seed0、EOS开启、一次同时提交200请求、无额外题目预热及重试；原答案解析器和200分母保持一致。长度终止单列且按原解析评分，不删除失败或无法解析项。

报告明确标为固定200题子集；其正确率、吞吐、时延和成本结论不称为官方12,032题全集成绩、统计质量等价或五数据集整体验收。当前运行前冻结的r4.3.2代码、题目及原始回执保持原样；本次补充不新跑或改写既有结果。

## HLE 固定分层200题（本轮已确认）

默认HLE范围为`--hle-profile fixed200-v1`：Gold文本64/图像10、Revision文本113/图像13，共200题、23图像题。先按四层源题数比例用最大余数法分配名额，再在原冻结分层顺序选择`floor(i*N/quota)`，最后保留原全集相对顺序；不按成绩、耗时或先返回顺序选题。这是确定性分层等距子集，不声称随机抽样代表性。

全部题号及原任务SHA256见[冻结清单](protocols/hle-fixed200.json)。准备时同时保留完整源数据并验证原哈希，冻结200题任务、1611题范围外排除记录、sample-scope.json和最终contract；两臂solver和共同B1裁判使用同一计划。solver16384/judge8192、并发16、temperature0/seed17、thinking=false、原提示词及strict_json保持原样；错误、长度终止及未决裁判保留分母，四层分别报告。显式`--hle-profile full1811`恢复全集211图像题；诊断4题另行标记，不能代替固定200题。

[历史611题先例](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/main/reports/hle-verified-qwen35-20261015/README.md)是文本各200加全部211图像的短输出性能测试（max256），同份报告的质量评分仍为1811题。本轮200题是新版本的完整回答/质量/稳定性执行子集，不借用短输出预算，不冒充历史质量抽样、官方全集或五项整体通过。16K结论仍按实际连续覆盖及原可比窗口核验。

旧r4.3.2全量HLE在用户改定抽样范围后受控停止并保留原1811题计划及全部部分输出；不截取先返回的200题。已完成的MMLU固定200和Frontier160保持原记录，新的HLE200两臂重新执行。首次第三方从本入口执行时，MMLU200、Frontier160和HLE200一起冻结并按序执行。

## SWE / Terminal 公开真实轨迹的快速性能回放

默认一键入口已接入两项公开trace，准确任务及轮次在运行前冻结；单独运行用`--datasets swe,terminal`。每项32任务、最多64请求/臂，原消息与工具历史不截断、不补写；目标模板输入≤16384、输出上限2048、并发8、EOS保留，不按历史或本轮成绩筛题。SWE使用官方claude-sonnet-4-paper轨迹，Terminal使用固定revision的pi
provider请求原件。工具包不附带原始对话全文，按URL/SHA256恢复并复用用户缓存。

[来源、抽样及大纲执行补充](protocols/PUBLIC-AGENT-TRACE-SCOPE.md) ·
[机器清单](protocols/public-agent-trace-fixed32-v1.json)。报告新增两项真实吞吐/TTFT/TPOT及分位数配对和PNG/SVG，失败与未返回保留分母。回放不执行任务环境，不产生SWE/Terminal解决率，也不把有限样本分位数充当连续窗口稳定性。CPU工作区就绪状态不再是trace性能的阻塞；旧章节中的“尚未接入”仅描述此前r4.3.7及更早的执行范围。当前HLE200冻结运行保持原样，新trace配对需在其释放设备后单独执行；代码与准备验证不能冒充本机trace实测。

## 显式复用已完成项

同一机器已有封存结果时，可在原一键命令中重复传入 `--reuse-completed-campaign /absolute/path/to/sealed-campaign`。控制器在发出新请求前核对
原封存文件、执行源码、运行时/模型/物理卡身份、任务合同与预算，并从原始流复算。 仅复用明确完成且释放资源的阶段；存在多个完成候选时拒绝歧义，不自动挑最好结果。
原父批次可能因其他阶段失败或停止，原状态及未决评分完整保留。 相符的阶段标记 `reused_verified`，只补跑缺项；全部已完成时新增推理阶段为零。
同机原始路径必须保持可读，不能把跨批次归并写成一次新的连续测量。

## 第三方现场采集

[课题四第三方执行包](third-party-test-kit/)
提供现场采集、复算和出图入口。先阅读包内的[参考算力标定与复算说明](third-party-test-kit/%E5%8F%82%E8%80%83%E7%AE%97%E5%8A%9B%E6%A0%87%E5%AE%9A%E4%B8%8E%E5%A4%8D%E7%AE%97.md)，再冻结本次工况。

`capture-live-work` 采集本轮的调度、双卡执行及依赖证据。正式的 `run-live-reference`
还需配置完整必要算量和独立标定的认证模块。当前包只覆盖部分算量，缺失认证的工况无法判定。独立验收与历史数据的复算命令见各自说明。

## 有效算力参考随一键执行

`one_click.py` 默认生成本轮固定MMLU200/max256/EOS、每臂三轮的自动认证计划，五项性能服务退出后依次执行已交付的组件资格、当前 HTTP 工作量观察及两侧原生同步计时。无需手写现场计划；HTTP观察耗时不进入效率。旧 out256 参考统计只保留原适用域，既有59.7573%不作为当前结果。必要算量、参考图或下界、独立审计、两侧F/T对应和封存复算全部具备后，流程才可输出效率与50%/90%结论；缺项时自动报告具体原因、保留原件并返回非零。

需要新增完整回答工况的现场参考时，显式提供 `--compute-reference-plan /path/execution-plan.json --compute-reference-sha256 <冻结摘要>`。该计划仍须绑定当前模型、设备、token输入与生成预算。`--compute-reference-only --reference-campaign /path/to/sealed-campaign` 仅追加现场参考，不重跑原性能阶段。现场补测的状态与既有合格参考分别记录。

## 实验性容量调优 MOD

[Capacity tuning](mods/capacity-tuning/README.md) 提供 r6/MTP2 的 E24、E26、E32 显式参数提案、原生实验控制器适配接口及吞吐—TPOT/TTFT tradeoff 出图。全部重复及正确数同时披露；质量与容量稳定性尚未验收，默认不启用，不改变生产参数。当前源包和图见[优化实验 Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/performance-exploration-20261115-r1)。

## 实验性 Tradeoff 调优 MOD

[Tradeoff tuning](mods/tradeoff-tuning/README.md) 提供吞吐/TTFT/TPOT/正确数/截断数的统一实测图和显式预算筛选。v3保留24次物理模型试验（22次完整200题、2次启动失败），全部重复与负例保留，历史引用去重；容量、布局、原生多流等组合的条件收益按原生对照解释。默认关闭、无自动部署；下载[优化实验 Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/performance-exploration-20261115-r1) 中的 tradeoff-tuning-mod-v0.1.2.zip 可离线复现图，质量与稳定性仍待独立验收。

### 真实长上下文与多租户补测

`one_click.py --context-tenant-only` 使用冻结的公开 SWE/Terminal 原始请求，自动准备 4K–8K、16K–24K、32K–48K 三组输入，顺序运行 B0/B1 和四租户共享服务，封存 SSE 与设备采样并绘制配对曲线。每组沿用 60 秒预热及 300 秒正式窗口，固定输出 256；这是独立压力工况，不修改五数据集样本或质量预算。五个独立任务与八条并发通道分别记录。

```sh
python one_click.py --context-tenant-only \
  --deployment /path/to/r6-deployment --baseline-deployment /path/to/original018-sdk \
  --b1-python /path/to/fixed-b1-sdk/bin/python --model /path/to/model \
  --physical-devices 0,1 --devices 0,1 --cache /path/to/trace-cache --out /path/to/new-supplement
```

默认实时跟随日志。已有合格专项可用 `--reuse-b0 /path/to/sealed-B0` 或 `--reuse-b1 /path/to/sealed-B1` 复用，复核完整清单、原始流、模型、设备及执行源码后才接受。失败保留原记录且不自动重试。本专项不包含有效算力独立参考；标准五项与该专项分别记录，有效算力由上述默认自动流程单列生成并核验当前适用域。

### 参考算力上界的聚合公式

`C_ref_upper = sum(F_upper) / (k * sum(T_ref_lower) * 10^12)`，单位TFLOP/s/卡，k=2。全三轮分别求和后相除，不平均逐轮速率、不挑最快轮。

默认参考复算原完整独立组件资源图及关键路径，原参考 F 上界除以原参考时间下界得到每卡参考上界。效率下界以本轮每卡有效算力下界除以该独立分母，不消去不同 F；不使用目标端到端时间或目标效率构造参考。dense 末段图只作为独立诊断。

B1原参考三轮ΣF上界=5438638334450385 FLOPs、ΣT参考下界=140.60316791316519秒，得到19.34038334687175052309470403284461724716 TFLOP/s/卡。B0对应5488258327989369 FLOPs、102.3231709195348958960064秒，得到26.81825767648090573566921216250915821371 TFLOP/s/卡。

本轮分子下界为 `C_current_lower=sum(F_current_lower)/(k*sum(T_current)*10^12)`；效率下界为 `100*C_current_lower/C_ref_upper`。只有必要工作量相同且F可合法消去时，才可化简为 `100*sum(T_ref_lower)/sum(T_current)`。参考缓存保留公式、全部原统计量、适用域及原件摘要；本轮分子改变后重新计算效率。

原生参考采集器对每次采集、每个预热请求和正式请求分配独立 `cache_salt`，避免共享前缀命中改变必要工作域。测量前保存 `prefix-cache-isolation.json`，记录缓存隔离策略、盐清单和 token 清单摘要；不改变输入 token、生成预算或计时边界。该修复只影响新的原生参考采集，不改写已有五数据集或专项结果。

## 自动检查与发布

每个 PR 和 main 提交执行 [Evaluation contract](.github/workflows/evaluation-contract.yml)：固定五项/full、MMLU200/quick、默认参考分母核验，覆盖源码、配置、单文件入口及实际 Release 包。发布前重新下载验收，故障注入确认历史范围错误会失败。说明及分支保护能力限制见 [CI 门禁](ci/README.md)。


### 课题四 V2 有效算力采集

默认 `--reference-profile submission-v2`：MMLU 固定200题、每题最多256词元、正常EOS，B0/B1各三轮。完整回答的质量、性能与成本仍使用既定8192输出预算。两种预算分别写入执行计划，不能相互替代。

先按预定形状域完成独立标定并冻结统计，再开始正式计时；归档冻结时刻及统计哈希，复算拒绝先计时后标定的送测记录。每轮以自己的公开原生输出推导必要算量，并与同次双卡同步计时、源码及参考图关联。三轮先分别累加必要算量、参考时间下界和实测时间，再复算每卡算力区间及效率下界。每轮的重置、编译边界、原始输出、算量、参考图和哈希均保留；缺一轮返回非零。

`--reference-profile full-answer-v1` 只用于显式复现此前8192预算、单轮的开发协议。原8192容量证明是256预算的保守上界；新资格检查仍逐轮核验实际预算、冷缓存重置与公开输出，不能把旧计时改名为送测三轮。

已完成的五项性能批次可仅追加参考阶段：

```sh
python3 one_click.py --engine-profile r8 --compute-reference-only \
  --reference-campaign /path/to/sealed-five-profile-run \
  --deployment /path/to/frozen-r8-sdk --model /path/to/model \
  --physical-devices 1,6 --devices 0,1 --out /path/to/new-reference-run
```

卡号按本机实际映射填写。既有分母原件只在源码、形状、精度、分布及统计协议全部适用时复用；更换输出预算或轮数不能复用旧原生时长。完整实机验收与工具包正式发布分别记录，CPU回归不替代实机结果。

## Qwen3.5-9B 4.1 原件复算

对同一份 BF16 checkpoint 的 original018 B0 与 r11 RC4 B1，使用 `one_click.py reference-9b` 复用封存的 200 道 MMLU-Pro 和每题 256 token 的原件。提供已有时钟缓存、原件图、算量和逻辑依赖清单，以及 `--conservative-reference-evidence` 指向的历史标定面板。入口检查原始工况、原件和缓存哈希，仅派发独立时钟缓存中实际缺失的用例。`reference-9b --help` 列出路径参数。

正式保守结果写入 `compute-reference/conservative-reference.json`，包括六轮的原墙钟、下界、最长路径与原件清单。成功退出码为 0。4.1 校验或标定故障返回 2，并照常出具封存的 4.2–4.4 报告。参考时长和效率按下界签发，精确值及里程碑达标判定另有证据要求。完整口径见[4.1 复算方法](protocols/m01-confirmed-reference-method.md)。
