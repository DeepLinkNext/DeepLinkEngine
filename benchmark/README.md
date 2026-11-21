# 测试与复核

[课题四 4.1–4.3 现场测试与历史预检](submission-review.md)对应 V5 的三项技术指标。正式判定使用第三方见证下现场运行的当次记录。

## 完整测试与第三方验证

新测试默认安装上游最新稳定引擎；在配置中填写 `engine_version` 可复现指定稳定版。整轮测试的 `engine-selection.json`、`evaluation.json` 和报告记录同一版本；恢复已开始的测试及复算历史报告不会重新解析最新版本。历史 R8 的参考证据只在原有适用域检查通过时复用原始分母，不把旧效率百分数或旧五项成绩记到 R9。

完整数据正文从[本仓库固定数据包](../data/README.md)自动恢复并校验，安装前完成准备。离线环境设置 `DEEPLINKENGINE_DATA_ARCHIVES` 指向已下载的五包目录。

按[首页说明](../README.md#测试与报告)填写配置后，在仓库根目录执行：

```sh
python run_all.py --config evaluation.json --rerun-b0
```

是否重跑 B0 默认选择“否”（`rerun_b0: false`）。上例用于首次测量，显式 `--rerun-b0` 才启动固定 B0。后续省略该开关，并用 `--reuse-completed-run` 指定旧结果；已保存的 B0 上下文和原生计时来源随恢复记录沿用，也可分别用 `--reuse-context-baseline-arm`、`--reuse-compute-baseline-arm` 指定。缺少匹配原件时在模型启动前报错，不自动退回重跑。`--no-rerun-b0` 可覆盖配置中的 `true`；重跑选项不与复用来源并用。

入口依次完成引擎安装与校验、固定五项 B0/B1 测试、长上下文和四租户测试，输出原始记录、校验值、JSON 结果及 HTML 报告。第三方见证运行，核对版本、配置、当次原始记录和报告后出具结论。4.1 随一键流程自动生成本轮原生双卡计时、必要算量与原完整参考分母复用认证，报告保留其方法身份。若现场方案要求重新独立标定完整参考，按[现场采集说明](indicator-retest.md)执行；自动复用报告不冒充现场新标定结果。历史预检不替代第三方正式判定。

模型启动后，入口在服务空闲、尚未预热时再次清理可回收缓存，并确认至少 24 GiB 原始内存余量后继续。临时页在预热前全部释放；模型配置、每卡 4 GiB 状态预算及原有内存保护保持不变。

## 恢复与结果复用

不填写版本即使用最新稳定版；配置中的 `engine_version` 可锁定历史稳定版。测试客户端保留已有的固定数据、参考算力、启动内存保护和恢复流程，按本轮发行包绑定模型适配器、运行几何和 MOD 清单；原有 R8 工具原件和历史结果保持封存。更换 B1 稳定版并复用原有结果时，在新输出目录的配置中指定 `engine_version`，使用 `--reuse-completed-run`。脚本仅允许复用通过原始数据审计的 B0 独立阶段，并进一步核验同模型、设备、基线源码、启动参数和预热协议；裁判服务属于 B1，不能作为 B0 跨版复用。一个已封存轮次同时包含 B0 和 B1 时，跨版入口会从相同工作负载中筛出可核验的 B0 原件，MMLU 两臂配对可单独核验其 B0 侧。原目录恢复保持原版本，旧轮缺独立版本锁时从已保存的安装、配置和执行计划查证。同版本恢复会重新核验并复用原 SDK 安装，保持其物理身份。若已完成记录分布在多个运行目录，可追加多个 `--reuse-completed-campaign`，并用 `--reuse-mmlu-baseline-arm` 指定固定 B0 的 200 题原始答卷。 已封存的 B0 上下文及四租户专项可用 `--reuse-context-baseline-arm` 指定；只有相同真实输入、预算、预热协议、模型、设备、固定镜像及已审阅控制源码组合才接受。入口会核验封存文件、同模型及权重、设备、题序、预算、预热和实际启动参数；固定 B0 的安装路径变化只在完整源码清单与固定镜像哈希一致时接受。原始运行日期、路径和结果始终保留，复用选项也随恢复记录保存。

续跑可以更换工作缓存目录。旧安装解释器仍按原轮回执记录的缓存位置核验，新轮使用当前配置的缓存。

固定 B0 已完成的原生算力标定可用 `--reuse-compute-baseline-arm` 指定原始 `native-baseline-timing` 目录。脚本核验封存、三次计时、模型、设备、采样和固定镜像后，只新测 B1；旧 B0 的时间、数据与当时的配对引擎记录原样保留。复用校验失败直接报错，不自动重跑 B0。

## 公开结果复核

| 任务 | 入口 | 输入及产出 |
| --- | --- | --- |
| 公开材料快速复核 | `python tools/review_results.py` | 仓库汇总和固定清单。输出五项吞吐、TTFT、TPOT、答题正确数和四组专项成本、利用率对照。 |
| 4.1 从已公开的标定记录复算 | `python benchmark/scripts/recompute_m01_qualified_pair.py --check` | [公开算量、计时和参考路径附件](reports/m01-qualified-pair-20261103/evidence)。逐项核验并复算每芯片 TFLOPS 和参考达成率。 |
| 4.4 历史成本与利用率测算 | [费率、现场采集及历史复算](indicator-4.4.md) | 按公开八卡整机月租估算 1,875 元/卡月，以两臂封存的占卡时长、输出量和同窗采样复算。 |
| 从原始运行封存重新生成 4.2 汇总 | [固定五项摘要生成器](reports/r8-fixed-five-20261118/reproduce_publication.py) | 需要原始 `summary.json` 及它引用的封存目录，输出新生成的公开 JSON。 |
| 从原始运行封存重新生成 4.3 和四租户汇总 | [上下文与四租户摘要生成器](reports/r8-context-tenant-20261119/reproduce_publication.py) | 需要原始专项 ZIP，输出 JSON 与配对曲线。 |

公开复算入口使用 Python 3.11 及以上，无需加载模型。原始运行封存包含请求和响应，需要相应资料访问权限。公开摘要保留来源校验值。仓库的 benchmark 副本固定在[公开上游提交](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/4d539805a08268253719ec9894d86716b7290c34)，逐文件 SHA-256 见 [provenance.json](provenance.json)。

取得原始封存后，可从仓库根目录执行：

```sh
python benchmark/reports/r8-fixed-five-20261118/reproduce_publication.py \
  --summary /path/to/original/summary.json --output /path/to/recomputed-five.json
python benchmark/reports/r8-context-tenant-20261119/reproduce_publication.py \
  --raw /path/to/original-context-pair.zip --out /path/to/new-context-report
```

## 昇腾实机配对测试

使用仓库根目录 `one_click.py`。该入口校验仓库内 [one-click-kit](one-click-kit/one_click.py) 的全部工具源码，自动获取本仓库公开 Release 的固定参考证据。工具和证据的字节数、SHA-256 及来源版本见 [one-click-lock.json](one-click-lock.json)。冻结安装包和旧 `tools-r4.tar.gz` 保留原件，新的测试由修正版工具执行。

B0 为固定的 `original018-cann8.5.1`，由工具核验上游镜像摘要并恢复，或指定已按该版本制备的 SDK。B1 使用[版本清单指定的引擎](../engine/README.md#版本与下载)。两臂使用同一套 Qwen3.5-35B-A3B 权重和同一对 910B2。物理卡号按实机分配填写。

1. 按[引擎安装说明](../engine/README.md)安装 B1。复制根目录[配置示例](../evaluation.example.json)，填写本机权重、B1 部署目录、空的输出目录和两张物理卡号。已有 B0 SDK 时可追加 `baseline_deployment`。已有题目数据时可追加 `dataset_assets`，其下为 `hle/`、`frontier/` 及 `public-agent-traces/`，下载和核验规则由工具执行。
2. 直接运行下面的 `run` 命令。工具自动下载并逐项核验参考证据，不需要访问原私有仓库，也不需要配置令牌。已有经校验的固定证据目录时仍可设置 `VLLM_HUST_PROJECT_EVIDENCE`；手工传入的目录同样必须符合固定清单。
3. 检查 `<out>-report/report.html`、`<out>/compute-reference/automatic-result.json`、`matched-reference-pair.json`、曲线和校验清单。可先用 `prepare` 检查环境并冻结计划；它不执行正式推理，写入 `<out>-prepare`。正式 `run` 使用配置中的 `<out>`。

```sh
python one_click.py verify
python one_click.py run --config /path/to/evaluation.json --rerun-b0
```

`test_mode: "full"` 运行固定五项，`"quick"` 仅运行 MMLU-Pro 200 题，两种模式都自动执行参考认证。默认 `reference_profile: "submission-v2"` 为固定 MMLU200/out256，每臂三轮；不能将该参考分母套到 out8192 完整回答。达成率使用本轮必要算量和全部三轮同步墙钟，除以重新核验的原完整参考每卡上界；保留两边各自的算量和参考时间，不直接复用历史达成率。工况或证据不匹配时返回非零，缺项单列。该认证明确标记为原完整参考分母的同工况复用，详见[参考认证说明](one-click-kit/third-party-test-kit/参考算力标定与复算.md)。

报告保留全部五项结果，评分使用同权重模型辅助裁判，未决保留在分母并单列。短输出参考样本的事后得分不能代替完整回答质量。4.3 和 4.4 的四租户结果来自另一次专项运行，[专项协议和原始汇总](reports/r8-context-tenant-20261119/summary-public-v1.json)保存输入档位、到达窗口、排空、队列及采样口径；这些专项不在默认五项流程里冒充已执行。专项摘要生成器可在有原封存 ZIP 时重建报告。


## 一键验收汇总

`run_all.py` 在固定五项、参考算力、4K/16K/32K 和四租户执行结束后，自动校验封存原件并生成 `full-report/report.html`。四项指标分别显示里程碑与项目最终目标的“达成目标 / 未达成目标”，五数据集共用一张配对图；公式、质量门槛、原始日志异常和详细报告放在展开项。执行完成与指标达标分别记录，性能目标由同一个固定数据集同时满足该项条件即可。

只重新出报告、避免重跑推理：

```sh
python run_all.py --config evaluation.json --report-only
```

输出为新的 `full-report-<标识>` 目录，保留原报告与封存数据。 CPU 入口只取报告所用的路径及费用清单字段。两臂原件与配对汇总均已封存时，绘图失败可进入完整报告独立重核并保留原失败回执。 4.4 正式成本使用按原臂封存绑定的实际费用凭据，里程碑固定为 MMLU200 完整回答。完整入口会生成带封存摘要的凭据模板。填写[实际费用清单](indicator-4.4.md#随封存回执提交实际费用)后，指定配置项 `actual_cost` 或存为运行目录内的 `actual-cost.json`，同一入口会自动读取。报告单列同费率估算。MMLU、HLE 总体和 Frontier 两赛道使用固定计划分母，正确率绝对下降上限为 5 个百分点；未决计非成功。上下文判定同时核查服务日志、OOM 事件和采样覆盖，写盘失败不能由吞吐曲线掩盖。

模型启动或预热失败、正式客户端尚未启动时，可用 `python run_all.py --config evaluation.json --resume-before-inference` 接续。入口核验封存原件、清理状态及零正式客户端记录，保留失败原件；已有正式客户端或请求记录时拒绝此模式。


## 自动内存准备与中断接续

完整测试入口自动回收已校验的安装归档及模型文件缓存。服务启动前的临时回收最多分配 40 GiB，按干净、未映射且未锁定的文件缓存量收紧上限，按原有 4 GiB 安全余量计算已有空闲内存和可回收缓存的合计容量；临时页全部释放后才允许启动模型。输入准备和设备等待结束后，紧邻服务启动再次回收缓存，检查至少 36 GiB 原始余量和 OOM 事件；保留 16 GiB 匿名内存贯穿模型加载，在服务健康、尚未发送首条预热请求时释放，给驱动锁页分配提供实际余量。如果加载期间该缓冲区挤占了原有 4 GiB 工作集安全余量，每次只释放自身 8 GiB 缓冲并立即重新执行原保护检查，尽量保留余下缓冲到服务健康；预热前要求至少 12 GiB 原始余量，覆盖两张卡各 4 GiB 状态预算及原有 4 GiB 安全余量。缓冲区不进入预热或正式计时，B0/B1 使用相同流程；不改变容器限制、每 rank 状态预算或推理参数。记录保存在各臂的 `checkpoint-cache-startup.json`、`startup-memory-reserve.json` 和 `memory-samples.jsonl`。清理失败时停止并保留原因，不发送正式请求。

容器内存计数异常时，脚本保留原始值并最多重采样约 2 秒。每次仍检查新增 OOM；持续异常则按原始占用执行相同的 4 GiB 保护，不把异常计数当作可用缓存。

某一阶段失败但此前已有完成记录时，保持原输出目录不动，在配置中填写新的 `out`，然后执行：

```sh
python run_all.py --config evaluation-next.json --reuse-completed-run /path/to/previous-run
```

入口重新核验封存、原始请求、模型、设备、缓存和运行配置。只有通过核验的完整阶段会复用；失败原件留在原目录，后续阶段在新目录执行。报告标明复用来源和原运行身份。零正式推理的首次启动失败仍可使用前述 `--resume-before-inference`。

已完成的 MMLU 配对若复用了早先的 B0，后续恢复会沿冻结配对清单解析 B0 原件路径并重新校验封存记录，整体复用这次配对；无需复制原件或重跑 B0。权重或封存记录不符时停止，不自动重测。

若容器中断导致顶层仍记录“运行中”，入口先核对原控制进程的 PID 和启动身份、子控制器退出记录及失败服务的清理回执。确认原控制器已停止、子控制器已回收且服务已释放后，可在新的输出目录复用通过原始数据审计的完整阶段，原失败记录保持不变。新运行原子写入顶层记录并记录进程身份。设备更换仍须遵守原配对协议；报错会列出原卡号和本轮卡号，不自动改写历史设备或重跑 B0。

上下文复用的报告沿 `context-tenant/execution.json` 和配对清单核验实际原件路径、封存摘要、角色及清理状态。复用 B0 保留原目录，完整报告同时记录它的真实来源，不要求在本轮目录复制一份 B0。

服务启动阶段，控制器会按原始 cgroup 已用量提前释放自己的可选预留内存，给驱动锁页分配留出空间；缓存折算后的工作集保护和 OOM 检查继续生效。预热前再次检查实际内存余量，临时预留不进入正式计时。

换卡续跑可在完整入口配置中显式设置 `"allow_device_change": true`，
同时指定当前的 `physical_devices` 和已封存的 `--reuse-completed-run`。
该选项仅用于固定 B0 复用，原卡与当前卡都需为 Ascend 910B2，逻辑卡映射须一致。
已完成阶段保留原设备与计时，新增阶段使用当前设备。报告标记跨卡比较及其实际设备。
