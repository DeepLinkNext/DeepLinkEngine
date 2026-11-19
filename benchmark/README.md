# 测试与复核

[课题四 4.1–4.3 现场测试与历史预检](submission-review.md)对应 V5 的三项技术指标。正式判定使用第三方见证下现场运行的当次记录。

## 完整测试与第三方验证

按[首页说明](../README.md#一键安装与测试)填写配置后，在仓库根目录执行：

```sh
python run_all.py --config evaluation.json
```

入口依次完成引擎安装与校验、固定五项 B0/B1 测试、长上下文和四租户测试，输出原始记录、校验值、JSON 结果及 HTML 报告。第三方见证运行，核对版本、配置、当次原始记录和报告后出具结论。4.1 按[现场采集说明](indicator-retest.md)单独执行进程内双卡计时、全模型有效算量与独立参考标定；仓库内的历史配对复算不用于正式判定。

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

五项实机评测脚本随 [r8 评测工具归档](https://github.com/DeepLinkNext/DeepLinkEngine/releases/download/r8-frozen-20261117/tools-r4.tar.gz) 提供。入口是 `one_click.py`，数据核验由 `scripts/dataset_contract.py` 和 `scripts/agent_trace.py` 执行，配对流程由 `scripts/third_party.py` 调度，报告由 `scripts/report_execution.py` 生成。

B0 为固定的 `original018-cann8.5.1`，由工具核验上游镜像摘要并恢复，或指定已按该版本制备的 SDK。B1 为 [冻结 r8](../engine/README.md)。两臂使用同一套 Qwen3.5-35B-A3B 权重和同一对 910B2。物理卡号按实机分配填写。

1. 按[引擎安装说明](../engine/README.md)安装 B1。复制[配置示例](evaluation-r8.example.json)，填写本机权重、B1 部署目录、空的输出目录和两张物理卡号。已有 B0 SDK 时可追加 `baseline_deployment`。已有题目数据时可追加 `dataset_assets`，其下为 `hle/`、`frontier/` 及 `public-agent-traces/`，下载和核验规则由归档工具执行。
2. 正式工具还会核验一份固定的项目证据包。把经授权取得并解包的 `benchmark-evidence` 目录设为环境变量 `VLLM_HUST_PROJECT_EVIDENCE`。已有源发布访问权时，归档工具也可从源 Release 恢复。此输入与本仓库公开的 4.1 附件、4.2 汇总分别有各自的版本校验。
3. 用相同配置先准备运行计划，再执行两臂。检查输出目录的执行回执及 HTML 报告。`prepare` 做环境和文件核验并冻结计划，完整配置的 `run` 才会启动推理。

```sh
vllm-hust-engine evaluate --installation /path/to/r8-installation -- prepare --config /path/to/evaluation.json
vllm-hust-engine evaluate --installation /path/to/r8-installation -- run --config /path/to/evaluation.json
```

`test_mode: "full"` 运行固定五项，`"quick"` 仅运行 MMLU-Pro 200 题。4.1 单独使用 MMLU-Pro 输出上限 256 的同步墙钟和参考路径证据。历史 4.3 和 4.4 的四租户结果来自预检专项，[专项协议和原始汇总](reports/r8-context-tenant-20261119/summary-public-v1.json)保存输入档位、到达窗口、排空、队列及采样口径。专项摘要生成器可在有原封存 ZIP 时重建报告。
