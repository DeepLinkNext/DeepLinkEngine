# DeepLinkEngine

DeepLinkEngine fork 自 [vllm-hust-engine](https://vllm-hust.sage.org.ai/engine.html)。

提供冻结 r8 引擎的一键安装、配对测试和报告生成。课题四正式验收在第三方见证下现场运行，按当次采集的数据判定。

## 一键安装与测试

准备双卡昇腾 910B2、模型权重及[运行环境](engine/README.md#安装)。复制[配置示例](benchmark/run-all.example.json)，填写模型路径、物理卡号、缓存和新的结果目录：

```sh
cp benchmark/run-all.example.json evaluation.json
python run_all.py --config evaluation.json
```

依次安装并校验引擎，执行 B0/B1 固定五项、长上下文和四租户测试，生成结果与报告。所需数据及项目证据的获取方式见[测试说明](benchmark/README.md#昇腾实机配对测试)。

结果目录包含 `results.json`、`fixed-five-report/report.html` 和 `context-tenant/paired-report/report.html`，同时保留原始记录与校验值。第三方见证现场运行并复核当次原始记录。4.1 的进程内计时、有效算量和独立参考标定按[现场采集说明](benchmark/indicator-retest.md)单独执行。

| 我想查看 | 从这里开始 |
| --- | --- |
| 下载 r8、查看版本和运行环境 | [冻结 r8 Release](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/r8-frozen-20261117) · [引擎安装与校验](engine/README.md) |
| 查看五项固定题单或回放清单 | [评测范围与数据入口](#五项评测) |
| 复算公开结果或查看实机测试命令 | [测试与复核指南](benchmark/README.md) |

## 仓库结构

| 目录 | 内容与入口 |
| --- | --- |
| [`engine/`](engine/README.md) | 冻结 r8 的安装说明、发行包锁定信息及带哈希的安装清单。引擎源码随 Release 发行包提供。 |
| [`datasets/`](datasets/README.md) | 五项固定题单、回放清单、评测协议和清单校验脚本。 |
| [`benchmark/`](benchmark/README.md) | 实机测试配置、公开结果、证据来源和复算脚本。 |
| [`evidence/`](evidence/mmlu-r8-window.json) | MMLU-Pro 同窗占卡、输出 token、利用率及成本复核记录。 |
| [`tools/`](tools/verify_snapshot.py) | 仓库快照校验及公开结果检查工具。 |

## 获取仓库

```sh
git clone https://github.com/DeepLinkNext/DeepLinkEngine.git
cd DeepLinkEngine
```

## 公开结果复核

使用 Python 3.11 及以上版本，在仓库根目录执行。以下命令只读取仓库中的公开材料，无需安装推理引擎。

```sh
python datasets/verify.py
python tools/verify_snapshot.py
python tools/review_results.py
python benchmark/scripts/recompute_m01_qualified_pair.py --check
```

第一条校验题单和回放协议，第二条校验发布文件及来源，第三条打印五项对照和四组专项结果并复核其计算，第四条用已公开的算量、计时和参考路径附件复算 4.1。

## 五项评测

| 评测 | 固定题单或回放清单 | 每臂请求 | 输出 |
| --- | --- | ---: | --- |
| MMLU-Pro | [200 题，按冻结题序](datasets/mmlu-fixed200.json) | 200 | 正确数、吞吐、TTFT、TPOT |
| HLE-Verified | [200 题，Gold/Revision × 文本/图像](datasets/hle-fixed200.json) | 200 | 正确数、吞吐、TTFT、TPOT |
| FrontierScience | [Olympiad 100 题和 Research 60 题](datasets/frontier-full160.json) | 160 | 正确数、吞吐、TTFT、TPOT |
| SWE-bench Pro | [32 个任务的离线回放清单](datasets/individual/swe-bench-pro-fixed32.json) | 64 | 吞吐、TTFT、TPOT |
| Terminal-Bench 2.1 | [32 个任务的离线回放清单](datasets/individual/terminal-bench-2.1-fixed32.json) | 48 | 吞吐、TTFT、TPOT |

两项回放使用清单中的模型请求，保留历史轮次和工具描述并关闭工具执行。公开清单记录原始来源、版本、顺序及校验值。原题、答案及回放正文按各来源的获取条件取得。完整口径见[冻结评测协议](datasets/standard.json)和[抽样及计分说明](datasets/README.md)。

## 现场测试与历史预检

[课题四 4.1–4.3 现场测试与历史预检](benchmark/submission-review.md)汇总正式工况、判据和历史数据。下表的公开记录用于预检对照。

| 指标 | 公开记录 |
| --- | --- |
| 4.1 有效算力 | [MMLU-Pro 固定 200 题、输出上限 256 的算量和参考时间](benchmark/reports/m01-qualified-pair-20261103/README.md) |
| 4.2 吞吐、时延与质量 | [固定五项 B0/B1 配对结果](benchmark/reports/r8-fixed-five-20261118/summary-public-v2.json) |
| 4.3 长上下文 | [4K–8K、16K–24K、32K–48K 的连续负载结果](benchmark/reports/r8-context-tenant-20261119/summary-public-v1.json) |
| 4.4 成本与利用率 | [现场采集、费率及历史测算](benchmark/indicator-4.4.md) · [MMLU-Pro 同窗记录](evidence/mmlu-r8-window.json) · [四租户结果](benchmark/reports/r8-context-tenant-20261119/summary-public-v1.json) |

4.1 的参考分母与其同步墙钟工况绑定。4.4 采用估算费率 **1,875 元/卡月**，按 30 天折算。历史预检中，MMLU-Pro 的每百万输出 token 估算成本为 **2.858 → 1.001 元**，降幅 **64.99%**。报价来源、四租户结果及复算公式见[成本测算说明](benchmark/indicator-4.4.md)。
