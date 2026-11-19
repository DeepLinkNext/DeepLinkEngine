# DeepEngine

DeepEngine 归档课题四的 **冻结 r8 推理引擎、五项评测清单和复核脚本**。公开材料可直接下载，可在普通电脑上校验和复算。

| 我想查看 | 从这里开始 |
| --- | --- |
| 下载 r8、查看版本和运行环境 | [冻结 r8 Release](https://github.com/DeepLinkNext/DeepEngine/releases/tag/r8-frozen-20261117) · [引擎安装与校验](engine/README.md) |
| 查看五项固定题单或回放清单 | [评测范围与数据入口](#五项评测) |
| 复算公开结果或查看实机测试命令 | [测试与复核指南](benchmark/README.md) |

## 快速复核

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

## 测试结果

| 指标 | 公开记录 |
| --- | --- |
| 4.1 有效算力 | [MMLU-Pro 固定 200 题、输出上限 256 的算量和参考时间](benchmark/reports/m01-qualified-pair-20261103/README.md) |
| 4.2 吞吐、时延与质量 | [固定五项 B0/B1 配对结果](benchmark/reports/r8-fixed-five-20261118/summary-public-v2.json) |
| 4.3 长上下文 | [4K–8K、16K–24K、32K–48K 的连续负载结果](benchmark/reports/r8-context-tenant-20261119/summary-public-v1.json) |
| 4.4 成本与利用率 | [第三方测试记录及复算](benchmark/indicator-4.4.md) · [MMLU-Pro 同窗记录](evidence/mmlu-r8-window.json) · [四租户结果](benchmark/reports/r8-context-tenant-20261119/summary-public-v1.json) |

4.1 的参考分母与其同步墙钟工况绑定。4.4 在测试记录中注明每卡月度综合费率及取值依据，两臂适用同一费率时可直接用每输出 token 的占卡时长计算成本降幅。
