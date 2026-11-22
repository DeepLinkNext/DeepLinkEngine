# r10 原生 RMS 与连续写回候选实测

两份候选均已完成固定五项和 4K、16K 两档合成 C8，运行于双 Ascend 910B2，使用 Qwen3.5-35B-A3B、BF16、TP2。
两份报告复用正式固定 B0，即 vLLM 0.18.0 / vLLM Ascend 0.18.0（CANN 8.5.1）。
固定五项每臂 672 条推理请求，C8 分档独立计时并完整排空。SWE 和 Terminal 使用公开请求轨迹，统计推理服务性能。

| 候选 | 运行时身份 | 本仓库机器可读汇总 | Benchmark 原报告 |
| --- | --- | --- | --- |
| 原生 RMS | `full-answer-b1-20261120-r10-native-rms` | [汇总](native-rms.json) | [原生 RMS 完整报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/82f7cb195ac3076b485fad875ef3c6c72c5cf612/reports/r10-native-rms-fixed-five-20261121/README.md) |
| 连续写回 | `full-answer-b1-20261121-r10-contiguous-restore` | [汇总](contiguous-restore.json) | [连续写回完整报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/82f7cb195ac3076b485fad875ef3c6c72c5cf612/reports/r10-contiguous-restore-fixed-five-20261121/README.md) |

## 候选实测

吞吐为**双卡合计的输出 token/s**。正确数使用固定计划分母，未决保留在分母。

| 数据集或工况 | 原生 RMS（tok/s） | 连续写回（tok/s） | 原生 RMS 正确数 | 连续写回正确数 |
| --- | ---: | ---: | ---: | ---: |
| MMLU-Pro 200 | 1,490.33 | 1,489.39 | 144/200 | 134/200 |
| HLE-Verified 200 | 1,035.50 | 1,044.42 | 65/200 | 61/200 |
| FrontierScience 160 | 1,010.97 | 996.20 | 66/160 | 64/160 |
| SWE-bench Pro 公开轨迹（64 条） | 112.23 | 117.14 | — | — |
| Terminal-Bench 2.1 公开轨迹（48 条） | 244.11 | 245.20 | — | — |
| 合成闭环 4K / C8 | 389.41 | 410.85 | — | — |
| 合成闭环 16K / C8 | 160.69 | 177.01 | — | — |

连续写回相对原生 RMS 的 4K / C8 和 16K / C8 双卡输出吞吐分别提高 5.51% 和 10.16%。本轮 MMLU-Pro、HLE-Verified、FrontierScience 的答对数均低于原生 RMS。
HLE 和 FrontierScience 的裁判请求独立于求解器吞吐。各份汇总保留本轮 B0 判分、原始 r10 判分及对应候选判分，HLE 的 B0 两轮分别为 58/200 和 57/200。

两份 JSON 保留逐臂请求数、失败、吞吐、时延分位数、裁判记录和原始封存摘要。
[原始 r10](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/82f7cb195ac3076b485fad875ef3c6c72c5cf612/reports/r10-fixed-five-20261120/README.md)和[早期原生 RMS 四项复查](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/82f7cb195ac3076b485fad875ef3c6c72c5cf612/reports/r10-native-rms-20261121/README.md)可按原运行身份核对。
[来源清单](../../provenance.json)固定两份机器可读汇总的源提交、字节数和 SHA-256。

这两份是候选性能与质量记录。DeepLinkEngine 的稳定入口见[引擎安装说明](../../../engine/README.md)，完整送测判定见[公开 B0/B1 页面](https://vllm-hust.sage.org.ai/b0-b1.html?release=r9)。
