# DeepEngine

本仓库归档冻结 r8、课题四的五项固定评测清单和可公开的复核材料。各项引用固定版本，便于逐条比对。

## 冻结引擎

[下载冻结 r8 源码及评测工具](https://github.com/DeepLinkNext/DeepEngine/releases/tag/r8-frozen-20261117)。发行包包含冻结 SDK 和配套评测工具，内容与 [PyPI 2026.11.17.8.post2](https://pypi.org/project/vllm-hust-engine/2026.11.17.8.post2/) 逐字节一致。上游冻结提交为 `c6059a6d3d1cf7ea16340d79cccf7bb1a13fa661`，[冻结安装包、运行时轮包与校验值](engine/r8-lock.json)记录了具体版本及来源。

在已配置 CPython 3.12、Linux aarch64、CANN 9.1.0、Torch 2.13.0+cpu、Torch-NPU 2.13.0rc1、Triton-Ascend 3.6.0、Transformers 5.14.1 和 tokenizers 0.22.2 的环境中，从仓库根目录安装固定版本：

```sh
python -m pip install --no-deps -r engine/r8-pip-requirements-post2.txt
vllm-hust-engine --help
```

完整推理复测需要模型权重、两张昇腾 910B2 和冻结配置。通过引擎 CLI 的 `verify`、`install`、`evaluate` 子命令使用归档工具，模型专用优化由冻结 launcher 选择。[环境与启动说明](https://vllm-hust.sage.org.ai/engine.html#install-runtime)

## 五项数据清单

| 工作负载 | 固定清单 | 每个对照组的请求数 |
| --- | --- | ---: |
| MMLU-Pro | [固定 200 题](datasets/mmlu-fixed200.json) | 200 |
| HLE-Verified | [固定 200 题及四个分层](datasets/hle-fixed200.json) | 200 |
| FrontierScience | [Olympiad 100 题、Research 60 题](datasets/frontier-full160.json) | 160 |
| SWE-bench Pro | [32 个任务的离线回放清单](datasets/individual/swe-bench-pro-fixed32.json) | 64 |
| Terminal-Bench 2.1 | [32 个任务的离线回放清单](datasets/individual/terminal-bench-2.1-fixed32.json) | 48 |

[冻结评测协议](datasets/standard.json)和[样本选择说明](datasets/README.md)记录来源版本、任务或请求 ID、顺序、哈希和生成参数。MMLU-Pro、HLE-Verified、FrontierScience 同时统计回答质量和推理性能。SWE-bench Pro 和 Terminal-Bench 2.1 按清单回放公开执行记录中的模型请求，保留历史轮次及工具描述，设置 `tool_choice=none`，统计推理性能。原始题目、答案和回放正文按各自来源的获取条件取得。

## 复核材料

| 事项 | 数据与复算入口 |
| --- | --- |
| 指标 4.1 有效算力 | [同工况参考标定汇总](benchmark/reports/m01-qualified-pair-20261103/summary.json)、[计算脚本](benchmark/scripts/recompute_m01_qualified_pair.py)、[上游复算说明](benchmark/reports/m01-qualified-pair-20261103/README.md) |
| 指标 4.2 固定五项 | [逐项性能和质量、运行身份及封存摘要](benchmark/reports/r8-fixed-five-20261118/summary-public-v2.json)、[公开摘要生成脚本](benchmark/reports/r8-fixed-five-20261118/reproduce_publication.py) |
| 指标 4.3 和 4.4 上下文与四租户工况 | [结果及各窗统计](benchmark/reports/r8-context-tenant-20261119/summary-public-v1.json)、[公开摘要生成脚本](benchmark/reports/r8-context-tenant-20261119/reproduce_publication.py) |
| 指标 4.4 MMLU-Pro | [同窗占卡、输出 token 数、采样时长与利用率](evidence/mmlu-r8-window.json) |

4.1 汇总对应其所列的已封存标定工况。4.2 至 4.4 的 r8 结果分别使用各自汇总中的工况、计时边界和统计量。4.4 同费率结果是根据占卡时长及生成 token 数计算的成本估算，实际费率和费用凭据待核对。公开摘要生成脚本的输入是对应轮次的原始运行封存，资料获取条件见各上游说明。公开数据来源及哈希见 [benchmark/provenance.json](benchmark/provenance.json)。

离线检查本仓库版本、五项清单和占卡计算：

```sh
python datasets/verify.py
python tools/verify_snapshot.py
python tools/verify_snapshot.py --engine /path/to/vllm_hust_engine-2026.11.17.8.post2.tar.gz
```
