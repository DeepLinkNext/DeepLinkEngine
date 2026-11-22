# 固定五项数据正文

数据包与源码同属本仓库，统一入口为 [固定数据 Release](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/fixed-five-data-20261120-v1)。[bundles.json](bundles.json) 锁定每包大小、SHA-256、原始文件及逐文件校验值。

| 数据包 | 实际内容 | 执行范围 |
| --- | --- | --- |
| MMLU | 完整输入、答案记录、请求计划及固定分词器元数据 | 原固定200题 |
| HLE | Gold、Revision原始JSONL，图像以原始data URL内嵌 | 固定四层200题：64/10/113/13 |
| Frontier | Olympiad、Research原始题目、答案和评分细则 | 100＋60题 |
| SWE | 32项任务的32份原始公开轨迹，包含历史对话及工具描述 | 原固定64请求 |
| Terminal | 32项任务的32份原始请求轨迹 | 原固定48请求 |

HLE保留原始1811题分片，以便沿用原有来源哈希和分层抽样校验；实际执行仍为原先确定的200题。两项轨迹的原文件同时供长上下文及四租户输入构建使用。题目、答案、图像、工具历史及评分程序均未改写。

## 一键使用

`python run_all.py --config evaluation.json` 在安装引擎前准备数据。独立的 `benchmark/one_click.py` 的 `prepare` 与 `run` 同样自动准备所选范围；quick仅需MMLU包。下载只访问本仓库Release的固定资产API及下载直链，成功后逐包、逐文件核验并在配置的cache目录恢复。正常测试不再访问Hugging Face、其镜像或原始轨迹站点。

仅准备数据：

```sh
python tools/prepare_datasets.py --cache /data/deeplinkengine/cache --context
```

离线机器先从上面的Release下载五个tar.gz，放在同一目录，然后执行：

```sh
export DEEPLINKENGINE_DATA_ARCHIVES=/data/fixed-five-archives
python run_all.py --config evaluation.json
```

已有`dataset_assets`目录继续可用；入口补齐缺少的固定文件，对已有但哈希不同的文件报错并保留原件。缓存中损坏的数据同样不会被视作有效输入。MMLU输入还与安装SDK中的原始输入逐项核对。

## 来源、署名与使用条件

本仓库保留各来源的版本、版权和原始使用条件；引擎的软件许可不覆盖或改变数据许可。

- MMLU-Pro：[TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)，来源revision及原始输入哈希见[固定200题协议](../datasets/mmlu-fixed200.json)。
- HLE-Verified：[SKYLENAGE-AI/HLE-Verified](https://github.com/SKYLENAGE-AI/HLE-Verified/tree/b705e0fb541c025a1532ce0d60d70ae2f53b00e0)。保留原始问题、署名字段、答案和图像；不为原材料附加新的许可授权。
- FrontierScience：[openai/frontierscience](https://huggingface.co/datasets/openai/frontierscience/tree/25ed67db7da8f4591484e764008ff585544f5a30)，原发布标记为Apache-2.0。
- SWE轨迹：[ScaleAI官方claude-sonnet-4-paper轨迹](https://github.com/scaleapi/SWE-bench_Pro-os/blob/66f92766bba642462d4bbe5479e83f91f9211862/traj/README.md)；[MIT许可](https://github.com/scaleapi/SWE-bench_Pro-os/blob/66f92766bba642462d4bbe5479e83f91f9211862/LICENSE)。
- Terminal轨迹：[shadowpa0327/pi_agents_terminal_bench](https://huggingface.co/datasets/shadowpa0327/pi_agents_terminal_bench/tree/ccc3fad7ed2cf81f98094f8fdd001bdb480e096e)。冻结来源协议记录原发布者的`research-only`标记，随包保留该使用限制。

原始数据定位、逐请求正文哈希、任务选择及排除记录见[固定评测标准](../datasets/README.md)。模型权重、CANN和引擎安装包分别管理。

## 验证与重建

```sh
python tools/verify_data_bundle.py --cache /data/deeplinkengine/cache
```

验证器先恢复全部原件，再禁止网络访问，使用真实评测客户端重建HLE200、Frontier160、SWE64和Terminal48请求计划，并重新读取、校验每项计划；同时核验MMLU原始输入。CI执行相同检查。

归档构建器为[build_dataset_archives.py](../tools/build_dataset_archives.py)，仅接受已有、哈希吻合的SDK输入、原始数据分片和原始轨迹；采用固定tar元数据和gzip时间戳，不下载、不抽样、不改写正文。
