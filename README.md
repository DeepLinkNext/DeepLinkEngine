# DeepLinkEngine

面向昇腾的推理引擎，提供自动环境准备、安装、推理服务和调用示例。引擎源码随发行包提供。

## 安装与使用

默认安装和测试 [vLLM-HUST-Engine](https://pypi.org/project/vllm-hust-engine/) 最新稳定发行版，跳过预览版、开发版和已撤回文件。R9 支持 Linux aarch64、双昇腾 910B2、CANN 9.1.0，推理服务支持 Qwen3.5-35B-A3B 和书生 Intern-S2。准备模型权重及系统环境后执行；无法克隆时可[下载源码归档](https://api.github.com/repos/DeepLinkNext/DeepLinkEngine/tarball/main)：

```sh
git clone https://github.com/DeepLinkNext/DeepLinkEngine.git
cd DeepLinkEngine
python deeplink.py install --out /workspace/deeplink-install
python deeplink.py serve --installation /workspace/deeplink-install \
  --model /models/Qwen3.5-35B-A3B --physical-devices 0,7
```

新安装默认选择最新稳定版，并保存本次版本与校验记录；服务和失败恢复沿用该版本。复现历史版本可加 `--engine-version VERSION`。

安装会自动准备独立 Python 环境、补齐依赖并核验文件。卡号填写实际分配的两张物理卡，逻辑卡号自动探测。服务默认监听 `127.0.0.1:8000`，提供 OpenAI 兼容接口：

```sh
python examples/chat.py "你好，请介绍一下自己"
```

[安装与服务配置](engine/README.md) · [R9 版本说明](engine/versions/2026.11.19.9.md) · [官方稳定包](https://pypi.org/project/vllm-hust-engine/) · [R9 运行包](https://github.com/vLLM-HUST/vllm-hust-website/releases/tag/engine-wheels-r9-v1) · [技术资料](https://vllm-hust.sage.org.ai/engine.html)

## 测试与报告

将[示例配置](benchmark/run-all.example.json)另存为仓库根目录的 `evaluation.json`，填好后运行下面的命令。

```sh
python run_all.py
```

首次全量实测在配置中设 `"rerun_b0": true`。已有匹配的 B0 结果时，按[恢复与复用说明](benchmark/README.md#恢复与结果复用)接续。运行 `python run_all.py --help-all` 可查看复用和恢复参数。

不填写 `engine_version` 时使用最新稳定版；指定版本可复现历史运行。恢复、复算及跨版本结果复用见[测试指南](benchmark/README.md#恢复与结果复用)。

R9 的书生单模型验证见[版本说明](engine/versions/2026.11.19.9.md)，不作为完整五项 B0/B1 或项目最终验收结果。

入口自动完成环境准备、缓存回收、双臂预热、五项配对、有效算力参考、上下文与四租户测试，并生成 `full-report/report.html` 及 PNG/SVG 对比图。终端从读取配置起显示全流程阶段和原始日志。后台运行时，可用 `python run_all.py --status --json` 查询一次，或用 `python run_all.py --status --watch` 持续观察。启动时指定了 `--config` 的，查询时使用同一配置文件。

[固定数据包](data/README.md)和[B0 镜像原件](data/B0-README.md)均由测试入口从本仓库 Release 自动恢复并校验。[测试与复核指南](benchmark/README.md)包含固定五项、长上下文、四租户、报告生成和第三方验证；[现场测试说明](benchmark/docs/topic4-test-guide.pdf)提供见证流程。

## 目录

| 目录 | 内容 |
| --- | --- |
| [engine/](engine/README.md) | 安装、服务适配器、版本和依赖索引 |
| [examples/](examples/chat.py) | 推理 API 调用示例 |
| [data/](data/README.md) · [datasets/](datasets/README.md) | 固定数据包、评测题单与协议 |
| [benchmark/](benchmark/README.md) | 测试工具、公开结果与复算 |
| [official/](official/) | 官方基线评测、SDK 准备、F/T 审计和报告实现 |
| [tools/](tools/verify_snapshot.py) | 发布材料和校验工具 |
