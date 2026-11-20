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

需要完整配对测试时，填写[示例配置](benchmark/run-all.example.json)后执行：

```sh
python run_all.py --config evaluation.json --rerun-b0
```

是否重跑 B0 默认选择“否”（`rerun_b0: false`）。上例用于首次测量，显式 `--rerun-b0` 才启动固定 B0。后续省略该开关，并用 `--reuse-completed-run` 指定旧结果；已保存的 B0 上下文和原生计时来源随恢复记录沿用，也可分别用 `--reuse-context-baseline-arm`、`--reuse-compute-baseline-arm` 指定。缺少匹配原件时在模型启动前报错，不自动退回重跑。`--no-rerun-b0` 可覆盖配置中的 `true`；重跑选项不与复用来源并用。

不填写 `engine_version` 时使用最新稳定版；指定版本可复现历史运行。恢复、复算及跨版本结果复用见[测试指南](benchmark/README.md#恢复与结果复用)。

R9 的书生单模型验证见[版本说明](engine/versions/2026.11.19.9.md)，不作为完整五项 B0/B1 或项目最终验收结果。

入口自动完成环境准备、缓存回收、双臂预热、五项配对、有效算力参考、上下文与四租户测试，并生成 `full-report/report.html` 及 PNG/SVG 对比图。终端持续显示原始日志和阶段进度。

[固定数据包](data/README.md)和[B0 镜像原件](data/B0-README.md)均由测试入口从本仓库 Release 自动恢复并校验。[测试与复核指南](benchmark/README.md)包含固定五项、长上下文、四租户、报告生成和第三方验证；[现场测试说明](benchmark/docs/topic4-test-guide.pdf)提供见证流程。

## 目录

| 目录 | 内容 |
| --- | --- |
| [engine/](engine/README.md) | 安装、服务适配器、版本和依赖索引 |
| [examples/](examples/chat.py) | 推理 API 调用示例 |
| [data/](data/README.md) · [datasets/](datasets/README.md) | 固定数据包、评测题单与协议 |
| [benchmark/](benchmark/README.md) | 测试工具、公开结果与复算 |
| [tools/](tools/verify_snapshot.py) | 发布材料和校验工具 |
