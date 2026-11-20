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

每次新安装或新测试开始时解析一次稳定版本，校验官方 PyPI 源码包及其绑定的三份原生运行包，再从本地文件安装。整轮流程保存 `engine-selection.json`，环境切换、子进程、失败恢复和报告复算沿用同一份版本及哈希；中途发布新版本不会改变正在运行的测试。下载或校验失败会明确退出，不自动降回 R8。需要复现历史安装时可指定 `--engine-version 2026.11.17.8.post2`。

安装自动准备独立 Python 环境，同时校验 Ascend Triton 的 565 个已登记文件，依赖解析覆盖厂商文件时自动恢复。卡号填写实际分配的两张物理卡，逻辑卡号自动探测。服务默认监听 `127.0.0.1:8000`，提供 OpenAI 兼容接口：

```sh
python examples/chat.py "你好，请介绍一下自己"
```

[安装与服务配置](engine/README.md) · [R9 版本说明](engine/versions/2026.11.19.9.md) · [官方稳定包](https://pypi.org/project/vllm-hust-engine/) · [R9 运行包](https://github.com/vLLM-HUST/vllm-hust-website/releases/tag/engine-wheels-r9-v1) · [技术资料](https://vllm-hust.sage.org.ai/engine.html)

## 测试与报告

需要完整配对测试时，填写[示例配置](benchmark/run-all.example.json)后执行：

```sh
python run_all.py --config evaluation.json
```

不填写版本即使用最新稳定版；配置中的 `engine_version` 可锁定历史稳定版。测试客户端保留已有的固定数据、参考算力、启动内存保护和恢复流程，按本轮发行包绑定模型适配器、运行几何和 MOD 清单；原有 R8 工具原件和历史结果保持封存。更换 B1 稳定版并复用原有结果时，在新输出目录的配置中指定 `engine_version`，使用 `--reuse-completed-run`。脚本仅允许复用通过原始数据审计的 B0 独立阶段，并进一步核验同模型、设备、基线源码、启动参数和预热协议；裁判服务属于 B1，不能作为 B0 跨版复用。原目录恢复保持原冻结版本。 同版本恢复会重新核验并复用原 SDK 安装，保持其物理身份。若已完成记录分布在多个运行目录，可追加多个 `--reuse-completed-campaign`，并用 `--reuse-mmlu-baseline-arm` 指定固定 B0 的 200 题原始答卷。入口会核验封存文件、同模型及权重、设备、题序、预算、预热和实际启动参数；固定 B0 的安装路径变化只在完整源码清单与固定镜像哈希一致时接受。原始运行日期、路径和结果始终保留，复用选项也随恢复记录保存。

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
