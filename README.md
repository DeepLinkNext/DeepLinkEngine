# DeepLinkEngine

面向昇腾的推理引擎，提供自动环境准备、安装、推理服务和调用示例。引擎源码随发行包提供。

## 安装与使用

当前支持 Linux aarch64、双昇腾 910B2、CANN 9.1.0 和 Qwen3.5-35B-A3B。准备模型权重及系统运行环境后执行；无法克隆时可[下载源码归档](https://api.github.com/repos/DeepLinkNext/DeepLinkEngine/tarball/main)：

```sh
git clone https://github.com/DeepLinkNext/DeepLinkEngine.git
cd DeepLinkEngine
python deeplink.py install --out /workspace/deeplink-install
python deeplink.py serve --installation /workspace/deeplink-install \
  --model /models/Qwen3.5-35B-A3B --physical-devices 0,7
```

安装会自动准备独立 Python 环境，按锁定索引下载并校验本仓库 Release 附件，再从本地文件安装。除版本号外，还校验 Ascend Triton 的 565 个已登记文件，依赖解析覆盖厂商文件时自动恢复。卡号填写实际分配的两张物理卡，逻辑卡号自动探测。服务默认监听 `127.0.0.1:8000`，提供 OpenAI 兼容接口：

```sh
python examples/chat.py "你好，请介绍一下自己"
```

[安装与服务配置](engine/README.md) · [当前版本](engine/versions/2026.11.17.8.post2.md) · [下载中心](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/r8-runtime-dependencies-20261119-v1) · [技术资料](https://vllm-hust.sage.org.ai/engine.html)

## 测试与报告

需要完整配对测试时，填写[示例配置](benchmark/run-all.example.json)后执行：

```sh
python run_all.py --config evaluation.json
```

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
