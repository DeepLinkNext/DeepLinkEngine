# 引擎安装与版本

DeepLinkEngine 按发行版本管理引擎、依赖、评测工具和校验信息。

## 一键安装与测试

按[当前发行版的安装说明](versions/2026.11.17.8.post2.md)准备设备、模型权重及系统运行环境，在仓库根目录填写配置后执行：

```sh
cp benchmark/run-all.example.json evaluation.json
python run_all.py --config evaluation.json
```

脚本自动探测并准备 Python 环境，安装和校验引擎，执行配对测试并生成报告。完整流程见[项目首页](../README.md#一键安装与测试)。

## 版本与校验

- [当前发行版说明](versions/2026.11.17.8.post2.md)：环境要求、安装命令、源码来源和适用范围。
- [统一下载中心](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/r8-runtime-dependencies-20261119-v1)与[完整索引](dependencies.json)：安装包、依赖、工具及 SHA-256。
- [运行时版本清单](r8-lock.json)与[评测工具版本清单](../benchmark/one-click-lock.json)：自动安装和评测所用的具体版本。

[引擎技术资料](https://vllm-hust.sage.org.ai/engine.html)
