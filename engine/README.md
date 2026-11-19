# 冻结 r8 引擎

[下载冻结 r8 发行包](https://github.com/DeepLinkNext/DeepLinkEngine/releases/download/r8-frozen-20261117/vllm_hust_engine-2026.11.17.8.post2.tar.gz)。包与 [PyPI 2026.11.17.8.post2](https://pypi.org/project/vllm-hust-engine/2026.11.17.8.post2/) 的发行包一致。冻结引擎提交为 `c6059a6d3d1cf7ea16340d79cccf7bb1a13fa661`，随附评测工具提交为 `bd5448c9f266cc0301c3d4cef2a10b9ab98f7205`。发行包中的 `sdk-r8.tar.gz` 包含冻结 SDK、vLLM 和 vLLM-Ascend 源码、MOD 与部署配置，`tools-r4.tar.gz` 是该 r8 版本一并锁定的评测工具归档。

## 下载与校验

在仓库根目录运行，路径填入实际下载位置：

```sh
python tools/verify_snapshot.py --engine /path/to/vllm_hust_engine-2026.11.17.8.post2.tar.gz
```

脚本核验发行包 SHA-256、内部 SDK 和工具包身份，并将包内评测协议与仓库清单逐一对照。[Release](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/r8-frozen-20261117) 同时提供三个配套轮包及独立的评测工具包。发行包和轮包的来源、文件大小和 SHA-256 见 [r8-lock.json](r8-lock.json)。

```sh
python tools/verify_snapshot.py --tools /path/to/tools-r4.tar.gz
tar -xzf /path/to/tools-r4.tar.gz
python vllm-hust-engine/one_click.py prepare --help
```

解包后即可查看该冻结版本的 `one_click.py`、`scripts/`、`protocols/` 和 `examples/`。该归档保留历史身份；新的第三方测试请使用仓库根目录的 `python one_click.py run --config evaluation.json`，其[修正版工具及参考证据锁](../benchmark/one-click-lock.json)独立版本化，自动生成本轮参考认证。

## 安装

| 项目 | 固定环境 |
| --- | --- |
| 设备及模型 | 双昇腾 910B2，Qwen3.5-35B-A3B，BF16，TP2 |
| 运行平台 | Linux aarch64，CPython 3.12，CANN 9.1.0 |
| 预装依赖 | Torch 2.13.0+cpu、Torch-NPU 2.13.0rc1、Triton-Ascend 3.6.0、Transformers 5.14.1、tokenizers 0.22.2 |

一键入口会探测可用的 CPython 3.12 和 CANN 9.1.0，自动加载环境、在专用 venv 中补齐不匹配的 Python 依赖，再复核版本后继续。共享环境和驱动不作修改。先准备上述环境及独立的 Python 环境。驱动、CANN 和模型权重按各自渠道配置。以下命令锁定引擎与轮包的哈希，并保留预装厂商依赖：

```sh
python -m pip install --no-deps -r engine/r8-pip-requirements-deeplinkengine.txt
vllm-hust-engine verify
vllm-hust-engine install --out /path/to/new-r8-installation
```

安装清单从 DeepLinkEngine 的 Release 查找包，哈希与[上游安装清单](r8-pip-requirements-post2.txt)一致。`install` 要求全新的目标目录，安装收据写入 `installation.json`，同时校验 9,235 个运行时文件。实机配对使用 r8 冻结启动器加载模型专用优化，配置和命令见[配对测试](../benchmark/README.md#昇腾实机配对测试)。[上游环境说明](https://vllm-hust.sage.org.ai/engine.html#install-runtime)

## 版本与命名

仓库和发行入口使用 **DeepLinkEngine**。冻结发行包沿用 `vllm-hust-engine` 包名和 CLI，安装包及轮包的文件名、版本与 SHA-256 保持原有身份。锁文件中的 `deepengine_url` 字段，以及公开 JSON 中的 `deepengine-*` schema 标识，保留为已有数据格式的兼容字段；下载地址已指向本仓库的新名称。

## 自动环境准备

默认无需增加配置。依赖下载到配置中的缓存目录，修复记录随结果保存在 `environment.json`。仅准备环境可执行 `python run_all.py --config evaluation.json --prepare-environment`。已有离线包或非默认安装路径时，可选填写 `environment` 对象中的 `wheelhouse`、`python`、`cann_home`，相对路径以配置文件为准。

Triton-Ascend 3.6.0 使用[已核验的恢复包](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/r8-runtime-dependencies-20261119-v1)，下载时核对固定 SHA-256。包内代码和二进制保持原安装记录的字节；原始 wheel 缺失，恢复来源及新旧归档身份随 Release 公布。
