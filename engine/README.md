# 安装与推理服务

在仓库根目录执行。安装默认跟随上游最新稳定发行版。R10 的系统要求和安装核验见[发行说明](https://pypi.org/project/vllm-hust-engine/2026.11.21.10/)。引擎源码由发行包提供。

## 安装

```sh
python deeplink.py install --out /workspace/deeplink-install \
  --cache /workspace/deeplink-cache
```

自动探测 CPython 3.12、加载已安装的 CANN 9.1.0，在专用 venv 中准备依赖，再安装引擎并校验运行时文件。安装不需要权重或评测数据。目标目录须为新目录，安装收据为 `installation.json`。未指定 `--cache` 时，缓存放在安装目录的同级位置。

`engine-selection.json` 记录本次解析的正式版本、源码包及三份原生运行包的 SHA256；全流程沿用该记录。显式安装历史稳定版使用 `--engine-version VERSION`。服务使用指定安装目录的版本，不会在启动时更换引擎。R9 及以后的服务通过安装包提供的启动 API 选择模型及 worker，避免在下游另写 Qwen 或书生专用启动器。

缓存位于 NFS 时优先使用本机磁盘存放 SDK 和私有 Python，逐文件核对 SHA-256，保留持久原件，并在本机缓存丢失后恢复。可设置 `DEEPLINKENGINE_RUNTIME_CACHE` 指定至少 10 GiB 的本机目录；默认探测无可用本机盘时沿用 NFS。

可用 `--wheelhouse /path/to/packages` 指定已下载的包，`--python /path/to/python3.12`、`--cann-home /path/to/cann` 指定安装环境。已有满足版本要求的专用 venv 可通过 `--use-environment /path/to/venv/bin/python` 显式复用。Python 包可自动补齐。运行 R10 服务前准备昇腾驱动、CANN 9.1.0、ATB 和模型权重。服务读取 `/usr/local/Ascend/cann-9.1.0` 与 `/usr/local/Ascend/nnal/atb`，原生算子要求 `libstdc++` 提供 `GLIBCXX_3.4.29`。

## 启动服务

```sh
python deeplink.py serve --installation /workspace/deeplink-install \
  --model /models/Qwen3.5-35B-A3B --physical-devices 0,7 \
  --host 127.0.0.1 --port 8000 --served-model-name deeplink
```

物理卡号按实际分配填写，逻辑卡号由 `npu-smi` 探测；也可用 `--devices 0,1` 显式指定并核对映射。启动前校验源码和依赖并申请共享设备锁，所选卡有其他作业时退出；`--lease-timeout 300` 可等待。每次服务使用独立工作目录。R9 及以后可在重启时沿用同一个 `--cache` 目录来复用已落盘的编译产物。使用 Ctrl+C 关闭服务并释放设备锁。

R10 的 Ascend 后端遇到 AOT 打包报错 `'NoneType' object is not callable` 时，可在启动命令加 `--aot-cache-format graph`。此模式保留 vLLM AOT 图缓存，跳过后端独立产物的打包。首次切换使用新的 vLLM 编译缓存键，后续重启保持相同选项和缓存目录。需要在当前进程注册的 Triton 图仍会重新构建。

加 `--dry-run` 可生成启动计划，不加载模型、不占设备。当前后端的模型与并行配置由版本适配器管理。

## 调用

```sh
python examples/chat.py "你好" --base-url http://127.0.0.1:8000 --model deeplink
```

服务提供 `/health`、`/v1/models` 和 `/v1/chat/completions`。示例仅使用 Python 标准库，可用 `--max-tokens` 调整输出长度。

## 版本与下载

- [官方最新稳定源码包](https://pypi.org/project/vllm-hust-engine/)与[R10 ARM64 运行包](https://github.com/vLLM-HUST/vllm-hust-website/releases/tag/engine-wheels-r10-v1)
- [R9 发行说明](versions/2026.11.19.9.md)与[R8 历史说明](versions/2026.11.17.8.post2.md)
- [依赖与工具下载中心](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/r8-runtime-dependencies-20261119-v1)与[依赖与工具索引](dependencies.json)
- [基础依赖清单](r8-lock.json)与[测试工具版本清单](../benchmark/one-click-lock.json)
- [测试与复核](../benchmark/README.md)
