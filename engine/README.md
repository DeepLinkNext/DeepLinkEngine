# 安装与推理服务

在仓库根目录执行。当前设备、模型与系统要求见[版本说明](versions/2026.11.17.8.post2.md)。引擎源码由发行包提供。

## 安装

```sh
python deeplink.py install --out /workspace/deeplink-install \
  --cache /workspace/deeplink-cache
```

自动探测 CPython 3.12、加载已安装的 CANN 9.1.0，在专用 venv 中准备依赖，再安装引擎并校验运行时文件。安装不需要权重或评测数据。目标目录须为新目录，安装收据为 `installation.json`。未指定 `--cache` 时，缓存放在安装目录的同级位置。

缓存位于 NFS 时优先使用本机磁盘存放 SDK 和私有 Python，逐文件核对 SHA-256，保留持久原件，并在本机缓存丢失后恢复。可设置 `DEEPLINKENGINE_RUNTIME_CACHE` 指定至少 10 GiB 的本机目录；默认探测无可用本机盘时沿用 NFS。

可用 `--wheelhouse /path/to/packages` 指定已下载的包，`--python /path/to/python3.12`、`--cann-home /path/to/cann` 指定系统环境。已有满足版本要求的专用 venv 可通过 `--use-environment /path/to/venv/bin/python` 显式复用。Python 包可自动补齐；系统驱动、CANN 和模型权重需事先准备。

## 启动服务

```sh
python deeplink.py serve --installation /workspace/deeplink-install \
  --model /models/Qwen3.5-35B-A3B --physical-devices 0,7 \
  --host 127.0.0.1 --port 8000 --served-model-name deeplink
```

物理卡号按实际分配填写，逻辑卡号由 `npu-smi` 探测；也可用 `--devices 0,1` 显式指定并核对映射。启动前校验源码和依赖并申请共享设备锁，所选卡有其他作业时退出；`--lease-timeout 300` 可等待。每次服务使用独立缓存和工作目录，路径在启动时输出。使用 Ctrl+C 关闭服务并释放设备锁。

加 `--dry-run` 可生成启动计划，不加载模型、不占设备。当前后端的模型与并行配置由版本适配器管理。

## 调用

```sh
python examples/chat.py "你好" --base-url http://127.0.0.1:8000 --model deeplink
```

服务提供 `/health`、`/v1/models` 和 `/v1/chat/completions`。示例仅使用 Python 标准库，可用 `--max-tokens` 调整输出长度。

## 版本与下载

- [当前发行版说明](versions/2026.11.17.8.post2.md)
- [统一下载中心](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/r8-runtime-dependencies-20261119-v1)与[完整索引](dependencies.json)
- [运行时版本清单](r8-lock.json)与[测试工具版本清单](../benchmark/one-click-lock.json)
- [测试与复核](../benchmark/README.md)
