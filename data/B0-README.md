# 固定 B0 镜像原件

B0 为 ARM64 `quay.io/ascend/vllm-ascend:v0.18.0`，原始 OCI manifest 为：

```
sha256:2467d259ff068ca4edd7b50b849152910f3c05e435f314639fc06bcc7a00f9fa
```

原始 manifest、配置和全部 17 个压缩层已封存到[本仓库 B0 Release](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/b0-original018-oci-20261120-v1)，共约 5.93 GB。大 blob 按顺序分片；[索引](b0-oci.json)记录每片和完整原件的大小、SHA-256 与下载入口。

根目录一键评测自动准备这些原件，再交给既有冻结恢复器重建 B0。无需 Docker 服务，也不依赖测试时能访问 Quay 的 blob 服务。原始镜像、源码、二进制及镜像内使用条件均保留，恢复后仅应用已冻结的文本适配补丁。

已存在且哈希相符的 OCI blob 优先复用。可通过 `DEEPLINKENGINE_B0_OCI_CACHE` 指定保存 `sha256-<digest>` 文件的目录。默认也检查 `~/.cache/vllm-hust-engine/b0-image-018`。本次缓存目录中已有的原件仍会校验，不重复下载。

完全离线时，将 Release 的分片下载到同一目录并设置 `DEEPLINKENGINE_B0_ARCHIVES=/path/to/b0-parts`。每片和重组后的原始 blob 均须通过校验，随后核对完整原始 manifest 清单。缺片、错序、损坏和额外 blob 均不能进入模型测试。

仅准备原件：

```sh
python baseline_bundle.py --cache /path/to/cache
```

若旧入口在安装完成后的准备阶段失败、尚未产生推理任务，可保留原配置恢复：

```sh
python run_all.py --config evaluation.json --resume-bootstrap
```

该模式先核对原配置，重新校验环境、安装包和运行时源码，保留失败收据及日志，并在新目录重做准备。已经存在推理任务时会拒绝此模式，避免重复运行或混用结果。
