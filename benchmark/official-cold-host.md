# 从空白 Ascend 主机准备官方对照

网络较慢时，可在另一台电脑（含 Windows）只下载一个官方锁中已有的 OCI 文件，不解包、不安装、不启动模型。配置仅含 `cache`、新 `out` 和 `versions`：

```sh
python run_all.py --official-baselines --config download.json --download-only --download-transport curl --artifact-digest sha256:...
```

逐个文件传入目标主机的独立缓存，按同一 SHA256 核验，再由目标主机的公开入口自动安装。下载结果不是实机测试结果。

纯下载可选环境变量 `DEEPLINKENGINE_DOWNLOAD_RANGE_WORKERS=4`（默认 1，最多 4），按严格 HTTP Range 断点下载；最终仍核验完整 SHA256。已有单流下载前缀会继续原传输，不创建第二份大文件。

可识别的同锁未完成安装会自动另存原件后重建；已完整解包的安装只恢复元数据。未知目录或来源锁不符时保留原目录并退出。

不修改宿主驱动或共享 Python。宿主 Python 3.9 可先恢复私有官方镜像；`model_manifest` 可选，按固定修订和 SHA256 自动恢复模型：

```sh
python3 run_all.py --official-baselines --config prepare-images.json --prepare-environment
```

在普通引擎配置的 `environment.official_sdk` 指向已恢复的 **0.23** SDK，固定 `engine_version` 为公共裁判/协议版本，然后只准备 SDK 和题单：

```sh
python3 run_all.py --config prepare-protocol.json --prepare-protocol-only
```

此入口在 `prepare-fixed-five` 后退出；不启动 B0、B1、模型服务或正式评测。结果为 `prepared`，不是验收完成。其 `results.json` 给出 `protocol_run`、`protocol_deployment`、`protocol_python`，填入官方版本配置后执行原有 `--official-baselines` 完整入口。

模型锁：[Qwen3.5-35B-A3B](../engine/models/qwen3.5-35b-a3b.json)。14 个权重分片及 tokenizer、配置和模板 SHA 与原完整题单相同；来源为 ModelScope 官方仓库各文件的固定提交。宿主、驱动、固件和各镜像运行时仍分别记录，跨主机结果按环境分组。

宿主 glibc 较旧时，入口从固定官方 Ubuntu 基础层准备独立执行根，自动让 SDK 探测、完整流程和后续子进程使用其 glibc。记录同时保留宿主与执行层的系统身份、真实 PID/boot 和共享租约 inode。SDK 准备按版本互斥；活跃或身份未知的旧准备目录保留。
