# 从空白 Ascend 主机准备官方对照

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
