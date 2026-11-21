# 官方 B0 版本对照

保留原 0.18 结果，为官方版本分别记录运行环境和测试结果。当前使用 Qwen3.5-35B-A3B、BF16、TP2 和完整题单。

填写 [配置示例](official-baselines.example.json)，将 `protocol_*` 指向已有测试准备生成的题单、SDK 和安装记录中的 Python，选择新的输出目录后执行：

```sh
python run_all.py --official-baselines --config evaluation.json
```

只准备环境可加 `--prepare-environment`。入口按摘要下载官方 ARM64 镜像，在独立目录恢复 Python、CANN 和依赖，核验实际包版本与文件；不更改宿主驱动或共享 Python。

0.23.0、0.30.0 使用各自固定镜像。0.28 当前只有 Ascend 候选源码标签 `v0.28.0.rc1`，尚无本入口核定的正式镜像，明确记录为受阻；不替换为其他版本。

0.23 的 GDN 预编译覆盖当前 4096-token、24-sequence 调度范围内全部 87 种块数。脚本在计时前调用官方 Triton 的编译接口，共享同一镜像的编译缓存，再执行原 HTTP 预热与编译封存检查。官方后端对图模式的实际调整保存在原始服务日志中。

0.30 会在启动 Python 前自动配置包内 FLA 自定义算子目录。初始化发生致命错误时，入口保存原始错误并清理自己的服务。

完整题单为 MMLU 200、HLE 200、Frontier 160、SWE 64、Terminal 48，另测连续负载、4K/16K/32K、四租户及原生计时。HLE/Frontier 共用已安装的同一裁判，裁判流量不计入求解性能。源码计量适用性和新 F/T 未核验完成时，报告仍为未完成，不填效率或达标结论。失败不自动重试、不更改生成预算、不回退 eager。

每个版本保存独立 `campaign.json`、原始输出、日志、`report.json` 和 `report.html`。本入口及其 CPU 检查不构成完整实机验收或独立第三方认证。

连续负载已完成、随后 MMLU 在模型启动前失败时，可选择新输出目录和单个版本，增加 `--reuse-completed-continuous /原输出/版本/continuous`。入口核验完整封存、官方镜像、模型、输入、物理卡映射及源码适用性；原连续结果保留原时间，不重测或改写。已启动模型的后续失败不适用此恢复入口。

完整 solver 续跑：仅当 C8 与全部五项 solver 已封存、首个固定裁判在启动前因旧端口预检失败时，使用新 `out` 配置执行：

```sh
python run_all.py --official-baselines --config evaluation.json --reuse-completed-solvers /path/to/previous/0.23.0
```

入口核验原封存、分母、权重、输入、设备及逐项审查的控制器变化，保留原响应、时间与失败，不重复 solver 推理。原失败状态保留；不自动签发 F/T 或第三方资格。
