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

原生计时后，入口自动运行计时之外的独立算术观察，核对实际源码、加载模型和缓存容量，分别重建三次本轮 F/T，再按公开固定来源复算原完整参考分母。已有封存原生计时只需续补时，使用新的输出目录：

```sh
python run_all.py --official-baselines --config evaluation.json --complete-ft 原版本目录 --source-version-seal-sha256 原checksums摘要
```

原三次计时和数据集结果保持不动。已有完整独立观察可加 `--reuse-ft-companion 观察目录 --companion-seal-sha256 观察checksums摘要`，仅做 CPU 审核；结果中的第三方独立见证状态仍单独保留。

连续负载已完成、随后 MMLU 在模型启动前失败时，可选择新输出目录和单个版本，增加 `--reuse-completed-continuous /原输出/版本/continuous`。入口核验完整封存、官方镜像、模型、输入、物理卡映射及源码适用性；原连续结果保留原时间，不重测或改写。已启动模型的后续失败不适用此恢复入口。

仅重建已封存的 0.23 C8 报告，在普通 CPU 机器执行：

```sh
python run_all.py --official-baselines --report-only --config report-only.json
```

配置为 `{"source":"原输出/0.23.0","source_sha256":"原 checksums.json 的 SHA256","out":"新的报告目录"}`。输出包含 4K/16K 正式阶段、单列预热、网站观测 JSON 和 `portable-c8.tar.gz`。外部机器将 `source` 换为该归档、`source_sha256` 换为归档摘要，并填写 `original_version_seal_sha256` 为原版本封存摘要，使用同一命令复算，无需模型、SDK 或 NPU。归档中的源码仅作来源核验，不执行。

目录输入仍核验原版本全部封存文件。便携包仅附完整 C8、父轮身份及相关源码；未运行的真实上下文输入不分发，其原封存摘要保留于清单。便携复算核验完整 C8 闭包，不声称已复验未附文件的全部字节。

此模式保留原失败、未运行阶段和原时钟；命令成功只表示 CPU 重建完成，不代表全套通过。F、效率和第三方认证仍为空或未认证。

已有完整阶段报告只需派生新报告时，使用同一命令，配置显式选择：

```json
{"kind":"sealed-report-projection","source":"已结束的版本目录","report_sha256":"原report.json的SHA256","original_version_seal_sha256":"原checksums.json的SHA256","out":"新的CPU报告目录"}
```

该分支仅验证报告、campaign及所消费的小元数据，保留原状态、原数值和未测项，不重算全部SSE或验证未附payload。输入原件与旧曲线不改，输出另存选中原件、来源记录及未就绪的网站预览；不签新F/T或第三方认证。绘图先探测当前Matplotlib 3.10.8，缺失或损坏时自动修复独立CPU缓存环境，支持标准`PIP_INDEX_URL`/`PIP_FIND_LINKS`下载源，环境状态写入`plot-environment.json`。默认缓存在输出父目录的独立目录，避免HOME/XDG空间影响；`plot_cache`仍可显式指定报告目录外的缓存。仅投影JSON时可设`"plots":false`。最后发布的`projection-execution.json`绑定最终封存摘要，作为完成标记。原C8严格复算仍使用前述配置。
