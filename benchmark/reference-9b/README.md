## Qwen3.5-9B 4.1 原件复算

对同一份 BF16 checkpoint 的 original018 B0 与 r11 RC4 B1，使用 `benchmark/one_click.py reference-9b` 复用封存的 200 道 MMLU-Pro 和每题 256 token 的原件。提供已有时钟缓存、原件图、算量和逻辑依赖清单，以及 `--conservative-reference-evidence` 指向的历史标定面板。入口检查原始工况、原件和缓存哈希，仅派发独立时钟缓存中实际缺失的用例。`reference-9b --help` 列出路径参数。

正式保守结果写入 `compute-reference/conservative-reference.json`，包括六轮的原墙钟、下界、最长路径与原件清单。成功退出码为 0。4.1 校验或标定故障返回 2，并照常出具封存的 4.2–4.4 报告。参考时长和效率按下界签发，精确值及里程碑达标判定另有证据要求。完整口径见[4.1 复算方法](../one-click-kit/protocols/m01-confirmed-reference-method.md)。
