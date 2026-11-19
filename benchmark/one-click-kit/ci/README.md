# 评测范围与发布门禁

正式冻结引擎大版本的 PyPI 自动发布规则、附件准备和复跑方式见 [冻结引擎自动发布到 PyPI](https://github.com/vLLM-HUST/vllm-hust-engine/blob/main/ci/PYPI_RELEASE.md)。它与下述测试工具包发布相互独立。

工作流 **Evaluation contract** 对每个 PR、main 提交和 `engine-tools-v*` 标签执行；没有路径过滤，不会因为只改配置、文档或下载入口而漏跑。CPU 检查不代表 NPU 性能验收。

业务标准独立记录在 `protocols/evaluation-entry-contract-v1.json`：全量为 MMLU200、HLE 分层200、Frontier160、SWE 公开轨迹固定32任务、Terminal 公开轨迹固定32任务；快速仅 MMLU200。默认核验参考分母。三个抽样原件连同生成预算由 SHA256 固定。显式的专项或自定义 CLI 参数仍可使用。

门禁执行真实配置解析、命令转发和控制器 argparse，只截住最后开始恢复/推理的 `run()`：

- 检查默认/full/quick × prepare/run，以及直接传参；同样检查默认 r6 和显式 r8 单文件入口。
- 构建候选 tar.gz，使用实际下载器解压、核验外部固定的 manifest，然后在解压结果上复验入口；多次配置导入不得产生 `__pycache__` 或改变封存目录。
- 同源 PR/main 还下载两个入口实际固定的 Release 附件，按当前独立标准复验，避免只修源码、遗漏旧包。
- 故意恢复历史的“三数据集”默认或暗加 `--performance-only`，要求检查确实失败；同时执行完整 CPU 回归。

检查报告、JUnit 和候选包保留在 Actions artifact。普通 CI 候选包的 asset ID 为0，明确不能充当已发布下载入口。

## 发布

在已合并到 main 的提交上创建 `engine-tools-v<版本>` 标签，触发依赖门禁成功的 **Publish validated tools**。它先建草稿，上传已验收包，从 GitHub 重新下载核验 SHA256，填入真实 asset ID，再次验收两个下载入口后才发布。失败保留草稿供排查，不修改旧引擎 Release。CI 的验证不要求 NPU 或模型权重，也不把运行中的冻结引擎替换掉。

## 执行器选择与历史故障

当前评测与 PyPI workflow 统一使用仓库变量 `EVALUATION_RUNNER_LABELS` 选择独立 CPU runner，未配置时才使用 ubuntu-24.04。现有自托管 runner 已通过 PR #36 的 Evaluation contract；不要将正式 NPU 推理进程所在的容器注册为通用 PR runner。

历史故障：2026-11-17 PR #35 的首次工作流在分配 GitHub 托管执行器前，因付款/额度错误未能启动，不能记作该次 CI 通过。该记录不代表当前自托管 runner 状态。遇到排队或启动失败时，核对实际 runner、labels、作业日志和组织配额，按具体原因处理。

## main 保护

应将作业 **Evaluation contract** 设为 main 必需检查，按 [AGENTS.md](https://github.com/vLLM-HUST/vllm-hust-engine/blob/main/AGENTS.md) 的 PR → 最新 head 检查 → 合并 → main 回读流程交付。

2026-11-17 配置分支保护/规则集的 API 曾返回套餐限制403。这是历史记录；当前是否启用以仓库 Settings/API 实际状态为准，不能把“有 CI”表述为已启用强制保护，也不能把未启用保护当作绕过检查的理由。配置规则时保留既有其他检查与审查要求。

修改业务范围时必须先更新测试大纲及这个独立标准，再解释抽样、预算和参考适用域的变化；不能单独把测试预期改成当前实现的输出。

本地复现：

```sh
python3 -m pip install pytest==8.3.2
python3 -B ci/evaluation_contract.py --report reports/source-entry-contract.json
python3 -B -m pytest -q tests
python3 -B ci/build_tools.py --out dist --commit "$(git rev-parse HEAD)"
python3 -B ci/evaluation_contract.py --root dist/extracted --bootstrap dist/download-and-run.py --report reports/default-artifact-contract.json
python3 -B ci/evaluation_contract.py --root dist/extracted --bootstrap dist/download-and-run-r8.py --runner-profile r8 --report reports/r8-artifact-contract.json
```
