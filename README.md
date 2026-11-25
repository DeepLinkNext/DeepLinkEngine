# DeepLinkEngine

面向昇腾的推理引擎，提供自动环境准备、安装、推理服务和调用示例。引擎源码随发行包提供。

## 安装与使用

默认安装和测试 [vLLM-HUST-Engine](https://pypi.org/project/vllm-hust-engine/) 最新稳定发行版。当前稳定版 R10 修订版 2026.11.24.10 面向 Qwen3.5-35B-A3B BF16／TP2，启用 3072 混合图。适用于 CPython 3.12、Linux ARM64、CANN 9.1.0 和双昇腾 910B2。准备系统环境和模型权重后执行。也可以[下载源码归档](https://api.github.com/repos/DeepLinkNext/DeepLinkEngine/tarball/main)。

```sh
git clone https://github.com/DeepLinkNext/DeepLinkEngine.git
cd DeepLinkEngine
python deeplink.py install --out /workspace/deeplink-install
python deeplink.py serve --installation /workspace/deeplink-install \
  --model /models/Qwen3.5-35B-A3B --physical-devices 0,7
```

新安装默认选择最新稳定版，并保存本次版本与校验记录；服务和失败恢复沿用该版本。复现历史版本可加 `--engine-version VERSION`。

安装会自动准备独立 Python 环境、补齐依赖并核验文件。卡号填写实际分配的两张物理卡，逻辑卡号自动探测。服务默认监听 `127.0.0.1:8000`，提供 OpenAI 兼容接口：

```sh
python examples/chat.py "你好，请介绍一下自己"
```

[安装与服务配置](engine/README.md) · [R10 发行说明](https://pypi.org/project/vllm-hust-engine/2026.11.24.10/) · [R10 运行包](https://github.com/vLLM-HUST/vllm-hust-website/releases/tag/engine-wheels-r10-v1) · [引擎官网](https://vllm-hust.sage.org.ai/engine.html)

## 测试与报告

2026.11.24.10 的 Qwen3.5-35B-A3B BF16 TP2 已完成固定 original018 同卡全量验收。4.1、4.2、4.4 两档达标。4.3 的 16K 里程碑达标，32K 五个窗口的 TTFT P95 波动为 2.031 倍，超过 2 倍结题门槛。4K–8K、16K–24K、32K–48K 专项使用 Terminal-Bench 2.1 公开历史离线回放，并发 8。 [本轮完整结果](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/70330385dfff597e1f7b83af6768d0356fd51f42/reports/r10-35b-paired-full-20261125/README.md) · [复算来源与哈希](evidence/r10-35b-full-acceptance-20261125.json) · [发行版及早期 V5 记录](engine/versions/2026.11.24.10.md)

单组实验用 `--engine` 选择 B0/B1，用 `--controllers kv-control,instance-scaling` 选择 RIDE 控制器；省略即运行引擎基线。见 [单组实验命令与站点配置](benchmark/unified-commands.md)。

将[示例配置](benchmark/run-all.example.json)另存为仓库根目录的 `evaluation.json`，填好后运行下面的命令。

```sh
python run_all.py
```

若省略 `model`，脚本默认使用 `/models/Qwen3.5-35B-A3B`。模型存放在其他位置时，在配置中填写实际路径。

完整入口默认检查输出目录的同级封存。匹配的 B0 阶段会复用，其余阶段自动测量。也可按[恢复与复用说明](benchmark/README.md#恢复与结果复用)接续。运行 `python run_all.py --help-all` 可查看复用和恢复参数。

不填写 `engine_version` 时使用最新稳定版；指定版本可复现历史运行。恢复、复算及跨版本结果复用见[测试指南](benchmark/README.md#恢复与结果复用)。

R9 的书生单模型验证见[版本说明](engine/versions/2026.11.19.9.md)，不作为完整五项 B0/B1 或项目最终验收结果。

入口自动完成环境准备、缓存回收、双臂预热、五项配对、有效算力参考、上下文与四租户测试，并生成 `full-report/report.html` 及 PNG/SVG 对比图。后台运行时，用 `python run_all.py --status` 查看全流程和当前任务进度，加 `--watch` 持续观察。耗时分别显示当前任务与整轮验收，原生计时尚未写出开始时间时可查看距首次设备采样的时间。程序读取状态时可加 `--json`，取得完整阶段记录。启动时指定了 `--config` 的，查询时使用同一配置文件。

[固定数据包](data/README.md)和[B0 镜像原件](data/B0-README.md)均由测试入口从本仓库 Release 自动恢复并校验。[测试与复核指南](benchmark/README.md)包含固定五项、长上下文、四租户、报告生成和第三方验证；[现场测试说明](benchmark/docs/topic4-test-guide.pdf)提供见证流程。

已保存的 B0/B1 验收结果可用 `python run_all.py --results` 在终端查看，加 `--json` 可读取结构化结果。使用自定义配置时传入相同的 `--config`。

## 验收实验台网页

网页和测试服务运行在 Coder 工作区。从一台新的 Windows 电脑访问时，按以下步骤操作。

### 1. 在 Windows 浏览器登录 Coder

在本机默认浏览器打开平台，使用有权限访问 `vllm-hust-midterm-repro-01` 的账号登录。也可以在 PowerShell 中直接打开。[Coder 平台](https://coder.sage.org.ai:9450/)

```powershell
Start-Process 'https://coder.sage.org.ai:9450/'
```

### 2. 在本机 PowerShell 安装 Coder CLI 并登录

```powershell
winget install --id Coder.Coder --exact --source winget
```

安装完成后重新打开 PowerShell，再运行以下命令。登录时浏览器通常会打开授权页。继续使用第 1 步登录的账号，按页面和终端提示完成授权。如果终端要求输入令牌，从授权页复制到终端。如果浏览器没有自动打开，手动打开终端给出的链接。[Coder CLI 安装文档](https://coder.com/docs/install/cli) · [CLI 登录文档](https://coder.com/docs/reference/cli/login)

```powershell
coder version
coder login https://coder.sage.org.ai:9450/
```

### 3. 从本机 PowerShell 进入验收工作区

```powershell
coder ssh vllm-hust-midterm-repro-01
```

### 4. 在工作区 Linux 终端启动或确认 UI 服务

在第 3 步进入的远程终端执行。`run.sh --background` 会复用已在运行的实验台，服务未运行时会在后台启动。健康检查返回 `status: ok` 后即可退出远程终端。

```bash
cd /root/test/DeepLinkEngine/tools/acceptance-ui && \
  bash ./run.sh --background 8081 && \
  curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8081/api/health
```

```bash
exit
```

### 5. 在本机 PowerShell 保持端口转发

回到 Windows 的 PowerShell 提示符后执行整段命令。它将本机的 `127.0.0.1:18081` 转发到工作区的 `127.0.0.1:8081`，转发进程退出后会等 5 秒再重试。使用网页期间保持这个窗口开启。需要停止转发时关闭该窗口。[Coder 端口转发文档](https://coder.com/docs/reference/cli/port-forward)

```powershell
while ($true) {
    coder port-forward vllm-hust-midterm-repro-01 --tcp '127.0.0.1:18081:8081'
    Start-Sleep -Seconds 5
}
```

如果本机 `18081` 已被占用，可把命令和下一步地址中的 `18081` 一起改为 `18082`。工作区端口仍用 `8081`。如果 CLI 提示登录过期，先重新执行第 2 步的登录命令。

### 6. 在 Windows 浏览器打开验收实验台

等转发窗口显示 `Ready!`，打开网页。也可以另开一个 PowerShell 窗口执行下面的命令。[验收实验台](http://127.0.0.1:18081/)

```powershell
Start-Process 'http://127.0.0.1:18081/'
```

关闭浏览器或本机转发窗口后，已经启动的实验仍在工作区运行。恢复转发后，从「运行记录」回到对应轮次。

打开实验台后，可选择 **预检准备** 或查看已有运行记录。更多测试步骤和部署配置见[验收实验台文档](tools/acceptance-ui/README.md#正式测试操作顺序)。

## 目录

| 目录 | 内容 |
| --- | --- |
| [engine/](engine/README.md) | 安装、服务适配器、版本和依赖索引 |
| [examples/](examples/chat.py) | 推理 API 调用示例 |
| [data/](data/README.md) · [datasets/](datasets/README.md) | 固定数据包、评测题单与协议 |
| [benchmark/](benchmark/README.md) | 测试工具、公开结果与复算 |
| [official/](official/) | 官方基线评测、SDK 准备、F/T 审计和报告实现 |
| [tools/](tools/verify_snapshot.py) | 发布材料和校验工具 |
