# 单组实验：引擎实例与 RIDE 控制器

在能管理本站引擎进程、访问 RIDE 和服务端点的 Linux 执行机上，进入 DeepLinkEngine 根目录。每条 benchmark 命令独立完成一组实验。引擎版本、初始实例数和每实例 TP 属于数据面；RIDE 按逗号分隔的类型集合启动控制器。省略 `--controllers` 即运行引擎基线。

```bash
common=(--config experiments/mmlu-pro-200-out256.json --site sites/local.json --replicas 1 --tp 2)
python deeplink.py benchmark "${common[@]}" --engine b0-original018
python deeplink.py benchmark "${common[@]}" --engine b1-r8
python deeplink.py benchmark "${common[@]}" --engine b1-r8 --controllers request-throttling
python deeplink.py benchmark "${common[@]}" --engine b1-r8 --controllers kv-control
python deeplink.py benchmark "${common[@]}" --engine b1-r8 --controllers kv-control,instance-scaling
python deeplink.py benchmark "${common[@]}" --engine b1-r8 --controllers request-throttling,kv-control,instance-scaling
```

加 `--dry-run` 查看控制器类型、具名实例、管理地址、客户端目标及引擎池，加 `--out DIR` 保存到新目录。零个控制器或仅启用后台控制器时，客户端直达本组引擎池。限流层使用本组后端并提供推理入口，MMLU 客户端带 program 身份并在该题结束后 release。类型顺序无需表述依赖关系。新后台控制器沿用 RIDE 插件注册和站点配置。当前支持直连与单个支持 program-release 协议的推理入口。

## 站点配置

已有站点在站点文件增加一个 `ride` 字段。新站点可给下节的 prepare_site 命令加 `--ride-config FILE`，把下面这份站点配置写入生成结果。每站一份，所有组合共用。此示例的地址、挂载路径、ID 和 SHA 由现场填写，控制器策略文件沿用各自原有格式。

```json
{
  "management_endpoint": "http://127.0.0.1:19100",
  "controllers": {
    "request-throttling": {
      "id": "throttle-main", "endpoint": "http://127.0.0.1:19001",
      "listen": "127.0.0.1:19001",
      "runtime_env": {"env_vars": {"THUNDERAGENT_TRANSPORT_TIMEOUT_SECONDS": "-1", "THUNDERAGENT_RESUME_TIMEOUT_SECONDS": "-1"}}
    },
    "kv-control": {
      "id": "myc-kv", "endpoint": "http://127.0.0.1:19082",
      "configuration": "/workspace/config/kv.json", "configuration_sha256": "<KV configuration SHA-256>"
    },
    "instance-scaling": {
      "id": "myc-instance", "endpoint": "http://127.0.0.1:19083",
      "configuration": "/workspace/config/instance.json", "configuration_sha256": "<Instance configuration SHA-256>"
    }
  }
}
```

站点先准备可访问的 RIDE 管理服务，管理端和执行环境安装所选类型，支持带 operation_id 和 expected_job_id 的条件启动、按作业身份停止及就绪探测。KV、Instance 的冻结策略文件须与实例 ID、端点、本组模型配置匹配。推理代理的后端由本组引擎池生成。运行器等所有控制器就绪后发题，将启动、就绪、停止和终态写入结果目录的 controllers 子目录。不同组若同时占用同一服务，后启动的一组会被拒绝。

Instance 的自动扩缩依赖真实可控的模型作业、请求排空和 KV 后端。现有引擎启动器托管站点前台进程。这里覆盖控制器启动和请求路径；允许操作真实模型作业的现场配置、缩容和端点切换需要独立验收。多引擎配置要求 engines 下逐实例提供 host、physical_cards、endpoint 和前台 argv；八实例 TP2 对应 16 卡。256 token 配置用于短输出服务检查。

## 代码位置

| 职责 | 入口 |
| --- | --- |
| 单组编排及公开参数 | [unified_run.py](unified_run.py) |
| 数据面的引擎放置、就绪和收尾 | [engine_pool.py](engine_pool.py) |
| 控制器选择、推理目标和 RIDE 作业生命周期 | [control_plane.py](control_plane.py)、[ride_management.py](ride_management.py) |
| 冻结题单、逐题流式请求、计时与离线评分 | [frozen_workload.py](frozen_workload.py)、[compare_groups.py](compare_groups.py) |

`run_all.py --engine …` 共用单组实现，`run_all.py --config evaluation.json` 保留历史完整评测。旧参数 `--route direct` 和 `--route request_throttling --controller-config FILE` 分别保留直连和原 RIDE local 部署。新入口直接使用本站 RIDE 管理服务。

## Control Plane §4.1 成对认证

在执行机选定包含对应一键入口的 request-throttling-controller checkout。两个 DeepLinkEngine 启动器把参数、当前目录和退出码原样交给限流层仓库的实验入口。两边共用同一套匹配、复算、门禁和报告代码。

```bash
export VLLM_HUST_CONTROLLER_ROOT=/workspace/request-throttling-controller
python deeplink.py control-plane run --scope 4.1 \
  --assets "$R8_ASSETS" --assets-sha256 "$R8_ASSETS_SHA256" \
  --checkpoints "$CHECKPOINTS_JSON" --output "$RESULTS_ROOT/control-plane-4.1"
```

也可把前缀换为 `python run_all.py control-plane`，或在限流层仓库直接调用原入口。已封存的原生计时和算量观察均匹配时，命令在 CPU 上复算。缺一项时提供站点、安装目录、运行后缀和授权截止，仅调度缺失阶段。终端与 `delivery/section-4.1.json` 提供同一组效率、参考上界及门槛结论。
单组请求用上文的 `benchmark`，其整批时延和吞吐另由 `compare` 读取。
[Control Plane 一键实验及原件清单](https://github.com/vLLM-HUST/request-throttling-controller/blob/main/experiments/vllm_hust_benchmark/README.md)

## 准备站点

原版 B0 的运行文件从仓库已有的 B0 镜像资产恢复；B1 使用已安装的发行版。以 R8、两张物理卡 6 和 7、容器内逻辑卡 0 和 1 为例。模型路径换成本站挂载位置。每个现场独立生成自己的站点文件。

```bash
python deeplink.py install --engine-version 2026.11.17.8.post2 --out /workspace/deeplink-r8
python tools/prepare_datasets.py --datasets mmlu --cache /workspace/deeplink-cache
python -m benchmark.prepare_site --installation /workspace/deeplink-r8 \
  --model /models/Qwen3.5-35B-A3B --physical-devices 6,7 --devices 0,1 \
  --cache /workspace/deeplink-cache --results /workspace/results --out sites/local.json
```

单组入口可从数据 Release 恢复 MMLU，并从原始题目顺序和请求正文重建输入。也可用站点的 `requests` 字段显式提供冻结 JSONL，两种方式都核对实验配置中的同一个 SHA-256。完整回答使用冻结协议的 8192 token 上限。已有的 out256 配置用于短输出服务测量。每题用独立 cache salt。

旧命令 `--route request_throttling --controller-config FILE` 使用 Controller 仓库提供的 RIDE local 依赖制备与启停脚本。将下列变量指向本站实际的 Controller 源码和依赖清单。制备会核验源包摘要并安装依赖。把示例中 `runner` 和 `installation` 改为实际绝对路径。无需在每次运行前手工起服务。

```bash
python "$CONTROL_REPO/scripts/prepare_ride_dependencies.py" \
  --manifest "$DEPENDENCY_MANIFEST" --installation /workspace/ride-local
cp configs/request-throttling.example.json configs/request-throttling.json
```

[Controller 依赖制备及 RIDE local 启停脚本](https://github.com/vLLM-HUST/request-throttling-controller/pull/7)

## 复核结果

```bash
python deeplink.py compare --deployment /workspace/deeplink-r8/sdk \
  /workspace/results/b0-direct /workspace/results/b0-controller \
  /workspace/results/b1-direct /workspace/results/b1-controller
```

把目录替换成四次实际输出路径。离线复核使用安装包的原始答案和提取器，逐题检查请求正文和响应流摘要。输出包含计划题数、完成数、答对数、达到长度上限数、整批用时、输出吞吐、首 token 及完整请求延迟的 P50/P95。验收 Controller 的实际放行与后端绑定还应检查 RIDE 回执和服务日志。

复现已制备的 RIDE R8 program-cache 实验环境时，可以在站点 B1 的 `argv` 中指定 `bash benchmark/serve_frozen_r8.sh LAUNCHER RUNTIME {output}/worker DEADLINE`。四项分别为 RIDE 冻结启动包所在目录、已核验运行时目录、本次新目录和站点服务租约截止 Unix 秒数。适配器校验冻结清单的摘要，再调用 RIDE 原有的运行时复用、模型预检和前台启动逻辑。可共享的启动代码和校验值随本仓库交付，机器上的模型、已安装依赖及分配卡号由站点提供。

MMLU 客户端沿用 Controller 仓冻结的 [MMLU 流式读取实现](https://github.com/vLLM-HUST/request-throttling-controller/blob/3bf91fbb2b5cdd8fd0bd5de94774c26ca651eb35/experiments/vllm_hust_benchmark/website/profile_streaming_transport.py)，增加用于两端关联请求的跟踪头。流须包含 HTTP 200、结束原因、有效 token 用量及终止事件。首个输出 token 包含引擎返回的 `reasoning` 或最终文本，首个最终文本时间单独保存。评分只读 `content`。到达长度上限而仅返回 `reasoning` 时，单独统计为完整响应、零最终答案，保留原始流。

四组实测、逐请求限流证据及原始输出预算的单题补验见[2026-11-24 单实例 TP2 真机验收报告](reports/unified-mmlu-20261124/README.md)。
