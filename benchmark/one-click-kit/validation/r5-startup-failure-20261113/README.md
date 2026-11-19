# r5 启动验收失败与 r6 修复候选

原活动 `five-datasets-r5-matched-20261113-r5` 完成 1/15 阶段，B0 的 C8 4K/16K 共225个正式请求成功，71次共同预热后双rank编译边界封闭。B1在HTTP健康检查之前失败，未提交正式请求，五数据集未执行。本轮四个目标无法作新配对验收；59.76%是已有M01/out256历史范围内的效率下界，不能冒充新r5结果。

2026-11-13 14:06:48 UTC，两个worker在 `serving_warmup.warmup -> position_reuse.prepare -> OwnedOptimisticOperand.refresh` 抛出 `ValueError: Active protected host progress snapshot required`。新增编译预热调用了原r4 M01真实请求状态保护，此时没有真实请求快照。外层只检查HTTP父进程是否退出，没有及时发现worker错误，最终到14:39才报告原3000秒截止。cgroup无OOM，模型和FULL图已加载；这不是HLE正式性能下降的测量。

r6显式冻结仅含预热适配的源码变化：只在compile-only positions预热中用私有浅拷贝runner，仍使用真实张量地址和原kernel封装；kernel只编译、不执行，真实请求仍要求M01受保护CPU快照。r4/r5原文件和哈希不变，r6从固定r4加明确overlay恢复，不升级最新引擎。回归用原r4校验过的保护模块证明预热不刷新真实状态、真实请求缺快照仍报错。CPU共108项通过，不代表NPU验收。

控制器新增启动期worker/EngineCore明确致命日志识别，保存首个原因和原日志，不更改超时、测试预算或守卫。仅在B0来源/设备/权重/共同预热实现/输入配方完全一致时允许显式复用已封存continuous-B0；旧回执保留真实身份。B1全部阶段仍须新测。

完整原始失败日志、清单、执行源码、B0正式流和报告将保存在同名诊断Release附件；无编译缓存、SDK和模型。五数据集标准及题号、评分分母、预算保持不变。
