# NumPy 1.26 兼容包

冻结厂商运行时要求 NumPy 1.26.4，而上游 OpenCV 5.0.0.93 的声明要求 NumPy 2。当前双卡环境中，原始 OpenCV 二进制已经能在 NumPy 1.26.4 下完成逐像素一致的 PNG 编解码。

`opencv-python-headless==5.0.0.93+deeplink.np126` 保留官方 ARM64 轮包的全部原生库和 Python 载荷，只调整版本标识、CPython 3.12 使用范围、NumPy 依赖声明及对应 RECORD。53 个已安装文件与官方轮包逐字节一致，58 个载荷文件在适配包中保持原哈希；[来源与逐文件记录](opencv-numpy126-provenance.json)随包提供。原使用条件保留。

该包随[中央依赖下载中心](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/r8-runtime-dependencies-20261119-v1)提供，一键安装自动下载验哈希并安装，随后按冻结版本约束补齐推理服务的完整传递依赖。冻结的四个核心发行包、SDK 和 MOD 源码不变。

CI 下载官方原包和兼容包，核对完整清单、逐文件内容及 RECORD：

```sh
python tools/verify_runtime_compatibility.py --cache /path/to/cache
```

## TorchVision 配对

图像题需要实际导入 `AutoProcessor`。一键安装固定使用官方 CPU ARM64 `torchvision==0.28.0+cpu`，其原始依赖明确要求 `torch==2.13.0`；轮包未经改写，SHA256 写入运行时清单。安装后运行 OpenCV 和 TorchVision 的 PNG 回环、CPU NMS、`AutoProcessor` 导入检查，原生扩展不兼容会在模型启动前报错。检查回执随依赖记录保存，不计入推理性能时间。
