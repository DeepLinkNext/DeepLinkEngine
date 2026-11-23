# Windows SSH 启动器

在 Windows 操作电脑上可使用上一级 README 的两条 SSH 命令。若希望双击启动，将本目录的两个启动文件复制到 Windows 的同一个文件夹，在旁边创建 config.local.json，并填入现场连接参数。

```json
{
  "ServerHost": "server.example.org",
  "SshPort": 22,
  "UserName": "tester",
  "RemotePath": "/srv/DeepLinkEngine/tools/acceptance-ui"
}
```

双击 **启动验收实验台.cmd** 后，启动器通过 OpenSSH 登录并检查远端服务，再打开本机浏览器。保持启动器窗口开启即可访问。配置文件仅用于本机，Git 会忽略它。

默认转发本机 18080 到远端 8081。可在 PowerShell 中覆盖配置项，例如更换本机端口。

```powershell
powershell -ExecutionPolicy Bypass -File .\launcher.ps1 -LocalPort 18081
```

启动器的后台转发需要可用的 SSH 密钥或 SSH agent。如果登录必须交互，请使用 README 的手动转发命令。按 Enter 退出启动器后，服务器上正在执行的测试会继续运行。
