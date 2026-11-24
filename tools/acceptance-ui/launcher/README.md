# Windows SSH 启动器

当前 Coder 验收工作区的连接步骤见[验收实验台说明](../../../README.md#验收实验台网页)。对于可直接使用 OpenSSH 登录的部署，可以将本目录的两个启动文件复制到 Windows 的同一个文件夹，在旁边创建 config.local.json，并填入现场连接参数。

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

启动器的后台转发需要可用的 SSH 密钥或 SSH agent。如果登录必须交互，请使用下方的手动转发命令。按 Enter 退出启动器后，服务器上正在执行的测试会继续运行。

## 用 OpenSSH 手动连接

将示例中的用户名、主机、SSH 端口和远程目录换成现场值。在 Windows 的 PowerShell 中先启动服务器端 UI，再保持运行第二条命令的窗口开启。

```powershell
ssh -p 22 tester@server.example.org 'cd /srv/DeepLinkEngine/tools/acceptance-ui && bash ./run.sh --background 8081'
```

```powershell
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -p 22 -L 127.0.0.1:18080:127.0.0.1:8081 tester@server.example.org
```

然后在 Windows 浏览器打开[验收实验台](http://127.0.0.1:18080/)。
