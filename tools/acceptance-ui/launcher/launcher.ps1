#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$NoGui,
    [switch]$NoBrowser,
    [string]$ServerHost,
    [ValidateRange(0, 65535)][int]$SshPort,
    [ValidateRange(1, 65535)][int]$LocalPort = 18080,
    [string]$UserName,
    [string]$RemotePath
)

$ErrorActionPreference = "Stop"
$settings = @{}
if ($PSScriptRoot) {
    $settingsPath = Join-Path $PSScriptRoot "config.local.json"
    if (Test-Path -LiteralPath $settingsPath -PathType Leaf) {
        $settings = Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
}
if (-not $ServerHost) { $ServerHost = $settings.ServerHost }
if (-not $UserName) { $UserName = $settings.UserName }
if (-not $RemotePath) { $RemotePath = $settings.RemotePath }
if (-not $SshPort) {
    $SshPort = if ($settings.SshPort) { [int]$settings.SshPort } else { 22 }
}
if (-not ($ServerHost -and $UserName -and $RemotePath) -or $SshPort -lt 1 -or $SshPort -gt 65535) {
    Write-Host "请按 README 配置 config.local.json，或传入 -ServerHost、-UserName、-SshPort 和 -RemotePath。"
    exit 2
}

$destination = "$UserName@$ServerHost"
$url = "http://127.0.0.1:$LocalPort/"
$commonArgs = @("-p", "$SshPort", "-o", "ConnectTimeout=12", "-o", "StrictHostKeyChecking=ask")
$quotedPath = "'" + $RemotePath.Replace("'", "'\''") + "'"
$remote = "cd $quotedPath && ./run.sh --background"
$forward = "127.0.0.1:$($LocalPort):127.0.0.1:8081"

function Test-UiHealth {
    try {
        $response = Invoke-RestMethod -Uri ($url + "api/health") -TimeoutSec 2
        return $response.service -eq "deeplink-acceptance-ui" -and $response.status -eq "ok"
    } catch {
        return $false
    }
}

function Test-LocalPort {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect("127.0.0.1", $LocalPort)
        return $true
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Host "当前电脑需要安装 OpenSSH 客户端。"
    exit 2
}

# Use the console for the initial SSH authentication and host-key confirmation.
& ssh @commonArgs $destination $remote
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (Test-LocalPort) {
    Write-Host "本地端口 $LocalPort 已被占用。请使用现有隧道，或指定 -LocalPort 18081。"
    exit 2
}

$tunnel = $null
try {
    $sshPath = (Get-Command ssh).Source
    $argsForForward = $commonArgs + @("-NT", "-L", $forward, "-o", "BatchMode=yes", "-o",
        "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", $destination)
    $tunnel = Start-Process -FilePath $sshPath -ArgumentList $argsForForward -WindowStyle Hidden -PassThru
    $ready = $false
    for ($i = 0; $i -lt 20; $i++) {
        if ($tunnel.HasExited) { break }
        if (Test-UiHealth) { $ready = $true; break }
        Start-Sleep -Milliseconds 350
    }
    if (-not $ready) {
        Write-Host "隧道未建立。确认这台电脑可用 SSH 密钥登录，或按 README 手动建立隧道。"
        exit 1
    }

    Write-Host "RESULT=connected $url"
    if (-not $NoBrowser -and -not $NoGui) {
        Start-Process $url
    }
    if (-not $NoGui) {
        Write-Host "请保持此窗口开启。按 Enter 断开后，可从运行记录回到当前轮次。"
        Read-Host | Out-Null
    }
} finally {
    if ($tunnel -and -not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue
    }
}
