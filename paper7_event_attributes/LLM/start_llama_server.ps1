# 论文七：启动本机 Qwen3.5-9B Q8_0 服务，供两阶段属性抽取脚本调用。
# 默认只监听本机；-Model 可指定已有 GGUF，-Cpu 可禁用 GPU 分层加载。
param(
    [string]$LlamaServer = "llama-server.exe",
    [string]$Model = (Join-Path $PSScriptRoot "models\Qwen3.5-9B-Q8_0\Qwen3.5-9B-Q8_0.gguf"),
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8080,
    [int]$ContextSize = 4096,
    [string]$GpuLayers = "all",
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "GGUF model not found: $Model"
}

if ([IO.Path]::IsPathRooted($LlamaServer) -or $LlamaServer.Contains([IO.Path]::DirectorySeparatorChar)) {
    $serverExe = (Resolve-Path -LiteralPath $LlamaServer).Path
} else {
    $command = Get-Command $LlamaServer -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "llama-server.exe is not in PATH. Pass its full path with -LlamaServer."
    }
    $serverExe = $command.Source
}

# 关闭模型思考模式，使回复直接用于作者的语言分类和属性 JSON 解析。
$offloadLayers = if ($Cpu) { "0" } else { $GpuLayers }
$flashAttention = if ($Cpu) { "auto" } else { "on" }
$serverArgs = @(
    "-m", (Resolve-Path -LiteralPath $Model).Path,
    "--host", $HostAddress,
    "--port", $Port,
    "-c", $ContextSize,
    "-ngl", $offloadLayers,
    "--jinja",
    "--reasoning", "off",
    "--reasoning-budget", "0",
    "-fa", $flashAttention,
    "--no-warmup"
)

Write-Host "llama-server : $serverExe"
Write-Host "model        : $Model"
Write-Host "endpoint     : http://${HostAddress}:$Port"
Write-Host "context      : $ContextSize"
Write-Host "GPU layers   : $offloadLayers"
Write-Host "Stop with Ctrl+C."

& $serverExe @serverArgs
exit $LASTEXITCODE
