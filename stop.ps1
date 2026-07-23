$ErrorActionPreference = "Stop"
$pidFile = Join-Path $PSScriptRoot ".vla_lens\server.json"
if (-not (Test-Path $pidFile)) {
    Write-Host "No launcher-managed alice blue process was found."
    exit 0
}
$state = Get-Content $pidFile -Raw | ConvertFrom-Json
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($state.pid)" -ErrorAction SilentlyContinue
if ($process -and $process.CommandLine -match "uvicorn app\.main:app") {
    Stop-Process -Id $state.pid
    Write-Host "alice blue stopped." -ForegroundColor Green
} else {
    Write-Host "The recorded process is no longer running."
}
Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
