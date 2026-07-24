$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$candidates = @(
    (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    (Get-Command python3 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

if (-not $candidates) {
    Write-Host "Python was not found; no managed service could be stopped." -ForegroundColor Yellow
    exit 1
}

& $candidates[0] -m app.cli stop
exit $LASTEXITCODE
