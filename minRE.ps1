$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    $Python = "python"
}

Push-Location $Root
try {
    & $Python -m app.minre @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
