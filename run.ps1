param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$NoModel,
    [switch]$Foreground,
    [switch]$Setup
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-AlicePython {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return $venvPython
    }
    foreach ($name in @("python", "python3")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $resolved = & $launcher.Source -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            return $resolved.Trim()
        }
    }
    throw "Python 3.12 was not found. Install it first, then run .\run.ps1 -Setup."
}

try {
    if ($Setup -and -not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
        $basePython = Find-AlicePython
        Write-Host "Creating .venv with $basePython" -ForegroundColor Cyan
        & $basePython -m venv ".venv"
        if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv." }
    }

    $python = Find-AlicePython
    if ($Setup) {
        Write-Host "Installing project dependencies..." -ForegroundColor Cyan
        & $python -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
        & $python -m pip install -r "requirements.txt"
        if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
    }

    & $python -c "import fastapi,uvicorn,cv2,mediapipe,ultralytics,httpx,h5py,pyarrow,imageio_ffmpeg,scipy,torch" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Project dependencies are incomplete. Run .\run.ps1 -Setup once."
    }

    if ($Foreground) {
        $arguments = @("-m", "app.cli", "serve", "--port", "$Port")
        if ($NoModel) { $arguments += "--no-model" }
        & $python @arguments
        exit $LASTEXITCODE
    }

    $arguments = @("-m", "app.cli", "start", "--port", "$Port", "--max-port", "$($Port + 10)")
    if (-not $NoBrowser) { $arguments += "--browser" }
    if ($NoModel) { $arguments += "--no-model" }
    & $python @arguments
    exit $LASTEXITCODE
} catch {
    Write-Host "" 
    Write-Host "alice blue failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
