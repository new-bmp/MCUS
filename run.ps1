param(
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Get-VlaHealth([int]$CandidatePort) {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$CandidatePort/api/health" -TimeoutSec 3
        if ($response.ok -eq $true -and $response.service -eq "vla-lens") {
            return $response
        }
    } catch {
        return $null
    }
    return $null
}

function Open-VlaBrowser([string]$Url) {
    if ($NoBrowser) {
        return
    }
    $separator = if ($Url.Contains("?")) { "&" } else { "?" }
    $launchUrl = "$Url${separator}launch=$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
    try {
        Start-Process $launchUrl
    } catch {
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "start", '""', $launchUrl -WindowStyle Hidden
    }
}

try {
    $python = Get-Command python -ErrorAction Stop
    $runtimeDir = Join-Path $PSScriptRoot ".vla_lens"
    New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

    # Reuse a healthy existing alice blue service.
    $existingHealth = Get-VlaHealth $Port
    if ($existingHealth) {
        $url = "http://127.0.0.1:$Port/"
        Write-Host "alice blue is already running." -ForegroundColor Green
        Write-Host $url -ForegroundColor Cyan
        Open-VlaBrowser $url
        exit 0
    }

    # If the preferred port belongs to another application, select the next free port.
    $selectedPort = $null
    foreach ($candidate in $Port..($Port + 10)) {
        $listener = Get-NetTCPConnection -LocalPort $candidate -State Listen -ErrorAction SilentlyContinue
        if (-not $listener) {
            $selectedPort = $candidate
            break
        }
        $candidateHealth = Get-VlaHealth $candidate
        if ($candidateHealth) {
            $url = "http://127.0.0.1:$candidate/"
            Write-Host "alice blue is already running." -ForegroundColor Green
            Write-Host $url -ForegroundColor Cyan
            Open-VlaBrowser $url
            exit 0
        }
    }
    if ($null -eq $selectedPort) {
        throw "No free port was found between $Port and $($Port + 10)."
    }

    Write-Host "Checking Python dependencies..." -ForegroundColor Cyan
    & $python.Source -c "import fastapi, uvicorn, cv2, ultralytics, httpx, h5py, pyarrow, imageio_ffmpeg, scipy" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing missing dependencies. This may take several minutes..." -ForegroundColor Yellow
        & $python.Source -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
        if ($LASTEXITCODE -ne 0) {
            throw "Dependency installation failed."
        }
    }

    $stdoutLog = Join-Path $runtimeDir "server.out.log"
    $stderrLog = Join-Path $runtimeDir "server.err.log"
    $pidFile = Join-Path $runtimeDir "server.json"
    $url = "http://127.0.0.1:$selectedPort/"

    Write-Host "Starting alice blue on port $selectedPort..." -ForegroundColor Cyan
    Write-Host "The first model warm-up can take 20-60 seconds." -ForegroundColor DarkGray
    $process = Start-Process `
        -FilePath $python.Source `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$selectedPort") `
        -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -WindowStyle Hidden `
        -PassThru

    @{ pid = $process.Id; port = $selectedPort; url = $url; started_at = (Get-Date).ToString("o") } |
        ConvertTo-Json | Set-Content -Path $pidFile -Encoding UTF8

    $health = $null
    for ($attempt = 1; $attempt -le 360; $attempt++) {
        Start-Sleep -Milliseconds 500
        $process.Refresh()
        if ($process.HasExited) {
            throw "The backend process exited during startup."
        }
        $health = Get-VlaHealth $selectedPort
        if ($health) {
            break
        }
        if ($attempt % 10 -eq 0) {
            Write-Host "Waiting for model and API... $([math]::Round($attempt / 2))s" -ForegroundColor DarkGray
        }
    }

    if (-not $health) {
        throw "The service did not become ready within 180 seconds."
    }

    Write-Host "alice blue is ready." -ForegroundColor Green
    Write-Host "Model: $($health.models.local.family) / loaded=$($health.models.local.loaded)" -ForegroundColor Green
    Write-Host $url -ForegroundColor Cyan
    Open-VlaBrowser $url
    Start-Sleep -Seconds 2
    exit 0
} catch {
    Write-Host "" 
    Write-Host "alice blue failed to start: $($_.Exception.Message)" -ForegroundColor Red
    $errorLog = Join-Path $PSScriptRoot ".vla_lens\server.err.log"
    if (Test-Path $errorLog) {
        Write-Host "" 
        Write-Host "Last server log entries:" -ForegroundColor Yellow
        Get-Content $errorLog -Tail 30
    }
    exit 1
}
