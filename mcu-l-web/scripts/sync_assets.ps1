param(
    [string]$AndroidAssets = "..\..\mcu-l-android\assets"
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$source = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $AndroidAssets))
$public = Join-Path $root "public"
$staticFiles = Join-Path $root "staticfiles"
if (-not (Test-Path -LiteralPath $source)) { throw "Android assets directory not found: $source" }
if (Test-Path -LiteralPath $public) { Remove-Item -LiteralPath $public -Recurse -Force }
New-Item -ItemType Directory -Force -Path $public | Out-Null
Get-ChildItem -LiteralPath $source -File | Copy-Item -Destination $public -Force
$logos = Join-Path $source "vendor-logos"
Get-ChildItem -LiteralPath $logos -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $public ("vendor-" + $_.Name)) -Force
}
if (Test-Path -LiteralPath $staticFiles) {
    # A running static server can keep the directory handle open on Windows.
    # Preserve the root and replace its contents so deployment sync still works.
    Get-ChildItem -LiteralPath $staticFiles -Force | Remove-Item -Recurse -Force
} else {
    New-Item -ItemType Directory -Force -Path $staticFiles | Out-Null
}
Get-ChildItem -LiteralPath $public -File | Where-Object { $_.Name -notin @("catalog.js", "index.html") } | Copy-Item -Destination $staticFiles -Force
python (Join-Path $PSScriptRoot "build_staticfiles.py") --source $public --output $staticFiles
if ($LASTEXITCODE -ne 0) { throw 'Static catalog bundle generation failed.' }
@'
/*.js
  Content-Type: application/javascript; charset=utf-8
  Cache-Control: public, max-age=3600

/index.html
  Cache-Control: no-cache
'@ | Set-Content -LiteralPath (Join-Path $staticFiles "_headers") -Encoding UTF8
$zipPath = Join-Path $root "MCUS-staticfiles.zip"
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Compress-Archive -Path (Join-Path $staticFiles "*") -DestinationPath $zipPath -CompressionLevel Optimal
Write-Output "Synced $source to $public and $staticFiles"
Write-Output "Deploy archive: $zipPath"
