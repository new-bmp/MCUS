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
if (Test-Path -LiteralPath $staticFiles) { Remove-Item -LiteralPath $staticFiles -Recurse -Force }
New-Item -ItemType Directory -Force -Path $staticFiles | Out-Null
Get-ChildItem -LiteralPath $public -File | Copy-Item -Destination $staticFiles -Force
Write-Output "Synced $source to $public and $staticFiles"
