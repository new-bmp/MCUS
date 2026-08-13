param(
    [switch]$SkipCatalog
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$WorkspaceRoot = [IO.Path]::GetFullPath((Split-Path -Parent $ProjectRoot))
$ToolRoot = Join-Path $WorkspaceRoot '.tools\android-build'
$SdkRoot = Join-Path $ToolRoot 'sdk'
$JdkRoot = Get-ChildItem -LiteralPath (Join-Path $ToolRoot 'jdk-extracted') -Directory | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'bin\javac.exe') } | Select-Object -First 1 -ExpandProperty FullName
if (-not $JdkRoot) { throw 'Local JDK 17 is missing.' }

$BuildTools = Join-Path $SdkRoot 'build-tools\35.0.0'
$AndroidJar = Join-Path $SdkRoot 'platforms\android-35\android.jar'
$Aapt2 = Join-Path $BuildTools 'aapt2.exe'
$D8 = Join-Path $BuildTools 'd8.bat'
$ZipAlign = Join-Path $BuildTools 'zipalign.exe'
$ApkSigner = Join-Path $BuildTools 'apksigner.bat'
$Javac = Join-Path $JdkRoot 'bin\javac.exe'
$Jar = Join-Path $JdkRoot 'bin\jar.exe'
$KeyTool = Join-Path $JdkRoot 'bin\keytool.exe'
$env:JAVA_HOME = $JdkRoot

foreach ($required in @($AndroidJar, $Aapt2, $D8, $ZipAlign, $ApkSigner, $Javac, $Jar, $KeyTool)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing build dependency: $required" }
}

if (-not $SkipCatalog) {
    $CatalogRoot = Join-Path $WorkspaceRoot 'mcu-l-catalog\data\combined'
    & python (Join-Path $ProjectRoot 'scripts\generate_catalog.py') --catalog $CatalogRoot --output (Join-Path $ProjectRoot 'assets\catalog.js')
    if ($LASTEXITCODE -ne 0) { throw 'Catalog generation failed.' }
}

$BuildRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot 'build'))
if (-not $BuildRoot.StartsWith($ProjectRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe build path.' }
if (Test-Path -LiteralPath $BuildRoot) { Remove-Item -LiteralPath $BuildRoot -Recurse -Force }
$ClassRoot = Join-Path $BuildRoot 'classes'
$DexRoot = Join-Path $BuildRoot 'dex'
$DistRoot = Join-Path $ProjectRoot 'dist'
New-Item -ItemType Directory -Force -Path $BuildRoot, $ClassRoot, $DexRoot, $DistRoot | Out-Null

# AAPT2 on Windows stores nested asset names with backslashes. WebView requests
# URL paths with forward slashes, so stage vendor logos as flat asset names.
$AssetRoot = Join-Path $BuildRoot 'assets'
New-Item -ItemType Directory -Force -Path $AssetRoot | Out-Null
Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'assets') -File | Copy-Item -Destination $AssetRoot
Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'assets\vendor-logos') -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $AssetRoot ("vendor-" + $_.Name))
}

$CompiledResources = Join-Path $BuildRoot 'resources.zip'
& $Aapt2 compile --dir (Join-Path $ProjectRoot 'res') -o $CompiledResources
if ($LASTEXITCODE -ne 0) { throw 'Resource compilation failed.' }

$UnsignedApk = Join-Path $BuildRoot 'MCUS-unsigned.apk'
& $Aapt2 link -o $UnsignedApk --manifest (Join-Path $ProjectRoot 'AndroidManifest.xml') -I $AndroidJar -A $AssetRoot --min-sdk-version 24 --target-sdk-version 35 --version-code 8 --version-name '0.6.1' $CompiledResources
if ($LASTEXITCODE -ne 0) { throw 'APK resource linking failed.' }

$JavaFiles = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'src') -Recurse -Filter '*.java' | Select-Object -ExpandProperty FullName
& $Javac -encoding UTF-8 -source 8 -target 8 -bootclasspath $AndroidJar -d $ClassRoot $JavaFiles
if ($LASTEXITCODE -ne 0) { throw 'Java compilation failed.' }

$ClassFiles = Get-ChildItem -LiteralPath $ClassRoot -Recurse -Filter '*.class' | Select-Object -ExpandProperty FullName
& $D8 --lib $AndroidJar --min-api 24 --output $DexRoot $ClassFiles
if ($LASTEXITCODE -ne 0) { throw 'DEX compilation failed.' }

Push-Location $DexRoot
try {
    & $Jar uf $UnsignedApk 'classes.dex'
    if ($LASTEXITCODE -ne 0) { throw 'Adding classes.dex failed.' }
} finally {
    Pop-Location
}

$AlignedApk = Join-Path $BuildRoot 'MCUS-aligned.apk'
& $ZipAlign -f -p 4 $UnsignedApk $AlignedApk
if ($LASTEXITCODE -ne 0) { throw 'APK alignment failed.' }

$KeyStore = Join-Path $ToolRoot 'mcus-debug.keystore'
if (-not (Test-Path -LiteralPath $KeyStore)) {
    & $KeyTool -genkeypair -keystore $KeyStore -storepass android -alias mcusdebugkey -keypass android -keyalg RSA -keysize 2048 -validity 10000 -dname 'CN=MCUS Debug,O=new.bmp,C=CN'
    if ($LASTEXITCODE -ne 0) { throw 'Debug keystore creation failed.' }
}

$FinalApk = Join-Path $DistRoot 'MCUS-0.6.1-debug.apk'
& $ApkSigner sign --ks $KeyStore --ks-pass pass:android --key-pass pass:android --out $FinalApk $AlignedApk
if ($LASTEXITCODE -ne 0) { throw 'APK signing failed.' }
& $ApkSigner verify --verbose --print-certs $FinalApk
if ($LASTEXITCODE -ne 0) { throw 'APK verification failed.' }

Write-Output "APK=$FinalApk"
Get-Item -LiteralPath $FinalApk | Select-Object FullName, Length, LastWriteTime
