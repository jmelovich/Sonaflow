<#
.SYNOPSIS
    Build a self-contained offline Sonaflow installer (no remote host).

.DESCRIPTION
    Two steps (much faster than embedding 50k+ files directly in Inno):

    1. Build-Payloads.ps1  - zip the stage (+ optional legacy model bundle) into archives
    2. ISCC                - embed those zips with store compression (~1-5 min)

    By default, public HuggingFace models (SD-Turbo, CLAP, taesd, Qwen) are downloaded
    during install — not bundled. Use -BundleModels only for a fully offline model cache.

    Output: installer\build\dist\Sonaflow-Setup-<ver>-offline.exe (+ .bin if >4 GB).
    Copy the entire dist\ folder to the target PC and run the .exe.

.EXAMPLE
    .\Build-OfflineInstaller.ps1 -Version 0.1.0

.EXAMPLE
    .\Build-OfflineInstaller.ps1 -Version 0.1.0 -BundleModels
    # Legacy: embed model zips in the installer instead of downloading during install.

.EXAMPLE
    .\Build-OfflineInstaller.ps1 -Version 0.1.0 -ReusePayloads
    # Skip re-zipping if payloads already exist from a prior run.
#>
[CmdletBinding()]
param(
    [string]$Version = "0.1.0",
    [string]$StageDir,
    [switch]$BundleModels,
    [switch]$SkipQwen,
    [switch]$RefreshQwen,
    [switch]$ReusePayloads,
    [switch]$SkipCompile,
    [string]$IsccPath
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "Common.ps1")

if (-not $StageDir) {
    $StageDir = Join-Path $ScriptDir "..\build\stage"
}

$InstallerRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$StageDir = (Resolve-Path $StageDir).Path
$PayloadDir = Join-Path $InstallerRoot "build\payloads"
New-Item -ItemType Directory -Force -Path $PayloadDir | Out-Null
$PayloadDir = (Resolve-Path $PayloadDir).Path

Write-Step "Checking staged install layout"
if (-not (Test-Path (Join-Path $StageDir "Sonaflow.exe"))) {
    throw @"
Staged layout not found at: $StageDir

Run Milestone 1 first:
  cd installer\tools
  .\Freeze-Env.ps1 -StreamDiffusionPath <path> -Dest ..\build\runtime\main-env
  .\Stage-Local.ps1 -StreamDiffusionPath <path> -FrozenMainEnv ..\build\runtime\main-env -StageDir ..\build\stage
"@
}
Write-Ok "Stage: $StageDir ($([math]::Round((Get-ChildItem $StageDir -Recurse | Measure-Object Length -Sum).Sum / 1GB, 1)) GB)"

$includeModels = $BundleModels.IsPresent
if ($includeModels) {
    $ModelsHubDir = "$env:LOCALAPPDATA\Sonaflow\models\hub"
    if (-not (Test-Path $ModelsHubDir)) {
        Write-Warn2 "Models hub not found at: $ModelsHubDir"
        Write-Warn2 "Continuing without bundled models (installer will download during install)."
        $includeModels = $false
    }
    else {
        $ModelsHubDir = (Resolve-Path $ModelsHubDir).Path
        Write-Ok "Bundling models from: $ModelsHubDir"
    }
}

$includeQwen = -not $SkipQwen
$ProjectRoot = (Resolve-Path (Join-Path $InstallerRoot "..")).Path
$QwenEnvDir = Join-Path $InstallerRoot "build\runtime\qwen-env"

if ($includeQwen) {
    $srcQwen = Join-Path $ProjectRoot ".venv-qwen"
    if (-not (Test-Path $srcQwen)) {
        Write-Warn2 "Qwen dev venv not found at $srcQwen - run setup.ps1 -Profile Full first."
        Write-Warn2 "Continuing without Qwen component."
        $includeQwen = $false
    }
    elseif (-not (Test-Path $QwenEnvDir) -or $RefreshQwen) {
        Write-Step "Freezing Qwen sidecar runtime"
        & (Join-Path $ScriptDir "Freeze-QwenEnv.ps1") `
            -ProjectRoot $ProjectRoot -Dest $QwenEnvDir
    }
    else {
        Write-Ok "Frozen qwen-env: $QwenEnvDir"
    }
}

$coreZip = Join-Path $PayloadDir "core-runtime-$Version.zip"
$modelsZip = Join-Path $PayloadDir "models-core-$Version.zip"
$qwenZip = Join-Path $PayloadDir "qwen-runtime-$Version.zip"
$needPayloads = -not $ReusePayloads
if ($ReusePayloads) {
    if (-not (Test-Path $coreZip)) { $needPayloads = $true }
    if ($includeModels -and -not (Test-Path $modelsZip)) { $needPayloads = $true }
    if ($includeQwen -and -not (Test-Path $qwenZip)) { $needPayloads = $true }
    if (-not $needPayloads) { Write-Ok "Reusing existing payload zips in $PayloadDir" }
}

if ($needPayloads) {
    $onlyQwen = (Test-Path $coreZip) `
        -and ((-not $includeModels) -or (Test-Path $modelsZip)) `
        -and $includeQwen -and (-not (Test-Path $qwenZip))
    if ($onlyQwen) {
        Write-Step "Packaging qwen-runtime zip only (core/models payloads reused)"
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $tmpQ = Join-Path $env:TEMP ("sf-qwen-" + [guid]::NewGuid().ToString("N"))
        $qDst = Join-Path $tmpQ "runtime\qwen-env"
        New-Item -ItemType Directory -Force -Path (Split-Path $qDst) | Out-Null
        Copy-Item $QwenEnvDir $qDst -Recurse -Force
        if (Test-Path $qwenZip) { Remove-Item $qwenZip -Force }
        [System.IO.Compression.ZipFile]::CreateFromDirectory(
            $tmpQ, $qwenZip, [System.IO.Compression.CompressionLevel]::Optimal, $false)
        Remove-Item $tmpQ -Recurse -Force -ErrorAction SilentlyContinue
        Write-Ok "qwen-runtime -> $(Split-Path $qwenZip -Leaf)"
    }
    else {
        Write-Step "Creating payload ZIPs"
        $payloadArgs = @{
            StageDir     = $StageDir
            Version      = $Version
            BaseUrl      = "local"
            OutDir       = $PayloadDir
            GeneratedDir = (Join-Path $InstallerRoot "generated")
        }
        if ($includeModels) { $payloadArgs['ModelsHubDir'] = $ModelsHubDir }
        if ($includeQwen) { $payloadArgs['QwenEnvDir'] = $QwenEnvDir }
        & (Join-Path $ScriptDir "Build-Payloads.ps1") @payloadArgs
        if (-not (Test-Path $coreZip)) { throw "Build-Payloads did not produce $coreZip" }
    }
}
else {
    if (-not (Test-Path $coreZip)) { throw "Missing $coreZip" }
}

# --- VC++ redist ---
$redistDir = Join-Path $InstallerRoot "redist"
$redistExe = Join-Path $redistDir "vc_redist.x64.exe"
if (-not (Test-Path $redistExe)) {
    Write-Step "Downloading Visual C++ redistributable (~25 MB)"
    New-Item -ItemType Directory -Force -Path $redistDir | Out-Null
    Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" `
        -OutFile $redistExe -UseBasicParsing
}
Write-Ok "VC++ redist ready"

if ($SkipCompile) {
    Write-Warn2 "-SkipCompile set; payload zips ready in $PayloadDir"
    return
}

if (-not $IsccPath) {
    foreach ($c in @(
            "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "C:\Program Files\Inno Setup 7\ISCC.exe")) {
        if (Test-Path $c) { $IsccPath = $c; break }
    }
}
if (-not $IsccPath) { throw "ISCC.exe not found. Install Inno Setup 6+." }
Write-Ok "ISCC: $IsccPath"

$payloadIss = ($PayloadDir -replace '\\', '/')
$defines = @(
    "/DAppVersion=$Version",
    "/DPayloadDir=$payloadIss"
)
if ($includeModels) { $defines += "/DIncludeModels" }
if ($includeQwen) { $defines += "/DIncludeQwen" }

Write-Step "Compiling installer (embeds pre-built ZIPs; usually 1-5 minutes)"
Push-Location $InstallerRoot
try {
    & $IsccPath @defines (Join-Path $InstallerRoot "sonaflow-offline.iss")
    if ($LASTEXITCODE -ne 0) { throw "ISCC failed (exit $LASTEXITCODE)" }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Offline installer ready in:" -ForegroundColor Green
Write-Host "  $(Join-Path $InstallerRoot 'build\dist')" -ForegroundColor White
Write-Host ""
Write-Host "Copy ALL files in that folder (Setup.exe + any .bin files) to the target PC." -ForegroundColor White
Write-Host "Then run Sonaflow-Setup-$Version-offline.exe" -ForegroundColor White
Write-Host "Optional setup notes (RTX Video SDK, etc.): {install}\docs\OPTIONAL_SETUP.txt" -ForegroundColor White
