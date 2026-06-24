<#

.SYNOPSIS

    Build a self-contained offline Sonaflow installer (no remote host).



.DESCRIPTION

    Builds (or reuses) a CUDA-specific package venv, freezes it, stages the install

    layout, zips payloads, and compiles the Inno Setup installer.



    Requires NV_RTX_VIDEO_SDK on the build machine (RTX Video SDK with

    include/nvsdk_ngx.h) so nvngx_vsr.dll can be bundled for end-user upscaling.



    Use -Cuda to target a GPU profile without changing your dev .venv:

      cu128  -> harvests .venv (default dev env)

      cu121  -> builds/harvests .venv-cu121 (isolated from dev)



.EXAMPLE

    .\Build-OfflineInstaller.ps1 -Version 0.1.0



.EXAMPLE

    .\Build-OfflineInstaller.ps1 -Version 0.1.0 -Cuda cu121

    # RTX 30/40 installer; creates .venv-cu121 if missing, leaves .venv untouched.



.EXAMPLE

    .\Build-OfflineInstaller.ps1 -Version 0.1.0 -Cuda cu121 -RefreshEnv -RefreshStage

    # Force rebuild of package venv, frozen runtime, and stage.



.EXAMPLE

    .\Build-OfflineInstaller.ps1 -Version 0.1.0 -ReusePayloads

    # Skip re-zipping if payloads already exist from a prior run.

#>

[CmdletBinding()]

param(

    [string]$Version = "0.1.0",

    [ValidateSet("cu128", "cu121")]

    [string]$Cuda = "cu128",

    [string]$StageDir,

    [switch]$BundleModels,

    [switch]$SkipQwen,

    [switch]$RefreshEnv,

    [switch]$RefreshStage,

    [switch]$ReusePayloads,

    [switch]$SkipCompile,

    [string]$IsccPath

)



$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

. (Join-Path $ScriptDir "Common.ps1")

. (Join-Path $ScriptDir "PackageEnv.ps1")



$InstallerRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path

$ProjectRoot = (Resolve-Path (Join-Path $InstallerRoot "..")).Path

$paths = Get-InstallerBuildPaths -InstallerRoot $InstallerRoot -Version $Version -Cuda $Cuda



if (-not $StageDir) {

    $StageDir = $paths.StageDir

}

else {

    $StageDir = (Resolve-Path $StageDir).Path

}



$PayloadDir = $paths.PayloadDir

New-Item -ItemType Directory -Force -Path $PayloadDir, $paths.DistDir | Out-Null

$PayloadDir = (Resolve-Path $PayloadDir).Path



Write-Step "CUDA profile: $Cuda (package venv: $(Get-PackageVenvRelName -Cuda $Cuda))"

Write-Ok "Stage target: $StageDir"

Write-Ok "Output: $($paths.OutputBaseName).exe (+ .bin if >4 GB)"



function Test-FullRtxVideoSdkRoot {
    param([string]$Root)
    if (-not $Root) { return $false }
    return Test-Path (Join-Path $Root "include\nvsdk_ngx.h")
}

Write-Step "Preflight: RTX Video SDK (required to bundle VSR upscaling)"
if (-not (Test-FullRtxVideoSdkRoot $env:NV_RTX_VIDEO_SDK)) {
    throw @"
NV_RTX_VIDEO_SDK is not set or does not point at a valid RTX Video SDK install.

Download and extract the SDK from:
  https://developer.nvidia.com/rtx-video-sdk

Then set (User environment variable recommended):
  NV_RTX_VIDEO_SDK=C:\RTX-Video-SDK

Current value: $(if ($env:NV_RTX_VIDEO_SDK) { $env:NV_RTX_VIDEO_SDK } else { '(not set)' })
"@
}
Write-Ok "NV_RTX_VIDEO_SDK=$($env:NV_RTX_VIDEO_SDK)"



$stageReady = (Test-Path (Join-Path $StageDir "Sonaflow.exe"))

if (-not $stageReady -or $RefreshStage) {

    Write-Step "Ensuring package Python environment"

    $null = Ensure-PackageMainEnv -ProjectRoot $ProjectRoot -Cuda $Cuda -Refresh:$RefreshEnv



    Write-Step "Freezing main runtime ($Cuda)"

    if ((Test-Path $paths.MainEnvDir) -and $RefreshStage) {

        Remove-Item $paths.MainEnvDir -Recurse -Force -ErrorAction SilentlyContinue

    }

    & (Join-Path $ScriptDir "Freeze-Env.ps1") -ProjectRoot $ProjectRoot -Cuda $Cuda -Dest $paths.MainEnvDir

    Write-Step "Staging install layout ($Cuda)"
    & (Join-Path $ScriptDir "Stage-Local.ps1") -FrozenMainEnv $paths.MainEnvDir -StageDir $StageDir

    $stageReady = $true

}

elseif (-not $stageReady) {

    throw @"

Staged layout not found at: $StageDir



Run with -RefreshStage, or manually:

  cd installer\tools

  .\Build-OfflineInstaller.ps1 -Version $Version -Cuda $Cuda -RefreshStage

"@

}

else {

    Write-Ok "Reusing stage: $StageDir"

}



Write-Step "Checking staged install layout"

Write-Ok "Stage: $StageDir ($([math]::Round((Get-ChildItem $StageDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1GB, 1)) GB)"



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

$coreZip = $paths.CoreZip
$modelsZip = $paths.ModelsZip
$needPayloads = -not $ReusePayloads
if ($ReusePayloads) {
    if (-not (Test-Path $coreZip)) { $needPayloads = $true }
    if ($includeModels -and -not (Test-Path $modelsZip)) { $needPayloads = $true }
    if (-not $needPayloads) { Write-Ok "Reusing existing payload zips in $PayloadDir" }
}

if ($needPayloads) {
    Write-Step "Creating payload ZIPs"
    $payloadArgs = @{
        StageDir       = $StageDir
        Version        = $Version
        BaseUrl        = "local"
        OutDir         = $PayloadDir
        GeneratedDir   = (Join-Path $InstallerRoot "generated")
        PayloadSuffix  = $paths.PayloadSuffix
        TorchCu        = $Cuda
    }
    if ($Cuda -eq "cu121") { $payloadArgs['GpuArch'] = "sm_89" }
    if ($includeModels) { $payloadArgs['ModelsHubDir'] = $ModelsHubDir }
    if ($includeQwen) { $payloadArgs['IncludeQwen'] = $true }
    & (Join-Path $ScriptDir "Build-Payloads.ps1") @payloadArgs
    if (-not (Test-Path $coreZip)) { throw "Build-Payloads did not produce $coreZip" }
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
if ($paths.IsccCudaSuffix) { $defines += "/DCudaSuffix=$($paths.IsccCudaSuffix)" }

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

Write-Host "  $($paths.DistDir)" -ForegroundColor White

Write-Host ""

Write-Host "CUDA profile: $Cuda" -ForegroundColor White

Write-Host "Installer:    $($paths.OutputBaseName).exe (+ .bin files if >4 GB)" -ForegroundColor White

Write-Host ""

Write-Host "Copy the entire folder above to the target PC, then run $($paths.OutputBaseName).exe" -ForegroundColor White

