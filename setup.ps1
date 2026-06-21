<#
.SYNOPSIS
    One-time setup for Sonaflow (main .venv, optional Qwen sidecar, settings).

.EXAMPLE
    .\setup.ps1

.EXAMPLE
    .\setup.ps1 -Profile Full -PersistEnv

.EXAMPLE
    .\setup.ps1 -Cuda cu121

.EXAMPLE
    .\setup.ps1 -Profile Minimal -SkipModels
#>
[CmdletBinding()]
param(
    [ValidateSet("Auto", "Minimal", "Standard", "Full")]
    [string]$Profile = "Auto",

    [ValidateSet("cu128", "cu121", "Auto")]
    [string]$Cuda = "Auto",

    [switch]$SkipQwen,
    [switch]$SkipTensorRT,
    [switch]$SkipModels,
    [switch]$SkipNativeBuild,
    [switch]$ForceSettings,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SetupDir = Join-Path $ProjectRoot "scripts\setup"

. (Join-Path $SetupDir "Common.ps1")
. (Join-Path $SetupDir "Prereqs.ps1")
. (Join-Path $SetupDir "GpuDetect.ps1")
. (Join-Path $SetupDir "MainEnv.ps1")
. (Join-Path $SetupDir "Sidecar.ps1")
. (Join-Path $SetupDir "Models.ps1")
. (Join-Path $SetupDir "Settings.ps1")
. (Join-Path $SetupDir "NativeBuild.ps1")

if (-not $Quiet) {
    Write-Host ""
    Write-Host "Sonaflow - developer setup" -ForegroundColor Cyan
    Write-Host "Project: $ProjectRoot"
}

$Lock = Get-SetupLock -ProjectRoot $ProjectRoot

Write-SetupStep "Checking prerequisites"
$prereqs = Test-Prereqs -Strict
if (-not $prereqs.Ok) {
    foreach ($issue in $prereqs.Issues) {
        Write-SetupFail $issue
    }
    exit 1
}

Write-SetupStep "Detecting GPU"
$gpu = Get-GpuInfo
$resolvedProfile = if ($Profile -eq "Auto") { $gpu.RecommendedProfile } else { $Profile }

$resolvedCuda = $Cuda
if ($Cuda -eq "Auto") {
    if ($gpu.Available -and $gpu.IsBlackwellOrNewer) {
        $resolvedCuda = "cu128"
    }
    else {
        $resolvedCuda = "cu121"
    }
}
if (-not $Quiet) {
    Write-Host "Using profile: $resolvedProfile" -ForegroundColor White
    Write-Host "Using CUDA profile: $resolvedCuda" -ForegroundColor White
}

Write-SetupStep "Main Python environment"
if ($resolvedProfile -eq "Minimal" -and -not $SkipTensorRT) {
    $SkipTensorRT = $true
}
Install-MainEnv -ProjectRoot $ProjectRoot -Lock $Lock -Cuda $resolvedCuda -SkipTensorRT:$SkipTensorRT

if (-not $SkipModels) {
    Write-SetupStep "Prefetching models (SD-Turbo + CLAP)"
    Install-Models -ProjectRoot $ProjectRoot -Lock $Lock
}

$wantQwen = ($resolvedProfile -eq "Full") -and -not $SkipQwen
if ($wantQwen) {
    Write-SetupStep "Qwen2-Audio sidecar"
    Install-Sidecar -ProjectRoot $ProjectRoot -Lock $Lock -Cuda $resolvedCuda -SkipModelDownload:$SkipModels
}
elseif (-not $SkipQwen -and $resolvedProfile -ne "Full") {
    Write-SetupWarn "Qwen sidecar skipped for profile $resolvedProfile - use -Profile Full to enable"
}

Write-SetupStep "Settings"
Install-Settings -ProjectRoot $ProjectRoot -Profile $resolvedProfile -Force:$ForceSettings

if (-not $SkipNativeBuild -and $resolvedProfile -ne "Minimal") {
    Write-SetupStep "Native bridges (optional)"
    Install-NativeBridges -ProjectRoot $ProjectRoot -SkipNativeBuild:$SkipNativeBuild
}

Write-SetupStep "Done"
Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Profile: $resolvedProfile"
Write-Host "  CUDA:    $resolvedCuda"
Write-Host ""
Write-Host "Run the visualizer:" -ForegroundColor White
Write-Host "  .\run.ps1 `"a vivid abstract nebula of swirling neon light`""
Write-Host ""
Write-Host "Health check:" -ForegroundColor White
Write-Host "  .\doctor.ps1"
Write-Host ""
