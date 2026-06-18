<#
.SYNOPSIS
    One-time setup for the music visualizer (StreamDiffusion env, optional Qwen sidecar, settings).

.EXAMPLE
    .\setup.ps1

.EXAMPLE
    .\setup.ps1 -Profile Full -PersistEnv

.EXAMPLE
    .\setup.ps1 -UseUpstreamStreamDiffusion

.EXAMPLE
    .\setup.ps1 -Profile Minimal -SkipModels
#>
[CmdletBinding()]
param(
    [ValidateSet("Auto", "Minimal", "Standard", "Full")]
    [string]$Profile = "Auto",

    [switch]$SkipQwen,
    [switch]$SkipTensorRT,
    [switch]$SkipModels,
    [switch]$SkipNativeBuild,
    [switch]$PersistEnv,
    [switch]$ForceSettings,
    [switch]$Update,
    [switch]$Quiet,

    [string]$StreamDiffusionRepo,
    [string]$StreamDiffusionRef,
    [switch]$UseUpstreamStreamDiffusion,
    [switch]$ForceBlackwellFork
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SetupDir = Join-Path $ProjectRoot "scripts\setup"

. (Join-Path $SetupDir "Common.ps1")
. (Join-Path $SetupDir "Prereqs.ps1")
. (Join-Path $SetupDir "GpuDetect.ps1")
. (Join-Path $SetupDir "StreamDiffusion.ps1")
. (Join-Path $SetupDir "MainEnv.ps1")
. (Join-Path $SetupDir "Sidecar.ps1")
. (Join-Path $SetupDir "Models.ps1")
. (Join-Path $SetupDir "Settings.ps1")
. (Join-Path $SetupDir "NativeBuild.ps1")

if (-not $Quiet) {
    Write-Host ""
    Write-Host "Music Visualizer - developer setup" -ForegroundColor Cyan
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
if (-not $Quiet) {
    Write-Host "Using profile: $resolvedProfile" -ForegroundColor White
}

Write-SetupStep "StreamDiffusion"
$sdSource = Resolve-StreamDiffusionRepo -Lock $Lock -Gpu $gpu `
    -RepoOverride $StreamDiffusionRepo -RefOverride $StreamDiffusionRef `
    -UseUpstream:$UseUpstreamStreamDiffusion -ForceBlackwellFork:$ForceBlackwellFork -Quiet:$Quiet
$sdPath = Install-StreamDiffusion -ProjectRoot $ProjectRoot -Lock $Lock `
    -Repo $sdSource.Repo -Ref $sdSource.Ref -Update:$Update
if ($PersistEnv) {
    Set-StreamDiffusionPathPersist -Path $sdPath
}

Write-SetupStep "Main Python environment"
if ($resolvedProfile -eq "Minimal" -and -not $SkipTensorRT) {
    $SkipTensorRT = $true
}
Install-MainEnv -ProjectRoot $ProjectRoot -StreamDiffusionPath $sdPath -Lock $Lock `
    -SkipTensorRT:$SkipTensorRT -UpstreamMode:($sdSource.UpstreamMode)

if (-not $SkipModels) {
    Write-SetupStep "Prefetching models (SD-Turbo + CLAP)"
    Install-Models -StreamDiffusionPath $sdPath -Lock $Lock
}

$wantQwen = ($resolvedProfile -eq "Full") -and -not $SkipQwen
if ($wantQwen) {
    Write-SetupStep "Qwen2-Audio sidecar"
    Install-Sidecar -ProjectRoot $ProjectRoot -Lock $Lock -SkipModelDownload:$SkipModels
}
elseif (-not $SkipQwen -and $resolvedProfile -ne "Full") {
    Write-SetupWarn "Qwen sidecar skipped for profile $resolvedProfile - use -Profile Full to enable"
}

Write-SetupStep "Settings"
Install-Settings -ProjectRoot $ProjectRoot -StreamDiffusionPath $sdPath -Profile $resolvedProfile -Force:$ForceSettings

if (-not $SkipNativeBuild -and $resolvedProfile -ne "Minimal") {
    Write-SetupStep "Native bridges (optional)"
    Install-NativeBridges -StreamDiffusionPath $sdPath -ProjectRoot $ProjectRoot -SkipNativeBuild:$SkipNativeBuild
}

Write-SetupStep "Done"
Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Profile:  $resolvedProfile"
Write-Host "  SD path:  $sdPath"
Write-Host ""
Write-Host "Run the visualizer:" -ForegroundColor White
Write-Host "  .\run.ps1 `"a vivid abstract nebula of swirling neon light`""
Write-Host ""
Write-Host "Health check:" -ForegroundColor White
Write-Host "  .\doctor.ps1"
Write-Host ""
