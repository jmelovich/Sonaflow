<#
.SYNOPSIS
    Isolated install smoke test — verifies setup.ps1 without touching your dev environment.

.DESCRIPTION
    Creates a throwaway workspace under %TEMP%, copies this repo (+ optional StreamDiffusion
    source), runs setup/doctor/(optional) benchmark, and writes a JSON report.
    Does NOT modify visualizer_settings.json, user env vars, or your dev checkouts.

.EXAMPLE
    .\test\install-smoke.ps1

.EXAMPLE
    .\test\install-smoke.ps1 -Profile Minimal -RunBenchmark -KeepWorkdir

.EXAMPLE
    .\test\install-smoke.ps1 -CloneStreamDiffusion   # test git clone path (needs network)
#>
[CmdletBinding()]
param(
    [ValidateSet("Minimal", "Standard")]
    [string]$Profile = "Minimal",

    [switch]$CloneStreamDiffusion,
    [string]$StreamDiffusionSource = "",
    [switch]$RunBenchmark,
    [switch]$KeepWorkdir,
    [string]$Workdir = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ReportDir = Join-Path $RepoRoot "test\smoke-results"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$WorkRoot = if ($Workdir) { $Workdir } else { Join-Path $env:TEMP "viz-install-smoke-$Timestamp" }
$LogFile = Join-Path $ReportDir "smoke-$Timestamp.log"
$ResultFile = Join-Path $ReportDir "smoke-$Timestamp.json"

# Preserve host environment (never persist changes to User/Machine).
$SavedSdPath = $env:STREAMDIFFUSION_PATH
if (-not $SavedSdPath) {
    $SavedSdPath = [Environment]::GetEnvironmentVariable("STREAMDIFFUSION_PATH", "User")
}

$report = @{
    started_at = (Get-Date -Format "o")
    profile = $Profile
    workdir = $WorkRoot
    clone_streamdiffusion = [bool]$CloneStreamDiffusion
    steps = @()
    success = $false
    error = $null
}

function Add-Step {
    param([string]$Name, [bool]$Ok, [string]$Detail = "")
    $report.steps += @{ name = $Name; ok = $Ok; detail = $Detail; at = (Get-Date -Format "o") }
    $mark = if ($Ok) { "PASS" } else { "FAIL" }
    $color = if ($Ok) { "Green" } else { "Red" }
    Write-Host ("  [{0}] {1}" -f $mark, $Name) -ForegroundColor $color
    if ($Detail) { Write-Host ("         {0}" -f $Detail) -ForegroundColor DarkGray }
}

function Write-Log {
    param([string]$Message)
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $Message"
    Add-Content -Path $LogFile -Value $line -Encoding utf8
    Write-Host $line
}

New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null

Write-Host ""
Write-Host "Install smoke test (isolated)" -ForegroundColor Cyan
Write-Host "  Workdir:  $WorkRoot"
Write-Host "  Profile:  $Profile"
Write-Host "  Report:   $ResultFile"
Write-Host ""
Write-Log "Starting isolated smoke test"

try {
    # --- Stage visualizer (no .git, no local settings, no sidecar venv) ---
    $vizDest = Join-Path $WorkRoot "Sonaflow"
    if (Test-Path $vizDest) { Remove-Item $vizDest -Recurse -Force }
    Write-Log "Copying visualizer to $vizDest"
    robocopy $RepoRoot $vizDest /E /XD .git .venv-qwen .cursor engines "test\sandbox\logs" "test\smoke-results" `
        /XF visualizer_settings.json *.dll /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy visualizer failed (exit $LASTEXITCODE)" }
    Add-Step "Stage visualizer copy" $true $vizDest

    # --- Stage StreamDiffusion ---
    $sdDest = Join-Path $WorkRoot "StreamDiffusion"
    if ($CloneStreamDiffusion) {
        Write-Log "Cloning StreamDiffusion from setup.lock.json ..."
        $lock = Get-Content (Join-Path $vizDest "setup.lock.json") -Raw | ConvertFrom-Json
        if (Test-Path $sdDest) { Remove-Item $sdDest -Recurse -Force }
        & git clone --branch $lock.streamdiffusion_ref --depth 1 $lock.streamdiffusion_repo $sdDest
        if ($LASTEXITCODE -ne 0) { throw "git clone StreamDiffusion failed" }
        Add-Step "Clone StreamDiffusion" $true "$($lock.streamdiffusion_repo) @ $($lock.streamdiffusion_ref)"
    }
    else {
        if (-not $StreamDiffusionSource) {
            $sibling = Join-Path (Split-Path -Parent $RepoRoot) "StreamDiffusion"
            if (Test-Path (Join-Path $sibling "pyproject.toml")) {
                $StreamDiffusionSource = $sibling
            }
        }
        if (-not $StreamDiffusionSource -or -not (Test-Path $StreamDiffusionSource)) {
            throw "StreamDiffusion source not found (use -CloneStreamDiffusion, -StreamDiffusionSource, or clone ../StreamDiffusion)"
        }
        Write-Log "Copying StreamDiffusion from $StreamDiffusionSource"
        if (Test-Path $sdDest) { Remove-Item $sdDest -Recurse -Force }
        robocopy $StreamDiffusionSource $sdDest /E /XD .git engines /XF *.engine /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy StreamDiffusion failed (exit $LASTEXITCODE)" }
        Add-Step "Stage StreamDiffusion copy" $true $sdDest
    }

    # Process-scoped only — never write to User/Machine env.
    $env:STREAMDIFFUSION_PATH = $sdDest

    # --- setup.ps1 ---
    Set-Location $vizDest
    $setupArgs = @(
        "-Profile", $Profile,
        "-SkipModels",
        "-SkipNativeBuild",
        "-ForceSettings",
        "-Quiet",
        "-ForceBlackwellFork"
    )
    if ($Profile -eq "Minimal") {
        $setupArgs += "-SkipTensorRT"
    }
    Write-Log "Running setup.ps1 $($setupArgs -join ' ')"
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & powershell -ExecutionPolicy Bypass -NoProfile -File (Join-Path $vizDest "setup.ps1") @setupArgs *>&1 | Tee-Object -FilePath $LogFile -Append | Out-Host
    $setupExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    Add-Step "setup.ps1" ($setupExit -eq 0) "exit=$setupExit"

    if ($setupExit -ne 0) {
        throw "setup.ps1 failed with exit $setupExit"
    }

    # --- doctor.ps1 (scoped to temp SD path via env in child process) ---
    Write-Log "Running doctor.ps1"
    $env:STREAMDIFFUSION_PATH = $sdDest
    $ErrorActionPreference = "Continue"
    & powershell -ExecutionPolicy Bypass -NoProfile -File (Join-Path $vizDest "doctor.ps1") *>&1 | Tee-Object -FilePath $LogFile -Append | Out-Host
    $doctorExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    Add-Step "doctor.ps1" ($doctorExit -eq 0) "exit=$doctorExit"

    if ($doctorExit -ne 0) {
        throw "doctor.ps1 failed with exit $doctorExit"
    }

    # --- optional benchmark (headless, no models prefetch needed but SD-Turbo downloads on first run) ---
    if ($RunBenchmark) {
        Write-Log "Running run.ps1 --benchmark 5"
        $env:STREAMDIFFUSION_PATH = $sdDest
        $ErrorActionPreference = "Continue"
        & powershell -ExecutionPolicy Bypass -NoProfile -File (Join-Path $vizDest "run.ps1") `
            --acceleration none --benchmark 5 --no-qwen --no-clap --no-upscale --audio-source none `
            *>&1 | Tee-Object -FilePath $LogFile -Append | Out-Host
        $benchExit = $LASTEXITCODE
        $ErrorActionPreference = $prevEap
        Add-Step "run.ps1 --benchmark 5" ($benchExit -eq 0) "exit=$benchExit"
    }

    $failed = @($report.steps | Where-Object { -not $_.ok })
    $report.success = ($failed.Count -eq 0)
    $report.finished_at = (Get-Date -Format "o")
}
catch {
    $report.error = $_.Exception.Message
    $report.success = $false
    $report.finished_at = (Get-Date -Format "o")
    Write-Log "FATAL: $_"
}
finally {
    # Restore host env exactly as before.
    if ($SavedSdPath) {
        $env:STREAMDIFFUSION_PATH = $SavedSdPath
    }
    else {
        Remove-Item Env:STREAMDIFFUSION_PATH -ErrorAction SilentlyContinue
    }

    $report | ConvertTo-Json -Depth 6 | Set-Content -Path $ResultFile -Encoding utf8

    if (-not $KeepWorkdir -and -not $Workdir) {
        if (Test-Path $WorkRoot) {
            Write-Log "Removing workdir $WorkRoot"
            Remove-Item $WorkRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    elseif ($KeepWorkdir) {
        Write-Host "Workdir kept: $WorkRoot" -ForegroundColor DarkGray
    }

    Write-Host ""
    if ($report.success) {
        Write-Host "SMOKE TEST PASSED" -ForegroundColor Green
        Write-Host "Report: $ResultFile"
        exit 0
    }
    else {
        Write-Host "SMOKE TEST FAILED" -ForegroundColor Red
        Write-Host "Log:    $LogFile"
        Write-Host "Report: $ResultFile"
        if (-not $KeepWorkdir) {
            Write-Host "Re-run with -KeepWorkdir to inspect the temp workspace." -ForegroundColor Yellow
        }
        exit 1
    }
}
