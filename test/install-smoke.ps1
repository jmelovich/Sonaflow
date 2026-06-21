<#
.SYNOPSIS
    Isolated install smoke test — verifies setup.ps1 without touching your dev environment.

.DESCRIPTION
    Creates a throwaway workspace under %TEMP%, copies this repo, runs setup/doctor/(optional)
    benchmark, and writes a JSON report. Does NOT modify visualizer_settings.json or user
    env vars in your dev checkout.

.EXAMPLE
    .\test\install-smoke.ps1

.EXAMPLE
    .\test\install-smoke.ps1 -Profile Minimal -RunBenchmark -KeepWorkdir
#>
[CmdletBinding()]
param(
    [ValidateSet("Minimal", "Standard")]
    [string]$Profile = "Minimal",

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

$report = @{
    started_at = (Get-Date -Format "o")
    profile = $Profile
    workdir = $WorkRoot
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
    $vizDest = Join-Path $WorkRoot "Sonaflow"
    if (Test-Path $vizDest) { Remove-Item $vizDest -Recurse -Force }
    Write-Log "Copying Sonaflow to $vizDest"
    robocopy $RepoRoot $vizDest /E /XD .git .venv .cursor engines "test\sandbox\logs" "test\smoke-results" `
        /XF visualizer_settings.json *.dll /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy Sonaflow failed (exit $LASTEXITCODE)" }
    Add-Step "Stage Sonaflow copy" $true $vizDest

    Set-Location $vizDest
    $setupArgs = @(
        "-Profile", $Profile,
        "-SkipModels",
        "-SkipNativeBuild",
        "-ForceSettings",
        "-Quiet"
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

    Write-Log "Running doctor.ps1"
    $ErrorActionPreference = "Continue"
    & powershell -ExecutionPolicy Bypass -NoProfile -File (Join-Path $vizDest "doctor.ps1") *>&1 | Tee-Object -FilePath $LogFile -Append | Out-Host
    $doctorExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    Add-Step "doctor.ps1" ($doctorExit -eq 0) "exit=$doctorExit"

    if ($doctorExit -ne 0) {
        throw "doctor.ps1 failed with exit $doctorExit"
    }

    if ($RunBenchmark) {
        Write-Log "Running run.ps1 --benchmark 5"
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
