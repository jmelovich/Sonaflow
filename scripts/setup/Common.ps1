# Shared helpers for setup.ps1, doctor.ps1, and run.ps1.

function Get-ProjectRoot {
    param([string]$StartPath = $PSScriptRoot)
    $dir = if ($StartPath) { $StartPath } else { $PSScriptRoot }
    while ($dir) {
        if (Test-Path (Join-Path $dir "music_visualizer\main.py")) {
            return (Resolve-Path $dir).Path
        }
        $parent = Split-Path -Parent $dir
        if (-not $parent -or $parent -eq $dir) { break }
        $dir = $parent
    }
    throw "Could not locate Sonaflow repo root (music_visualizer\main.py not found)."
}

function Get-SetupLock {
    param([string]$ProjectRoot)
    $lockPath = Join-Path $ProjectRoot "setup.lock.json"
    if (-not (Test-Path $lockPath)) {
        throw "setup.lock.json not found at $lockPath"
    }
    return Get-Content $lockPath -Raw | ConvertFrom-Json
}

function Write-SetupStep {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Write-SetupOk {
    param([string]$Message)
    Write-Host "  OK  $Message" -ForegroundColor Green
}

function Write-SetupWarn {
    param([string]$Message)
    Write-Host "  WARN  $Message" -ForegroundColor Yellow
}

function Write-SetupFail {
    param([string]$Message)
    Write-Host "  FAIL  $Message" -ForegroundColor Red
}
