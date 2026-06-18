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

function Resolve-StreamDiffusionPath {
    param([string]$ProjectRoot)
    if ($env:STREAMDIFFUSION_PATH -and (Test-Path (Join-Path $env:STREAMDIFFUSION_PATH "pyproject.toml"))) {
        return (Resolve-Path $env:STREAMDIFFUSION_PATH).Path
    }
    $sibling = Join-Path (Split-Path -Parent $ProjectRoot) "StreamDiffusion"
    if (Test-Path $sibling) {
        return (Resolve-Path $sibling).Path
    }
    $inRepo = Join-Path $ProjectRoot "StreamDiffusion"
    if (Test-Path $inRepo) {
        return (Resolve-Path $inRepo).Path
    }
    return $sibling
}

function Set-StreamDiffusionEnv {
    param([string]$Path)
    $env:STREAMDIFFUSION_PATH = $Path
    $env:PYTHONUTF8 = "1"
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
