<#
.SYNOPSIS
    Launch the realtime music visualizer inside the StreamDiffusion Poetry venv.

.EXAMPLE
    .\run.ps1 "a vivid abstract nebula of swirling neon light"

.EXAMPLE
    .\run.ps1 --acceleration none --benchmark 30

.NOTES
    All arguments are passed straight through to music_visualizer/main.py.
    Run .\setup.ps1 first on a fresh machine.
#>

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $projectRoot "scripts\setup\Common.ps1")

$sdPath = Resolve-StreamDiffusionPath -ProjectRoot $projectRoot
if (-not (Test-Path $sdPath)) {
    Write-Host "StreamDiffusion not found at: $sdPath" -ForegroundColor Red
    Write-Host "Run .\setup.ps1 from the repo root first, or set STREAMDIFFUSION_PATH." -ForegroundColor Yellow
    exit 1
}
Set-StreamDiffusionEnv -Path $sdPath

$mainScript = Join-Path $projectRoot "music_visualizer\main.py"
$requirements = Join-Path $projectRoot "music_visualizer\requirements.txt"

Push-Location $sdPath
try {
    # One-time / idempotent: ensure pinned deps are present. Probes both the
    # numpy 1.x pin (TensorRT/onnxruntime) and aubio (milestone 2 audio backend);
    # if either is missing/wrong, reinstall the whole requirements file.
    $probe = 'import numpy, aubio, sounddevice, soundfile; raise SystemExit(0 if numpy.__version__.startswith("1.") else 1)'
    if ($IsWindows -or $env:OS -match "Windows") {
        $probe = 'import numpy, aubio, sounddevice, soundfile, pyaudiowpatch; raise SystemExit(0 if numpy.__version__.startswith("1.") else 1)'
    }
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & poetry run python -c $probe 2>&1 | Out-Null
    $probeOk = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prevEap
    if (-not $probeOk) {
        Write-Host "Installing pinned deps (numpy 1.26.4 + audio backend)..." -ForegroundColor Yellow
        poetry run pip install -r $requirements --quiet
    }

    poetry run python $mainScript @args
}
finally {
    Pop-Location
}
