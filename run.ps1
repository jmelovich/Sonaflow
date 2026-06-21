<#
.SYNOPSIS
    Launch the realtime music visualizer inside Sonaflow's .venv.

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
. (Join-Path $projectRoot "scripts\setup\MainEnv.ps1")

$python = Get-VenvPython -ProjectRoot $projectRoot
$mainScript = Join-Path $projectRoot "music_visualizer\main.py"
$requirements = Join-Path $projectRoot "music_visualizer\requirements.txt"

$env:PYTHONUTF8 = "1"

Push-Location $projectRoot
try {
    $probe = 'import numpy, aubio, sounddevice, soundfile, streamdiffusion; raise SystemExit(0 if numpy.__version__.startswith("1.") else 1)'
    if ($IsWindows -or $env:OS -match "Windows") {
        $probe = 'import numpy, aubio, sounddevice, soundfile, streamdiffusion, pyaudiowpatch; raise SystemExit(0 if numpy.__version__.startswith("1.") else 1)'
    }
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python -c $probe 2>&1 | Out-Null
    $probeOk = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prevEap
    if (-not $probeOk) {
        Write-Host "Installing pinned deps ..." -ForegroundColor Yellow
        & (Join-Path (Split-Path $python) "pip.exe") install -r $requirements --quiet
    }

    & $python $mainScript @args
}
finally {
    Pop-Location
}
