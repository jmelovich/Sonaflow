<#
.SYNOPSIS
    Read-only health check for the music visualizer installation.

.EXAMPLE
    .\doctor.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SetupDir = Join-Path $ProjectRoot "scripts\setup"

. (Join-Path $SetupDir "Common.ps1")
. (Join-Path $SetupDir "GpuDetect.ps1")
. (Join-Path $SetupDir "StreamDiffusion.ps1")
. (Join-Path $SetupDir "NativeBuild.ps1")

$Lock = $null
$failures = 0
$warnings = 0

function Test-Doctor {
    param(
        [string]$Name,
        [scriptblock]$Check,
        [switch]$Optional
    )
    try {
        $ok = & $Check
        if ($ok) {
            Write-Host "  [PASS] $Name" -ForegroundColor Green
        }
        else {
            if ($Optional) {
                Write-Host "  [WARN] $Name" -ForegroundColor Yellow
                $script:warnings++
            }
            else {
                Write-Host "  [FAIL] $Name" -ForegroundColor Red
                $script:failures++
            }
        }
    }
    catch {
        if ($Optional) {
            Write-Host ("  [WARN] {0} - {1}" -f $Name, $_) -ForegroundColor Yellow
            $script:warnings++
        }
        else {
            Write-Host ("  [FAIL] {0} - {1}" -f $Name, $_) -ForegroundColor Red
            $script:failures++
        }
    }
}

Write-Host ""
Write-Host "Music Visualizer - doctor" -ForegroundColor Cyan
Write-Host ""

Write-Host "GPU" -ForegroundColor White
$gpu = Get-GpuInfo
Test-Doctor "NVIDIA GPU detected" { $gpu.Available } -Optional
if ($gpu.Available) {
    Write-Host "         $($gpu.Name), $($gpu.VramGb) GB, driver $($gpu.DriverVersion)" -ForegroundColor DarkGray
    if (-not $gpu.IsBlackwellOrNewer) {
        Write-Host "         GPU is not Blackwell-class — blackwell-tensorrt fork may not work (see docs/SETUP.md)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Tools" -ForegroundColor White
Test-Doctor "Python 3.10+" {
    & python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>$null | Out-Null
    $LASTEXITCODE -eq 0
}
Test-Doctor "git" { $null -ne (Get-Command git -ErrorAction SilentlyContinue) }
Test-Doctor "poetry" { $null -ne (Get-Command poetry -ErrorAction SilentlyContinue) }

Write-Host ""
Write-Host "StreamDiffusion" -ForegroundColor White
$script:sdPath = $null
Test-Doctor "StreamDiffusion checkout" {
    $script:sdPath = Resolve-StreamDiffusionPath -ProjectRoot $ProjectRoot
    Test-Path (Join-Path $script:sdPath "pyproject.toml")
}
if ($script:sdPath -and (Test-Path $script:sdPath)) {
    $checkout = Get-StreamDiffusionCheckoutInfo -CheckoutPath $script:sdPath
    if ($checkout.Branch -or $checkout.RemoteUrl) {
        $label = @()
        if ($checkout.Branch) { $label += "branch $($checkout.Branch)" }
        if ($checkout.RemoteUrl) { $label += $checkout.RemoteUrl }
        Write-Host ("         {0}" -f ($label -join ", ")) -ForegroundColor DarkGray
    }
    if ($checkout.IsBlackwellFork -and $gpu.Available -and -not $gpu.IsBlackwellOrNewer) {
        Write-Host "  [WARN] blackwell-tensorrt fork on a non-Blackwell GPU — use -UseUpstreamStreamDiffusion or manual upstream install" -ForegroundColor Yellow
        $script:warnings++
    }
    Push-Location $script:sdPath
    try {
        Test-Doctor "Poetry env + streamdiffusion import" {
            & poetry run python -c "import streamdiffusion, torch" 2>$null
            return ($LASTEXITCODE -eq 0)
        }
        Test-Doctor "Visualizer deps (numpy 1.x, aubio, opencv)" {
            & poetry run python -c "import numpy, aubio, cv2; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>$null
            return ($LASTEXITCODE -eq 0)
        }
        Test-Doctor "compel installed" {
            & poetry run python -c "import compel" 2>$null
            return ($LASTEXITCODE -eq 0)
        } -Optional
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "Settings" -ForegroundColor White
$settingsPath = Join-Path $ProjectRoot "visualizer_settings.json"
Test-Doctor "visualizer_settings.json" { Test-Path $settingsPath } -Optional
if (-not (Test-Path $settingsPath)) {
    Write-Host '         Run .\setup.ps1 or copy visualizer_settings.example.json' -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Qwen sidecar" -ForegroundColor White
$qwenPython = Join-Path $ProjectRoot ".venv-qwen\Scripts\python.exe"
Test-Doctor "Sidecar venv (.venv-qwen)" { Test-Path $qwenPython } -Optional
if (Test-Path $qwenPython) {
    Test-Doctor 'Sidecar imports (transformers, bitsandbytes)' {
        & $qwenPython -c 'import transformers, bitsandbytes' 2>$null | Out-Null
        $LASTEXITCODE -eq 0
    } -Optional
}

Write-Host ""
Write-Host "Models (HuggingFace cache)" -ForegroundColor White
try {
    $Lock = Get-SetupLock -ProjectRoot $ProjectRoot
}
catch {
    Write-Host "  [WARN] setup.lock.json missing" -ForegroundColor Yellow
    $warnings++
}

if ($Lock -and $script:sdPath -and (Test-Path $script:sdPath)) {
    Push-Location $script:sdPath
    try {
        foreach ($modelKey in @("sd_model", "clap_model")) {
            $modelId = $Lock.$modelKey
            Test-Doctor "Model cached: $modelId" {
                $py = ('from huggingface_hub import try_to_load_from_cache; import sys; r=try_to_load_from_cache("{0}", "config.json"); sys.exit(0 if r else 1)' -f $modelId)
                poetry run python -c $py 2>$null | Out-Null
                $LASTEXITCODE -eq 0
            } -Optional
        }
        if (Test-Path $qwenPython) {
            Test-Doctor "Model cached: $($Lock.qwen_model)" {
                $qid = $Lock.qwen_model
                $py = ('from huggingface_hub import try_to_load_from_cache; import sys; r=try_to_load_from_cache("{0}", "config.json"); sys.exit(0 if r else 1)' -f $qid)
                & $qwenPython -c $py 2>$null | Out-Null
                $LASTEXITCODE -eq 0
            } -Optional
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "Native bridges" -ForegroundColor White
$glDll = Join-Path $ProjectRoot "music_visualizer\display\native\glpresent_bridge.dll"
$vsrDll = Join-Path $ProjectRoot "music_visualizer\generation\native\rtxvsr_bridge.dll"
Test-Doctor "glpresent_bridge.dll" { Test-Path $glDll } -Optional
Test-Doctor "rtxvsr_bridge.dll" { Test-Path $vsrDll } -Optional
$nativeTools = Test-NativeBuildTools
if (-not $nativeTools.Ok) {
    Write-Host "         MSVC: $($nativeTools.Msvc), CUDA toolkit: $($nativeTools.Cuda)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "RTX Video SDK (upscaling)" -ForegroundColor White
Test-Doctor "NV_RTX_VIDEO_SDK set" {
    $env:NV_RTX_VIDEO_SDK -and (Test-Path $env:NV_RTX_VIDEO_SDK)
} -Optional
if ($env:NV_RTX_VIDEO_SDK) {
    Write-Host "         $($env:NV_RTX_VIDEO_SDK)" -ForegroundColor DarkGray
}
else {
    Write-Host '         Optional - required for RTX VSR upscaling (see docs/SETUP.md)' -ForegroundColor DarkGray
}

Write-Host ""
if ($failures -eq 0) {
    Write-Host ("Summary: {0} warning(s), 0 failures - ready to run .\run.ps1" -f $warnings) -ForegroundColor Green
    exit 0
}
else {
    Write-Host ("Summary: {0} failure(s), {1} warning(s) - run .\setup.ps1 to fix" -f $failures, $warnings) -ForegroundColor Red
    exit 1
}
