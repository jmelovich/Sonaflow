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
. (Join-Path $SetupDir "MainEnv.ps1")
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
Write-Host "Sonaflow - doctor" -ForegroundColor Cyan
Write-Host ""

Write-Host "GPU" -ForegroundColor White
$gpu = Get-GpuInfo
Test-Doctor "NVIDIA GPU detected" { $gpu.Available } -Optional
if ($gpu.Available) {
    Write-Host "         $($gpu.Name), $($gpu.VramGb) GB, driver $($gpu.DriverVersion)" -ForegroundColor DarkGray
    if ($gpu.IsBlackwellOrNewer) {
        Write-Host "         Blackwell-class GPU — default CUDA profile is cu128" -ForegroundColor DarkGray
    }
    else {
        Write-Host "         Non-Blackwell GPU — use .\setup.ps1 -Cuda cu121 if cu128 fails" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Tools" -ForegroundColor White
Test-Doctor "Python 3.10+" {
    & python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>$null | Out-Null
    $LASTEXITCODE -eq 0
}

Write-Host ""
Write-Host "Sonaflow environment" -ForegroundColor White
$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Test-Doctor "Main venv (.venv)" { Test-Path $venvPython }
$vendoredPkg = Join-Path $ProjectRoot "streamdiffusion\__init__.py"
Test-Doctor "Vendored streamdiffusion package" { Test-Path $vendoredPkg }

if (Test-Path $venvPython) {
    $env:PYTHONPATH = $ProjectRoot
    Test-Doctor "Imports (torch, streamdiffusion, numpy 1.x)" {
        & $venvPython -c "import numpy, torch, streamdiffusion; exit(0 if numpy.__version__.startswith('1.') else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    Test-Doctor "Visualizer deps (aubio, opencv, sounddevice)" {
        & $venvPython -c "import aubio, cv2, sounddevice, soundfile" 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    Test-Doctor "compel installed" {
        & $venvPython -c "import compel" 2>$null
        return ($LASTEXITCODE -eq 0)
    } -Optional
    Test-Doctor "sd_wrapper import" {
        & $venvPython -c "from music_visualizer.generation.sd_wrapper import StreamDiffusionWrapper" 2>$null
        return ($LASTEXITCODE -eq 0)
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
Write-Host "Qwen2-Audio" -ForegroundColor White
if (Test-Path $venvPython) {
    Test-Doctor "transformers >= 4.45 (main venv)" {
        & $venvPython -c 'import transformers; v=tuple(int(x) for x in transformers.__version__.split(".")[:2]); exit(0 if v>=(4,45) else 1)' 2>$null | Out-Null
        $LASTEXITCODE -eq 0
    } -Optional
    Test-Doctor "Qwen imports (transformers, bitsandbytes, Qwen2Audio)" {
        & $venvPython -c 'import transformers, bitsandbytes; from transformers import Qwen2AudioForConditionalGeneration' 2>$null | Out-Null
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

if ($Lock -and (Test-Path $venvPython)) {
    foreach ($modelKey in @("sd_model", "clap_model")) {
        $modelId = $Lock.$modelKey
        Test-Doctor "Model cached: $modelId" {
            $py = ('from huggingface_hub import try_to_load_from_cache; import sys; r=try_to_load_from_cache("{0}", "config.json"); sys.exit(0 if r else 1)' -f $modelId)
            & $venvPython -c $py 2>$null | Out-Null
            $LASTEXITCODE -eq 0
        } -Optional
    }
    if ($Lock.qwen_model) {
        Test-Doctor "Model cached: $($Lock.qwen_model)" {
            $qid = $Lock.qwen_model
            $py = ('from huggingface_hub import try_to_load_from_cache; import sys; r=try_to_load_from_cache("{0}", "config.json"); sys.exit(0 if r else 1)' -f $qid)
            & $venvPython -c $py 2>$null | Out-Null
            $LASTEXITCODE -eq 0
        } -Optional
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
