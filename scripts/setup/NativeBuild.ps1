# Optionally compile native bridge DLLs when MSVC + CUDA toolkit are available.

function Test-NativeBuildTools {
    $result = @{ Msvc = $false; Cuda = $false; Ok = $false }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath 2>$null
        if ($vsPath -and (Test-Path (Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"))) {
            $result.Msvc = $true
        }
    }

    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "include\cudaGL.h"))) {
        $result.Cuda = $true
    }
    else {
        $cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if (Test-Path $cudaBase) {
            $versions = Get-ChildItem $cudaBase -Directory -Filter "v*" | Sort-Object Name -Descending
            foreach ($v in $versions) {
                if (Test-Path (Join-Path $v.FullName "include\cudaGL.h")) {
                    $result.Cuda = $true
                    break
                }
            }
        }
    }

    $result.Ok = $result.Msvc -and $result.Cuda
    return $result
}

function Install-NativeBridges {
    param(
        [string]$StreamDiffusionPath,
        [string]$ProjectRoot,
        [switch]$SkipNativeBuild
    )

    if ($SkipNativeBuild) {
        Write-SetupWarn 'Skipped native bridge build (-SkipNativeBuild)'
        return
    }

    $tools = Test-NativeBuildTools
    if (-not $tools.Ok) {
        if (-not $tools.Msvc) {
            Write-SetupWarn 'MSVC not found - glpresent/rtxvsr DLLs will build on first run (or install VS Build Tools).'
        }
        if (-not $tools.Cuda) {
            Write-SetupWarn 'CUDA toolkit not found - native bridges need CUDA_PATH or a default install.'
        }
        return
    }

    Push-Location $StreamDiffusionPath
    try {
        Write-Host "Building glpresent_bridge.dll ..."
        poetry run python (Join-Path $ProjectRoot "music_visualizer\display\native\build_glpresent.py")
        if ($LASTEXITCODE -eq 0) {
            Write-SetupOk "glpresent_bridge.dll built"
        }
        else {
            Write-SetupWarn 'glpresent build failed - OpenCV fallback will be used.'
        }

        if ($env:NV_RTX_VIDEO_SDK -or (Get-Content (Join-Path $ProjectRoot "visualizer_settings.json") -ErrorAction SilentlyContinue | Select-String "rtx")) {
            Write-Host "Building rtxvsr_bridge.dll ..."
            poetry run python (Join-Path $ProjectRoot "music_visualizer\generation\native\build_rtxvsr.py")
            if ($LASTEXITCODE -eq 0) {
                Write-SetupOk "rtxvsr_bridge.dll built"
            }
            else {
                Write-SetupWarn 'rtxvsr build failed - disable upscale or set NV_RTX_VIDEO_SDK.'
            }
        }
        else {
            Write-SetupWarn 'NV_RTX_VIDEO_SDK not set - skipping rtxvsr build (enable upscale after installing RTX Video SDK).'
        }
    }
    finally {
        Pop-Location
    }
}
