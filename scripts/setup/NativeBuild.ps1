# Optionally compile native bridge DLLs when MSVC + CUDA toolkit are available.

function Get-CudaToolkitDownloadUrl {
    param(
        [ValidateSet("cu128", "cu121")]
        [string]$Cuda = "cu128"
    )
    if ($Cuda -eq "cu121") {
        return "https://developer.nvidia.com/cuda-12-1-0-download-archive"
    }
    return "https://developer.nvidia.com/cuda-12-8-0-download-archive"
}

function Test-NativeBuildTools {
    $result = @{ Msvc = $false; Cuda = $false; Ok = $false }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installs = @(& $vswhere -latest -products * -property installationPath 2>$null)
        foreach ($vsPath in $installs) {
            if (-not $vsPath) { continue }
            $classic = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
            $vsdevcmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
            $hasCl = Get-ChildItem (Join-Path $vsPath "VC\Tools\MSVC") -Recurse -Filter "cl.exe" -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -match 'HostX64\\x64\\cl\.exe' } |
                Select-Object -First 1
            if ((Test-Path $classic) -or ((Test-Path $vsdevcmd) -and $hasCl)) {
                $result.Msvc = $true
                break
            }
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
        [string]$ProjectRoot,
        [ValidateSet("cu128", "cu121")]
        [string]$Cuda = "cu128",
        [switch]$SkipNativeBuild
    )

    if ($SkipNativeBuild) {
        Write-SetupWarn 'Skipped native bridge build (-SkipNativeBuild)'
        return
    }

    $tools = Test-NativeBuildTools
    if (-not $tools.Ok) {
        if (-not $tools.Msvc) {
            Write-SetupWarn 'MSVC not found - native bridge DLLs need a minimal VS Build Tools install (~2 GB).'
            Write-Host '         Bootstrapper: https://aka.ms/vs/17/release/vs_BuildTools.exe' -ForegroundColor DarkGray
            Write-Host '         Individual components: MSVC v143 x64/x86 build tools + Windows 11 SDK' -ForegroundColor DarkGray
        }
        if (-not $tools.Cuda) {
            $cudaUrl = Get-CudaToolkitDownloadUrl -Cuda $Cuda
            $cudaVer = if ($Cuda -eq "cu121") { "12.1" } else { "12.8" }
            Write-SetupWarn "CUDA toolkit not found - install CUDA $cudaVer for native bridge builds:"
            Write-Host "         $cudaUrl" -ForegroundColor DarkGray
            Write-Host "         (OpenCV fallback works without it; use --no-gl-present to skip GPU presenter.)" -ForegroundColor DarkGray
        }
        return
    }

    . (Join-Path $PSScriptRoot "MainEnv.ps1")
    $python = Get-VenvPython -ProjectRoot $ProjectRoot

    Push-Location $ProjectRoot
    try {
        Write-Host "Building glpresent_bridge.dll ..."
        & $python (Join-Path $ProjectRoot "music_visualizer\display\native\build_glpresent.py")
        if ($LASTEXITCODE -eq 0) {
            Write-SetupOk "glpresent_bridge.dll built"
        }
        else {
            Write-SetupWarn 'glpresent build failed - pass --no-gl-present to use OpenCV, or install MSVC + CUDA toolkit.'
        }

        if ($env:NV_RTX_VIDEO_SDK -or (Get-Content (Join-Path $ProjectRoot "visualizer_settings.json") -ErrorAction SilentlyContinue | Select-String "rtx")) {
            Write-Host "Building rtxvsr_bridge.dll ..."
            & $python (Join-Path $ProjectRoot "music_visualizer\generation\native\build_rtxvsr.py")
            if ($LASTEXITCODE -eq 0) {
                Write-SetupOk "rtxvsr_bridge.dll built"
            }
            else {
                Write-SetupWarn 'rtxvsr build failed - disable upscale or set NV_RTX_VIDEO_SDK.'
            }
        }
        else {
            Write-SetupWarn 'NV_RTX_VIDEO_SDK not set - skipping rtxvsr build.'
            Write-Host "         Download RTX Video SDK: https://developer.nvidia.com/rtx-video-sdk" -ForegroundColor DarkGray
            Write-Host "         Set NV_RTX_VIDEO_SDK to the SDK root, or pass --no-upscale at runtime." -ForegroundColor DarkGray
        }
    }
    finally {
        Pop-Location
    }
}
