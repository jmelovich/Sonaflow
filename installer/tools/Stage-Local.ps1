<#
.SYNOPSIS
    Assemble the M1 install layout locally so the frozen runtime + launcher can
    be exercised end-to-end without an installer.

.DESCRIPTION
    Produces, under -StageDir:

        Sonaflow.exe                      (compiled launcher)
        app\music_visualizer\...          (this repo's package)
        app\streamdiffusion\...           (vendored StreamDiffusion package)
        runtime\main-env\...              (frozen env from Freeze-Env.ps1)
        native\*.dll                      (prebuilt bridges)
        thirdparty\rtx-video-sdk\...    (bundled nvngx_vsr.dll)

    Mutable data (settings, models, engines) is created at runtime under
    %LOCALAPPDATA%\Sonaflow by the launcher; nothing is written into the stage.

.EXAMPLE
    .\Stage-Local.ps1 -FrozenMainEnv ..\build\runtime\main-env -StageDir ..\build\stage
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$FrozenMainEnv,
    [string]$StageDir = (Join-Path $PSScriptRoot "..\build\stage"),
    [switch]$SkipLauncher
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Common.ps1")

$ProjectRoot = Get-ProjectRootFromTools
$FrozenMainEnv = (Resolve-Path $FrozenMainEnv).Path
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
$StageDir = (Resolve-Path $StageDir).Path

function Copy-Tree {
    param([string]$Src, [string]$Dst, [string[]]$ExcludeDirs = @())
    New-Item -ItemType Directory -Force -Path $Dst | Out-Null
    $xd = @()
    foreach ($d in $ExcludeDirs) { $xd += "/XD"; $xd += $d }
    $rc = Start-Process robocopy -ArgumentList (@(
            "`"$Src`"", "`"$Dst`"", "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/R:1", "/W:1", "/MT:16"
        ) + $xd) -NoNewWindow -Wait -PassThru
    if ($rc.ExitCode -ge 8) { throw "robocopy failed copying $Src (exit $($rc.ExitCode))" }
}

Write-Step "Staging app package"
Copy-Tree -Src (Join-Path $ProjectRoot "music_visualizer") `
    -Dst (Join-Path $StageDir "app\music_visualizer") `
    -ExcludeDirs @("__pycache__")
Write-Ok "app\music_visualizer"

Write-Step "Staging vendored streamdiffusion under app\"
Copy-Tree -Src (Join-Path $ProjectRoot "streamdiffusion") `
    -Dst (Join-Path $StageDir "app\streamdiffusion") `
    -ExcludeDirs @("__pycache__")
Write-Ok "app\streamdiffusion"

Write-Step "Staging frozen runtime\main-env"
Copy-Tree -Src $FrozenMainEnv -Dst (Join-Path $StageDir "runtime\main-env")
Write-Ok "runtime\main-env"

function Get-BuildPython {
    param([string]$Root, [string]$FrozenEnv)
    foreach ($rel in @(".venv\Scripts\python.exe", ".venv-cu121\Scripts\python.exe")) {
        $candidate = Join-Path $Root $rel
        if (Test-Path $candidate) { return $candidate }
    }
    $frozenPy = Join-Path $FrozenEnv "python.exe"
    if (Test-Path $frozenPy) { return $frozenPy }
    throw "No Python interpreter found for native bridge builds (need .venv or frozen runtime)."
}

Write-Step "Building rtxvsr_bridge.dll (required for packaged upscaling)"
$buildPy = Get-BuildPython -Root $ProjectRoot -FrozenEnv $FrozenMainEnv
$buildScript = Join-Path $ProjectRoot "music_visualizer\generation\native\build_rtxvsr.py"
$prevPyPath = $env:PYTHONPATH
$env:PYTHONPATH = $ProjectRoot
try {
    Push-Location $ProjectRoot
    & $buildPy $buildScript
    if ($LASTEXITCODE -ne 0) {
        throw "rtxvsr_bridge.dll build failed (exit $LASTEXITCODE). Ensure NV_RTX_VIDEO_SDK, MSVC, and CUDA toolkit are installed."
    }
}
finally {
    Pop-Location
    if ($null -ne $prevPyPath) { $env:PYTHONPATH = $prevPyPath }
    else { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
}
Write-Ok "rtxvsr_bridge.dll built"

Write-Step "Staging native bridges"
$nativeDst = Join-Path $StageDir "native"
New-Item -ItemType Directory -Force -Path $nativeDst | Out-Null
$dllSources = @(
    (Join-Path $ProjectRoot "music_visualizer\display\native"),
    (Join-Path $ProjectRoot "music_visualizer\generation\native")
)
$copied = 0
foreach ($srcDir in $dllSources) {
    if (Test-Path $srcDir) {
        Get-ChildItem $srcDir -Filter *.dll -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item $_.FullName (Join-Path $nativeDst $_.Name) -Force
            $copied++
        }
    }
}
if ($copied -gt 0) { Write-Ok "$copied native DLL(s) staged" }
else { Write-Warn2 "No native DLLs staged besides rtxvsr (OpenCV / first-run build may apply for others)" }

$stagedRtxVsr = Join-Path $nativeDst "rtxvsr_bridge.dll"
if (-not (Test-Path $stagedRtxVsr)) {
    throw "rtxvsr_bridge.dll missing from stage at $stagedRtxVsr"
}

Write-Step "Staging bundled RTX VSR runtime"
& (Join-Path $PSScriptRoot "Stage-RtxVsr.ps1") -StageDir $StageDir

if (-not $SkipLauncher) {
    Write-Step "Building launcher into stage root"
    & (Join-Path $PSScriptRoot "..\launcher\Build-Launcher.ps1") -OutDir $StageDir
}

Write-Host ""
Write-Host "Stage ready: $StageDir" -ForegroundColor Green
Write-Host "Smoke test (headless, no audio/qwen):" -ForegroundColor White
Write-Host "  & `"$([System.IO.Path]::Combine($StageDir,'Sonaflow.exe'))`" --acceleration none --benchmark 5 --no-qwen --no-clap --audio-source none"
