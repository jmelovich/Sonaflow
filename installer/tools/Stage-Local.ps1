<#
.SYNOPSIS
    Assemble the M1 install layout locally so the frozen runtime + launcher can
    be exercised end-to-end without an installer.

.DESCRIPTION
    Produces, under -StageDir:

        Sonaflow.exe                      (compiled launcher)
        app\music_visualizer\...          (this repo's package)
        streamdiffusion\...               (SD checkout minus .git/.venv/engines)
        runtime\main-env\...              (frozen env from Freeze-Env.ps1)
        native\*.dll                      (prebuilt bridges, if present)

    Mutable data (settings, models, engines) is created at runtime under
    %LOCALAPPDATA%\Sonaflow by the launcher; nothing is written into the stage.

.EXAMPLE
    .\Stage-Local.ps1 -StreamDiffusionPath H:\_dev\StreamDiffusion -FrozenMainEnv ..\build\runtime\main-env -StageDir ..\build\stage
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$StreamDiffusionPath,
    [Parameter(Mandatory = $true)][string]$FrozenMainEnv,
    [string]$StageDir = (Join-Path $PSScriptRoot "..\build\stage"),
    [switch]$SkipLauncher
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Common.ps1")

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$StreamDiffusionPath = (Resolve-Path $StreamDiffusionPath).Path
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

Write-Step "Staging StreamDiffusion (minus .git / .venv / engines)"
Copy-Tree -Src $StreamDiffusionPath `
    -Dst (Join-Path $StageDir "streamdiffusion") `
    -ExcludeDirs @(".git", ".venv", "engines", "__pycache__")
Write-Ok "streamdiffusion"

Write-Step "Staging frozen runtime\main-env"
Copy-Tree -Src $FrozenMainEnv -Dst (Join-Path $StageDir "runtime\main-env")
Write-Ok "runtime\main-env"

Write-Step "Staging native bridges (if built)"
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
else { Write-Warn2 "No prebuilt DLLs found (OpenCV fallback / first-run build will apply)" }

if (-not $SkipLauncher) {
    Write-Step "Building launcher into stage root"
    & (Join-Path $PSScriptRoot "..\launcher\Build-Launcher.ps1") -OutDir $StageDir
}

Write-Host ""
Write-Host "Stage ready: $StageDir" -ForegroundColor Green
Write-Host "Smoke test (headless, no audio/qwen):" -ForegroundColor White
Write-Host "  & `"$([System.IO.Path]::Combine($StageDir,'Sonaflow.exe'))`" --acceleration none --benchmark 5 --no-qwen --no-clap --audio-source none"
