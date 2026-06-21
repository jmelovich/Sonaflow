<#
.SYNOPSIS
    Build a relocatable "main-env" by harvesting the working Poetry venv's
    packages onto a standalone CPython. No re-download of torch/TensorRT.

.DESCRIPTION
    1. Locates the source Poetry venv (or accepts -SourceVenv).
    2. Downloads + extracts a relocatable CPython (python-build-standalone).
    3. Copies the venv's Lib\site-packages onto the standalone interpreter.
    4. De-editable-izes the StreamDiffusion install (copies src\streamdiffusion
       in place of the editable .pth shim) so no absolute dev path is baked in.
    5. Smoke-tests imports and scans for leaked build-machine paths.

    The result is a self-contained folder invoked by explicit interpreter path
    (no activation), matching how run/launcher call python.

.EXAMPLE
    .\Freeze-Env.ps1 -StreamDiffusionPath H:\_dev\StreamDiffusion -Dest ..\build\runtime\main-env
#>
[CmdletBinding()]
param(
    [string]$StreamDiffusionPath = $env:STREAMDIFFUSION_PATH,
    [string]$SourceVenv,
    [Parameter(Mandatory = $true)][string]$Dest,
    [string]$PythonUrl,
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Common.ps1")

if (-not $PythonUrl) { $PythonUrl = $script:DefaultStandalonePythonUrl }
if (-not $StreamDiffusionPath) {
    throw "Provide -StreamDiffusionPath or set STREAMDIFFUSION_PATH."
}
$StreamDiffusionPath = (Resolve-Path $StreamDiffusionPath).Path

Write-Step "Resolving source Poetry venv"
$srcVenv = Get-SourceVenvPath -StreamDiffusionPath $StreamDiffusionPath -Explicit $SourceVenv
$srcSitePackages = Join-Path $srcVenv "Lib\site-packages"
if (-not (Test-Path $srcSitePackages)) {
    throw "Source site-packages not found: $srcSitePackages"
}
$srcPython = Join-Path $srcVenv "Scripts\python.exe"
if (-not (Test-Path $srcPython)) { $srcPython = Join-Path $srcVenv "python.exe" }
$srcMinor = Get-PythonMinor -PythonExe $srcPython
Write-Ok "Source venv: $srcVenv (Python $srcMinor)"

# --- standalone interpreter ---
$Dest = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Dest))
if ((Test-Path $Dest) -and $KeepExisting) {
    Write-Warn2 "Dest exists and -KeepExisting set; reusing $Dest"
}
else {
    Expand-StandalonePython -Url $PythonUrl -Dest $Dest
}
$destPython = Join-Path $Dest "python.exe"
if (-not (Test-Path $destPython)) { throw "Standalone python missing at $destPython" }
$destMinor = Get-PythonMinor -PythonExe $destPython
Write-Ok "Standalone interpreter: $destPython (Python $destMinor)"

if ($srcMinor -ne $destMinor) {
    throw "Python minor mismatch: source venv is $srcMinor but standalone is $destMinor. " +
          "Pass -PythonUrl for a matching $srcMinor build (extension modules are ABI-tied to the minor version)."
}

$destSitePackages = Join-Path $Dest "Lib\site-packages"
New-Item -ItemType Directory -Force -Path $destSitePackages | Out-Null

# --- copy site-packages (merge onto the standalone's pip/setuptools) ---
Write-Step "Copying site-packages (this is the large step)"
# /E recurse incl. empty, /NJH /NJS quiet headers, /R:1 /W:1 fail fast, multi-thread.
$rc = Start-Process robocopy -ArgumentList @(
    "`"$srcSitePackages`"", "`"$destSitePackages`"", "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/R:1", "/W:1", "/MT:16"
) -NoNewWindow -Wait -PassThru
# robocopy exit codes < 8 are success.
if ($rc.ExitCode -ge 8) { throw "robocopy failed (exit $($rc.ExitCode))" }
Write-Ok "site-packages copied"

# --- de-editable-ize streamdiffusion ---
Write-Step "Resolving StreamDiffusion package (replace editable shim with real source)"
$sdSrc = $null
foreach ($cand in @(
        (Join-Path $StreamDiffusionPath "src\streamdiffusion"),
        (Join-Path $StreamDiffusionPath "streamdiffusion"))) {
    if (Test-Path (Join-Path $cand "__init__.py")) { $sdSrc = $cand; break }
}
if (-not $sdSrc) { throw "Could not find the streamdiffusion package under $StreamDiffusionPath (looked in src\ and root)." }

# Remove PEP 660 editable shims so the copied package is the import target.
Get-ChildItem $destSitePackages -Filter "__editable__*streamdiffusion*" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item $_.FullName -Force; Write-Warn2 "removed shim $($_.Name)" }
Get-ChildItem $destSitePackages -Filter "__editable___streamdiffusion*" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item $_.FullName -Force; Write-Warn2 "removed shim $($_.Name)" }
$egglink = Join-Path $destSitePackages "streamdiffusion.egg-link"
if (Test-Path $egglink) { Remove-Item $egglink -Force; Write-Warn2 "removed streamdiffusion.egg-link" }

$destSdPkg = Join-Path $destSitePackages "streamdiffusion"
if (Test-Path $destSdPkg) { Remove-Item $destSdPkg -Recurse -Force }
$rc2 = Start-Process robocopy -ArgumentList @(
    "`"$sdSrc`"", "`"$destSdPkg`"", "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/R:1", "/W:1", "/MT:16"
) -NoNewWindow -Wait -PassThru
if ($rc2.ExitCode -ge 8) { throw "robocopy (streamdiffusion) failed (exit $($rc2.ExitCode))" }
Write-Ok "streamdiffusion copied from $sdSrc"

# Drop any *.pth that still references the source venv / SD path.
$leakPaths = @($srcVenv, $StreamDiffusionPath)
Get-ChildItem $destSitePackages -Filter "*.pth" -ErrorAction SilentlyContinue | ForEach-Object {
    $content = Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue
    foreach ($lp in $leakPaths) {
        if ($content -and $content.Contains($lp)) {
            Remove-Item $_.FullName -Force
            Write-Warn2 "removed leaking .pth $($_.Name)"
            break
        }
    }
}

# --- smoke test ---
Write-Step "Smoke-testing the frozen environment"
$probe = @"
import sys
mods = ['numpy', 'torch', 'streamdiffusion', 'cv2', 'aubio', 'sounddevice', 'soundfile', 'compel']
ok = True
for m in mods:
    try:
        __import__(m)
        print('  import', m, 'OK')
    except Exception as e:
        ok = False
        print('  import', m, 'FAILED:', repr(e))
try:
    import torch
    print('  torch', torch.__version__, 'cuda_avail=', torch.cuda.is_available())
except Exception as e:
    print('  torch version check failed:', repr(e))
sys.exit(0 if ok else 1)
"@
$probeFile = Join-Path $env:TEMP ("sf-probe-" + [guid]::NewGuid().ToString("N") + ".py")
Set-Content -Path $probeFile -Value $probe -Encoding UTF8
$env:PYTHONUTF8 = "1"
& $destPython $probeFile
$probeOk = ($LASTEXITCODE -eq 0)
Remove-Item $probeFile -Force -ErrorAction SilentlyContinue
if (-not $probeOk) { Write-Err2 "Smoke test reported failed imports (see above)." }
else { Write-Ok "Imports OK" }

# --- leaked-path scan (non-fatal warning) ---
Write-Step "Scanning for leaked build-machine paths in text shims"
$needle = $srcVenv.Replace('\', '\\')
$hits = Get-ChildItem $destSitePackages -Recurse -Include *.pth, *.egg-link, *.txt -ErrorAction SilentlyContinue |
    Select-String -SimpleMatch $srcVenv -List -ErrorAction SilentlyContinue
if ($hits) {
    Write-Warn2 "Found references to the source venv path in:"
    $hits | ForEach-Object { Write-Host "      $($_.Path)" }
}
else {
    Write-Ok "No source-venv path leaks in .pth/.egg-link/.txt"
}

Write-Host ""
Write-Host "Frozen main-env ready at: $Dest" -ForegroundColor Green
Write-Host "  Interpreter: $destPython"
