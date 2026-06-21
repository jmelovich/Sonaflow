<#
.SYNOPSIS
    Build a relocatable qwen-env by harvesting .venv-qwen onto standalone CPython.

.EXAMPLE
    .\Freeze-QwenEnv.ps1 -ProjectRoot H:\_dev\Sonaflow -Dest ..\build\runtime\qwen-env

.EXAMPLE
    .\Freeze-QwenEnv.ps1 -Cuda cu121 -Dest ..\build\runtime\cu121\qwen-env
#>
[CmdletBinding()]
param(
    [string]$ProjectRoot,
    [string]$SourceVenv,
    [ValidateSet("cu128", "cu121")]
    [string]$Cuda = "cu128",
    [Parameter(Mandatory = $true)][string]$Dest,
    [string]$PythonUrl,
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "Common.ps1")

if (-not $PythonUrl) { $PythonUrl = $script:DefaultStandalonePythonUrl }
if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
}
else {
    $ProjectRoot = (Resolve-Path $ProjectRoot).Path
}

if (-not $SourceVenv) {
    . (Join-Path $ScriptDir "PackageEnv.ps1")
    $venvRel = Get-PackageQwenVenvRelName -Cuda $Cuda
    $SourceVenv = Join-Path $ProjectRoot $venvRel
}
if (-not (Test-Path $SourceVenv)) {
    throw @"
Qwen sidecar venv not found at: $SourceVenv

Create it first:
  cd $ProjectRoot
  .\setup.ps1 -Profile Full
  # or: .\setup_sidecar.ps1
"@
}

$srcSitePackages = Join-Path $SourceVenv "Lib\site-packages"
if (-not (Test-Path $srcSitePackages)) {
    throw "Source site-packages not found: $srcSitePackages"
}
$srcPython = Join-Path $SourceVenv "Scripts\python.exe"
if (-not (Test-Path $srcPython)) { $srcPython = Join-Path $SourceVenv "python.exe" }
$srcMinor = Get-PythonMinor -PythonExe $srcPython
Write-Ok "Source qwen venv: $SourceVenv (Python $srcMinor)"

$Dest = Resolve-PackagePath -Path $Dest
if ((Test-Path $Dest) -and $KeepExisting) {
    Write-Warn2 "Dest exists and -KeepExisting set; reusing $Dest"
}
else {
    Expand-StandalonePython -Url $PythonUrl -Dest $Dest
}

$destPython = Join-Path $Dest "python.exe"
$destMinor = Get-PythonMinor -PythonExe $destPython
if ($srcMinor -ne $destMinor) {
    throw "Python minor mismatch: source $srcMinor vs standalone $destMinor"
}

$destSitePackages = Join-Path $Dest "Lib\site-packages"
New-Item -ItemType Directory -Force -Path $destSitePackages | Out-Null

Write-Step "Copying qwen sidecar site-packages"
$rc = Start-Process robocopy -ArgumentList @(
    "`"$srcSitePackages`"", "`"$destSitePackages`"", "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/R:1", "/W:1", "/MT:16"
) -NoNewWindow -Wait -PassThru
if ($rc.ExitCode -ge 8) { throw "robocopy failed (exit $($rc.ExitCode))" }

Write-Step "Smoke-testing frozen qwen-env"
$probe = @"
import sys
mods = ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'librosa', 'soundfile']
ok = True
for m in mods:
    try:
        __import__(m)
        print('  import', m, 'OK')
    except Exception as e:
        ok = False
        print('  import', m, 'FAILED:', repr(e))
sys.exit(0 if ok else 1)
"@
$probeFile = Join-Path $env:TEMP ("sf-qwen-probe-" + [guid]::NewGuid().ToString("N") + ".py")
Set-Content -Path $probeFile -Value $probe -Encoding UTF8
$env:PYTHONUTF8 = "1"
& $destPython $probeFile
$probeOk = ($LASTEXITCODE -eq 0)
Remove-Item $probeFile -Force -ErrorAction SilentlyContinue
if (-not $probeOk) { throw "Qwen env smoke test failed." }
Write-Ok "Qwen sidecar env ready at $Dest"
