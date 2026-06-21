<#
.SYNOPSIS
    Compile Sonaflow.exe (the launcher stub) with MSVC.

.DESCRIPTION
    Locates the Visual Studio C++ toolchain via vswhere, then compiles
    launcher.c into a small console executable. Optionally embeds an icon if
    one is present (icon.ico + launcher.rc).

.EXAMPLE
    .\Build-Launcher.ps1 -OutDir ..\build\stage
#>
[CmdletBinding()]
param(
    [string]$OutDir = (Join-Path $PSScriptRoot "..\build\stage"),
    [string]$Name = "Sonaflow.exe"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\tools\Common.ps1")

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere not found. Install Visual Studio Build Tools (Desktop development with C++)."
}
$vsPath = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $vsPath) { throw "No VC++ toolchain found via vswhere." }
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found under $vsPath" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
$src = Join-Path $PSScriptRoot "launcher.c"
$exe = Join-Path $OutDir $Name
$obj = Join-Path $env:TEMP ("launcher-" + [guid]::NewGuid().ToString("N") + ".obj")

$rc = ""
$icon = Join-Path $PSScriptRoot "icon.ico"
$rcFile = Join-Path $PSScriptRoot "launcher.rc"
if ((Test-Path $icon) -and (Test-Path $rcFile)) {
    $res = Join-Path $env:TEMP ("launcher-" + [guid]::NewGuid().ToString("N") + ".res")
    $rc = "rc /nologo /fo `"$res`" `"$rcFile`" && "
}

Write-Step "Compiling $Name"
# /O2 optimize, /MT static CRT so the exe has no VC runtime dependency of its own.
$cl = "cl /nologo /O2 /MT /W3 `"$src`" /Fe`"$exe`" /Fo`"$obj`" /link"
if ($rc) {
    $resPath = $res
    $cl = "cl /nologo /O2 /MT /W3 `"$src`" /Fe`"$exe`" /Fo`"$obj`" /link `"$resPath`""
}
$cmd = "`"$vcvars`" >nul && $rc$cl"
& cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "Launcher compilation failed (exit $LASTEXITCODE)." }
Remove-Item $obj -Force -ErrorAction SilentlyContinue

Write-Ok "Built $exe"
