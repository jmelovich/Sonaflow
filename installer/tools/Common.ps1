# Shared helpers for the Sonaflow packaging tools.

function Write-Step  { param([string]$Msg) Write-Host "==> $Msg" -ForegroundColor Cyan }
function Write-Ok    { param([string]$Msg) Write-Host "  [ok] $Msg" -ForegroundColor Green }
function Write-Warn2 { param([string]$Msg) Write-Host "  [warn] $Msg" -ForegroundColor Yellow }
function Write-Err2  { param([string]$Msg) Write-Host "  [fail] $Msg" -ForegroundColor Red }

# Default python-build-standalone (relocatable CPython) for the main env.
# Override with -PythonUrl. The "install_only" archive extracts to a `python/`
# directory containing python.exe at its root.
$script:DefaultStandalonePythonUrl =
    "https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-3.10.15+20241016-x86_64-pc-windows-msvc-install_only.tar.gz"

function Get-ProjectRootFromTools {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Resolve-PackagePath {
    <#
    .SYNOPSIS Resolve a path that may be relative to the current directory or absolute.
    #>
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

function Get-SourceVenvPath {
    <#
    .SYNOPSIS Resolve the venv to harvest packages from (dev .venv or package venv).
    #>
    param(
        [string]$ProjectRoot,
        [string]$Explicit,
        [ValidateSet("cu128", "cu121")]
        [string]$Cuda = "cu128"
    )
    if ($Explicit) {
        if (-not (Test-Path $Explicit)) { throw "Source venv not found: $Explicit" }
        return (Resolve-Path $Explicit).Path
    }
    if (-not $ProjectRoot) {
        $ProjectRoot = Get-ProjectRootFromTools
    }
    . (Join-Path $PSScriptRoot "PackageEnv.ps1")
    $venvRel = Get-PackageVenvRelName -Cuda $Cuda
    $venv = Join-Path $ProjectRoot $venvRel
    $python = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path $python)) {
        throw "Package venv not found at $venv. Run Ensure-PackageMainEnv or Build-OfflineInstaller.ps1 -RefreshEnv first."
    }
    return (Resolve-Path $venv).Path
}

function Get-PythonMinor {
    param([string]$PythonExe)
    $v = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0) { throw "Failed to read python version from $PythonExe" }
    return $v.Trim()
}

function Expand-StandalonePython {
    <#
    .SYNOPSIS Download + extract a relocatable CPython into $Dest.
    #>
    param(
        [string]$Url,
        [string]$Dest
    )
    $tmp = Join-Path $env:TEMP ("sf-py-" + [guid]::NewGuid().ToString("N") + ".tar.gz")
    Write-Step "Downloading standalone CPython"
    Write-Host "    $Url"
    Invoke-WebRequest -Uri $Url -OutFile $tmp -UseBasicParsing
    try {
        $stage = Join-Path $env:TEMP ("sf-pyx-" + [guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Force -Path $stage | Out-Null
        Write-Step "Extracting CPython"
        tar -xzf $tmp -C $stage
        if ($LASTEXITCODE -ne 0) { throw "tar extraction failed" }
        $pyRoot = Join-Path $stage "python"
        if (-not (Test-Path (Join-Path $pyRoot "python.exe"))) {
            throw "Unexpected archive layout (no python\python.exe under $stage)"
        }
        if (Test-Path $Dest) { Remove-Item $Dest -Recurse -Force }
        New-Item -ItemType Directory -Force -Path (Split-Path $Dest) | Out-Null
        Move-Item $pyRoot $Dest
        Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
    }
    finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
}
