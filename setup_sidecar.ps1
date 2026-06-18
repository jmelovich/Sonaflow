# Backward-compatible wrapper — prefer .\setup.ps1 -Profile Full
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $Root "scripts\setup\Common.ps1")
. (Join-Path $Root "scripts\setup\Sidecar.ps1")

$Lock = Get-SetupLock -ProjectRoot $Root
Install-Sidecar -ProjectRoot $Root -Lock $Lock
