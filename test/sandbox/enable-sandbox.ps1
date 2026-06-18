# Enable Windows Sandbox (requires Administrator).
# Run from an elevated PowerShell:
#   .\test\sandbox\enable-sandbox.ps1

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "SandboxResolve.ps1")

$state = Get-SandboxFeatureState
Write-Host "Windows Sandbox feature: $state"

if ($state -eq "Enabled") {
    $exe = Get-WindowsSandboxExe
    if ($exe) {
        Write-Host "Launcher found: $exe" -ForegroundColor Green
        Write-Host "Try: .\test\sandbox\launch-sandbox-test.ps1 -Profile Minimal" -ForegroundColor White
        exit 0
    }
    Write-Host ""
    Write-Host "Feature is Enabled but WindowsSandbox.exe is not linked." -ForegroundColor Red
    Write-Host "This is a known Win11 24H2 issue. Try:" -ForegroundColor Yellow
    Write-Host "  1. Settings -> Apps -> Microsoft Edge -> Modify -> Repair" -ForegroundColor White
    Write-Host "  2. Turn Windows features OFF: uncheck Windows Sandbox, reboot" -ForegroundColor White
    Write-Host "  3. Turn Windows Sandbox back ON, reboot again" -ForegroundColor White
    Write-Host "  4. Install latest Windows updates (KB5074105+ fixed this for some builds)" -ForegroundColor White
    Write-Host ""
    Write-Host "Or run repair: DISM /Online /Cleanup-Image /RestoreHealth && sfc /scannow" -ForegroundColor DarkGray
    exit 1
}

Write-Host "Enabling Windows Sandbox ..."
Enable-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM -All -NoRestart
Write-Host ""
Write-Host "Enabled. Reboot, then run:" -ForegroundColor Yellow
Write-Host "  .\test\sandbox\launch-sandbox-test.ps1 -Profile Minimal" -ForegroundColor White
