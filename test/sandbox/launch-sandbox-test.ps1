# Launch automated Windows Sandbox test and poll for results on the host.
param(
    [int]$TimeoutMinutes = 90,
    [ValidateSet("Minimal", "Standard")]
    [string]$Profile = "Minimal"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $Root "SandboxResolve.ps1")

$RepoRoot = Split-Path -Parent (Split-Path -Parent $Root)
$LogDir = Join-Path $Root "logs"
$Wsb = Join-Path $Root "visualizer-test-auto.wsb"
$SandboxExe = Get-WindowsSandboxExe

if (-not $SandboxExe) {
    $state = Get-SandboxFeatureState
    Write-Host ""
    Write-Host "Windows Sandbox launcher not found (feature state: $state)." -ForegroundColor Red
    Write-Host ""
    if ($state -eq "Enabled") {
        Write-Host "The feature shows Enabled but the exe is not linked - common on Win11 24H2." -ForegroundColor Yellow
        Write-Host "Repair steps:" -ForegroundColor Yellow
        Write-Host "  1. Repair Microsoft Edge (Settings -> Apps -> Edge -> Modify -> Repair)" -ForegroundColor White
        Write-Host "  2. Disable Windows Sandbox in 'Turn Windows features on or off', reboot" -ForegroundColor White
        Write-Host "  3. Re-enable Windows Sandbox, reboot again" -ForegroundColor White
        Write-Host "  4. Run elevated: .\test\sandbox\enable-sandbox.ps1" -ForegroundColor White
    }
    else {
        Write-Host "Enable it (Administrator + reboot):" -ForegroundColor Yellow
        Write-Host "  Enable-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM -All" -ForegroundColor White
    }
    Write-Host ""
    exit 2
}

# Ensure log dir exists and is clean
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Remove-Item (Join-Path $LogDir "*") -Force -ErrorAction SilentlyContinue

# Patch profile and host paths in wsb logon command by writing a temp wsb
$wsbContent = Get-Content $Wsb -Raw
$wsbContent = $wsbContent -replace '-Profile Minimal', "-Profile $Profile"
$sdPath = Join-Path (Split-Path -Parent $RepoRoot) "StreamDiffusion"
$wsbContent = $wsbContent.Replace('__SONAFLOW_REPO__', $RepoRoot)
$wsbContent = $wsbContent.Replace('__STREAMDIFFUSION_REPO__', $sdPath)
$wsbContent = $wsbContent.Replace('__SANDBOX_LOG_DIR__', $LogDir)
$tempWsb = Join-Path $LogDir "run-$Profile.wsb"
Set-Content -Path $tempWsb -Value $wsbContent -Encoding utf8

Write-Host "Launching Windows Sandbox (profile: $Profile) ..." -ForegroundColor Cyan
Write-Host "Logs will appear in: $LogDir" -ForegroundColor DarkGray
Write-Host "This may take 30-90 minutes (winget + poetry install). Polling every 30s ..." -ForegroundColor Yellow

Start-Process -FilePath $SandboxExe -ArgumentList $tempWsb

$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$resultFile = Join-Path $LogDir "sandbox-result.json"

while ((Get-Date) -lt $deadline) {
    if (Test-Path $resultFile) {
        $result = Get-Content $resultFile -Raw | ConvertFrom-Json
        Write-Host ""
        Write-Host "=== Sandbox test finished ===" -ForegroundColor Cyan
        foreach ($step in $result.steps) {
            $mark = if ($step.ok) { "PASS" } else { "FAIL" }
            $color = if ($step.ok) { "Green" } else { "Red" }
            Write-Host ("  [{0}] {1}" -f $mark, $step.name) -ForegroundColor $color
            if ($step.detail) { Write-Host ("         {0}" -f $step.detail) -ForegroundColor DarkGray }
        }
        if ($result.error) {
            Write-Host "Error: $($result.error)" -ForegroundColor Red
        }
        if ($result.success) {
            Write-Host "OVERALL: PASSED" -ForegroundColor Green
            exit 0
        }
        else {
            Write-Host "OVERALL: FAILED" -ForegroundColor Red
            $logFile = Join-Path $LogDir "sandbox-test.log"
            if (Test-Path $logFile) {
                Write-Host "--- last 40 lines of sandbox-test.log ---" -ForegroundColor DarkGray
                Get-Content $logFile -Tail 40
            }
            exit 1
        }
    }
    Start-Sleep -Seconds 30
    $elapsed = [int]((Get-Date) - ($deadline.AddMinutes(-$TimeoutMinutes))).TotalMinutes
    Write-Host ("  ... waiting ({0} min elapsed, timeout {1} min)" -f $elapsed, $TimeoutMinutes) -ForegroundColor DarkGray
}

Write-Error "Timed out after $TimeoutMinutes minutes waiting for $resultFile"
