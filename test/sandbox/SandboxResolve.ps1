# Resolve WindowsSandbox.exe (System32 link may be missing on Win11 24H2).
function Get-WindowsSandboxExe {
    $system32 = Join-Path $env:SystemRoot "System32\WindowsSandbox.exe"
    if (Test-Path $system32) {
        return $system32
    }
    $fromWinSxS = cmd /c "dir /s /b $env:SystemRoot\WinSxS\WindowsSandbox.exe 2>nul" |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and (Test-Path $_) } |
        Select-Object -First 1
    if ($fromWinSxS) {
        Write-Host "Using WinSxS launcher: $fromWinSxS" -ForegroundColor Yellow
        Write-Host "System32 link missing - Sandbox feature may be partially installed." -ForegroundColor Yellow
        return $fromWinSxS
    }
    return $null
}

function Get-SandboxFeatureState {
    try {
        $feat = Get-CimInstance Win32_OptionalFeature -Filter "Name='Containers-DisposableClientVM'" -ErrorAction Stop
        switch ($feat.InstallState) {
            1 { return "Enabled" }
            2 { return "Disabled" }
            3 { return "EnablePending" }
            4 { return "DisablePending" }
            default { return "Unknown($($feat.InstallState))" }
        }
    }
    catch {
        return "Unknown"
    }
}
