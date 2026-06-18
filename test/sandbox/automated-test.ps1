# Runs inside Windows Sandbox. Installs prereqs, runs setup.ps1, doctor.ps1, and a smoke benchmark.
# Writes a transcript to the mapped logs folder on the host.

param(
    [ValidateSet("Minimal", "Standard")]
    [string]$Profile = "Minimal"
)

$ErrorActionPreference = "Continue"
$LogDir = "C:\logs"
$RepoSrc = "C:\repo"
$WorkDir = "C:\test"
$LogFile = Join-Path $LogDir "sandbox-test.log"
$ResultFile = Join-Path $LogDir "sandbox-result.json"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Start-Transcript -Path $LogFile -Force

$result = @{
    started_at = (Get-Date -Format "o")
    profile = $Profile
    steps = @()
    success = $false
    error = $null
}

function Add-Step {
    param([string]$Name, [bool]$Ok, [string]$Detail = "")
    $entry = @{ name = $Name; ok = $Ok; detail = $Detail; at = (Get-Date -Format "o") }
    $result.steps += $entry
    $color = if ($Ok) { "Green" } else { "Red" }
    Write-Host ("[{0}] {1}" -f $(if ($Ok) { "PASS" } else { "FAIL" }), $Name) -ForegroundColor $color
    if ($Detail) { Write-Host "       $Detail" -ForegroundColor DarkGray }
}

try {
    Write-Host "=== Sandbox automated test (profile: $Profile) ===" -ForegroundColor Cyan

    # Copy repo to writable workspace
    if (Test-Path $WorkDir) { Remove-Item $WorkDir -Recurse -Force }
    Copy-Item $RepoSrc $WorkDir -Recurse
    Set-Location $WorkDir
    Add-Step "Copy repo to C:\test" $true

    # Use pre-mapped StreamDiffusion if present (sandbox wsb maps host fork read-only)
    if (Test-Path "C:\StreamDiffusion\pyproject.toml") {
        $env:STREAMDIFFUSION_PATH = "C:\StreamDiffusion"
        Write-Host "Using mapped StreamDiffusion at C:\StreamDiffusion"
        Add-Step "Mapped StreamDiffusion" $true
    }

    # Install prerequisites via winget (sandbox has network; may take several minutes)
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        Add-Step "winget available" $false "winget not found in sandbox image"
        throw "winget not available"
    }
    Add-Step "winget available" $true

    Write-Host "Installing Python 3.11 via winget ..."
    & winget install --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements --disable-interactivity
    Add-Step "Install Python" ($LASTEXITCODE -eq 0) "exit=$LASTEXITCODE"

    Write-Host "Installing Git via winget ..."
    & winget install --id Git.Git --accept-package-agreements --accept-source-agreements --disable-interactivity
    Add-Step "Install Git" ($LASTEXITCODE -eq 0) "exit=$LASTEXITCODE"

    # Refresh PATH for new installs
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")

    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        # Common winget Python location
        $candidates = @(
            "$env:LocalAppData\Programs\Python\Python311\python.exe",
            "$env:ProgramFiles\Python311\python.exe"
        )
        foreach ($c in $candidates) {
            if (Test-Path $c) {
                $env:Path = (Split-Path $c) + ";" + $env:Path
                break
            }
        }
    }

    Add-Step "Python on PATH" ($null -ne (Get-Command python -ErrorAction SilentlyContinue))
    Add-Step "git on PATH" ($null -ne (Get-Command git -ErrorAction SilentlyContinue))

    Write-Host "Installing Poetry ..."
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | & python -
    $env:Path = "$env:APPDATA\Python\Scripts;${env:APPDATA}\pypoetry\venv\Scripts;" + $env:Path
    Add-Step "Poetry installed" ($null -ne (Get-Command poetry -ErrorAction SilentlyContinue))

    # GPU check (optional)
    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidia) {
        $smi = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
        Add-Step "nvidia-smi" $true ($smi -join "; ")
    }
    else {
        Add-Step "nvidia-smi" $false "GPU not visible in sandbox"
    }

    # Run setup
    Set-Location $WorkDir
    $setupArgs = @("-Profile", $Profile, "-SkipModels", "-SkipNativeBuild", "-ForceSettings")
    if ($Profile -eq "Minimal") {
        $setupArgs += "-SkipTensorRT"
    }
    Write-Host "Running setup.ps1 $($setupArgs -join ' ') ..."
    & powershell -ExecutionPolicy Bypass -File (Join-Path $WorkDir "setup.ps1") @setupArgs
    Add-Step "setup.ps1" ($LASTEXITCODE -eq 0) "exit=$LASTEXITCODE"

    # Doctor
    & powershell -ExecutionPolicy Bypass -File (Join-Path $WorkDir "doctor.ps1")
    $doctorOk = $LASTEXITCODE -eq 0
    Add-Step "doctor.ps1" $doctorOk "exit=$LASTEXITCODE"

    # Smoke benchmark (Minimal path - no GPU required)
    if ($doctorOk) {
        Write-Host "Running smoke benchmark ..."
        & powershell -ExecutionPolicy Bypass -File (Join-Path $WorkDir "run.ps1") `
            --acceleration none --benchmark 5 --no-qwen --no-clap --no-upscale --audio-source none
        Add-Step "run.ps1 --benchmark 5" ($LASTEXITCODE -eq 0) "exit=$LASTEXITCODE"
    }

    $failed = @($result.steps | Where-Object { -not $_.ok })
    $result.success = ($failed.Count -eq 0)
    $result.finished_at = (Get-Date -Format "o")
}
catch {
    $result.error = $_.Exception.Message
    $result.success = $false
    $result.finished_at = (Get-Date -Format "o")
    Write-Host "FATAL: $_" -ForegroundColor Red
}
finally {
    Stop-Transcript
    $result | ConvertTo-Json -Depth 6 | Set-Content -Path $ResultFile -Encoding utf8
    Write-Host ""
    if ($result.success) {
        Write-Host "SANDBOX TEST PASSED - results in $LogDir" -ForegroundColor Green
    }
    else {
        Write-Host "SANDBOX TEST FAILED - see $LogFile and $ResultFile" -ForegroundColor Red
    }
}
