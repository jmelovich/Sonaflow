# Clone or update the StreamDiffusion checkout (default Blackwell fork or upstream).

function Get-StreamDiffusionCheckoutInfo {
    param([string]$CheckoutPath)

    $info = @{
        Branch = $null
        RemoteUrl = $null
        IsBlackwellFork = $false
        IsUpstreamFork = $false
    }

    if (-not (Test-Path (Join-Path $CheckoutPath ".git"))) {
        return $info
    }

    Push-Location $CheckoutPath
    try {
        $branch = (& git rev-parse --abbrev-ref HEAD 2>$null)
        if ($LASTEXITCODE -eq 0) { $info.Branch = $branch.Trim() }

        $remote = (& git remote get-url origin 2>$null)
        if ($LASTEXITCODE -eq 0) { $info.RemoteUrl = $remote.Trim() }

        if ($info.RemoteUrl -like "*jmelovich/StreamDiffusion*") {
            $info.IsBlackwellFork = $true
        }
        if ($info.RemoteUrl -like "*cumulo-autumn/StreamDiffusion*") {
            $info.IsUpstreamFork = $true
        }
        if ($info.Branch -eq "blackwell-tensorrt") {
            $info.IsBlackwellFork = $true
        }
    }
    finally {
        Pop-Location
    }

    return $info
}

function Resolve-StreamDiffusionRepo {
    param(
        [object]$Lock,
        [object]$Gpu,
        [string]$RepoOverride,
        [string]$RefOverride,
        [switch]$UseUpstream,
        [switch]$ForceBlackwellFork,
        [switch]$Quiet
    )

    $blackwellRepo = $Lock.streamdiffusion_repo
    $blackwellRef = $Lock.streamdiffusion_ref
    $upstreamRepo = if ($Lock.streamdiffusion_upstream_repo) {
        $Lock.streamdiffusion_upstream_repo
    }
    else {
        "https://github.com/cumulo-autumn/StreamDiffusion.git"
    }
    $upstreamRef = if ($Lock.streamdiffusion_upstream_ref) {
        $Lock.streamdiffusion_upstream_ref
    }
    else {
        "main"
    }

    if ($UseUpstream) {
        Write-SetupOk "Using upstream StreamDiffusion: $upstreamRepo @ $upstreamRef"
        return @{
            Repo = $upstreamRepo
            Ref = $upstreamRef
            UpstreamMode = $true
            IsBlackwellFork = $false
        }
    }

    $repo = if ($RepoOverride) { $RepoOverride } else { $blackwellRepo }
    $ref = if ($RefOverride) { $RefOverride } else { $blackwellRef }
    $upstreamMode = ($repo -like "*cumulo-autumn/StreamDiffusion*")

    $isDefaultBlackwellFork = ($repo -eq $blackwellRepo) -and ($ref -eq $blackwellRef)

    if ($isDefaultBlackwellFork -and $Gpu.Available -and -not $Gpu.IsBlackwellOrNewer -and -not $ForceBlackwellFork) {
        Write-SetupWarn @"
The default StreamDiffusion branch (blackwell-tensorrt) was tested only on an RTX 5080
(Blackwell TensorRT + PyTorch cu128). Your GPU is $($Gpu.Name) — this fork may not work.
"@
        if (-not $Quiet) {
            Write-Host ""
            Write-Host "  [1] Continue with blackwell-tensorrt anyway (unsupported on this GPU)" -ForegroundColor Yellow
            Write-Host "  [2] Clone upstream cumulo-autumn/StreamDiffusion instead (recommended)" -ForegroundColor Green
            Write-Host "  [3] Cancel setup" -ForegroundColor White
            Write-Host ""
            $choice = Read-Host "Choice [1/2/3] (default 2)"
            switch ($choice) {
                '1' {
                    Write-SetupWarn "Continuing with blackwell-tensorrt on a non-Blackwell GPU."
                }
                '3' {
                    throw "Setup cancelled. Use -UseUpstreamStreamDiffusion or see docs/SETUP.md for manual install."
                }
                default {
                    $repo = $upstreamRepo
                    $ref = $upstreamRef
                    $upstreamMode = $true
                    Write-SetupOk "Switching to upstream: $repo @ $ref"
                }
            }
        }
        else {
            Write-SetupWarn "Non-interactive mode: continuing with blackwell-tensorrt. Pass -UseUpstreamStreamDiffusion to use upstream instead."
        }
    }
    elseif ($isDefaultBlackwellFork -and $Gpu.Available -and $Gpu.IsBlackwellOrNewer) {
        Write-SetupOk "Blackwell-class GPU detected — using blackwell-tensorrt fork."
    }

    return @{
        Repo = $repo
        Ref = $ref
        UpstreamMode = [bool]$upstreamMode
        IsBlackwellFork = -not $upstreamMode
    }
}

function Install-StreamDiffusion {
    param(
        [string]$ProjectRoot,
        [object]$Lock,
        [string]$Repo,
        [string]$Ref,
        [switch]$Update
    )

    $parent = Split-Path -Parent $ProjectRoot
    $target = Join-Path $parent "StreamDiffusion"

    # Honour a pre-set path (e.g. sandbox mapped folder or existing checkout).
    if ($env:STREAMDIFFUSION_PATH -and (Test-Path (Join-Path $env:STREAMDIFFUSION_PATH "pyproject.toml"))) {
        $target = $env:STREAMDIFFUSION_PATH
        Write-SetupOk "Using STREAMDIFFUSION_PATH: $target"
    }
    elseif (-not (Test-Path $target)) {
        Write-Host "Cloning StreamDiffusion to $target ..."
        Write-Host "  repo: $Repo"
        Write-Host "  ref:  $Ref"
        & git clone --branch $Ref --depth 1 $Repo $target
        if ($LASTEXITCODE -ne 0) {
            throw "git clone failed for $Repo (ref $Ref)."
        }
        Write-SetupOk "Cloned StreamDiffusion ($Ref)"
    }
    else {
        Write-SetupOk "StreamDiffusion already present at $target"
        if ($Update) {
            Push-Location $target
            try {
                Write-Host "Fetching latest for branch $Ref ..."
                & git fetch origin $Ref --depth 1 2>$null
                & git checkout $Ref 2>$null
                & git pull origin $Ref 2>$null
            }
            finally {
                Pop-Location
            }
        }
    }

    if (-not (Test-Path (Join-Path $target "pyproject.toml"))) {
        throw "StreamDiffusion checkout at $target does not look valid (pyproject.toml missing)."
    }
    $pyproject = Get-Content (Join-Path $target "pyproject.toml") -Raw
    if ($pyproject -notmatch '\[project\]' -and $pyproject -notmatch '\[tool\.poetry\.name\]') {
        throw @"
StreamDiffusion checkout at $target has an invalid pyproject.toml (missing [project]).
Use the Blackwell fork: https://github.com/jmelovich/StreamDiffusion/tree/blackwell-tensorrt
Or upstream: https://github.com/cumulo-autumn/StreamDiffusion
"@
    }

    Set-StreamDiffusionEnv -Path (Resolve-Path $target).Path
    return $env:STREAMDIFFUSION_PATH
}
