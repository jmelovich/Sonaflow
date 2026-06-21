# Create Sonaflow's main .venv and install pinned dependencies.

function Get-VenvPython {
    param(
        [string]$ProjectRoot,
        [string]$VenvRelName = ".venv"
    )
    $py = Join-Path $ProjectRoot (Join-Path $VenvRelName "Scripts\python.exe")
    if (-not (Test-Path $py)) {
        throw "Python venv not found at $py — run setup.ps1 first."
    }
    return (Resolve-Path $py).Path
}

function Install-MainEnv {
    param(
        [string]$ProjectRoot,
        [object]$Lock,
        [ValidateSet("cu128", "cu121")]
        [string]$Cuda = "cu128",
        [string]$VenvRelName = ".venv",
        [switch]$SkipTensorRT
    )

    $requirements = Join-Path $ProjectRoot "music_visualizer\requirements.txt"
    if (-not (Test-Path $requirements)) {
        throw "Requirements file not found: $requirements"
    }

    $venvDir = Join-Path $ProjectRoot $VenvRelName
    $python = Join-Path $venvDir "Scripts\python.exe"
    $pip = Join-Path $venvDir "Scripts\pip.exe"

    if (-not (Test-Path $python)) {
        Write-Host "Creating Sonaflow venv at $venvDir ..."
        & python -m venv $venvDir
        if ($LASTEXITCODE -ne 0) { throw "python -m venv failed for $venvDir" }
        Write-SetupOk "Created $VenvRelName"
    }
    else {
        Write-SetupOk "Using existing .venv"
    }

    $torchIndexUrl = if ($Cuda -eq "cu121") {
        $Lock.torch_index_url_upstream
    }
    else {
        $Lock.torch_index_url
    }
    if (-not $torchIndexUrl) {
        throw "Missing torch index URL in setup.lock.json for CUDA profile $Cuda"
    }

    Write-Host "Upgrading pip ..."
    & $python -m pip install --upgrade pip wheel setuptools
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

    Write-Host "Installing PyTorch ($Cuda) from $torchIndexUrl ..."
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $pip uninstall -y torch torchvision torchaudio 2>&1 | Out-Null
    & $pip install torch torchvision torchaudio --index-url $torchIndexUrl
    $ErrorActionPreference = $prevEap
    if ($LASTEXITCODE -ne 0) {
        throw "torch install failed (index: $torchIndexUrl)."
    }
    Write-SetupOk "PyTorch installed from $torchIndexUrl"

    if (-not $SkipTensorRT) {
        Write-Host "Installing TensorRT 10 (tensorrt-cu12) from NVIDIA index ..."
        & $pip install `
            "tensorrt-cu12==10.16.1.11" `
            "tensorrt-cu12-libs==10.16.1.11" `
            "tensorrt-cu12-bindings==10.16.1.11" `
            --extra-index-url https://pypi.nvidia.com
        if ($LASTEXITCODE -ne 0) {
            throw "TensorRT install failed — try -SkipTensorRT or see docs/SETUP.md"
        }
        Write-SetupOk "TensorRT 10 installed"
    }
    else {
        Write-SetupWarn "Skipping TensorRT install (-SkipTensorRT)"
    }

    Write-Host "Installing Sonaflow requirements ..."
    & $pip install -r $requirements
    if ($LASTEXITCODE -ne 0) {
        throw "pip install -r requirements.txt failed."
    }
    Write-SetupOk "Requirements installed"

    Write-Host "Installing compel (no-deps) ..."
    & $pip install --no-deps "compel==$($Lock.compel)" "pyparsing>=3.0"
    if ($LASTEXITCODE -ne 0) {
        throw "compel install failed."
    }
    Write-SetupOk "compel $($Lock.compel) installed"

    Write-Host "Smoke-testing imports ..."
    $env:PYTHONPATH = $ProjectRoot
    & $python -c @"
import numpy, torch, streamdiffusion, cv2, aubio, sounddevice, soundfile
from music_visualizer.generation.sd_wrapper import StreamDiffusionWrapper
print('imports ok', torch.__version__, numpy.__version__)
"@
    if ($LASTEXITCODE -ne 0) {
        throw "Import smoke test failed."
    }
    Write-SetupOk "Import smoke test passed"
}
