# Install StreamDiffusion Poetry env + visualizer requirements.

function Install-MainEnv {
    param(
        [string]$ProjectRoot,
        [string]$StreamDiffusionPath,
        [object]$Lock,
        [switch]$SkipTensorRT,
        [switch]$UpstreamMode
    )

    $requirements = Join-Path $ProjectRoot "music_visualizer\requirements.txt"
    if (-not (Test-Path $requirements)) {
        throw "Requirements file not found: $requirements"
    }

    $torchIndexUrl = if ($UpstreamMode -and $Lock.torch_index_url_upstream) {
        $Lock.torch_index_url_upstream
    }
    else {
        $Lock.torch_index_url
    }

    Push-Location $StreamDiffusionPath
    try {
        Write-Host "Installing StreamDiffusion Poetry environment (this may take several minutes) ..."
        if ($SkipTensorRT -or $UpstreamMode) {
            poetry install
        }
        else {
            poetry install --with tensorrt
        }
        if ($LASTEXITCODE -ne 0) {
            Write-SetupWarn 'poetry install failed (stale lock?); running poetry lock then retrying ...'
            poetry lock
            if ($LASTEXITCODE -ne 0) {
                throw "poetry lock failed in $StreamDiffusionPath"
            }
            if ($SkipTensorRT -or $UpstreamMode) {
                poetry install
            }
            else {
                poetry install --with tensorrt
            }
        }
        if ($LASTEXITCODE -ne 0) {
            throw "poetry install failed in $StreamDiffusionPath"
        }
        Write-SetupOk "StreamDiffusion Poetry env ready"

        if ($UpstreamMode) {
            Write-Host "Installing PyTorch ($($Lock.torch_index_upstream)) for upstream StreamDiffusion ..."
        }
        else {
            Write-Host "Installing PyTorch cu128 (Blackwell / RTX 50) ..."
        }
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        poetry run pip uninstall -y torch torchvision torchaudio 2>&1 | Out-Null
        poetry run pip install torch torchvision --index-url $torchIndexUrl 2>&1 | Out-Null
        $ErrorActionPreference = $prevEap
        if ($LASTEXITCODE -ne 0) {
            throw "torch install failed (index: $torchIndexUrl)."
        }
        Write-SetupOk "PyTorch installed from $torchIndexUrl"

        if ($UpstreamMode) {
            Write-Host "Installing upstream TensorRT extras (if available) ..."
            $ErrorActionPreference = "Continue"
            poetry run pip install -e ".[tensorrt]" 2>&1 | Out-Null
            poetry run python -m streamdiffusion.tools.install-tensorrt 2>&1 | Out-Null
            $ErrorActionPreference = $prevEap
            if ($LASTEXITCODE -eq 0) {
                Write-SetupOk "Upstream TensorRT extension installed"
            }
            else {
                Write-SetupWarn "TensorRT extension install skipped or failed — use --acceleration none or see docs/SETUP.md"
            }
        }

        Write-Host "Pinning huggingface-hub, transformers, numpy for diffusers 0.24 ..."
        poetry run pip install huggingface-hub==0.25.2 transformers==4.36.2 "numpy==$($Lock.numpy)" onnxscript
        if ($LASTEXITCODE -ne 0) {
            throw "StreamDiffusion dependency pins failed."
        }
        Write-SetupOk "StreamDiffusion dependency pins applied"

        Write-Host "Installing visualizer requirements ..."
        poetry run pip install -r $requirements
        if ($LASTEXITCODE -ne 0) {
            throw "pip install -r requirements.txt failed."
        }
        Write-SetupOk "Visualizer requirements installed"

        Write-Host "Installing compel (no-deps) + pyparsing ..."
        poetry run pip install --no-deps "compel==$($Lock.compel)" "pyparsing>=3.0"
        if ($LASTEXITCODE -ne 0) {
            throw "compel install failed."
        }
        Write-SetupOk "compel $($Lock.compel) installed"

        Write-Host "Installing torchaudio ($($Lock.torchaudio)) ..."
        poetry run pip install --no-deps "torchaudio==$($Lock.torchaudio)" --index-url $torchIndexUrl
        if ($LASTEXITCODE -ne 0) {
            Write-SetupWarn 'torchaudio install failed - CLAP will use a linear resampling fallback.'
        }
        else {
            Write-SetupOk "torchaudio $($Lock.torchaudio) installed"
        }
    }
    finally {
        Pop-Location
    }
}
