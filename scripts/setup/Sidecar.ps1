# Create the isolated Qwen2-Audio sidecar venv and download the model checkpoint.

function Install-Sidecar {
    param(
        [string]$ProjectRoot,
        [object]$Lock,
        [ValidateSet("cu128", "cu121")]
        [string]$Cuda = "cu128",
        [string]$VenvRelName = ".venv-qwen",
        [switch]$SkipModelDownload
    )

    $VenvDir = Join-Path $ProjectRoot $VenvRelName
    $Python = Join-Path $VenvDir "Scripts\python.exe"
    $ReqFile = Join-Path $ProjectRoot "music_visualizer\qwen_sidecar\requirements.txt"

    if (-not (Test-Path $Python)) {
        Write-Host "Creating sidecar venv at $VenvDir ..."
        & python -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            throw "python -m venv failed for $VenvDir"
        }
    }
    else {
        Write-SetupOk "Sidecar venv already exists"
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

    Write-Host "Upgrading pip in sidecar venv ..."
    & $Python -m pip install --upgrade pip wheel setuptools
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed in sidecar venv." }

    Write-Host "Installing PyTorch ($Cuda) in sidecar venv ..."
    & $Python -m pip install torch torchvision --index-url $torchIndexUrl
    if ($LASTEXITCODE -ne 0) { throw "torch install failed in sidecar venv." }

    Write-Host "Installing sidecar requirements ..."
    & $Python -m pip install -r $ReqFile
    if ($LASTEXITCODE -ne 0) { throw "sidecar requirements install failed." }
    Write-SetupOk "Sidecar Python env ready ($Python)"

    if (-not $SkipModelDownload) {
        $modelId = $Lock.qwen_model
        Write-Host "Downloading $modelId (~16 GB on first run) ..."
        & $Python -c "from huggingface_hub import snapshot_download; snapshot_download('$modelId')"
        if ($LASTEXITCODE -ne 0) {
            throw "Qwen model download failed for $modelId"
        }
        Write-SetupOk "Qwen model cached"
    }
    else {
        Write-SetupWarn 'Skipped Qwen model download (-SkipModels)'
    }

    return $Python
}
