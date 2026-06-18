# Prefetch HuggingFace models into the StreamDiffusion Poetry cache.

function Install-Models {
    param(
        [string]$StreamDiffusionPath,
        [object]$Lock
    )

    $models = @($Lock.sd_model, $Lock.clap_model)

    $modelList = ($models | ForEach-Object { "'$_'" }) -join ", "
    $script = @"
from huggingface_hub import snapshot_download
for model_id in ($modelList):
    print(f'Downloading {model_id}...')
    snapshot_download(model_id)
print('Done.')
"@

    Push-Location $StreamDiffusionPath
    try {
        Write-Host "Prefetching models: $($models -join ', ') ..."
        poetry run python -c $script
        if ($LASTEXITCODE -ne 0) {
            throw "Model prefetch failed."
        }
        Write-SetupOk "Models prefetched"
    }
    finally {
        Pop-Location
    }
}
