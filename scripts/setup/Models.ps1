# Prefetch HuggingFace models into the HuggingFace cache (via Sonaflow .venv).

function Install-Models {
    param(
        [string]$ProjectRoot,
        [object]$Lock
    )

    . (Join-Path $PSScriptRoot "MainEnv.ps1")
    $python = Get-VenvPython -ProjectRoot $ProjectRoot

    $models = @($Lock.sd_model, $Lock.clap_model)

    $modelList = ($models | ForEach-Object { "'$_'" }) -join ", "
    $script = @"
from huggingface_hub import snapshot_download
for model_id in ($modelList):
    print(f'Downloading {model_id}...')
    snapshot_download(model_id)
print('Done.')
"@

    Write-Host "Prefetching models: $($models -join ', ') ..."
    & $python -c $script
    if ($LASTEXITCODE -ne 0) {
        throw "Model prefetch failed."
    }
    Write-SetupOk "Models prefetched"
}
