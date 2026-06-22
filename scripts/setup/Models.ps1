# Prefetch HuggingFace models into the HuggingFace cache (via Sonaflow .venv).

function Install-Models {
    param(
        [string]$ProjectRoot,
        [object]$Lock,
        [switch]$IncludeQwen
    )

    . (Join-Path $PSScriptRoot "MainEnv.ps1")
    $python = Get-VenvPython -ProjectRoot $ProjectRoot

    $models = @($Lock.sd_model, $Lock.clap_model)
    if ($IncludeQwen -and $Lock.qwen_model) {
        $models += $Lock.qwen_model
    }

    $prefetchScript = Join-Path $PSScriptRoot "hf_prefetch.py"
    Write-Host "Prefetching models: $($models -join ', ') ..."
    & $python $prefetchScript @models
    if ($LASTEXITCODE -ne 0) {
        throw "Model prefetch failed."
    }
    Write-SetupOk "Models prefetched"
}
