# Build isolated package venvs for installer freezing (without touching dev .venv).

function Get-PackageVenvRelName {
    param(
        [ValidateSet("cu128", "cu121")]
        [string]$Cuda = "cu128"
    )
    if ($Cuda -eq "cu128") { return ".venv" }
    return ".venv-$Cuda"
}

function Get-PackageQwenVenvRelName {
    param(
        [ValidateSet("cu128", "cu121")]
        [string]$Cuda = "cu128"
    )
    if ($Cuda -eq "cu128") { return ".venv-qwen" }
    return ".venv-qwen-$Cuda"
}

function Get-InstallerBuildPaths {
    <#
    .SYNOPSIS Resolve stage/runtime/payload/dist paths for a CUDA profile.
    cu128 keeps legacy paths (build/stage, core-runtime-<ver>.zip) for compatibility.
    #>
    param(
        [string]$InstallerRoot,
        [string]$Version = "0.1.0",
        [ValidateSet("cu128", "cu121")]
        [string]$Cuda = "cu128"
    )

    $build = Join-Path $InstallerRoot "build"
    $payloadDir = Join-Path $build "payloads"
    $suffix = if ($Cuda -eq "cu128") { "" } else { "-$Cuda" }

    if ($Cuda -eq "cu128") {
        $stageDir = Join-Path $build "stage"
        $mainEnvDir = Join-Path $build "runtime\main-env"
        $qwenEnvDir = Join-Path $build "runtime\qwen-env"
    }
    else {
        $stageDir = Join-Path $build "stage-$Cuda"
        $mainEnvDir = Join-Path $build "runtime\$Cuda\main-env"
        $qwenEnvDir = Join-Path $build "runtime\$Cuda\qwen-env"
    }

    $outputBaseName = "Sonaflow-Setup-$Version$suffix-offline"

    return @{
        Cuda           = $Cuda
        PayloadSuffix  = $suffix
        StageDir       = $stageDir
        MainEnvDir     = $mainEnvDir
        QwenEnvDir     = $qwenEnvDir
        PayloadDir     = $payloadDir
        CoreZip        = Join-Path $payloadDir "core-runtime-$Version$suffix.zip"
        ModelsZip      = Join-Path $payloadDir "models-core-$Version$suffix.zip"
        QwenZip        = Join-Path $payloadDir "qwen-runtime-$Version$suffix.zip"
        DistDir        = Join-Path $build "dist\$outputBaseName"
        OutputBaseName = $outputBaseName
        IsccCudaSuffix = $suffix
    }
}

function Ensure-PackageMainEnv {
    param(
        [string]$ProjectRoot,
        [ValidateSet("cu128", "cu121")]
        [string]$Cuda = "cu128",
        [switch]$Refresh
    )

    $SetupDir = Join-Path $ProjectRoot "scripts\setup"
    . (Join-Path $SetupDir "Common.ps1")
    . (Join-Path $SetupDir "MainEnv.ps1")

    $Lock = Get-SetupLock -ProjectRoot $ProjectRoot
    $venvRel = Get-PackageVenvRelName -Cuda $Cuda
    $venvPath = Join-Path $ProjectRoot $venvRel
    $python = Join-Path $venvPath "Scripts\python.exe"

    if ($Refresh -or -not (Test-Path $python)) {
        Write-Step "Building package venv $venvRel (CUDA $Cuda) ..."
        Install-MainEnv -ProjectRoot $ProjectRoot -Lock $Lock -Cuda $Cuda -VenvRelName $venvRel
    }
    else {
        Write-Ok "Using existing package venv $venvRel ($venvPath)"
    }
    return (Resolve-Path $venvPath).Path
}

function Ensure-PackageQwenEnv {
    param(
        [string]$ProjectRoot,
        [ValidateSet("cu128", "cu121")]
        [string]$Cuda = "cu128",
        [switch]$Refresh,
        [switch]$SkipModelDownload
    )

    $SetupDir = Join-Path $ProjectRoot "scripts\setup"
    . (Join-Path $SetupDir "Common.ps1")
    . (Join-Path $SetupDir "Sidecar.ps1")

    $Lock = Get-SetupLock -ProjectRoot $ProjectRoot
    $venvRel = Get-PackageQwenVenvRelName -Cuda $Cuda
    $venvPath = Join-Path $ProjectRoot $venvRel
    $python = Join-Path $venvPath "Scripts\python.exe"

    if ($Refresh -or -not (Test-Path $python)) {
        Write-Step "Building Qwen package venv $venvRel (CUDA $Cuda) ..."
        Install-Sidecar -ProjectRoot $ProjectRoot -Lock $Lock -Cuda $Cuda `
            -VenvRelName $venvRel -SkipModelDownload:$SkipModelDownload
    }
    else {
        Write-Ok "Using existing Qwen package venv $venvRel ($venvPath)"
    }
    return (Resolve-Path $venvPath).Path
}
