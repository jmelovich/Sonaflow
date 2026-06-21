<#
.SYNOPSIS
    Split a staged install layout (+ models, optional Qwen, optional engines)
    into versioned download payloads, then emit:
      * dist\manifest.json          - for the release host + app-side engine fetch
      * generated\payloads.iss      - ISPP defines + engine table embedded into the installer

.DESCRIPTION
    The web bootstrapper is built alongside these payloads, so it can embed the
    exact version / URLs / sha256 / available-engine list at compile time and
    avoid runtime JSON parsing. This script produces both the host artifacts and
    that compile-time include.

    Archive internal layouts (so the installer extracts to clean targets):
      core-runtime-<ver>.zip   -> {app}            (Sonaflow.exe, app\, runtime\main-env\, native\)
      models-core-<ver>.zip    -> {data}\models    (hub\models--stabilityai--sd-turbo, --madebyollin--taesd, --laion--larger_clap_music)
      qwen-runtime-<ver>.zip    -> {app}            (runtime\qwen-env\)            [optional]
      qwen-model-<ver>.zip      -> {data}\models    (hub\models--Qwen--Qwen2-Audio-7B-Instruct) [optional]
      engine-<res>-<ver>.zip    -> {data}\engines   (<res>\<descriptor>\unet.engine + .built_resolution)

.EXAMPLE
    .\Build-Payloads.ps1 -StageDir ..\build\stage -Version 0.1.0 `
        -BaseUrl https://downloads.example.com/sonaflow/0.1.0 `
        -ModelsHubDir $env:LOCALAPPDATA\Sonaflow\models\hub -OutDir ..\build\dist
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$StageDir,
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$BaseUrl,
    [string]$ModelsHubDir,
    [string[]]$CoreModels = @(
        "models--stabilityai--sd-turbo",
        "models--madebyollin--taesd",
        "models--laion--larger_clap_music"
    ),
    [string]$QwenEnvDir,
    [string]$QwenModelHubDir,
    [string]$QwenModel = "models--Qwen--Qwen2-Audio-7B-Instruct",
    [string]$EngineDir,
    [string]$OutDir = (Join-Path $PSScriptRoot "..\build\dist"),
    [string]$GeneratedDir = (Join-Path $PSScriptRoot "..\generated"),
    [string]$GpuArch = "sm_120",
    [string]$TorchCu = "cu128",
    [string]$Trt = "10.x",
    [string]$DriverMin = "570.20"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Common.ps1")
Add-Type -AssemblyName System.IO.Compression.FileSystem

$BaseUrl = $BaseUrl.TrimEnd("/")
$StageDir = (Resolve-Path $StageDir).Path
New-Item -ItemType Directory -Force -Path $OutDir, $GeneratedDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
$GeneratedDir = (Resolve-Path $GeneratedDir).Path

$payloads = [System.Collections.ArrayList]::new()
$engines = [System.Collections.ArrayList]::new()

function New-Zip {
    param([string]$SourceDir, [string]$ZipPath)
    if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
    Write-Host "    zipping -> $(Split-Path $ZipPath -Leaf)"
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $SourceDir, $ZipPath,
        [System.IO.Compression.CompressionLevel]::Optimal, $false)
}

function Add-Payload {
    param([string]$Id, [string]$ZipPath)
    $hash = (Get-FileHash -Algorithm SHA256 -Path $ZipPath).Hash.ToLower()
    $size = (Get-Item $ZipPath).Length
    $name = Split-Path $ZipPath -Leaf
    [void]$payloads.Add([ordered]@{
            id     = $Id
            file   = $name
            url    = "$BaseUrl/$name"
            sha256 = $hash
            size   = $size
        })
    Write-Ok "$Id  ($([math]::Round($size/1MB,1)) MB)  sha256=$($hash.Substring(0,12))..."
}

# --- core-runtime: the whole stage (sans any qwen runtime that lives elsewhere) ---
Write-Step "Packaging core-runtime"
$coreSonaExe = Join-Path $StageDir "Sonaflow.exe"
if (-not (Test-Path $coreSonaExe)) {
    Write-Warn2 "Sonaflow.exe not found in stage - run Stage-Local.ps1 first (continuing anyway)."
}
$coreZip = Join-Path $OutDir "core-runtime-$Version.zip"
New-Zip -SourceDir $StageDir -ZipPath $coreZip
Add-Payload -Id "core-runtime" -ZipPath $coreZip

# --- models-core: selected HF snapshots into a hub\ tree ---
if ($ModelsHubDir -and (Test-Path $ModelsHubDir)) {
    Write-Step "Packaging models-core"
    $tmpHub = Join-Path $env:TEMP ("sf-hub-" + [guid]::NewGuid().ToString("N"))
    $hubDst = Join-Path $tmpHub "hub"
    New-Item -ItemType Directory -Force -Path $hubDst | Out-Null
    foreach ($m in $CoreModels) {
        $src = Join-Path $ModelsHubDir $m
        if (Test-Path $src) {
            Copy-Item $src (Join-Path $hubDst $m) -Recurse -Force
            Write-Host "    + $m"
        }
        else { Write-Warn2 "model not present, skipped: $m" }
    }
    Get-ChildItem $ModelsHubDir -Filter "version*.txt" -ErrorAction SilentlyContinue |
        ForEach-Object { Copy-Item $_.FullName (Join-Path $hubDst $_.Name) -Force }
    $modelsZip = Join-Path $OutDir "models-core-$Version.zip"
    New-Zip -SourceDir $tmpHub -ZipPath $modelsZip
    Add-Payload -Id "models-core" -ZipPath $modelsZip
    Remove-Item $tmpHub -Recurse -Force -ErrorAction SilentlyContinue
}
else {
    Write-Warn2 "No -ModelsHubDir given/found - skipping models-core payload."
}

# --- optional qwen runtime ---
$qwenAvailable = $false
if ($QwenEnvDir -and (Test-Path $QwenEnvDir)) {
    Write-Step "Packaging qwen-runtime"
    $tmpQ = Join-Path $env:TEMP ("sf-qwen-" + [guid]::NewGuid().ToString("N"))
    $qDst = Join-Path $tmpQ "runtime\qwen-env"
    New-Item -ItemType Directory -Force -Path (Split-Path $qDst) | Out-Null
    Copy-Item $QwenEnvDir $qDst -Recurse -Force
    $qZip = Join-Path $OutDir "qwen-runtime-$Version.zip"
    New-Zip -SourceDir $tmpQ -ZipPath $qZip
    Add-Payload -Id "qwen-runtime" -ZipPath $qZip
    Remove-Item $tmpQ -Recurse -Force -ErrorAction SilentlyContinue
    $qwenAvailable = $true
}

if ($QwenModelHubDir -and (Test-Path (Join-Path $QwenModelHubDir $QwenModel))) {
    Write-Step "Packaging qwen-model"
    $tmpQM = Join-Path $env:TEMP ("sf-qwenm-" + [guid]::NewGuid().ToString("N"))
    $qmHub = Join-Path $tmpQM "hub"
    New-Item -ItemType Directory -Force -Path $qmHub | Out-Null
    Copy-Item (Join-Path $QwenModelHubDir $QwenModel) (Join-Path $qmHub $QwenModel) -Recurse -Force
    $qmZip = Join-Path $OutDir "qwen-model-$Version.zip"
    New-Zip -SourceDir $tmpQM -ZipPath $qmZip
    Add-Payload -Id "qwen-model" -ZipPath $qmZip
    Remove-Item $tmpQM -Recurse -Force -ErrorAction SilentlyContinue
}

# --- optional prebuilt TensorRT engines (one zip per resolution) ---
if ($EngineDir -and (Test-Path $EngineDir)) {
    Write-Step "Packaging prebuilt engines"
    Get-ChildItem $EngineDir -Directory | Where-Object { $_.Name -match '^\d+x\d+$' } | ForEach-Object {
        $res = $_.Name
        # Only ship a resolution that actually has a built unet.engine somewhere inside.
        $hasEngine = Get-ChildItem $_.FullName -Recurse -Filter "unet.engine" -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $hasEngine) { Write-Warn2 "no unet.engine under $res - skipped"; return }
        $tmpE = Join-Path $env:TEMP ("sf-eng-" + [guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Force -Path $tmpE | Out-Null
        Copy-Item $_.FullName (Join-Path $tmpE $res) -Recurse -Force
        $eZip = Join-Path $OutDir "engine-$res-$Version.zip"
        New-Zip -SourceDir $tmpE -ZipPath $eZip
        $hash = (Get-FileHash -Algorithm SHA256 -Path $eZip).Hash.ToLower()
        $size = (Get-Item $eZip).Length
        [void]$engines.Add([ordered]@{
                gpu_arch   = $GpuArch
                trt        = $Trt
                torch_cu   = $TorchCu
                driver_min = $DriverMin
                resolution = $res
                file       = (Split-Path $eZip -Leaf)
                url        = "$BaseUrl/$(Split-Path $eZip -Leaf)"
                sha256     = $hash
                size       = $size
            })
        Remove-Item $tmpE -Recurse -Force -ErrorAction SilentlyContinue
        Write-Ok "engine $res ($([math]::Round($size/1MB,1)) MB)"
    }
}

# --- manifest.json (host + app-side) ---
$manifest = [ordered]@{
    schema   = 1
    app      = "Sonaflow"
    version  = $Version
    base_url = $BaseUrl
    payloads = $payloads
    engines  = $engines
}
$manifestPath = Join-Path $OutDir "manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8
Write-Ok "manifest.json -> $manifestPath"

# --- generated\payloads.iss (ISPP defines + engine table for the installer) ---
$sb = [System.Text.StringBuilder]::new()
[void]$sb.AppendLine("; AUTO-GENERATED by Build-Payloads.ps1 - do not edit by hand.")
[void]$sb.AppendLine("#define HavePayloads")
[void]$sb.AppendLine("#define AppVersion `"$Version`"")
[void]$sb.AppendLine("#define PayloadBaseUrl `"$BaseUrl`"")
function Emit-Define {
    param($id, $prefix)
    $p = $payloads | Where-Object { $_.id -eq $id }
    if ($p) {
        [void]$sb.AppendLine("#define ${prefix}File `"$($p.file)`"")
        [void]$sb.AppendLine("#define ${prefix}Url `"$($p.url)`"")
        [void]$sb.AppendLine("#define ${prefix}Sha `"$($p.sha256)`"")
        [void]$sb.AppendLine("#define ${prefix}Size `"$($p.size)`"")
        return $true
    }
    return $false
}
[void](Emit-Define "core-runtime" "CoreRuntime")
if (Emit-Define "models-core" "ModelsCore") { [void]$sb.AppendLine("#define HaveModelsCore") }
if (Emit-Define "qwen-runtime" "QwenRuntime") { [void]$sb.AppendLine("#define HaveQwen") }
[void](Emit-Define "qwen-model" "QwenModel")
$payloadsIss = Join-Path $GeneratedDir "payloads.iss"
Set-Content -Path $payloadsIss -Value $sb.ToString() -Encoding ASCII
Write-Ok "payloads.iss -> $payloadsIss"

# --- generated\engines.iss (Pascal lookup, #included inside [Code]) ---
$eb = [System.Text.StringBuilder]::new()
[void]$eb.AppendLine("{ AUTO-GENERATED by Build-Payloads.ps1 - do not edit by hand. }")
[void]$eb.AppendLine("function FindPrebuiltEngine(Res: String; var Url, Sha: String): Boolean;")
[void]$eb.AppendLine("begin")
[void]$eb.AppendLine("  Result := False;")
foreach ($e in $engines) {
    [void]$eb.AppendLine("  if Res = '$($e.resolution)' then begin")
    [void]$eb.AppendLine("    Url := '$($e.url)'; Sha := '$($e.sha256)'; Result := True; Exit;")
    [void]$eb.AppendLine("  end;")
}
[void]$eb.AppendLine("end;")
$enginesIss = Join-Path $GeneratedDir "engines.iss"
Set-Content -Path $enginesIss -Value $eb.ToString() -Encoding ASCII
Write-Ok "engines.iss -> $enginesIss ($($engines.Count) engine(s))"

Write-Host ""
Write-Host "Payloads ready in: $OutDir" -ForegroundColor Green
Write-Host "Upload the *.zip + manifest.json to: $BaseUrl" -ForegroundColor White
Write-Host "Then compile the installer (generated\payloads.iss is now populated)." -ForegroundColor White
