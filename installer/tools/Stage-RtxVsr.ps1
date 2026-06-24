<#
.SYNOPSIS
    Copy RTX Video SDK runtime files into the staged install layout.

.DESCRIPTION
    Requires a full RTX Video SDK install on the build machine (NV_RTX_VIDEO_SDK).
    Copies nvngx_vsr.dll into thirdparty/rtx-video-sdk/ and ships NOTICES.txt.
    Fails if the SDK or feature DLL is missing.

.EXAMPLE
    .\Stage-RtxVsr.ps1 -StageDir ..\build\stage
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$StageDir,
    [string]$SdkRoot = $env:NV_RTX_VIDEO_SDK
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Common.ps1")

$StageDir = (Resolve-Path $StageDir).Path
$InstallerRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Test-FullRtxVideoSdkRoot {
    param([string]$Root)
    if (-not $Root) { return $false }
    return Test-Path (Join-Path $Root "include\nvsdk_ngx.h")
}

function Get-VsrFeatureDll {
    param([string]$Root)
    foreach ($variant in @("rel", "dev")) {
        $dll = Join-Path $Root "bin\Windows\x64\$variant\nvngx_vsr.dll"
        if (Test-Path $dll) {
            return @{ Path = $dll; Variant = $variant }
        }
    }
    return $null
}

if (-not (Test-FullRtxVideoSdkRoot $SdkRoot)) {
    throw @"
RTX Video SDK not found on the build machine.

Download and extract the SDK from:
  https://developer.nvidia.com/rtx-video-sdk

Set NV_RTX_VIDEO_SDK to the SDK root (directory containing include/, bin/, lib/).
Current value: $(if ($SdkRoot) { $SdkRoot } else { '(not set)' })
"@
}

$feature = Get-VsrFeatureDll $SdkRoot
if (-not $feature) {
    throw "nvngx_vsr.dll not found under $SdkRoot\bin\Windows\x64\rel or dev\"
}

Write-Step "Staging bundled RTX VSR runtime (from $($feature.Variant) build)"

$bundleRoot = Join-Path $StageDir "thirdparty\rtx-video-sdk"
$bundleDllDir = Join-Path $bundleRoot "bin\Windows\x64\rel"
New-Item -ItemType Directory -Force -Path $bundleDllDir | Out-Null

Copy-Item $feature.Path (Join-Path $bundleDllDir "nvngx_vsr.dll") -Force

$noticesSrc = Join-Path $InstallerRoot "thirdparty\rtx-video-sdk\NOTICES.txt"
if (-not (Test-Path $noticesSrc)) {
    throw "Missing NVIDIA attribution file: $noticesSrc"
}
Copy-Item $noticesSrc (Join-Path $bundleRoot "NOTICES.txt") -Force

Write-Ok "thirdparty\rtx-video-sdk\bin\Windows\x64\rel\nvngx_vsr.dll"
