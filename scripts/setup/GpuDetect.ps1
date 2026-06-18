# Detect NVIDIA GPU via nvidia-smi and recommend a setup profile.

function Test-IsBlackwellOrNewerGpu {
    param(
        [string]$GpuName,
        [string]$ComputeCap = $null
    )

    if ($GpuName) {
        # Consumer Blackwell (RTX 50 series) and datacenter Blackwell naming.
        if ($GpuName -match 'RTX\s*50|5080|5090|5070|5060|Blackwell|\bB200\b|\bB100\b|GB200') {
            return $true
        }
    }

    if ($ComputeCap -and $ComputeCap -match '^(\d+)\.(\d+)') {
        $major = [int]$Matches[1]
        if ($major -ge 12) {
            return $true
        }
    }

    return $false
}

function Get-GpuInfo {
    $info = @{
        Available = $false
        Name = $null
        DriverVersion = $null
        ComputeCap = $null
        VramMb = 0
        VramGb = 0.0
        IsBlackwellOrNewer = $false
        RecommendedProfile = "Minimal"
    }

    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        Write-SetupWarn 'nvidia-smi not found - using Minimal profile (no GPU assumptions).'
        return $info
    }

    try {
        $query = & nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $query) {
            Write-SetupWarn 'nvidia-smi failed - using Minimal profile.'
            return $info
        }
        $line = ($query | Select-Object -First 1).Trim()
        if ($line -match '^(.+?),\s*([\d.]+),\s*(\d+),\s*([\d.]+)') {
            $info.Available = $true
            $info.Name = $Matches[1].Trim()
            $info.DriverVersion = $Matches[2].Trim()
            $info.VramMb = [int]$Matches[3]
            $info.ComputeCap = $Matches[4].Trim()
        }
        elseif ($line -match '^(.+?),\s*([\d.]+),\s*(\d+)') {
            $info.Available = $true
            $info.Name = $Matches[1].Trim()
            $info.DriverVersion = $Matches[2].Trim()
            $info.VramMb = [int]$Matches[3]
        }
        if ($info.Available) {
            $info.VramGb = [math]::Round($info.VramMb / 1024.0, 1)
            $info.IsBlackwellOrNewer = Test-IsBlackwellOrNewerGpu -GpuName $info.Name -ComputeCap $info.ComputeCap
            $arch = if ($info.IsBlackwellOrNewer) { "Blackwell+" } else { "pre-Blackwell" }
            Write-SetupOk "GPU: $($info.Name) ($($info.VramGb) GB VRAM, driver $($info.DriverVersion), $arch)"
        }
    }
    catch {
        Write-SetupWarn "Could not parse nvidia-smi output: $_"
        return $info
    }

    if (-not $info.Available) {
        return $info
    }

    if ($info.VramGb -lt 8) {
        $info.RecommendedProfile = "Minimal"
    }
    elseif ($info.VramGb -lt 14) {
        $info.RecommendedProfile = "Standard"
    }
    else {
        $info.RecommendedProfile = "Full"
    }

    return $info
}

function Get-ProfileSettings {
    param(
        [ValidateSet("Minimal", "Standard", "Full")]
        [string]$Profile
    )

    switch ($Profile) {
        "Minimal" {
            return @{
                acceleration = "none"
                width = 512
                height = 512
                audio_source = "mic"
                clap = $false
                qwen = $false
                upscale = $false
                gl_present = $false
            }
        }
        "Standard" {
            return @{
                acceleration = "tensorrt"
                width = 512
                height = 512
                audio_source = "loopback"
                clap = $true
                qwen = $false
                upscale = $false
                gl_present = $true
            }
        }
        "Full" {
            return @{
                acceleration = "tensorrt"
                width = 512
                height = 512
                audio_source = "loopback"
                clap = $true
                qwen = $true
                upscale = $true
                gl_present = $true
                upscale_factor = 4
            }
        }
    }
}
