# Create visualizer_settings.json from the example template + profile overrides.

function Install-Settings {
    param(
        [string]$ProjectRoot,
        [string]$StreamDiffusionPath,
        [ValidateSet("Minimal", "Standard", "Full")]
        [string]$Profile,
        [switch]$Force
    )

    $example = Join-Path $ProjectRoot "visualizer_settings.example.json"
    $target = Join-Path $ProjectRoot "visualizer_settings.json"

    if (-not (Test-Path $example)) {
        throw "Example settings not found: $example"
    }

    if ((Test-Path $target) -and -not $Force) {
        Write-SetupOk 'visualizer_settings.json already exists (use -ForceSettings to overwrite)'
        return $target
    }

    $settings = Get-Content $example -Raw | ConvertFrom-Json
    $overrides = Get-ProfileSettings -Profile $Profile
    foreach ($key in $overrides.Keys) {
        $settings.$key = $overrides[$key]
    }
    # Resolved at runtime when null; store absolute path for clarity after setup.
    $settings.streamdiffusion_path = $StreamDiffusionPath

    $json = $settings | ConvertTo-Json -Depth 20
    Set-Content -Path $target -Value $json -Encoding utf8
    Write-SetupOk "Created visualizer_settings.json (profile: $Profile)"
    return $target
}

function Set-StreamDiffusionPathPersist {
    param([string]$Path)
    [Environment]::SetEnvironmentVariable("STREAMDIFFUSION_PATH", $Path, "User")
    Write-SetupOk "Persisted STREAMDIFFUSION_PATH for current user"
}
