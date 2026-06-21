param(
    [Parameter(Mandatory = $true)][string]$ZipPath,
    [Parameter(Mandatory = $true)][string]$Dest
)

# Robust ZIP extraction for Windows PowerShell 5.1 (.NET Framework).
# Avoids ZipFile.ExtractToDirectory(src, dst, overwrite), which does not exist
# on .NET Framework; uses per-entry ExtractToFile with overwrite instead.

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null

if (-not (Test-Path -LiteralPath $ZipPath)) {
    Write-Error "ZIP not found: $ZipPath"
    exit 2
}

[void][System.IO.Directory]::CreateDirectory($Dest)
$rootFull = [System.IO.Path]::GetFullPath($Dest)
$sep = [System.IO.Path]::DirectorySeparatorChar
if (-not $rootFull.EndsWith($sep)) { $rootFull += $sep }

$zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
try {
    foreach ($entry in $zip.Entries) {
        $target = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($Dest, $entry.FullName))
        # Guard against zip-slip: skip anything that escapes the destination root.
        if (-not $target.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        if ($entry.FullName.EndsWith('/') -or $entry.FullName.EndsWith('\')) {
            [void][System.IO.Directory]::CreateDirectory($target)
            continue
        }
        $dir = [System.IO.Path]::GetDirectoryName($target)
        if ($dir -and -not (Test-Path -LiteralPath $dir)) {
            [void][System.IO.Directory]::CreateDirectory($dir)
        }
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
    }
}
finally {
    $zip.Dispose()
}

exit 0
