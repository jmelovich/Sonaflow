# Check Python 3.10+. Returns a hashtable of tool paths.

function Test-Prereqs {
    param(
        [switch]$Strict
    )

    $result = @{
        Python = $null
        Ok = $true
        Issues = @()
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        $result.Ok = $false
        $result.Issues += "Python 3.10+ not found on PATH. Install from https://www.python.org/downloads/ and enable 'Add to PATH'."
    }
    else {
        $verOut = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -ne 0) {
            $result.Ok = $false
            $result.Issues += "Could not read Python version."
        }
        else {
            $parts = $verOut.Trim().Split(".")
            $major = [int]$parts[0]
            $minor = [int]$parts[1]
            if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
                $result.Ok = $false
                $result.Issues += "Python $verOut found; 3.10+ required."
            }
            else {
                $result.Python = $pythonCmd.Source
                Write-SetupOk "Python $verOut ($($pythonCmd.Source))"
            }
        }
    }

    return $result
}
