# Check Python 3.10+, git, and poetry. Returns a hashtable of tool paths.

function Test-Prereqs {
    param(
        [switch]$Strict
    )

    $result = @{
        Python = $null
        Git = $null
        Poetry = $null
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

    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if (-not $gitCmd) {
        $result.Ok = $false
        $result.Issues += "git not found. Install Git for Windows: https://git-scm.com/download/win"
    }
    else {
        $result.Git = $gitCmd.Source
        Write-SetupOk "git ($($gitCmd.Source))"
    }

    $poetryCmd = Get-Command poetry -ErrorAction SilentlyContinue
    if (-not $poetryCmd) {
        if ($Strict) {
            $result.Ok = $false
            $result.Issues += "Poetry not found. Install: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -"
        }
        else {
            Write-SetupWarn "Poetry not on PATH (required before run.ps1). Install via https://python-poetry.org/docs/#installation"
        }
    }
    else {
        $result.Poetry = $poetryCmd.Source
        Write-SetupOk "poetry ($($poetryCmd.Source))"
    }

    return $result
}
