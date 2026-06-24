# Windows Sandbox setup test checklist

Use this to verify `setup.ps1` on a **clean Windows environment** without polluting your dev machine.

## Prerequisites (host)

- Windows 10/11 **Pro, Enterprise, or Education** (Sandbox is not on Home edition)
- Enable **Windows Sandbox** (one-time, Administrator + reboot):

  ```powershell
  # Run PowerShell as Administrator:
  Enable-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM -All
  # Or:
  .\test\sandbox\enable-sandbox.ps1
  ```

  Reboot after enabling. Confirm: `Test-Path C:\Windows\System32\WindowsSandbox.exe`

## Automated test (recommended)

From the repo root on the **host** (not inside the sandbox):

```powershell
.\test\sandbox\launch-sandbox-test.ps1 -Profile Minimal
```

This launches `visualizer-test-auto.wsb`, which:
- Maps this repo + your local `StreamDiffusion` fork read-only into the sandbox
- Installs Python/Git/Poetry via winget inside the sandbox
- Runs `setup.ps1`, `doctor.ps1`, and a 5-frame benchmark
- Writes results to `test/sandbox/logs/sandbox-result.json`

After pushing an updated `poetry.lock` to [jmelovich/StreamDiffusion `blackwell-tensorrt`](https://github.com/jmelovich/StreamDiffusion/tree/blackwell-tensorrt), remove the StreamDiffusion `<MappedFolder>` from `visualizer-test-auto.wsb` to test the full git-clone path.

## Manual test loop

Each sandbox session starts fresh — no leftover venvs or settings from your dev PC.

### 1. Launch sandbox

Double-click `visualizer-test.wsb` or run:

```powershell
start test\sandbox\visualizer-test.wsb
```

The logon command copies the repo to `C:\test` (writable) and opens PowerShell there.

### 2. Install host prerequisites inside sandbox

The sandbox image has **no Python, git, or Poetry** preinstalled. Install them as you would on a new PC:

1. Python 3.10+ from [python.org](https://www.python.org/downloads/) — check **Add to PATH**
2. Git from [git-scm.com](https://git-scm.com/download/win)
3. Poetry: `(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -`
4. Close and reopen PowerShell so PATH updates

For GPU profiles, confirm the host GPU is visible:

```powershell
nvidia-smi
```

If `nvidia-smi` fails inside the sandbox, you can still test **Minimal** profile (no GPU).

### 3. Run setup

```powershell
cd C:\test
.\setup.ps1 -Profile Minimal -SkipModels
```

Capture output. Expected: prerequisites pass, StreamDiffusion clones to `C:\StreamDiffusion`, Poetry env installs, settings file created.

Repeat for other profiles as needed:

```powershell
.\setup.ps1 -Profile Standard
.\setup.ps1 -Profile Full -SkipModels   # skip 16 GB Qwen download if bandwidth-limited
```

### 4. Run doctor

```powershell
.\doctor.ps1
```

Save the output. Failures should be actionable (missing poetry, models not cached, etc.).

### 5. Smoke run

```powershell
# Minimal — no TensorRT, no window required for benchmark
.\run.ps1 --acceleration none --benchmark 5 --no-qwen --no-clap --no-upscale --audio-source none

# Standard/Full — first run may build TensorRT engines (slow)
.\run.ps1 --benchmark 5 --no-qwen --audio-source none
```

### 6. Close sandbox

Closing the sandbox discards all state. Re-run from step 1 to regression-test after setup script changes.

## Test matrix

| Step | Profile | Expectation |
|------|---------|-------------|
| Prereqs only | any | `doctor.ps1` reports missing tools before setup |
| `setup.ps1 -Profile Minimal -SkipModels` | Minimal | Completes; no Qwen sidecar |
| `setup.ps1 -Profile Standard` | Standard | TensorRT group installed; CLAP model prefetch |
| `setup.ps1 -Profile Full` | Full | Qwen model prefetch (unless `-SkipModels`; deps in main `.venv`) |
| `doctor.ps1` | any | Pass for installed components; warnings for optional RTX SDK / DLLs |
| `run.ps1 --benchmark 5` | Minimal | Exits with FPS line, no crash |

## Notes

- `<vGPU>Enable</vGPU>` passes through the host GPU; TensorRT engine builds are still **GPU-specific**.
- Mapped folder is **read-only**; always work in `C:\test` after the copy step.
- Network access inside the sandbox is required for `git clone`, `poetry install`, and HuggingFace downloads.
- RTX VSR upscaling is bundled in packaged installs (`thirdparty/rtx-video-sdk/`). Dev/sandbox upscale tests still require `NV_RTX_VIDEO_SDK` on the host when running from a git checkout.
