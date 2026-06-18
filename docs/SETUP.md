# Installation guide



This document covers everything `setup.ps1` automates and what you must install manually.



## Quick start



```powershell

git clone https://github.com/jmelovich/Sonaflow.git

cd Sonaflow

.\setup.ps1

.\run.ps1 "a vivid abstract nebula of swirling neon light"

```



`setup.ps1` clones the [jmelovich StreamDiffusion fork (`blackwell-tensorrt`)](https://github.com/jmelovich/StreamDiffusion/tree/blackwell-tensorrt) as a sibling directory (`../StreamDiffusion`), creates the Poetry environment, installs visualizer dependencies, optionally sets up the Qwen sidecar, and writes `visualizer_settings.json`.



> **Tested hardware:** The default `blackwell-tensorrt` branch was developed and verified **only on an NVIDIA GeForce RTX 5080** (Blackwell) with TensorRT 10+ and PyTorch cu128. Other GPUs are not supported for that branch. If you have an older GPU, use upstream StreamDiffusion instead (see below).



Check your install:



```powershell

.\doctor.ps1

```



## Verify setup scripts (without touching your dev environment)



Use the isolated smoke test — it copies both repos to `%TEMP%`, runs `setup.ps1` + `doctor.ps1` in a throwaway workspace, and restores your env vars afterward. Your dev checkouts, `visualizer_settings.json`, and Poetry venvs are not modified.



```powershell

# Fast check (~3–5 min): Minimal profile, no model downloads

.\test\install-smoke.ps1 -Profile Minimal



# Full check including a 5-frame benchmark (downloads SD-Turbo on first run)

.\test\install-smoke.ps1 -Profile Minimal -RunBenchmark -KeepWorkdir



# Test the git-clone path instead of copying your local StreamDiffusion fork

.\test\install-smoke.ps1 -CloneStreamDiffusion -Profile Minimal

```



Results are written to `test/smoke-results/smoke-<timestamp>.json`. Use `-KeepWorkdir` to inspect the temp copy if something fails.



For a truly clean Windows VM, see [`test/sandbox/CHECKLIST.md`](../test/sandbox/CHECKLIST.md) (requires working Windows Sandbox).



## StreamDiffusion dependency



### Default (RTX 5080 / Blackwell only)



This project defaults to a **maintained fork** tuned for Blackwell TensorRT:



| | |

|---|---|

| **Repo** | [github.com/jmelovich/StreamDiffusion](https://github.com/jmelovich/StreamDiffusion) |

| **Branch** | `blackwell-tensorrt` |

| **Tested on** | **NVIDIA GeForce RTX 5080 only** — TensorRT 10+, PyTorch cu128 |

| **Clone target** | `../StreamDiffusion` (sibling to this repo) |

| **Pinned in** | [`setup.lock.json`](../setup.lock.json) |



Fork-specific steps (Poetry, torch cu128, TensorRT pins) are in [INSTALL-WIN-RTX50.md](https://github.com/jmelovich/StreamDiffusion/blob/blackwell-tensorrt/docs/INSTALL-WIN-RTX50.md). `setup.ps1` automates the overlapping steps.



**Do not use `blackwell-tensorrt` on RTX 40xx/30xx or other pre-Blackwell GPUs** unless you are experimenting — engine builds and CUDA wheels are unlikely to work.



### Upstream (other NVIDIA GPUs)



For Ada, Ampere, and other pre-Blackwell cards, use the **original upstream** repo:



| | |

|---|---|

| **Repo** | [github.com/cumulo-autumn/StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) |

| **Branch** | `main` |



Automated setup:



```powershell

.\setup.ps1 -UseUpstreamStreamDiffusion

```



On a non-Blackwell GPU, `setup.ps1` **prompts** to switch to upstream when run interactively.



### Custom fork or branch



Point setup at any StreamDiffusion checkout:



```powershell

.\setup.ps1 -StreamDiffusionRepo "https://github.com/you/StreamDiffusion.git" -StreamDiffusionRef "my-branch"

```



Or set an existing checkout and skip cloning:



```powershell

$env:STREAMDIFFUSION_PATH = "C:\path\to\StreamDiffusion"

.\setup.ps1

```



Persist the path for future sessions:



```powershell

.\setup.ps1 -PersistEnv

# or manually:

[Environment]::SetEnvironmentVariable("STREAMDIFFUSION_PATH", "C:\path\to\StreamDiffusion", "User")

```



## Manual install (when automated setup fails)



Use this on a different machine or GPU class when `setup.ps1` cannot complete.



### A. Blackwell (RTX 5080) — manual blackwell-tensorrt path



```powershell

cd ..

git clone --branch blackwell-tensorrt https://github.com/jmelovich/StreamDiffusion.git

cd StreamDiffusion

poetry install --with tensorrt

poetry run pip uninstall -y torch torchvision torchaudio

poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

poetry run pip install huggingface-hub==0.25.2 transformers==4.36.2 numpy==1.26.4 onnxscript

cd ..\Sonaflow

$env:STREAMDIFFUSION_PATH = (Resolve-Path ..\StreamDiffusion).Path

.\setup.ps1 -SkipModels   # skips re-clone; installs visualizer deps + settings

.\doctor.ps1

.\run.ps1 --acceleration none --benchmark 5 --no-qwen --no-clap --audio-source none

```



See the fork's [INSTALL-WIN-RTX50.md](https://github.com/jmelovich/StreamDiffusion/blob/blackwell-tensorrt/docs/INSTALL-WIN-RTX50.md) for TensorRT troubleshooting.



### B. Pre-Blackwell GPU — manual upstream path



```powershell

cd ..

git clone https://github.com/cumulo-autumn/StreamDiffusion.git

cd StreamDiffusion

poetry install

poetry run pip uninstall -y torch torchvision torchaudio

poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

poetry run pip install -e ".[tensorrt]"

poetry run python -m streamdiffusion.tools.install-tensorrt

poetry run pip install huggingface-hub==0.25.2 transformers==4.36.2 numpy==1.26.4

cd ..\Sonaflow

$env:STREAMDIFFUSION_PATH = (Resolve-Path ..\StreamDiffusion).Path

poetry run pip install -r music_visualizer\requirements.txt   # run from StreamDiffusion dir

poetry run pip install --no-deps compel==2.0.3 pyparsing>=3.0

copy visualizer_settings.example.json visualizer_settings.json

.\doctor.ps1

.\run.ps1 --acceleration none --benchmark 5 --no-qwen --no-clap --audio-source none

```



If Poetry is not available upstream, follow [upstream install docs](https://github.com/cumulo-autumn/StreamDiffusion#installation) (venv + `pip install git+...`), then set `STREAMDIFFUSION_PATH` and install `music_visualizer/requirements.txt` into that same environment.



### C. Verify a manual install



```powershell

cd $env:STREAMDIFFUSION_PATH

poetry run python -c "import streamdiffusion, torch; print(torch.__version__, torch.cuda.get_device_name(0))"

cd path\to\Sonaflow

.\doctor.ps1

```



All core checks should pass. Warnings for optional components (Qwen, RTX SDK, native DLLs) are normal.



## Hardware requirements



| Tier | GPU | VRAM | Profile | Features |

|------|-----|------|---------|----------|

| Minimum | Any (CPU fallback) | — | `Minimal` | PyTorch path only, mic/file audio, OpenCV window |

| Recommended | NVIDIA RTX | 8–12 GB | `Standard` | TensorRT, system audio loopback, CLAP tags |

| Full | NVIDIA RTX | 16+ GB | `Full` | + Qwen2-Audio captions, RTX VSR upscaling, GL presenter |



**TensorRT profile note:** Default TensorRT engines target **RTX 5080 (Blackwell)**. Other GPUs should use `-UseUpstreamStreamDiffusion` or `--acceleration none` until engines are built for their architecture.



Setup auto-selects a profile from detected VRAM. Override with:



```powershell

.\setup.ps1 -Profile Full

.\setup.ps1 -Profile Minimal -SkipModels

```



## Prerequisites (manual)



### Required



| Tool | Version | Install |

|------|---------|---------|

| **Windows** | 10/11 | Primary platform |

| **Python** | 3.10+ | [python.org](https://www.python.org/downloads/) — enable “Add to PATH” |

| **Git** | any recent | [git-scm.com](https://git-scm.com/download/win) |

| **Poetry** | 1.8+ | [python-poetry.org](https://python-poetry.org/docs/#installation) |

| **NVIDIA driver** | recent | Required for GPU inference |



### Required for TensorRT (Standard / Full profiles)



- **NVIDIA GPU** with CUDA support

- First run builds TensorRT engines per resolution (~5–15 minutes, then cached under `../StreamDiffusion/engines/`)

- **Blackwell fork:** RTX 5080-class GPU only (tested)

- **Upstream:** follow cumulo-autumn TensorRT install for your GPU generation



### Optional — Qwen2-Audio sidecar (Full profile)



- ~16 GB disk for `Qwen/Qwen2-Audio-7B-Instruct`

- 16 GB VRAM recommended to run Qwen + StreamDiffusion together



### Optional but highly recommended — RTX VSR upscaling (Full profile)



1. Download the [NVIDIA RTX Video SDK](https://developer.nvidia.com/rtx-video-sdk) (license acceptance required)

2. Set environment variable:



   ```powershell

   [Environment]::SetEnvironmentVariable("NV_RTX_VIDEO_SDK", "C:\path\to\RTX-Video-SDK", "User")

   ```



3. Driver **R570.20+** and an RTX GPU



### Optional — native bridge pre-build (GL presenter / VSR)



Built automatically on first use if missing, or during `setup.ps1` when tools are present:



| Tool | Purpose |

|------|---------|

| **Visual Studio Build Tools** (C++) | Compile `glpresent_bridge.dll`, `rtxvsr_bridge.dll` |

| **CUDA Toolkit** | Headers/libs for CUDA→OpenGL interop |



Set `CUDA_PATH` if the toolkit is not in the default location.



## What setup.ps1 does



1. Verifies Python, git, poetry

2. Detects GPU (VRAM + Blackwell class) → suggests profile and warns on fork/GPU mismatch

3. Clones `../StreamDiffusion` from pinned URL/ref (or upstream / custom repo)

4. Runs `poetry install` (+ TensorRT group for Blackwell fork)

5. Installs [`music_visualizer/requirements.txt`](../music_visualizer/requirements.txt) + compel + torchaudio

6. Prefetches SD-Turbo and CLAP models (unless `-SkipModels`)

7. For **Full** profile: creates `.venv-qwen`, downloads Qwen model

8. Writes `visualizer_settings.json` from [`visualizer_settings.example.json`](../visualizer_settings.example.json)

9. Optionally builds native DLLs



### Useful flags



| Flag | Effect |

|------|--------|

| `-Profile Auto\|Minimal\|Standard\|Full` | Hardware profile |

| `-UseUpstreamStreamDiffusion` | Clone [cumulo-autumn/StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) instead of blackwell fork |

| `-StreamDiffusionRepo URL` | Custom git remote |

| `-StreamDiffusionRef branch` | Custom branch or tag |

| `-ForceBlackwellFork` | Skip non-Blackwell GPU warning (use at your own risk) |

| `-SkipQwen` | Skip Qwen sidecar even on Full |

| `-SkipTensorRT` | `poetry install` without TensorRT group |

| `-SkipModels` | Skip HuggingFace downloads |

| `-SkipNativeBuild` | Skip DLL compile during setup |

| `-PersistEnv` | Save `STREAMDIFFUSION_PATH` to user environment |

| `-ForceSettings` | Overwrite existing `visualizer_settings.json` |

| `-Update` | `git pull` StreamDiffusion checkout |

| `-Quiet` | Non-interactive (no fork prompt; keeps default fork) |



## Troubleshooting



### `StreamDiffusion path does not exist`



Run `.\setup.ps1` or set:



```powershell

$env:STREAMDIFFUSION_PATH = "C:\path\to\StreamDiffusion"

```



### Wrong fork for my GPU



Non-Blackwell GPU + TensorRT failures on `blackwell-tensorrt`:



```powershell

.\setup.ps1 -UseUpstreamStreamDiffusion -ForceSettings

# or start fresh:

Remove-Item ..\StreamDiffusion -Recurse -Force   # if safe to delete

.\setup.ps1 -UseUpstreamStreamDiffusion

```



Use `--acceleration none` in `run.ps1` to bypass TensorRT while debugging.



### NumPy 2.x / TensorRT engine build failures



The project pins `numpy==1.26.4` and `opencv-python<4.12`. Re-run:



```powershell

cd ..\StreamDiffusion

poetry run pip install -r ..\Sonaflow\music_visualizer\requirements.txt

```



Or use `.\run.ps1`, which auto-repairs missing pins.



### aubio build fails on Windows



Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with “Desktop development with C++”, then retry setup. As a fallback, build aubio from source and `pip install` the wheel.



### No system audio (loopback)



- Requires **PyAudioWPatch** (Windows only)

- Try `--audio-source mic` or `--audio-source file path\to\track.wav`

- Enable “Stereo Mix” or use a virtual cable if WASAPI loopback fails



### TensorRT engine rebuild after resolution change



Engines are tied to `width×height`. Changing resolution in settings triggers a new build on first run. Delete stale engines under `StreamDiffusion/engines/` if you see dimension mismatch errors.



### Qwen sidecar not found



```powershell

.\setup.ps1 -Profile Full

# or legacy:

.\setup_sidecar.ps1

```



### RTX VSR / upscale disabled



Set `NV_RTX_VIDEO_SDK`, rebuild with `setup.ps1`, or run with `--no-upscale`.



### First run is slow



Expected: model downloads, TensorRT engine compilation, optional native DLL build. Subsequent runs use cached artifacts.



## Testing on a clean machine (Windows Sandbox)



See [`test/sandbox/CHECKLIST.md`](../test/sandbox/CHECKLIST.md) for validating setup on a disposable Windows instance.



## Version pins



Dependency versions are centralized in [`setup.lock.json`](../setup.lock.json). Bump deliberately when upgrading PyTorch, StreamDiffusion, or model IDs.

