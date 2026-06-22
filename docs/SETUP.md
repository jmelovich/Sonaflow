# Installation guide

This document covers what `setup.ps1` automates and what you must install manually.

## Quick start

```powershell
git clone https://github.com/jmelovich/Sonaflow.git
cd Sonaflow
.\setup.ps1
.\run.ps1 "a vivid abstract nebula of swirling neon light"
```

`setup.ps1` creates Sonaflow's `.venv`, installs PyTorch (CUDA wheels), TensorRT 10,
and all pinned dependencies. StreamDiffusion is **vendored inside this repo** at
`streamdiffusion/` — no sibling checkout or Poetry required.

Check your install:

```powershell
.\doctor.ps1
```

## CUDA profiles (`-Cuda`)

Sonaflow ships one codebase with TensorRT 10 for all RTX 30/40/50 GPUs. The only
build-time difference is the PyTorch wheel index:

| Profile | Torch index | Typical GPU |
|---------|-------------|-------------|
| `cu128` (default on Blackwell) | `download.pytorch.org/whl/cu128` | RTX 50-series |
| `cu121` | `download.pytorch.org/whl/cu121` | RTX 30/40-series |

```powershell
.\setup.ps1                    # Auto: cu128 on Blackwell, cu121 otherwise
.\setup.ps1 -Cuda cu128        # Force Blackwell wheels
.\setup.ps1 -Cuda cu121        # Force Ada/Ampere wheels (also test on RTX 5080)
```

Index URLs are pinned in [`setup.lock.json`](../setup.lock.json).

> **Tested hardware:** Primary development target is **NVIDIA GeForce RTX 5080**
> (Blackwell) with TensorRT 10 and PyTorch cu128. RTX 30/40 builds use the same
> TensorRT 10 codepath with cu121 torch wheels. **Do not use cu121 on RTX 50-series**
> — PyTorch cu121 wheels lack `sm_120` kernels and CUDA will fail at runtime.

## Hardware profiles (`-Profile`)

Setup auto-detects GPU VRAM and picks a profile:

| Profile | VRAM | What you get |
|---------|------|--------------|
| **Minimal** | any / no GPU | PyTorch inference, mic audio, OpenCV window (no TensorRT) |
| **Standard** | 8–12 GB | TensorRT @ 512², CLAP tags, system audio loopback |
| **Full** | 16+ GB | + Qwen2-Audio captions, RTX VSR upscaling, GL presenter |

```powershell
.\setup.ps1 -Profile Full
.\setup.ps1 -Profile Minimal -SkipModels   # faster, no HuggingFace downloads
```

## Verify setup scripts (isolated smoke test)

The smoke test copies this repo to `%TEMP%`, runs `setup.ps1` + `doctor.ps1`, and
does not modify your dev checkout:

```powershell
.\test\install-smoke.ps1 -Profile Minimal
.\test\install-smoke.ps1 -Profile Minimal -RunBenchmark -KeepWorkdir
```

Results: `test/smoke-results/smoke-<timestamp>.json`.

## Optional dev override

To test against an external StreamDiffusion checkout instead of the vendored copy:

```powershell
$env:STREAMDIFFUSION_PATH = "C:\path\to\StreamDiffusion"
.\run.ps1 --benchmark 5
```

Or set `streamdiffusion_path` in `visualizer_settings.json`. The external checkout's
`src/` (or repo root) is prepended to `sys.path` and shadows the vendored package.

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| **Python** | 3.10+ | On PATH; used to create `.venv` |
| **NVIDIA driver** | R570+ recommended | For TensorRT / RTX features |
| **MSVC + CUDA toolkit** | optional | Compile native bridge DLLs (GPU presenter, RTX upscale). PyTorch ships its own CUDA runtime — the toolkit is only needed as a **compiler SDK**. |

Poetry and a separate StreamDiffusion clone are **not** required.

### Native bridge DLLs (optional)

Inference works without these — PyTorch only needs a recent NVIDIA **driver**. The CUDA **toolkit** is separate: headers + `cuda.lib` to compile `glpresent_bridge.dll` and `rtxvsr_bridge.dll`.

| CUDA profile | Toolkit to install |
|--------------|-------------------|
| `cu128` (default, RTX 50) | [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) |
| `cu121` (RTX 30/40) | [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) |

On Windows, choose the **exe (network)** installer. After install, open a **new** terminal so `CUDA_PATH` is set. You also need **MSVC Build Tools** (see below).

Without the toolkit, run with `--no-gl-present` to use the OpenCV window instead. `setup.ps1` tries to pre-build the DLLs when MSVC + CUDA are detected.

#### MSVC Build Tools (minimal ~2 GB)

Sonaflow only needs `cl.exe` and Win32 headers to compile two small bridge DLLs — not the full **Desktop development with C++** workload (~6 GB).

1. Download the bootstrapper (~3 MB): [vs_BuildTools.exe](https://aka.ms/vs/17/release/vs_BuildTools.exe)
2. In the installer, open **Individual components** and select only:
   - **MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)**
   - **Windows 11 SDK** (10.0.22621.0 or newer)
3. Install, then open a **new** terminal.

Or from a shell (after downloading `vs_BuildTools.exe`):

```powershell
.\vs_BuildTools.exe --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621
```

## What `setup.ps1` does

1. Verifies Python 3.10+
2. Detects GPU → profile + CUDA profile
3. Creates `.venv`, installs torch from the chosen CUDA index
4. Installs TensorRT 10 (`tensorrt-cu12` trio from NVIDIA index)
5. Installs `music_visualizer/requirements.txt` + `compel --no-deps`
6. Prefetches SD-Turbo + CLAP models (unless `-SkipModels`); Full profile also prefetches Qwen2-Audio
7. Writes `visualizer_settings.json`

### Common flags

| Flag | Effect |
|------|--------|
| `-Cuda cu128\|cu121\|Auto` | PyTorch wheel index |
| `-Profile Minimal\|Standard\|Full\|Auto` | Feature set |
| `-SkipTensorRT` | Skip TensorRT install (Minimal forces this) |
| `-SkipModels` | Skip HuggingFace prefetch |
| `-SkipQwen` | Skip Qwen model prefetch on Full (Qwen deps still in main `.venv`) |
| `-SkipNativeBuild` | Skip optional DLL build |
| `-ForceSettings` | Overwrite `visualizer_settings.json` |

## Run

```powershell
.\run.ps1 "neon city rain at night"
.\run.ps1 --acceleration none --benchmark 30
```

First TensorRT run builds engines (~5–15 min), cached under `engines/` in the repo
(or `%LOCALAPPDATA%\Sonaflow\engines` when packaged).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `.venv` not found | Run `.\setup.ps1` |
| NumPy / TensorRT errors | Re-run `.\setup.ps1` (pins numpy 1.26.4 + TRT 10) |
| `aubio` build fails (MSVC) | Re-run `.\setup.ps1` — Windows uses prebuilt `aubio-ledfx` wheels (same `import aubio`) |
| CUDA mismatch on RTX 30/40 | `.\setup.ps1 -Cuda cu121 -ForceSettings` |
| Qwen not working | `.\setup.ps1 -Profile Full` (deps in main `.venv`; subprocess uses same interpreter) |
| HF model download fails (WinError 1314) | Re-run `.\setup.ps1` — setup pre-probes symlink support before parallel downloads; or enable Windows Developer Mode |
| `cudaGL.h` / CUDA toolkit missing | Install [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) (cu128) or [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) (cu121); or pass `--no-gl-present` |
| MSVC / `cl.exe` not found | Minimal install (~2 GB): [vs_BuildTools.exe](https://aka.ms/vs/17/release/vs_BuildTools.exe) + **MSVC v143 x64/x86 build tools** + **Windows 11 SDK** (Individual components); or pass `--no-gl-present` / `--no-upscale` |
| RTX upscale fails | Download [RTX Video SDK](https://developer.nvidia.com/rtx-video-sdk), set `NV_RTX_VIDEO_SDK`, or pass `--no-upscale` |

### Stale TensorRT engines

Engines are tied to `width×height` and GPU architecture. Delete stale engines under
`engines/` (or `%LOCALAPPDATA%\Sonaflow\engines`) if you see dimension mismatch errors.

## Building an installer

```powershell
cd installer\tools

# cu128 (default — uses dev .venv, RTX 50)
.\Build-OfflineInstaller.ps1 -Version 0.1.0

# cu121 (isolated .venv-cu121 — RTX 30/40; does not touch dev .venv)
.\Build-OfflineInstaller.ps1 -Version 0.1.0 -Cuda cu121
```

See [`installer/README.md`](../installer/README.md) for `-RefreshEnv`, `-RefreshStage`, and `-SkipQwen`.

## Attribution

The `streamdiffusion/` package is vendored from [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)
(Apache 2.0). See [`NOTICE`](../NOTICE) in the repo root.
