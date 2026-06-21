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
| **MSVC + CUDA toolkit** | optional | Only for pre-building native bridge DLLs |

Poetry and a separate StreamDiffusion clone are **not** required.

## What `setup.ps1` does

1. Verifies Python 3.10+
2. Detects GPU → profile + CUDA profile
3. Creates `.venv`, installs torch from the chosen CUDA index
4. Installs TensorRT 10 (`tensorrt-cu12` trio from NVIDIA index)
5. Installs `music_visualizer/requirements.txt` + `compel --no-deps`
6. Prefetches SD-Turbo + CLAP models (unless `-SkipModels`)
7. Optionally creates `.venv-qwen` sidecar (`-Profile Full`)
8. Writes `visualizer_settings.json`

### Common flags

| Flag | Effect |
|------|--------|
| `-Cuda cu128\|cu121\|Auto` | PyTorch wheel index |
| `-Profile Minimal\|Standard\|Full\|Auto` | Feature set |
| `-SkipTensorRT` | Skip TensorRT install (Minimal forces this) |
| `-SkipModels` | Skip HuggingFace prefetch |
| `-SkipQwen` | Skip Qwen sidecar even on Full |
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
| CUDA mismatch on RTX 30/40 | `.\setup.ps1 -Cuda cu121 -ForceSettings` |
| Qwen sidecar missing | `.\setup.ps1 -Profile Full` |
| RTX upscale fails | Install RTX Video SDK, set `NV_RTX_VIDEO_SDK` |

### Stale TensorRT engines

Engines are tied to `width×height` and GPU architecture. Delete stale engines under
`engines/` (or `%LOCALAPPDATA%\Sonaflow\engines`) if you see dimension mismatch errors.

## Attribution

The `streamdiffusion/` package is vendored from [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)
(Apache 2.0). See [`NOTICE`](../NOTICE) in the repo root.
