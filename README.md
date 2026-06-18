# Sonaflow: Realtime Diffusion-Based AI Music Visualizer

A realtime visualizer that drives Stable Diffusion image generation, reacts to audio, and optionally upscales to 4K. Repository: [github.com/jmelovich/Sonaflow](https://github.com/jmelovich/Sonaflow). See [`docs/SETUP.md`](./docs/SETUP.md) for installation and [`docs/settings.md`](./docs/settings.md) for configuration.

## Quick start

```powershell
git clone https://github.com/jmelovich/Sonaflow.git
cd Sonaflow
.\setup.ps1
.\run.ps1 "a vivid abstract nebula of swirling neon light"
```

Verify the install:

```powershell
.\doctor.ps1
```

## Hardware profiles

Setup auto-detects GPU VRAM and picks a profile. Override with `-Profile`:

| Profile | VRAM | What you get |
|---------|------|--------------|
| **Minimal** | any / no GPU | PyTorch inference, mic audio, OpenCV window |
| **Standard** | 8–12 GB | TensorRT @ 512², CLAP tags, system audio loopback |
| **Full** | 16+ GB | + Qwen2-Audio captions, RTX VSR upscaling, GL presenter |

```powershell
.\setup.ps1 -Profile Full
.\setup.ps1 -Profile Minimal -SkipModels   # faster, no HuggingFace downloads
```

## Run

```powershell
# Realtime window (TensorRT — first run builds engines ~5–15 min, then cached)
.\run.ps1 "neon city rain at night"

# Faster iteration (no engine build, slower inference)
.\run.ps1 --acceleration none "abstract waves"

# Headless throughput check
.\run.ps1 --acceleration none --benchmark 30
```

### Live prompt injection

While the window is open, type a new prompt in the **terminal** and press Enter. Type `q` or press `q` / `Esc` in the window to quit. Press **F1** for the settings panel.

## Key options

| Flag | Default | Notes |
|------|---------|-------|
| `--mode img2img\|txt2img` | `img2img` | img2img = morphing feedback loop |
| `--acceleration none\|xformers\|tensorrt` | `tensorrt` | `none` skips engine build |
| `--audio-source loopback\|mic\|file\|none` | `loopback` | System audio (Windows WASAPI) |
| `--no-clap` / `--no-qwen` | CLAP off, Qwen on | Audio-driven prompt injection |
| `--no-upscale` | upscale on (Full) | Requires RTX Video SDK |
| `--width` / `--height` | `512` | New resolution triggers engine rebuild |
| `--benchmark N` | off | Headless FPS over N frames |

Settings persist in `visualizer_settings.json` (created by setup; gitignored). See [`docs/settings.md`](./docs/settings.md).

## Layout

```
Sonaflow/
├── setup.ps1              # One-time developer setup
├── doctor.ps1             # Health check
├── run.ps1                # Launcher (StreamDiffusion Poetry venv)
├── setup.lock.json        # Pinned fork URL, torch index, models
├── music_visualizer/
│   ├── main.py            # CLI, threading, orchestration
│   ├── config.py          # Config dataclass
│   ├── audio/             # Capture, aubio, reactive motion
│   ├── analysis/          # CLAP, Qwen client
│   ├── generation/        # StreamDiffusion, RTX VSR
│   ├── display/           # OpenCV / GL presenter
│   └── qwen_sidecar/      # Isolated Qwen2-Audio server
└── docs/
    └── SETUP.md           # Full install guide + troubleshooting
```

## External dependency

Runs inside the **StreamDiffusion Poetry venv**. By default `setup.ps1` clones the [jmelovich fork, `blackwell-tensorrt`](https://github.com/jmelovich/StreamDiffusion/tree/blackwell-tensorrt) to `../StreamDiffusion`.

> **Tested only on NVIDIA GeForce RTX 5080 (Blackwell)** with TensorRT 10+ and PyTorch cu128. Other GPUs: use `.\setup.ps1 -UseUpstreamStreamDiffusion` for [cumulo-autumn/StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion), or see [`docs/SETUP.md`](./docs/SETUP.md) for manual install.

Override checkout with `$env:STREAMDIFFUSION_PATH`, `-StreamDiffusionRepo`, or `--streamdiffusion-path`. Fork install details: [INSTALL-WIN-RTX50.md](https://github.com/jmelovich/StreamDiffusion/blob/blackwell-tensorrt/docs/INSTALL-WIN-RTX50.md).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| StreamDiffusion not found | Run `.\setup.ps1` |
| NumPy / TensorRT build errors | Pinned to numpy 1.26.4 — re-run `.\run.ps1` or see SETUP.md |
| Qwen sidecar missing | `.\setup.ps1 -Profile Full` |
| RTX upscale fails | Install RTX Video SDK, set `NV_RTX_VIDEO_SDK` |

Full details: [`docs/SETUP.md`](./docs/SETUP.md).
