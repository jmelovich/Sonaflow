# Settings file reference

The visualizer loads defaults from `visualizer_settings.json` in the repo root (gitignored). On first setup, `setup.ps1` copies [`visualizer_settings.example.json`](../visualizer_settings.example.json) and applies a hardware profile.

## Key paths

| Field | Default | Notes |
|-------|---------|-------|
| `streamdiffusion_path` | `null` → auto-resolved | Sibling `../StreamDiffusion` or `$STREAMDIFFUSION_PATH`. Default clone is jmelovich `blackwell-tensorrt` (RTX 5080 only); use upstream cumulo-autumn for other GPUs — see [SETUP.md](./SETUP.md). |
| `engine_dir` | `null` | TensorRT engines under `<streamdiffusion_path>/engines/{width}x{height}` |
| `qwen_python` | `null` | `.venv-qwen/Scripts/python.exe` (Windows) or `bin/python` (Linux) |
| `rtx_video_sdk_path` | `null` | Falls back to `$NV_RTX_VIDEO_SDK` |

## Profiles (applied by setup.ps1)

| Profile | Typical use |
|---------|-------------|
| **Minimal** | No GPU / dev smoke tests: `acceleration=none`, mic audio, no Qwen/upscale/GL presenter |
| **Standard** | 8–12 GB VRAM: 512×512 TensorRT, CLAP, loopback, no Qwen/upscale |
| **Full** | 16+ GB VRAM: Qwen sidecar, RTX VSR upscaling, GL presenter |

Edit `visualizer_settings.json` directly or press **F1** in the running app to open the settings panel. Changes that require a model reload trigger an in-process restart.
