# Settings file reference

The visualizer loads defaults from `visualizer_settings.json` in the repo root (gitignored). On first setup, `setup.ps1` copies [`visualizer_settings.example.json`](../visualizer_settings.example.json) and applies a hardware profile.

## Key paths

| Field | Default | Notes |
|-------|---------|-------|
| `streamdiffusion_path` | `null` → vendored | Optional dev override: external checkout path or `$STREAMDIFFUSION_PATH` prepended to `sys.path`. Default uses repo-root `streamdiffusion/`. |
| `engine_dir` | `null` | TensorRT engines under `engines/{width}x{height}` in repo root, or `%LOCALAPPDATA%\Sonaflow\engines` when packaged |
| `qwen_python` | `null` | Main `.venv` or packaged `runtime/main-env/python.exe` (override only) |
| `rtx_video_sdk_path` | `null` | Falls back to `$NV_RTX_VIDEO_SDK` |

## Profiles (applied by setup.ps1)

| Profile | Typical use |
|---------|-------------|
| **Minimal** | No GPU / dev smoke tests: `acceleration=none`, mic audio, no Qwen/upscale/GL presenter |
| **Standard** | 8–12 GB VRAM: 512×512 TensorRT, CLAP, loopback, no Qwen/upscale |
| **Full** | 16+ GB VRAM: Qwen2-Audio captions (same `.venv`), RTX VSR upscaling, GL presenter |

Edit `visualizer_settings.json` directly or press **F1** in the running app to open the settings panel. Changes that require a model reload trigger an in-process restart.
