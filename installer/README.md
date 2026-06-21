# Sonaflow packaging (`installer/`)

Tooling that turns the dev checkout into a near-normal Windows app install.
See [`../docs/PACKAGING.md`](../docs/PACKAGING.md) for the full plan and rationale.

This directory currently implements **Milestone 1**: a frozen, relocatable
runtime plus a launcher that runs Sonaflow from a self-contained install layout
(no Poetry, pip, or activation at runtime).

## Layout

```
installer/
├── tools/
│   ├── Common.ps1        # shared helpers + default standalone-CPython URL
│   ├── Freeze-Env.ps1    # build runtime\main-env from Sonaflow's .venv
│   └── Stage-Local.ps1   # assemble the install tree locally for testing
├── launcher/
│   ├── launcher.c        # Sonaflow.exe console launcher (sets env, runs main.py)
│   └── Build-Launcher.ps1
└── build/                # (generated) frozen env + staged layout
```

## Prerequisites (build machine)

- A working dev setup: `..\setup.ps1 -Profile Full` (or at least Standard) so
  `.venv` exists to harvest from.
- Visual Studio Build Tools with **Desktop development with C++** (to compile the
  launcher; same toolchain the native DLLs use).
- Internet access (downloads a ~30 MB standalone CPython once).

> The frozen runtime is **GPU/arch-specific** (RTX 50 / cu128 for v1). Build it on
> the target-class machine. See `docs/PACKAGING.md` §9 for multi-arch plans.

## Milestone 1 — build & run locally

```powershell
cd H:\_dev\Sonaflow
.\setup.ps1 -Profile Full          # creates .venv with torch + TensorRT

cd installer\tools
.\Freeze-Env.ps1 -Dest ..\build\runtime\main-env
.\Stage-Local.ps1 -FrozenMainEnv ..\build\runtime\main-env -StageDir ..\build\stage

..\build\stage\Sonaflow.exe --acceleration none --benchmark 5 --no-qwen --no-clap --audio-source none
```

If the benchmark completes, the frozen runtime + launcher are working end-to-end.
A real TensorRT run (drop `--acceleration none`) will build engines on first use
under `%LOCALAPPDATA%\Sonaflow\engines\{w}x{h}\...`.

## How the launcher wires packaged mode

`Sonaflow.exe` resolves the install root from its own path and sets, before
launching `runtime\main-env\python.exe app\music_visualizer\main.py`:

| Env var | Value | Consumed by |
|---|---|---|
| `PYTHONUTF8` | `1` | interpreter |
| `PYTHONPATH` | `<install>\app` | vendored `streamdiffusion` + `music_visualizer` |
| `SONAFLOW_DATA_DIR` | `%LOCALAPPDATA%\Sonaflow` | `utils/paths.py` (settings location) |
| `HF_HOME` | `<data>\models` | huggingface model cache |
| `SONAFLOW_ENGINE_DIR` | `<data>\engines` | `config.resolved_engine_dir` |
| `SONAFLOW_QWEN_PYTHON` | `<install>\runtime\qwen-env\python.exe` (if present) | `config.resolved_qwen_python` |

The working directory is set to `<install>\app`, matching how `main.py` bootstraps imports.

## Milestone 2 — offline installer (use this now)

**No remote host required.** One EXE bundles the staged app + runtimes. Public models
download from Hugging Face on first launch. Copy the installer to another PC and run it.

```
installer/
├── sonaflow-offline.iss           # offline full installer (embedded payloads)
├── sonaflow.iss                   # web bootstrapper (disabled until you have a host)
├── docs/OPTIONAL_SETUP.txt        # shown after install + Start Menu shortcut
├── tools/Build-OfflineInstaller.ps1
└── build/dist/                    # (generated) Sonaflow-Setup-<ver>-offline.exe
```

### Build the offline installer

```powershell
cd installer\tools

# Default: cu128 (uses your dev .venv — RTX 50 / Blackwell)
.\Build-OfflineInstaller.ps1 -Version 0.1.0

# RTX 30/40: build an isolated cu121 installer without touching .venv
.\Build-OfflineInstaller.ps1 -Version 0.1.0 -Cuda cu121

# Force rebuild package venv + stage + payloads
.\Build-OfflineInstaller.ps1 -Version 0.1.0 -Cuda cu121 -RefreshEnv -RefreshStage
```

| `-Cuda` | Package venv | Stage dir | Output |
|---------|--------------|-----------|--------|
| `cu128` (default) | `.venv` | `build/stage` | `build/dist/Sonaflow-Setup-0.1.0-offline/` |
| `cu121` | `.venv-cu121` | `build/stage-cu121` | `build/dist/Sonaflow-Setup-0.1.0-cu121-offline/` |

Qwen sidecar venvs follow the same pattern: `.venv-qwen` vs `.venv-qwen-cu121`.

**Two phases (do not confuse with a hang):**

| Phase | What | Typical time |
|---|---|---|
| 1. Payload ZIPs | `core-runtime` (+ optional `qwen-runtime`) in `build\payloads\` | ~15–30 min |
| 2. ISCC compile | Embeds those ZIPs (no re-compression) | ~1–5 min |

Use `-ReusePayloads` to skip phase 1 if zips already exist. Use `-BundleModels` only
if you need a fully offline install with models embedded (~+4 GB).

Output: `installer\build\dist\Sonaflow-Setup-<ver>-offline\` (or `...\Sonaflow-Setup-<ver>-cu121-offline\`) containing the `.exe` and any `.bin` files.

**Copy that entire subfolder** to the target PC and run the `.exe`. Internet is
**required on first launch** to download public models.

### What the offline installer does

1. **Prereqs** — warns if `nvidia-smi` is missing; RTX-50 notice for other GPUs.
2. **Resolution page** — snaps to generation size (same logic as `utils/resolution.py`).
3. **Installs** — app (including vendored `streamdiffusion`) + frozen runtime + native DLLs →
   `%LOCALAPPDATA%\Programs\Sonaflow`.
4. **Settings** — writes `visualizer_settings.json` with generation size snapped from
   your display resolution.
5. **VC++ runtime** — installed silently from bundled `vc_redist.x64.exe` if missing.
6. **Shortcuts + uninstaller** — Start Menu, optional desktop, uninstaller.

## Not yet implemented (later milestones)

- **M3** Remote prebuilt TensorRT engine cache keyed by resolution.
- **M5** Code signing, updates, multi-profile installer media (`-Cuda cu121` vs `cu128`).
