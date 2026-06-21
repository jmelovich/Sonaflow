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
│   ├── Freeze-Env.ps1    # build a relocatable runtime\main-env from the Poetry venv
│   └── Stage-Local.ps1   # assemble the install tree locally for testing
├── launcher/
│   ├── launcher.c        # Sonaflow.exe console launcher (sets env, runs main.py)
│   └── Build-Launcher.ps1
└── build/                # (generated) frozen env + staged layout
```

## Prerequisites (build machine)

- A working dev setup: `..\setup.ps1 -Profile Full` (or at least the Standard
  StreamDiffusion Poetry venv) so there is an env to harvest from.
- Visual Studio Build Tools with **Desktop development with C++** (to compile the
  launcher; same toolchain the native DLLs use).
- Internet access (downloads a ~30 MB standalone CPython once).

> The frozen runtime is **GPU/arch-specific** (RTX 50 / cu128 for v1). Build it on
> the target-class machine. See `docs/PACKAGING.md` §9 for multi-arch plans.

## Milestone 1 — build & run locally

```powershell
cd installer\tools

# 1. Freeze the main environment (copies the Poetry venv's packages onto a
#    relocatable CPython; de-editable-izes streamdiffusion; smoke-tests imports).
.\Freeze-Env.ps1 -StreamDiffusionPath H:\_dev\StreamDiffusion -Dest ..\build\runtime\main-env

# 2. Assemble the install layout + compile the launcher into it.
.\Stage-Local.ps1 -StreamDiffusionPath H:\_dev\StreamDiffusion `
    -FrozenMainEnv ..\build\runtime\main-env -StageDir ..\build\stage

# 3. Run it like an installed app (mutable data goes to %LOCALAPPDATA%\Sonaflow).
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
| `SONAFLOW_DATA_DIR` | `%LOCALAPPDATA%\Sonaflow` | `utils/paths.py` (settings location) |
| `HF_HOME` | `<data>\models` | huggingface model cache |
| `STREAMDIFFUSION_PATH` | `<install>\streamdiffusion` | `config.py` |
| `SONAFLOW_ENGINE_DIR` | `<data>\engines` | `config.resolved_engine_dir` |
| `SONAFLOW_QWEN_PYTHON` | `<install>\runtime\qwen-env\python.exe` (if present) | `config.resolved_qwen_python` |

The working directory is set to the bundled `streamdiffusion\` folder, matching
how `main.py` expects to run.

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
.\Build-OfflineInstaller.ps1 -Version 0.1.0
```

**Two phases (do not confuse with a hang):**

| Phase | What | Typical time |
|---|---|---|
| 1. Payload ZIPs | `core-runtime` (+ optional `qwen-runtime`) in `build\payloads\` | ~15–30 min |
| 2. ISCC compile | Embeds those ZIPs (no re-compression) | ~1–5 min |

The earlier approach embedded 50,000+ files directly in Inno and appeared frozen for
hours. The new approach zips once, then compiles quickly.

Use `-ReusePayloads` to skip phase 1 if zips already exist. Use `-BundleModels` only
if you need a fully offline install with models embedded (~+4 GB).

Output: `installer\build\dist\Sonaflow-Setup-<ver>-offline.exe` plus one or more
`Sonaflow-Setup-<ver>-offline-*.bin` files (Windows limits a single EXE to ~4 GB).
**Copy the entire `dist\` folder** to the target PC and run the `.exe`. Internet is
**required on first launch** to download public models.

Legacy fully-offline model bundle (no HF download on first launch either):

```powershell
.\Build-OfflineInstaller.ps1 -Version 0.1.0 -BundleModels
```

### What the offline installer does

1. **Prereqs** — warns if `nvidia-smi` is missing; RTX-50 notice for other GPUs.
2. **Resolution page** — snaps to generation size (same logic as `utils/resolution.py`).
3. **Installs** — app + frozen runtime + StreamDiffusion + native DLLs →
   `%LOCALAPPDATA%\Programs\Sonaflow`.
4. **Settings** — writes `visualizer_settings.json` with generation size snapped from
   your display resolution (same logic as `utils/resolution.py`).
5. **VC++ runtime** — installed silently from bundled `vc_redist.x64.exe` if missing.
6. **Shortcuts + uninstaller** — Start Menu, optional desktop, uninstaller.
7. **After install** — shows **Optional setup notes** (RTX Video SDK, driver, first-run
   engine build). Also saved as `{install}\docs\OPTIONAL_SETUP.txt`.

**Not bundled** (documented in Optional setup notes):

| Item | Why | What to do |
|---|---|---|
| NVIDIA driver | Hardware-specific | Install from nvidia.com (R570.20+ recommended) |
| TensorRT engines | GPU + driver specific | Built automatically on first launch (~5–15 min); never downloaded |
| RTX Video SDK | License; cannot redistribute | Download from developer.nvidia.com; set `NV_RTX_VIDEO_SDK` |
| Qwen sidecar runtime (~5 GB) | Bundled when Qwen component selected | Included in offline installer |

### Web bootstrapper (later, when you have a host)

`sonaflow.iss` + `Build-Payloads.ps1` download payloads from a URL at install time.
Use that once you publish zips + `manifest.json` to a release host. Until then, use
the offline installer above.

## Not yet implemented (later milestones)

- **M3** Remote prebuilt TensorRT engine cache keyed by resolution — manifest +
  engine table are in place (`Build-Payloads.ps1` / `engines.iss`); still needed:
  the engine prebuild farm and an app-side re-fetch on in-app resolution change.
- **M4** ~~Optional qwen-env~~ Done: frozen qwen-env + online model download during install.
- **M5** Code signing, updates, multi-arch (broader RTX) runtimes.
