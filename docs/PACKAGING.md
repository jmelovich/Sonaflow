# Sonaflow Windows Installer — Implementation Plan

Status: **M1 implemented**, **M2 offline installer implemented** (see `installer/README.md`)
Goal: ship Sonaflow as a near-normal Windows desktop install — user double-clicks
an installer, gets a Start Menu / desktop shortcut, and runs the app without ever
touching Python, Poetry, pip, Git, or a C++ compiler.

---

## 1. Scope & decisions

| Decision | Choice for v1 | Notes / future door |
|---|---|---|
| **GPU target** | NVIDIA **RTX 50-series only** (Blackwell, `sm_120`, cu128 torch, TensorRT 10) | Architecture must leave room to add more arch/cu builds (30/40-series, cu121) later — see §9. |
| **Qwen2-Audio captions (~16 GB)** | **Optional component** — excluded from the base download; fetched/enabled on demand | |
| **Delivery** | **Web bootstrapper**: small signed `.exe` that downloads payloads from a release host at install time | Keeps the distributed installer tiny; payloads are versioned on the host. |
| **TensorRT engines** | Installer **asks the user for a target display resolution**, snaps it to a canonical generation size, and tries to **download a prebuilt engine** for that size from the remote. Falls back to first-run on-device build. | Engines are GPU+TRT-specific; remote cache is an optimization, on-device build is the guaranteed path. |
| **Install model** | **Per-user** (no admin) under `%LOCALAPPDATA%\Programs\Sonaflow`; mutable data under `%LOCALAPPDATA%\Sonaflow` | Avoids UAC and keeps engine/model/settings writes legal. |
| **Installer toolchain** | **Inno Setup 6** (+ Inno Download Plugin / `DownloadTemporaryFile`) | Free, scriptable, great with large payloads + shortcuts + uninstaller. |

### Hard constraints we cannot bundle away
1. **An NVIDIA RTX GPU + recent driver is mandatory** for any usable experience. Torch wheels carry their own CUDA runtime, so no system CUDA toolkit is needed — but the driver is. No GPU ⇒ only the unusably slow CPU fallback.
2. **TensorRT engines are not universally portable.** They are tied to the GPU SM family + TensorRT version (and sometimes driver). The remote engine cache is best-effort; if a downloaded engine fails to deserialize, the app must rebuild it on device.

---

## 2. Why not just reproduce `setup.ps1` on the user's machine?

The old `setup.ps1` required Poetry, a sibling StreamDiffusion clone, Git, and multiple pip passes.
**Current dev setup** uses a single `.venv` + pip; StreamDiffusion is vendored at `streamdiffusion/`.
Packaging harvests that `.venv` via `installer/tools/Freeze-Env.ps1`.

**Strategy: do all slow/fragile work once on a build machine and ship the *result*.**
Freeze both Python environments, prebuild native DLLs, pre-bundle the small models, and host everything as versioned archives the bootstrapper downloads and lays down.

---

## 3. Target install layout

```
%LOCALAPPDATA%\Programs\Sonaflow\            (code + frozen runtimes — written once at install)
├── Sonaflow.exe                             # launcher stub (real exe, proper icon)
├── app\
│   ├── music_visualizer\...             # this repo's python package
│   └── streamdiffusion\...              # vendored StreamDiffusion package (Apache 2.0)
├── runtime\
│   ├── main-env\python.exe + Lib\site-packages\   # torch cu128, streamdiffusion, TensorRT, opencv, aubio, ...
│   └── qwen-env\python.exe  + Lib\site-packages\   # OPTIONAL component (transformers 4.45, bitsandbytes, ...)
├── native\
│   ├── glpresent_bridge.dll
│   └── rtxvsr_bridge.dll                     # only meaningful if user supplies RTX Video SDK
├── redist\VC_redist.x64.exe                  # installed if missing
└── unins000.exe

%LOCALAPPDATA%\Sonaflow\                      (mutable per-user data)
├── visualizer_settings.json
├── models\                                   # HF_HOME — sd-turbo, clap, (qwen if installed)
├── engines\{w}x{h}\<model_descriptor>\unet.engine + .built_resolution
└── logs\
```

Mutable paths are passed to the app via the launcher (env + settings) so **nothing is hardcoded** to a dev path like `H:\_dev`.

---

## 4. Freezing the two Python environments

The app needs **two** Python 3.10 environments (they have conflicting `transformers` pins and must stay separate):

- **main-env** — runs `main.py`: `torch` cu128, `streamdiffusion`, TensorRT 10, `onnxruntime`, `diffusers 0.24`, `transformers 4.36.2`, `numpy 1.26.4`, `compel`, `torchaudio`, `aubio`, `opencv`, `sounddevice`, `PyAudioWPatch`.
- **qwen-env** (`.venv-qwen` today) — sidecar subprocess: `torch` cu128, `transformers 4.45+`, `accelerate`, `bitsandbytes`, `librosa`, `soundfile`.

### Make them relocatable
Standard venvs hardcode absolute paths (`pyvenv.cfg`, shebangs, `activate`) and break when moved. Build each env on a **standalone, relocatable CPython** from [python-build-standalone](https://github.com/astral-sh/python-build-standalone) (`cpython-3.10 … install_only`, Windows x64), then install packages into its `Lib\site-packages`.

Relocation rules:
- We **never rely on activation** — the launcher and the existing code call interpreters by explicit path (`runtime\main-env\python.exe`, and `qwen_python`/`resolved_qwen_python()` for the sidecar). So no `activate`/shebang fixups needed.
- **Avoid editable installs.** Install `streamdiffusion` as a built wheel (`pip install <sd_repo>`), not `-e`, so no absolute `.pth`/`.egg-link` is baked in. Keep the SD source tree bundled separately only for assets it reads at runtime.
- Verify no absolute paths remain: grep `site-packages` for the build-machine path; fail the build if any are found.

### Harvest procedure (build machine)
1. `setup.ps1 -Profile Full` to create `.venv` (+ optional `.venv-qwen`).
2. `Freeze-Env.ps1` copies `.venv` site-packages onto standalone CPython and embeds the vendored `streamdiffusion/` package.
3. Smoke-test imports; scan for leaked build-machine paths.

---

## 5. Launcher (`Sonaflow.exe`)

Small compiled stub that:
1. Resolves install root + mutable data root.
2. Sets env: `PYTHONUTF8=1`, `PYTHONPATH=<install>\app`, `HF_HOME=%LOCALAPPDATA%\Sonaflow\models`, `SONAFLOW_ENGINE_DIR=...`.
3. `cwd` → `<install>\app` (vendored `streamdiffusion` + `music_visualizer` live here).
4. Exec `runtime\main-env\python.exe app\music_visualizer\main.py` with persisted settings.

This **replaces `run.ps1` + Poetry entirely**. The pip-probe/repair logic in `run.ps1` is dropped (deps are frozen).

App-side config wiring (small changes in this repo):
- `engine_dir` → `%LOCALAPPDATA%\Sonaflow\engines`
- `qwen_python` → `<install>\runtime\qwen-env\Scripts\python.exe` (only set when the Qwen component is installed)
- settings file path → mutable data dir
These are all already configurable via `Config` / `visualizer_settings.json` — packaged mode just supplies packaged defaults.

---

## 6. Resolution selection → TensorRT engine acquisition

### 6.1 Snap target display resolution → generation size
The installer asks the user for their **target display resolution** (free entry + common presets). We derive the generation size StreamDiffusion will build an engine for:

```
aspect      = display_w / display_h
gen_height  = 512                                  # fixed
gen_width   = round(512 * aspect / 64) * 64        # snap to multiple of 64
gen_width   = clamp(gen_width, 320, 1216)          # supported/tested range
```

Examples:
- `3440x1440` (21:9) → 512×2.389 = 1223 → **1216×512**
- `1920x1080` / `2560x1440` (16:9) → 512×1.778 = 910 → **896×512**
- `2560x1080` (≈21:9) → **1216×512**

The chosen `gen_width`/`gen_height` are written into `visualizer_settings.json` (`width`/`height`); display scaling/upscale (`upscale_factor`, `display_max_size`) handles the rest.

> This snap helper should live in **one** place (e.g. `music_visualizer/utils/resolution.py`) and be reused by both the installer and the app, so they never disagree.

### 6.2 Remote engine manifest
Host a JSON manifest describing available prebuilt engines. Key by everything that affects engine compatibility (v1 only populates Blackwell/cu128, but the key is future-proof):

```json
{
  "schema": 1,
  "engines": [
    {
      "gpu_arch": "sm_120",
      "trt": "10.x",
      "torch_cu": "cu128",
      "driver_min": "570.20",
      "model": "stabilityai/sd-turbo",
      "mode": "img2img",
      "batch": 1,
      "resolution": "1216x512",
      "url": "https://<host>/engines/sm_120/cu128/sd-turbo-img2img-b1-1216x512.zip",
      "sha256": "...",
      "built_resolution_stamp": "1216x512"
    }
  ]
}
```

### 6.3 Install-time flow
1. Compute `gen_w x gen_h` (§6.1).
2. Look up the manifest for `gpu_arch=sm_120, cu128, model=sd-turbo, mode=img2img, batch=1, resolution=<gen_w x gen_h>`.
3. **If found** → download + verify sha256 → extract into
   `%LOCALAPPDATA%\Sonaflow\engines\{gen_w}x{gen_h}\<model_descriptor>\` including `unet.engine` and the `.built_resolution` marker (whose contents must equal the resolution stamp the app checks — see `_invalidate_stale_tensorrt_engines` / `_mark_tensorrt_engines_built`).
4. **If not found** → skip; the app builds engines on first run (5–15 min, one-time).
5. **Runtime safety net**: if a downloaded engine fails to deserialize on the user's specific card/driver, delete it and rebuild on device. (The existing stale-engine invalidation logic is the basis for this.)

### 6.4 Producing the prebuilt engines (build pipeline)
On the RTX 50 build box, for each resolution we want to pre-cache: run the app once at that resolution to build engines, then zip the `engines/{w}x{h}/...` tree (with `.built_resolution`) and publish to the host with its sha256 + manifest entry. Start with the common set (`1216x512`, `896x512`) and expand.

---

## 7. Models

| Model | Size | Handling |
|---|---|---|
| SD-Turbo | ~2.5 GB | **Bundled** (core download), placed in `models\` HF cache |
| CLAP `larger_clap_music` | ~1.5 GB | **Bundled** (core download) |
| Qwen2-Audio-7B-Instruct | ~16 GB | **Optional** component / on-demand download into `models\`; only with `qwen-env` |

Harvest = copy the content-addressed `~/.cache/huggingface/hub/models--...` snapshots into the packaged `models\` archive. Runtime sets `HF_HOME` so resolution is offline.

---

## 8. Web bootstrapper (Inno Setup) flow

The distributed `Sonaflow-Setup.exe` is small (just the bootstrapper). At runtime it:

1. **Prerequisite checks**
   - OS = Windows 10/11 x64.
   - NVIDIA GPU + driver present (query `nvidia-smi` / registry); read driver version.
   - Blackwell check (warn if not RTX 50 for v1; do not hard-block — leave door open).
   - Free disk space for selected components.
   - VC++ runtime present (else install bundled `VC_redist.x64.exe`).
2. **Choices**
   - Install location (default per-user).
   - Components: `[x] Core (runtime + SD/CLAP + DLLs)` · `[ ] Qwen captions (+~16 GB)`.
   - **Target display resolution** (preset list + custom) → §6.1.
3. **Download** versioned payloads from the release host (resumable, sha256-verified):
   - `core-runtime` (main-env + app + streamdiffusion + native DLLs)
   - `models-core` (SD-Turbo + CLAP)
   - `qwen` (qwen-env + Qwen model) — only if selected
   - matching **engine archive** for chosen resolution if present (§6.3)
4. **Lay down** the install tree (§3), write `visualizer_settings.json` with packaged defaults + chosen resolution, generate launcher config.
5. **Shortcuts** (Start Menu + optional desktop) → `Sonaflow.exe`; register uninstaller.
6. **First run**: launcher detects missing engines → shows a "Preparing graphics engine (one-time, 5–15 min)…" console/splash while TensorRT compiles; subsequent launches are instant.

---

## 9. Leaving the door open for broader RTX (post-v1)

- Manifest keys already include `gpu_arch` / `torch_cu` / `trt` / `driver_min`.
- `runtime\` can hold **multiple** main-env variants (e.g. `main-env-cu128`, `main-env-cu121`); the bootstrapper picks the build matching the detected GPU arch, downloading only that one.
- Dev builds already support `-Cuda cu128|cu121` via `setup.ps1`; ship separate frozen payloads per profile when ready.
- Engine prebuild farm extends to additional `sm_*` targets; absent a prebuilt engine, on-device build remains the fallback for any supported card.

---

## 10. Required code changes in this repo (small)

1. `music_visualizer/utils/resolution.py` — shared aspect→generation-size snap helper (§6.1).
2. Engine fetch/verify helper + manifest client (used by installer; optionally by app for re-fetch on resolution change).
3. "Packaged mode" config defaults: resolve `engine_dir`, `HF_HOME`, `qwen_python`, `streamdiffusion_path`, settings path from the install layout/env instead of dev defaults.
4. Launcher stub project (`installer/launcher/`).
5. Runtime safety net: on TRT engine deserialize failure, invalidate + rebuild (extend existing stale-engine logic).

> None of these touch the core generation/audio pipeline; they're packaging glue.

---

## 11. Build & release pipeline (one-time tooling on RTX 50 box)

```
1. setup.ps1 -Profile Full                    # known-good deps + model cache
2. tools/freeze_env.ps1                        # build relocatable main-env (+ qwen-env)
3. tools/build_native.ps1                      # compile glpresent/rtxvsr DLLs
4. tools/harvest_models.ps1                    # copy SD-Turbo/CLAP (and Qwen) HF snapshots
5. tools/build_engines.ps1 -Res 1216x512,896x512   # prebuild + zip TRT engines
6. tools/stage_payloads.ps1                    # assemble + sha256 + manifest.json
7. tools/publish.ps1                           # upload payloads + manifest to release host
8. installer/sonaflow.iss (Inno) -> Sonaflow-Setup.exe   # the small bootstrapper
9. signtool                                    # code-sign launcher + installer (avoid SmartScreen)
```

---

## 12. Risks & validation

- **Engine cross-card portability within sm_120** (5070/5080/5090): must validate. If unreliable, treat remote engines as opportunistic only and lean on first-run build.
- **Relocatable env correctness**: torch/TensorRT DLL loading after move; verify no absolute build-machine paths leak.
- **Payload size / download UX**: core likely ~6–10 GB; need resumable, verified downloads and clear progress.
- **Driver/TRT version drift**: a downloaded engine built against TRT X may fail on a user with a much newer/older driver — runtime rebuild fallback is mandatory.
- **SmartScreen**: unsigned installer will be flagged; budget for a code-signing cert.
- **RTX Video SDK** (upscaling) remains user-supplied (license-gated, non-redistributable) — `rtxvsr_bridge.dll` ships but upscale stays off until the user provides the SDK.

---

## 13. Milestones

1. **M1 — Frozen runtime + launcher (offline, single machine).** Relocatable main-env, launcher exe, packaged-mode config, runs locally from the install layout. *(Proves the freeze works.)*
2. **M2 — Inno bootstrapper + core payload hosting.** Prereq checks, component UI, resolution prompt, download/extract/lay-down, shortcuts, uninstaller, first-run engine build.
3. **M3 — Remote engine cache.** Manifest + prebuilt engines for common resolutions + runtime rebuild fallback.
4. **M4 — Qwen optional component.** qwen-env payload + on-demand model fetch.
5. **M5 — Polish.** Code signing, updates/versioning, broader-RTX scaffolding (§9).
```
