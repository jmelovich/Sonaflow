"""Configuration for the music visualizer.

A single :class:`Config` dataclass holds all runtime settings. CLI arguments map
1:1 onto these fields (see ``main.py``). Only the fields needed for milestone 1
(realtime generation + display + prompt injection) are wired up so far; more are
added as the audio pipeline lands.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


def _repo_root() -> Path:
    """Sonaflow repository root (parent of ``music_visualizer/``)."""
    return Path(__file__).resolve().parents[1]


from music_visualizer.bootstrap_paths import bootstrap_import_paths, repo_root


def _default_streamdiffusion_path() -> Optional[str]:
    """Optional dev override for an external StreamDiffusion checkout."""
    env = os.environ.get("STREAMDIFFUSION_PATH", "").strip()
    return env or None


@dataclass
class Config:
    # --- StreamDiffusion integration ---
    streamdiffusion_path: Optional[str] = field(default_factory=_default_streamdiffusion_path)
    engine_dir: Optional[str] = None  # defaults to <repo>/engines or SONAFLOW_ENGINE_DIR

    # --- Generation ---
    model_id: str = "stabilityai/sd-turbo"
    mode: Literal["img2img", "txt2img"] = "img2img"
    width: int = 512
    height: int = 512
    # If None, resolved per-mode in __post_init__:
    #   txt2img -> [0]   (single-step SD-Turbo, fresh frame each call)
    #   img2img -> [10]  (single-step self-feedback morph)
    # NOTE: img2img MUST be a single step. Multi-step t_index_lists engage
    # StreamDiffusion's staggered "stream batch" buffer, which assumes a fresh
    # external input each frame. In a self-feedback loop it pipelines several
    # independent trajectories at once and they interleave/oscillate on screen.
    # Lower index = stronger per-frame denoise = crisper and stays locked to the
    # prompt; higher index lets the feedback image dominate and drift off-prompt.
    t_index_list: Optional[List[int]] = None
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt"
    # cfg_type "self" (RCFG) diverges to black in a self-feedback loop, so default
    # to "none" (also the right choice for SD-Turbo, which wants little/no CFG).
    cfg_type: Literal["none", "full", "self", "initialize"] = "none"
    guidance_scale: float = 1.0
    delta: float = 0.5
    seed: int = 2
    warmup: int = 10
    # img2img feedback: False avoids extra noise in the latent buffer (reduces flicker).
    do_add_noise: Optional[bool] = None

    # --- Feedback motion (img2img only, Deforum-style) ---
    # A geometric transform applied to the feedback frame each cycle. Without it
    # the self-feedback loop converges to a near-static image. These give it
    # continuous motion. All amounts are PER FRAME.
    #   feedback_zoom:   per-frame scale (>1 zoom in, <1 zoom out, 1.0 = off)
    #   feedback_pan_x:  per-frame horizontal drift as fraction of width
    #   feedback_pan_y:  per-frame vertical drift as fraction of height
    #   feedback_rotate: per-frame rotation in degrees (positive = clockwise)
    # If left at sentinel defaults, img2img gets a gentle zoom so motion is
    # visible out of the box; txt2img ignores these entirely.
    feedback_zoom: Optional[float] = None
    feedback_pan_x: float = 0.0
    feedback_pan_y: float = 0.0
    feedback_rotate: float = 0.0
    feedback_padding: Literal["reflection", "border", "zeros"] = "reflection"
    # Per-frame noise drift [0, 1]. StreamDiffusion adds a FIXED noise tensor each
    # img2img frame, which makes the self-feedback loop deterministic -> it locks
    # onto a fixed-point attractor (looks frozen with tiny looping artifacts).
    # Drifting that noise via a variance-preserving random walk removes the fixed
    # point so the visuals keep evolving. 0 = locked (old behavior), ~0.1-0.3 =
    # lively but coherent, 1.0 = fully fresh noise each frame (most chaotic).
    feedback_noise_drift: Optional[float] = 0.05

    # --- Audio (milestone 2) ---
    # audio_source selects where realtime audio comes from:
    #   "none"     -> no audio, motion driven only by defaults/CLI (milestone 1)
    #   "file"     -> play + analyze audio_file
    #   "loopback" -> capture current system output (Windows WASAPI loopback)
    #   "mic"      -> capture default input device (microphone / line-in)
    audio_source: Literal["none", "file", "loopback", "mic"] = "loopback"
    audio_file: Optional[str] = None
    audio_device: Optional[int] = None  # sounddevice device index (None = default)
    sample_rate: int = 44100            # used for live capture; file uses its own
    hop_size: int = 512                 # samples per analysis block (~12ms @ 44100)
    fft_size: int = 1024                # aubio FFT window (>= hop_size)
    # Realtime device buffer (samples), DECOUPLED from hop_size. The output/input
    # callback uses this, not hop_size: a 512-sample buffer (~12ms) underruns the
    # instant a GPU/GIL stall exceeds it, producing gibberish noise once CLAP +
    # generation contend. A larger buffer (~93ms) tolerates those stalls. The
    # analyzer still splits each block into hop_size chunks, so analysis is
    # unaffected. Raise further if you still hear glitches under load.
    audio_blocksize: int = 4096         # ~93ms @ 44100; device buffer resilience

    # --- Reactive mapping (audio features -> feedback motion) ---
    # Enabled whenever audio_source != "none"; --no-react forces it off so manual
    # /zoom etc. still work while audio plays.
    reactive: bool = True
    react_zoom_gain: float = 0.008      # bass energy -> extra per-frame zoom
    react_beat_zoom: float = 0.08      # beat impulse -> zoom punch (peak)
    # Scale the beat zoom punch by bass energy at the hit: 0 = full punch every
    # beat (legacy), 1 = punch proportional to bass [0,1] at that beat.
    react_beat_bass_scale: float = 1.0
    react_zoom_max: float = 1.1         # hard cap on reactive zoom
    react_noise_gain: float = 0.5       # onset strength -> extra noise_drift
    react_bpm_noise_pivot: float = 80.0   # BPM at which scale is 1.0
    react_bpm_noise_mult_min: float = 0.8   # noise scale at very slow BPM
    react_bpm_noise_mult_max: float = 1.2   # noise scale at high BPM
    react_bpm_noise_high_bpm: float = 200.0 # BPM at which mult reaches max
    react_rotate_gain: float = 0.02     # treble -> degrees/frame (max when treble=1)
    react_rotate_max: float = 0.02      # hard cap on |rotate| degrees/frame
    react_beat_rotate: float = 0.015    # loud-beat impulse added to rotate (same sign)
    react_beat_loud_rms: float = 0.35   # min rms for a beat to count as "loud"
    react_beat_loud_mid: float = 0.45     # min mid energy on beat for rotate punch
    react_beat_loud_treb: float = 0.45    # min treble energy on beat for rotate punch
    react_pan_gain: float = 0.01        # mid energy -> pan orbit radius (frac/frame)
    react_smoothing_tau: float = 0.12   # EMA time constant (s) for band/onset
    react_beat_decay_tau: float = 0.18  # beat impulse decay time constant (s)
    react_pan_orbit_rate: float = 0.6   # pan drift angular speed (rad/s)
    react_norm_decay: float = 0.999     # rolling-peak decay for feature normalize

    # --- Beat detection (aubio tempo; optional grid fill-in) ---
    # aubio defaults: threshold 0.3, silence -90 dB.
    beat_tempo_threshold: float = 0.3
    beat_silence_db: float = -90.0
    # Off by default — synthetic grid beats drift off the music when BPM is wrong.
    beat_grid_lock: bool = False
    beat_bpm_min: float = 60.0
    beat_bpm_max: float = 200.0
    beat_bpm_smoothing: float = 0.1     # EMA rate when aubio reports a beat

    # --- CLAP prompt injection (audio -> auto prompt tags) ---
    # A CLAP model scores short rolling audio windows against a vocabulary of
    # visual/mood phrases and injects the best matches as compel-weighted tags
    # appended to the (still editable) base prompt. Enabled whenever audio is
    # active; disable with --no-clap. Smaller window = less lag, lower confidence.
    clap: bool = False
    clap_model_id: str = "laion/larger_clap_music"
    # CLAP runs on CPU by default: its audio encoder is a ~80ms GPU burst that,
    # on a single GPU shared with StreamDiffusion, stalls generation once per
    # cycle (a ~1s stutter). On CPU it runs in the orchestrator thread off the
    # generation GPU, so video stays smooth -- at the cost of extra tag latency,
    # which is harmless since tags are already smoothed over seconds. Set
    # --clap-device cuda to put it back on the GPU (faster tags, may stutter).
    clap_device: str = "cpu"
    clap_window_sec: float = 4.0       # audio window scored each cycle (encoder
                                       # cost scales with this; smaller = smaller
                                       # per-cycle GPU hitch against generation)
    clap_interval_sec: float = 1.0     # seconds between CLAP scorings
    clap_top_k: int = 3                # phrases retrieved per cycle
    clap_max_tags: int = 3             # max tags appended to the prompt
    clap_enter_threshold: float = 0.12 # smoothed score to activate a tag
    clap_exit_threshold: float = 0.06  # smoothed score below which a tag drops
    clap_score_tau: float = 1.5        # EMA time constant (s) for tag scores
    clap_min_hold_sec: float = 1.5     # minimum time a tag stays active
    clap_weight_min: float = 0.7       # compel weight at the enter threshold
    clap_weight_max: float = 1.4       # compel weight at strong confidence
    clap_weight_step: float = 0.1      # quantize weights; larger = fewer prompt
                                       # re-encodes (less stutter), coarser emphasis
    clap_softmax_temp: float = 0.02    # CLAP similarity softmax temperature
    # Hubness calibration: subtract each phrase's running-average similarity so
    # the same generic "hub" phrases don't win for every track. Disable to rank
    # on raw similarity. Smaller alpha = slower baseline (more global prior).
    clap_calibrate: bool = True
    clap_calib_alpha: float = 0.001    # EMA rate for the per-phrase baseline
    clap_calib_warmup: int = 8         # cycles before trusting the baseline
    # Spike protection (cosine-sim units): damps the baseline's *upward* update
    # for phrases spiking above their baseline, so a strong, unique match held
    # over a long stretch isn't slowly absorbed into its own baseline and
    # discarded. 0 = original symmetric EMA; larger = less protection.
    clap_calib_spike_guard: float = 0.05
    clap_vocab_path: Optional[str] = None  # JSON array override for the vocab

    # --- Qwen2-Audio captioning (sidecar subprocess, same .venv) ---
    # Runs Qwen2-Audio-7B-Instruct in a child process (modern transformers +
    # bitsandbytes) and injects a natural-language visual caption into the base
    # prompt. Enable with --no-qwen to disable when audio is active.
    qwen: bool = True
    qwen_model_id: str = "Qwen/Qwen2-Audio-7B-Instruct"
    qwen_python: Optional[str] = None  # defaults to main .venv / packaged main-env python
    # GPU load is dominated by (a) the audio encoder prefill, which scales with
    # the window length, and (b) the LM decode, which costs one full 7B forward
    # per generated token. Keeping the window short and the output capped makes
    # each inference far cheaper, so the fps dip while Qwen runs is smaller and
    # briefer. Qwen2-Audio only ships at 7B, so these knobs are the main levers.
    qwen_window_sec: float = 8.0       # past audio window fed to the model
    qwen_min_interval_sec: float = 4.0   # min seconds between inferences (0 = free-run)
    qwen_max_new_tokens: int = 32        # cap output length (fewer decode steps)
    # Greedy decoding is deterministic and collapses to one generic caption, so
    # sample instead -- each cycle produces fresh imagery that tracks the music.
    qwen_do_sample: bool = True
    qwen_temperature: float = 0.9
    qwen_top_p: float = 0.9
    qwen_system_prompt: Optional[str] = (
        "Describe the musical feel and atmosphere only as visual art prompts"
    )
    qwen_instruction: str = (
        "In one short vivid phrase, describe this music as a visual scene: mood, "
        "colors, environment, lighting, energy. No instruments or musical terms."
    )
    qwen_4bit: bool = True
    # First inference is cold and contends with StreamDiffusion for the GPU, so
    # it can take much longer than steady-state. A generous timeout avoids
    # killing a slow-but-working request (which otherwise triggers a reconnect
    # storm against a sidecar still busy with the abandoned request).
    qwen_timeout_sec: float = 120.0
    qwen_port: int = 0  # 0 = let sidecar pick an ephemeral port
    # Crossfade successive Qwen captions in CLIP/compel embedding space instead
    # of swapping instantly. Incompatible with CLAP (which also owns embeds).
    qwen_blend: bool = True
    qwen_blend_sec: Optional[float] = 2.0
    # Hold the current Qwen caption while CLAP audio similarity to the anchor
    # window stays above a threshold; skip inference until the music changes
    # notably or a max-hold timeout elapses.
    qwen_hold: bool = False
    qwen_hold_threshold: float = 0.9       # hold while score >= this (see sim_power)
    qwen_hold_sim_power: float = 1.0       # score = max(0, cosine) ** power
    qwen_hold_max_sec: float = 20.0        # force refresh even if still similar
    qwen_hold_check_interval_sec: float = 1.0  # how often to re-check while holding
    # While holding, still run Qwen each interval and cache the caption; publish
    # immediately on switch instead of waiting for a fresh inference.
    qwen_hold_prefetch: bool = False

    # --- Content centering (img2img flat-fill recovery) ---
    # When zoom drifts into solid-color padding (black, white, etc.), recenters
    # content via corrective pan.
    content_centering: bool = True
    center_black_threshold: float = 0.02   # min RGB deviation from fill = content
    center_region_frac: float = 0.40     # center square size (fraction of frame)
    center_black_trigger: float = 0.93   # center must be this fraction flat fill
    center_pan_gain: float = 0.20        # corrective pan per unit centroid offset
    center_pan_max: float = 0.025        # max corrective pan per axis per frame
    center_smooth_tau: float = 0.20      # EMA time constant (s) for correction
    center_min_lit_frac: float = 0.015   # minimum lit pixels to consider
    center_analysis_size: int = 64       # downsample size for analysis
    center_audio_bass_w: float = 0.20    # bass weight in centering strength
    center_audio_mid_w: float = 0.40     # mid weight in centering strength
    center_audio_treb_w: float = 0.40    # treble weight in centering strength
    center_audio_min_scale: float = 0.35 # strength at quietest (still some pull)

    # --- Prompt ---
    prompt: str = ""
    negative_prompt: str = "low quality, blurry, distorted, watermark, text"
    # Comma-separated words to block in Qwen captions and CLAP tags (not the
    # manual base prompt). Case-insensitive whole-word match.
    prompt_blacklist: str = ""
    # "strip" removes matched substrings; "discard" rejects the whole prompt.
    prompt_blacklist_mode: Literal["discard", "strip"] = "strip"
    # When True, each entry must match as a whole word/phrase (``synth`` won't
    # match inside ``synthesizer``). When False, allow within-word substring hits.
    prompt_blacklist_whole_words: bool = True

    # --- Upscaling (NVIDIA RTX VSR) ---
    # RTX Video Super Resolution upscales each generated frame on-GPU before
    # display (e.g. 512 -> 2048). Runs via a small native NGX bridge DLL built on
    # first use; requires NV_RTX_VIDEO_SDK to be set and an RTX GPU.
    upscale: bool = True
    upscale_factor: int = 8              # output = generation size * factor (if no explicit size)
    output_width: Optional[int] = None   # explicit upscaled width (overrides factor)
    output_height: Optional[int] = None  # explicit upscaled height (overrides factor)
    vsr_quality: int = 4                 # VSR quality 0..4 (0=bicubic, 4=best/slowest)
    rtx_video_sdk_path: Optional[str] = None  # defaults to $NV_RTX_VIDEO_SDK

    # --- Display ---
    window_title: str = "Music Visualizer"
    fullscreen: bool = False
    show_overlay: bool = True
    display_scale: float = 1.0  # upscale factor applied at display time (cv2 resize)
    # Cap the frame sent to the display path (longest side, pixels). VSR still
    # runs at the full upscale resolution for quality, but the result is
    # downscaled on-GPU to this size before the CPU transfer + OpenCV display.
    # The window is only ~1600px anyway, so this avoids shipping a 4096^2 (~48MB)
    # frame over PCIe every frame. Set 0 to disable (ship full upscaled frames,
    # e.g. for full-res capture/recording).
    display_max_size: int = 1600
    # Present frames straight from the GPU via CUDA->OpenGL interop (skips the
    # device->host copy and cv2.imshow). Windows-only; falls back to the OpenCV
    # window automatically if it can't initialize or when --fullscreen is set.
    gl_present: bool = True

    # --- Runtime ---
    log_level: str = "INFO"
    benchmark: int = 0  # if > 0, run N frames headless and report FPS, then exit
    # Cap how fast StreamDiffusion is invoked (frames/sec). The loop waits for
    # each slot *before* calling generate(), so the img2img pass itself is
    # gated (not just presentation afterward). Keeps per-frame feedback motion
    # and audio-reactive dt consistent. 0 disables (run uncapped).
    gen_fps_cap: float = 30.0

    def resolved_engine_dir(self) -> str:
        # TensorRT engines are static-shape: a given engine only accepts the
        # width/height (and batch/mode) it was built for. StreamDiffusion keys
        # its engine subfolders by model/batch/mode but NOT resolution, so two
        # different resolutions would otherwise collide on the same engine and
        # fail with a static-dimension mismatch. Encode the resolution into the
        # base dir so each size gets its own engine tree (a new one builds on
        # first use of that size).
        # In packaged installs the program tree may be read-only, so the launcher
        # points engines at a writable per-user dir via SONAFLOW_ENGINE_DIR.
        base = (self.engine_dir or os.environ.get("SONAFLOW_ENGINE_DIR")
                or str(repo_root() / "engines"))
        if self.acceleration == "tensorrt":
            return os.path.join(base, f"{self.width}x{self.height}")
        return base

    def resolved_output_size(self) -> tuple[int, int]:
        """Upscaled (width, height): explicit overrides, else generation * factor."""
        w = self.output_width or self.width * self.upscale_factor
        h = self.output_height or self.height * self.upscale_factor
        return int(w), int(h)

    def resolved_rtx_sdk_path(self) -> Optional[str]:
        return self.rtx_video_sdk_path or os.environ.get("NV_RTX_VIDEO_SDK")

    def resolved_qwen_python(self) -> str:
        """Path to the Python interpreter used for the Qwen sidecar subprocess."""
        if self.qwen_python:
            return self.qwen_python
        if env := os.environ.get("SONAFLOW_QWEN_PYTHON"):
            return env
        root = _repo_root()
        packaged_py = root.parent / "runtime" / "main-env" / "python.exe"
        if packaged_py.is_file():
            return str(packaged_py)
        if sys.platform == "win32":
            main_py = root / ".venv" / "Scripts" / "python.exe"
        else:
            main_py = root / ".venv" / "bin" / "python"
        try:
            Path(sys.executable).resolve().relative_to((root / ".venv").resolve())
            return sys.executable
        except ValueError:
            pass
        if main_py.is_file():
            return str(main_py)
        return str(main_py)

    def __post_init__(self) -> None:
        bootstrap_import_paths()
        if self.t_index_list is None:
            self.t_index_list = [0] if self.mode == "txt2img" else [10]
        if self.do_add_noise is None:
            self.do_add_noise = self.mode != "img2img"
        # Default img2img to a very gentle forward zoom; audio-reactive mapping
        # modulates around this baseline up to react_zoom_max.
        if self.feedback_zoom is None:
            self.feedback_zoom = 1.0025 if self.mode == "img2img" else 1.0
        if self.feedback_noise_drift is None:
            self.feedback_noise_drift = 0.05 if self.mode == "img2img" else 0.0
        # Guard: img2img self-feedback only behaves with a single denoising step.
        if self.mode == "img2img" and len(self.t_index_list) > 1:
            import warnings

            warnings.warn(
                f"img2img self-feedback with t_index_list={self.t_index_list} "
                "(multi-step) interleaves trajectories and will visibly oscillate. "
                "Use a single index (e.g. [32]) for a stable morph.",
                stacklevel=2,
            )
        # txt2img only supports cfg_type "none" in StreamDiffusion.
        if self.mode == "txt2img" and self.cfg_type != "none":
            self.cfg_type = "none"
        # Reactive motion warps the img2img feedback frame; txt2img has none, so
        # audio would have nothing to drive. Warn rather than silently no-op.
        if self.audio_source != "none" and self.reactive and self.mode == "txt2img":
            import warnings

            warnings.warn(
                "Audio-reactive motion only affects the img2img feedback loop, "
                "but mode is 'txt2img'. Run with --mode img2img to see audio drive "
                "zoom/pan/rotate/noise.",
                stacklevel=2,
            )
        if self.qwen_blend_sec is None:
            self.qwen_blend_sec = float(self.qwen_min_interval_sec)
