"""Parameter schema for the settings UI.

Each :class:`ParamSpec` maps 1:1 to a :class:`~music_visualizer.config.Config`
field and drives widget type, ranges, tooltips, and restart behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Tuple

from music_visualizer.audio.routing import (
    Route,
    normalize_routes,
    parse_route_key,
    route_field_value,
    set_route_field,
)
from music_visualizer.analysis.vocab_lists import (
    VocabList,
    normalize_vocab_lists,
    parse_vocab_key,
    vocab_field_value,
    set_vocab_field,
)


class ParamKind(str, Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    ENUM = "enum"
    TEXT = "text"


class ApplyStrategy(str, Enum):
    LIVE = "live"
    RESTART = "restart"


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    group: str
    kind: ParamKind
    min_val: float = 0.0
    max_val: float = 1.0
    step: float = 0.01
    options: Tuple[str, ...] = ()
    tooltip: str = ""
    requires_restart: bool = False
    apply: ApplyStrategy = ApplyStrategy.LIVE


def _s(
    key: str,
    label: str,
    group: str,
    kind: ParamKind,
    *,
    min_val: float = 0.0,
    max_val: float = 1.0,
    step: float = 0.01,
    options: Tuple[str, ...] = (),
    tooltip: str = "",
    requires_restart: bool = False,
) -> ParamSpec:
    apply = ApplyStrategy.RESTART if requires_restart else ApplyStrategy.LIVE
    return ParamSpec(
        key=key,
        label=label,
        group=group,
        kind=kind,
        min_val=min_val,
        max_val=max_val,
        step=step,
        options=options,
        tooltip=tooltip,
        requires_restart=requires_restart,
        apply=apply,
    )


def build_route_specs(cfg: Any) -> Tuple[ParamSpec, ...]:
    """Build labeled gain + enable controls for each route in the routes file.

    Route *structure* (source/amp/target/curve/mode) lives in the external
    routes file; the UI only exposes the runtime-tunable ``gain`` (a labeled
    slider with per-route range) and ``enable`` toggle.
    """
    routes = normalize_routes(getattr(cfg, "react_routes", None))
    specs: list[ParamSpec] = []
    for i, raw in enumerate(routes):
        route = Route.from_dict(raw)
        label = route.display_label(i)
        group = route.group or "Audio Routes"
        binding = f"{route.source} -> {route.target}"
        if route.mode != "add":
            binding += f" ({route.mode})"
        tip = route.tooltip or binding
        specs.append(_s(
            f"route{i}_enable", f"{label} (on)", group, ParamKind.BOOL,
            tooltip=f"Enable: {binding}.",
        ))
        specs.append(_s(
            f"route{i}_gain", label, group, ParamKind.FLOAT,
            min_val=route.gain_min, max_val=route.gain_max, step=route.gain_step,
            tooltip=tip,
        ))
    return tuple(specs)


def build_vocab_specs(cfg: Any) -> Tuple[ParamSpec, ...]:
    """Build per-vocab-list CLAP controls from the external vocab files."""
    lists = normalize_vocab_lists(getattr(cfg, "clap_vocab_lists", None))
    specs: list[ParamSpec] = []
    for i, raw in enumerate(lists):
        vl = VocabList.from_dict(raw)
        label = vl.name or vl.id
        group = f"CLAP — {label}"
        tuning = f"CLAP — {label} (tuning)"
        specs.extend([
            _s(
                f"vocab{i}_enable", "Enabled", group, ParamKind.BOOL,
                tooltip=f"Score audio against the {label} phrase list.",
            ),
            _s(
                f"vocab{i}_interval_sec", "Search interval (s)", group, ParamKind.FLOAT,
                min_val=0.1, max_val=30.0, step=0.1,
                tooltip="Seconds between CLAP searches for this list.",
            ),
            _s(
                f"vocab{i}_top_k", "Top K retrieved", group, ParamKind.INT,
                min_val=1, max_val=20, step=1,
                tooltip="Phrases considered from this list each search.",
            ),
            _s(
                f"vocab{i}_max_tags", "Max injected tags", group, ParamKind.INT,
                min_val=0, max_val=10, step=1,
                tooltip="Maximum tags from this list appended to the prompt.",
            ),
            _s(
                f"vocab{i}_enter_threshold", "Enter threshold", tuning, ParamKind.FLOAT,
                min_val=0.0, max_val=1.0, step=0.01,
                tooltip="Smoothed score needed to activate a tag from this list.",
            ),
            _s(
                f"vocab{i}_exit_threshold", "Exit threshold", tuning, ParamKind.FLOAT,
                min_val=0.0, max_val=1.0, step=0.01,
                tooltip="Smoothed score below which a tag from this list drops.",
            ),
            _s(
                f"vocab{i}_score_tau", "Score smoothing (s)", tuning, ParamKind.FLOAT,
                min_val=0.1, max_val=10.0, step=0.1,
                tooltip="EMA time constant for tag score smoothing.",
            ),
            _s(
                f"vocab{i}_min_hold_sec", "Min hold (s)", tuning, ParamKind.FLOAT,
                min_val=0.0, max_val=30.0, step=0.1,
                tooltip="Minimum time a tag stays active once entered.",
            ),
            _s(
                f"vocab{i}_weight_min", "Weight min", tuning, ParamKind.FLOAT,
                min_val=0.1, max_val=2.0, step=0.05,
                tooltip="Compel weight at the enter threshold.",
            ),
            _s(
                f"vocab{i}_weight_max", "Weight max", tuning, ParamKind.FLOAT,
                min_val=0.1, max_val=3.0, step=0.05,
                tooltip="Compel weight at strong confidence.",
            ),
            _s(
                f"vocab{i}_weight_step", "Weight step", tuning, ParamKind.FLOAT,
                min_val=0.01, max_val=0.5, step=0.01,
                tooltip="Quantize tag weights; larger = fewer prompt re-encodes.",
            ),
            _s(
                f"vocab{i}_softmax_temp", "Softmax temp", tuning, ParamKind.FLOAT,
                min_val=0.001, max_val=1.0, step=0.001,
                tooltip="CLAP similarity softmax temperature for this list.",
            ),
            _s(
                f"vocab{i}_calibrate", "Hubness calibration", tuning, ParamKind.BOOL,
                tooltip="Subtract per-phrase running average to reduce generic hub tags.",
            ),
            _s(
                f"vocab{i}_calib_alpha", "Calib alpha", tuning, ParamKind.FLOAT,
                min_val=0.00001, max_val=0.1, step=0.0001,
                tooltip="EMA rate for per-phrase baseline (smaller = slower).",
            ),
            _s(
                f"vocab{i}_calib_warmup", "Calib warmup cycles", tuning, ParamKind.INT,
                min_val=0, max_val=100, step=1,
                tooltip="Cycles before trusting the calibration baseline.",
            ),
            _s(
                f"vocab{i}_calib_spike_guard", "Calib spike guard", tuning, ParamKind.FLOAT,
                min_val=0.0, max_val=0.5, step=0.01,
                tooltip="Damps baseline upward updates for spiking phrases.",
            ),
        ])
    return tuple(specs)


_PROMPT_SPECS: Tuple[ParamSpec, ...] = (
    _s("prompt", "Base prompt", "Prompt", ParamKind.TEXT,
       tooltip="Manual foundation prompt. CLAP/Qwen append on top. Press Enter or "
               "click away to apply; updates the overlay base/prompt lines."),
    _s("negative_prompt", "Negative prompt", "Prompt", ParamKind.TEXT,
       tooltip="Terms to suppress in generation. Applied on next encode/restart."),
    _s("prompt_blacklist_strip", "Strip blacklist", "Prompt", ParamKind.TEXT,
       tooltip="Comma-separated words or phrases to remove from Qwen captions and CLAP tags "
               "(not the base prompt). Multi-word entries like 'electric guitar' are supported. "
               "Whole-word/phrase match, case-insensitive."),
    _s("prompt_blacklist_discard", "Discard blacklist", "Prompt", ParamKind.TEXT,
       tooltip="Comma-separated words or phrases that reject the whole Qwen/CLAP addition "
               "when matched (keeps the current prompt). Never filters the base prompt."),
    _s("prompt_blacklist_order", "Blacklist order", "Prompt", ParamKind.ENUM,
       options=("strip_first", "discard_first"),
       tooltip="strip_first = remove strip-list matches, then check discard list; "
               "discard_first = check discard list on the original text, then strip."),
    _s("prompt_blacklist_whole_words", "Blacklist whole words", "Prompt", ParamKind.BOOL,
       tooltip="On: each entry must match as a whole word/phrase (synth won't "
               "match inside synthesizer). Off: allow substring matches anywhere."),
)

_GENERATION_SPECS: Tuple[ParamSpec, ...] = (
    _s("model_id", "Model ID", "Generation", ParamKind.TEXT, requires_restart=True,
       tooltip="HuggingFace model id or local checkpoint path for StreamDiffusion."),
    _s("mode", "Mode", "Generation", ParamKind.ENUM,
       options=("img2img", "txt2img"), requires_restart=True,
       tooltip="img2img morphs the previous frame; txt2img samples fresh each frame."),
    _s("width", "Width", "Generation", ParamKind.INT, min_val=256, max_val=1536, step=64,
       requires_restart=True,
       tooltip="Latent generation width (multiple of 64). TensorRT rebuilds per size; very wide frames use more VRAM."),
    _s("height", "Height", "Generation", ParamKind.INT, min_val=256, max_val=1536, step=64,
       requires_restart=True,
       tooltip="Latent generation height (multiple of 64). TensorRT rebuilds per size; very tall frames use more VRAM."),
    _s("t_index_list", "T-index list", "Generation", ParamKind.TEXT, requires_restart=True,
       tooltip="Comma-separated denoise indices (e.g. '10'). Lower = more change per frame. "
               "img2img must use a single index."),
    _s("acceleration", "Acceleration", "Generation", ParamKind.ENUM,
       options=("none", "xformers", "tensorrt"), requires_restart=True,
       tooltip="tensorrt is fastest on RTX; first run builds engines (slow)."),
    _s("cfg_type", "CFG type", "Generation", ParamKind.ENUM,
       options=("none", "full", "self", "initialize"), requires_restart=True,
       tooltip="Classifier-free guidance mode. Use 'none' for SD-Turbo img2img."),
    _s("guidance_scale", "Guidance scale", "Generation", ParamKind.FLOAT,
       min_val=0.0, max_val=20.0, step=0.1, requires_restart=True,
       tooltip="CFG strength. SD-Turbo typically wants ~1.0."),
    _s("delta", "Delta", "Generation", ParamKind.FLOAT,
       min_val=0.0, max_val=1.0, step=0.05, requires_restart=True,
       tooltip="StreamDiffusion delta parameter for prompt switching smoothness."),
    _s("seed", "Seed", "Generation", ParamKind.INT, min_val=0, max_val=999_999_999, step=1,
       requires_restart=True, tooltip="Random seed for noise initialization."),
    _s("warmup", "Warmup passes", "Generation", ParamKind.INT, min_val=0, max_val=50, step=1,
       requires_restart=True, tooltip="Blank passes during pipeline init to stabilize latents."),
    _s("do_add_noise", "Add noise (img2img)", "Generation", ParamKind.BOOL, requires_restart=True,
       tooltip="Extra noise between feedback steps. Usually off for img2img stability."),
)

_CLAP_SPECS: Tuple[ParamSpec, ...] = (
    _s("clap", "Enable CLAP", "CLAP", ParamKind.BOOL, requires_restart=True,
       tooltip="Auto-inject tags from audio via CLAP scoring."),
    _s("clap_model_id", "Model", "CLAP", ParamKind.TEXT, requires_restart=True,
       tooltip="HuggingFace id for the CLAP model."),
    _s("clap_device", "Device", "CLAP", ParamKind.ENUM,
       options=("cpu", "cuda"), requires_restart=True,
       tooltip="cpu avoids GPU stutter; cuda gives faster tag updates."),
    _s("clap_window_sec", "Audio window (s)", "CLAP", ParamKind.FLOAT,
       min_val=1.0, max_val=30.0, step=0.5,
       tooltip="Seconds of audio scored each search. Smaller = less lag."),
)

_QWEN_SPECS: Tuple[ParamSpec, ...] = (
    _s("qwen", "Enable Qwen", "Qwen", ParamKind.BOOL, requires_restart=True,
       tooltip="Qwen2-Audio sidecar caption injection."),
    _s("qwen_model_id", "Model", "Qwen", ParamKind.TEXT, requires_restart=True,
       tooltip="HuggingFace id for Qwen2-Audio."),
    _s("qwen_python", "Sidecar Python", "Qwen", ParamKind.TEXT, requires_restart=True,
       tooltip="Override Python for Qwen subprocess (default: main .venv)."),
    _s("qwen_4bit", "4-bit quant", "Qwen", ParamKind.BOOL, requires_restart=True,
       tooltip="Load Qwen in 4-bit (~4.5 GB VRAM). Off needs more VRAM."),
    _s("qwen_port", "Fixed port", "Qwen", ParamKind.INT,
       min_val=0, max_val=65535, step=1, requires_restart=True,
       tooltip="TCP port for sidecar (0 = ephemeral)."),
    _s("qwen_window_sec", "Audio window (s)", "Qwen", ParamKind.FLOAT,
       min_val=1.0, max_val=60.0, step=0.5,
       tooltip="Past audio window fed to Qwen each inference."),
    _s("qwen_min_interval_sec", "Min interval (s)", "Qwen", ParamKind.FLOAT,
       min_val=0.0, max_val=120.0, step=0.5,
       tooltip="Minimum seconds between Qwen inferences (0 = free-run)."),
    _s("qwen_max_new_tokens", "Max new tokens", "Qwen", ParamKind.INT,
       min_val=8, max_val=256, step=1,
       tooltip="Cap output length (fewer = faster inference)."),
    _s("qwen_timeout_sec", "Timeout (s)", "Qwen", ParamKind.FLOAT,
       min_val=5.0, max_val=600.0, step=5.0,
       tooltip="Sidecar inference timeout."),
    _s("qwen_do_sample", "Sampling", "Qwen", ParamKind.BOOL,
       tooltip="Sample captions for variety. Off = greedy (repeats same caption)."),
    _s("qwen_temperature", "Temperature", "Qwen", ParamKind.FLOAT,
       min_val=0.1, max_val=2.0, step=0.05,
       tooltip="Sampling temperature (higher = more varied captions)."),
    _s("qwen_top_p", "Top-p", "Qwen", ParamKind.FLOAT,
       min_val=0.1, max_val=1.0, step=0.05,
       tooltip="Nucleus sampling top-p."),
    _s("qwen_system_prompt", "System prompt", "Qwen", ParamKind.TEXT,
       tooltip="System instruction for the Qwen sidecar."),
    _s("qwen_instruction", "User instruction", "Qwen", ParamKind.TEXT,
       tooltip="Per-request instruction for visual-scene captioning."),
    _s("qwen_blend", "Caption blend", "Qwen", ParamKind.BOOL,
       tooltip="Crossfade Qwen captions in embedding space (incompatible with CLAP)."),
    _s("qwen_blend_sec", "Blend duration (s)", "Qwen", ParamKind.FLOAT,
       min_val=0.1, max_val=30.0, step=0.1,
       tooltip="Seconds to crossfade between successive Qwen captions."),
    _s("qwen_hold", "Prompt hold", "Qwen", ParamKind.BOOL, requires_restart=True,
       tooltip="Hold Qwen captions while CLAP audio similarity to the anchor stays high."),
    _s("qwen_hold_threshold", "Hold threshold", "Qwen", ParamKind.FLOAT,
       min_val=0.0, max_val=1.0, step=0.01,
       tooltip="Hold while score >= this, where score = max(0, cosine) ** sim power."),
    _s("qwen_hold_sim_power", "Hold sim power", "Qwen", ParamKind.FLOAT,
       min_val=1.0, max_val=8.0, step=0.5,
       tooltip="Exponent on cosine similarity (1=linear, 2=squared, 3=cubed). Spreads high scores apart."),
    _s("qwen_hold_max_sec", "Hold max (s)", "Qwen", ParamKind.FLOAT,
       min_val=1.0, max_val=300.0, step=1.0,
       tooltip="Force a new Qwen caption after this many seconds even if similar."),
    _s("qwen_hold_check_interval_sec", "Hold check interval (s)", "Qwen", ParamKind.FLOAT,
       min_val=0.1, max_val=30.0, step=0.1,
       tooltip="How often to re-check audio similarity while holding."),
    _s("qwen_hold_prefetch", "Hold prefetch", "Qwen", ParamKind.BOOL,
       tooltip="While holding, keep inferring each interval and publish the cached caption on switch."),
)

_FEEDBACK_SPECS: Tuple[ParamSpec, ...] = (
    _s("feedback_zoom", "Zoom / frame", "Feedback Motion", ParamKind.FLOAT,
       min_val=0.95, max_val=1.15, step=0.0005,
       tooltip="Per-frame scale (>1 zooms in). Baseline when audio-reactive is on."),
    _s("feedback_pan_x", "Pan X / frame", "Feedback Motion", ParamKind.FLOAT,
       min_val=-0.05, max_val=0.05, step=0.001,
       tooltip="Horizontal drift as fraction of frame width per frame."),
    _s("feedback_pan_y", "Pan Y / frame", "Feedback Motion", ParamKind.FLOAT,
       min_val=-0.05, max_val=0.05, step=0.001,
       tooltip="Vertical drift as fraction of frame height per frame."),
    _s("feedback_rotate", "Rotate / frame", "Feedback Motion", ParamKind.FLOAT,
       min_val=-2.0, max_val=2.0, step=0.001,
       tooltip="Degrees of rotation per frame (positive = clockwise)."),
    _s("feedback_padding", "Padding mode", "Feedback Motion", ParamKind.ENUM,
       options=("reflection", "border", "zeros"), requires_restart=True,
       tooltip="How pixels outside the frame are filled during warp."),
    _s("feedback_noise_drift", "Noise drift", "Feedback Motion", ParamKind.FLOAT,
       min_val=0.0, max_val=1.0, step=0.01,
       tooltip="Random walk on init noise. Prevents the feedback loop locking to a fixed point. "
               "Higher = more chaotic evolution."),
)

_AUDIO_SPECS: Tuple[ParamSpec, ...] = (
    _s("audio_source", "Audio source", "Audio", ParamKind.ENUM,
       options=("none", "file", "loopback", "mic"), requires_restart=True,
       tooltip="Where realtime audio comes from. loopback = system output (WASAPI)."),
    _s("audio_file", "Audio file", "Audio", ParamKind.TEXT, requires_restart=True,
       tooltip="Path to audio file when source is 'file'."),
    _s("audio_device", "Device index", "Audio", ParamKind.INT,
       min_val=-1, max_val=99, step=1, requires_restart=True,
       tooltip="sounddevice device index (-1 = default)."),
    _s("audio_blocksize", "Device blocksize", "Audio", ParamKind.INT,
       min_val=256, max_val=16384, step=256, requires_restart=True,
       tooltip="Capture/playback buffer in samples. Larger = fewer glitches under GPU load."),
    _s("sample_rate", "Sample rate", "Audio", ParamKind.INT,
       min_val=8000, max_val=96000, step=1000, requires_restart=True,
       tooltip="Sample rate for live capture (file uses its native rate)."),
    _s("hop_size", "Hop size", "Audio", ParamKind.INT,
       min_val=128, max_val=4096, step=128, requires_restart=True,
       tooltip="Analysis block size in samples (~12ms at 512/44100)."),
    _s("fft_size", "FFT size", "Audio", ParamKind.INT,
       min_val=512, max_val=8192, step=256, requires_restart=True,
       tooltip="aubio tempo/onset FFT window (must be >= hop size). Keep small for tight beat timing."),
    _s("react_spectral_fft", "Spectral FFT size", "Audio", ParamKind.INT,
       min_val=0, max_val=16384, step=256, requires_restart=True,
       tooltip="Separate FFT for band energies/centroid (power of two >= hop). "
               "Larger = sharper bass resolution without hurting beat timing. 0 = use FFT size."),
    _s("reactive", "Audio-reactive motion", "Audio", ParamKind.BOOL,
       tooltip="Map audio features to zoom/pan/rotate/noise. Off keeps manual motion only."),
)

_BEAT_SPECS: Tuple[ParamSpec, ...] = (
    _s("beat_tempo_threshold", "Tempo threshold", "Beat Detection", ParamKind.FLOAT,
       min_val=0.05, max_val=1.0, step=0.01,
       tooltip="aubio tempo peak threshold. Lower = more beat detections."),
    _s("beat_silence_db", "Silence (dB)", "Beat Detection", ParamKind.FLOAT,
       min_val=-120.0, max_val=0.0, step=1.0,
       tooltip="Ignore beats when input is below this level."),
    _s("beat_grid_lock", "BPM grid lock", "Beat Detection", ParamKind.BOOL,
       tooltip="Fill missed beats on a BPM grid (experimental; can drift if BPM wrong)."),
    _s("beat_bpm_min", "BPM min", "Beat Detection", ParamKind.FLOAT,
       min_val=40.0, max_val=120.0, step=1.0,
       tooltip="Minimum BPM for grid lock."),
    _s("beat_bpm_max", "BPM max", "Beat Detection", ParamKind.FLOAT,
       min_val=120.0, max_val=300.0, step=1.0,
       tooltip="Maximum BPM for grid lock."),
    _s("beat_bpm_smoothing", "BPM smoothing", "Beat Detection", ParamKind.FLOAT,
       min_val=0.01, max_val=1.0, step=0.01,
       tooltip="EMA rate when aubio reports a new BPM on a beat."),
    _s("beat_adaptive_gate", "Adaptive beat gate", "Beat Detection", ParamKind.BOOL,
       tooltip="Confirm each aubio beat against a rolling onset threshold; rejects false "
               "beats on weak/steady content. Lets one setting work across songs."),
    _s("beat_gate_k", "Beat gate strictness", "Beat Detection", ParamKind.FLOAT,
       min_val=0.0, max_val=3.0, step=0.05,
       tooltip="Onset must exceed median + k*MAD to count as a beat. Higher = stricter "
               "(fewer beats); lower = more permissive. Only used when the gate is on."),
    _s("beat_octave_correct", "Octave correction", "Beat Detection", ParamKind.BOOL,
       tooltip="Fold aubio's BPM into one octave so half/double-tempo estimates don't "
               "make BPM-driven motion jump."),
    _s("beat_octave_lo", "Octave low BPM", "Beat Detection", ParamKind.FLOAT,
       min_val=50.0, max_val=140.0, step=1.0,
       tooltip="Low edge of the preferred octave; BPM folds into [this, 2x this). "
               "90 -> 90-180 covers most dance/pop."),
)

_REACTIVE_SPECS: Tuple[ParamSpec, ...] = (
    _s("react_zoom_max", "Max reactive zoom", "Audio Reactive", ParamKind.FLOAT,
       min_val=1.0, max_val=1.5, step=0.005,
       tooltip="Hard cap on per-frame zoom from reactive mapping."),
    _s("react_rotate_max", "Max rotate", "Audio Reactive", ParamKind.FLOAT,
       min_val=0.0, max_val=0.5, step=0.005,
       tooltip="Hard cap on |rotate| degrees/frame."),
    _s("react_pan_max", "Max pan / frame", "Audio Reactive", ParamKind.FLOAT,
       min_val=0.0, max_val=0.2, step=0.001,
       tooltip="Hard cap on |pan_x| and |pan_y| per frame."),
    _s("react_bpm_map_min", "BPM map min", "Audio Reactive", ParamKind.FLOAT,
       min_val=40.0, max_val=160.0, step=1.0,
       tooltip="BPM mapped to 0 for the bpm_mapped source."),
    _s("react_bpm_map_max", "BPM map max", "Audio Reactive", ParamKind.FLOAT,
       min_val=80.0, max_val=300.0, step=1.0,
       tooltip="BPM mapped to 1 for the bpm_mapped source."),
    _s("react_bpm_noise_pivot", "BPM noise pivot", "Audio Reactive", ParamKind.FLOAT,
       min_val=40.0, max_val=160.0, step=1.0,
       tooltip="BPM at which the bpm_noise_mult source equals 1.0."),
    _s("react_bpm_noise_mult_min", "BPM noise min mult", "Audio Reactive", ParamKind.FLOAT,
       min_val=0.1, max_val=2.0, step=0.05,
       tooltip="bpm_noise_mult source at very slow tempos."),
    _s("react_bpm_noise_mult_max", "BPM noise max mult", "Audio Reactive", ParamKind.FLOAT,
       min_val=0.1, max_val=2.0, step=0.05,
       tooltip="bpm_noise_mult source at high BPM."),
    _s("react_bpm_noise_high_bpm", "BPM noise high BPM", "Audio Reactive", ParamKind.FLOAT,
       min_val=100.0, max_val=300.0, step=1.0,
       tooltip="BPM where bpm_noise_mult reaches its maximum."),
    _s("react_beat_loud_rms", "Loud beat RMS min", "Audio Reactive", ParamKind.FLOAT,
       min_val=0.0, max_val=1.0, step=0.01,
       tooltip="Minimum RMS for beat_loud rotate punch."),
    _s("react_beat_loud_mid", "Loud beat mid min", "Audio Reactive", ParamKind.FLOAT,
       min_val=0.0, max_val=1.0, step=0.01,
       tooltip="Minimum mid energy for beat_loud rotate punch."),
    _s("react_beat_loud_treb", "Loud beat treble min", "Audio Reactive", ParamKind.FLOAT,
       min_val=0.0, max_val=1.0, step=0.01,
       tooltip="Minimum treble energy for beat_loud rotate punch."),
    _s("react_smoothing_tau", "Smoothing tau (s)", "Audio Reactive", ParamKind.FLOAT,
       min_val=0.01, max_val=2.0, step=0.01,
       tooltip="EMA time constant for continuous source smoothing."),
    _s("react_beat_decay_tau", "Beat decay tau (s)", "Audio Reactive", ParamKind.FLOAT,
       min_val=0.01, max_val=2.0, step=0.01,
       tooltip="How fast beat/beat_loud impulses decay."),
    _s("react_pan_orbit_rate", "Pan orbit rate", "Audio Reactive", ParamKind.FLOAT,
       min_val=0.0, max_val=5.0, step=0.05,
       tooltip="Angular speed of orbit_sin/orbit_cos sources (rad/s)."),
    _s("react_norm_decay", "Norm decay", "Audio Reactive", ParamKind.FLOAT,
       min_val=0.9, max_val=1.0, step=0.001,
       tooltip="Peak/ceiling release rate for feature normalization (near 1 = slow)."),
    _s("react_adaptive_norm", "Adaptive normalize", "Audio Reactive", ParamKind.BOOL,
       tooltip="dB-window normalization with a silence gate. Fixes the legacy peak "
               "normalizer's stale-peak fade and silence blow-up. Off = legacy peak."),
    _s("react_norm_range_db", "Norm range (dB)", "Audio Reactive", ParamKind.FLOAT,
       min_val=12.0, max_val=90.0, step=1.0,
       tooltip="Dynamic-range window below the tracked ceiling mapped to 0..1. "
               "Smaller = punchier/more contrast; larger = smoother. Adaptive mode only."),
    _s("react_norm_gate_db", "Norm gate (dB)", "Audio Reactive", ParamKind.FLOAT,
       min_val=-100.0, max_val=-20.0, step=1.0,
       tooltip="When the tracked ceiling falls below this, output is 0 (treated as "
               "silence). Raise toward -50 if quiet passages still jitter. Adaptive only."),
)

_CENTERING_SPECS: Tuple[ParamSpec, ...] = (
    _s("content_centering", "Content centering", "Content Centering", ParamKind.BOOL,
       tooltip="Pan correction when zoom drifts into large flat-color regions (black, white, etc.)."),
    _s("center_black_threshold", "Content threshold", "Content Centering", ParamKind.FLOAT,
       min_val=0.0, max_val=0.2, step=0.005,
       tooltip="Min per-channel deviation from the center fill color to count as content."),
    _s("center_region_frac", "Center region", "Content Centering", ParamKind.FLOAT,
       min_val=0.1, max_val=0.9, step=0.05,
       tooltip="Center square size as fraction of frame for fill detection. Uses edge-band mass on wide frames."),
    _s("center_black_trigger", "Fill trigger", "Content Centering", ParamKind.FLOAT,
       min_val=0.5, max_val=1.0, step=0.01,
       tooltip="Center must be this fraction flat fill (solid color) to trigger correction."),
    _s("center_pan_gain", "Pan gain", "Content Centering", ParamKind.FLOAT,
       min_val=0.0, max_val=1.0, step=0.01,
       tooltip="Corrective pan strength per unit centroid offset."),
    _s("center_pan_max", "Pan max / frame", "Content Centering", ParamKind.FLOAT,
       min_val=0.0, max_val=0.2, step=0.001,
       tooltip="Maximum corrective pan per axis per frame."),
    _s("center_smooth_tau", "Smooth tau (s)", "Content Centering", ParamKind.FLOAT,
       min_val=0.01, max_val=2.0, step=0.01,
       tooltip="EMA time constant for centering correction."),
    _s("center_min_lit_frac", "Min content fraction", "Content Centering", ParamKind.FLOAT,
       min_val=0.0, max_val=0.2, step=0.005,
       tooltip="Minimum content pixels required in the frame to compute centroid."),
    _s("center_analysis_size", "Analysis size", "Content Centering", ParamKind.INT,
       min_val=16, max_val=256, step=8,
       tooltip="Downsample size for centering analysis."),
    _s("center_audio_bass_w", "Audio bass weight", "Content Centering", ParamKind.FLOAT,
       min_val=0.0, max_val=1.0, step=0.05,
       tooltip="Bass weight in audio-driven centering strength."),
    _s("center_audio_mid_w", "Audio mid weight", "Content Centering", ParamKind.FLOAT,
       min_val=0.0, max_val=1.0, step=0.05,
       tooltip="Mid weight in audio-driven centering strength."),
    _s("center_audio_treb_w", "Audio treble weight", "Content Centering", ParamKind.FLOAT,
       min_val=0.0, max_val=1.0, step=0.05,
       tooltip="Treble weight in audio-driven centering strength."),
    _s("center_audio_min_scale", "Audio min scale", "Content Centering", ParamKind.FLOAT,
       min_val=0.0, max_val=1.0, step=0.05,
       tooltip="Centering strength at quietest audio."),
)

_DISPLAY_SPECS: Tuple[ParamSpec, ...] = (
    _s("window_title", "Window title", "Display", ParamKind.TEXT,
       tooltip="Native window title string."),
    _s("fullscreen", "Fullscreen", "Display", ParamKind.BOOL,
       tooltip="Borderless fullscreen (toggle also via F11)."),
    _s("show_overlay", "Debug overlay", "Display", ParamKind.BOOL,
       tooltip="FPS/BPM/prompt debug overlay (F7 toggles independently of settings UI)."),
    _s("gl_present", "GPU present", "Display", ParamKind.BOOL, requires_restart=True,
       tooltip="CUDA->OpenGL presenter (default). Off uses OpenCV fallback."),
    _s("display_scale", "Display scale", "Display", ParamKind.FLOAT,
       min_val=0.25, max_val=4.0, step=0.05,
       tooltip="Extra CPU resize at display time (OpenCV path only)."),
    _s("display_max_size", "Max display height", "Display", ParamKind.INT,
       min_val=0, max_val=4320, step=64, requires_restart=True,
       tooltip="Cap display height in px (width scales with aspect; VSR still runs full res). 0 = full."),
)

_UPSCALE_SPECS: Tuple[ParamSpec, ...] = (
    _s("upscale", "Enable upscale", "Upscaling", ParamKind.BOOL, requires_restart=True,
       tooltip="NVIDIA RTX VSR upscaling on GPU before display."),
    _s("upscale_factor", "Upscale factor", "Upscaling", ParamKind.INT,
       min_val=1, max_val=16, step=1, requires_restart=True,
       tooltip="Output = generation size x factor when width/height unset."),
    _s("output_width", "Output width", "Upscaling", ParamKind.INT,
       min_val=0, max_val=7680, step=64, requires_restart=True,
       tooltip="Explicit upscaled width (0 = use factor)."),
    _s("output_height", "Output height", "Upscaling", ParamKind.INT,
       min_val=0, max_val=4320, step=64, requires_restart=True,
       tooltip="Explicit upscaled height (0 = use factor)."),
    _s("vsr_quality", "VSR quality", "Upscaling", ParamKind.INT,
       min_val=0, max_val=4, step=1, requires_restart=True,
       tooltip="0=bicubic, 4=best/slowest RTX VSR quality."),
    _s("rtx_video_sdk_path", "RTX SDK path", "Upscaling", ParamKind.TEXT, requires_restart=True,
       tooltip="Path to NV RTX Video SDK root (or set NV_RTX_VIDEO_SDK env)."),
)

_RUNTIME_SPECS: Tuple[ParamSpec, ...] = (
    _s("gen_fps_cap", "Gen FPS cap", "Runtime", ParamKind.FLOAT,
       min_val=0.0, max_val=120.0, step=1.0,
       tooltip="Max StreamDiffusion loop rate. 0 = uncapped. Keeps per-frame motion consistent."),
    _s("log_level", "Log level", "Runtime", ParamKind.ENUM,
       options=("DEBUG", "INFO", "WARNING", "ERROR"),
       tooltip="Python logging verbosity."),
)

_INTEGRATION_SPECS: Tuple[ParamSpec, ...] = (
    _s("streamdiffusion_path", "StreamDiffusion path", "Integration", ParamKind.TEXT,
       requires_restart=True,
       tooltip="Path to the StreamDiffusion repo root."),
    _s("engine_dir", "TensorRT engine dir", "Integration", ParamKind.TEXT,
       requires_restart=True,
       tooltip="Directory for TensorRT engines (blank = auto)."),
    _s("react_routes_path", "Audio routes file", "Integration", ParamKind.TEXT,
       requires_restart=True,
       tooltip="Path to the audio routing matrix JSON (blank = <data dir>/audio_routes.json)."),
    _s("clap_vocabs_dir", "CLAP vocabs dir", "Integration", ParamKind.TEXT,
       requires_restart=True,
       tooltip="Directory of CLAP vocabulary list JSON files (blank = <data dir>/clap_vocabs/)."),
)

_STATIC_PARAM_SECTIONS: Tuple[Tuple[ParamSpec, ...], ...] = (
    _PROMPT_SPECS,
    _GENERATION_SPECS,
    _CLAP_SPECS,
    _QWEN_SPECS,
    _FEEDBACK_SPECS,
    _AUDIO_SPECS,
    _BEAT_SPECS,
    _REACTIVE_SPECS,
    _CENTERING_SPECS,
    _DISPLAY_SPECS,
    _UPSCALE_SPECS,
    _RUNTIME_SPECS,
    _INTEGRATION_SPECS,
)

STATIC_PARAM_SPECS: Tuple[ParamSpec, ...] = tuple(
    spec for section in _STATIC_PARAM_SECTIONS for spec in section
)
PARAM_SPECS: Tuple[ParamSpec, ...] = STATIC_PARAM_SPECS


def build_param_specs(cfg: Any) -> Tuple[ParamSpec, ...]:
    """Full UI spec list in a stable, user-facing section order."""
    return (
        _PROMPT_SPECS
        + _GENERATION_SPECS
        + _CLAP_SPECS
        + build_vocab_specs(cfg)
        + _QWEN_SPECS
        + _FEEDBACK_SPECS
        + _AUDIO_SPECS
        + _BEAT_SPECS
        + _REACTIVE_SPECS
        + build_route_specs(cfg)
        + _CENTERING_SPECS
        + _DISPLAY_SPECS
        + _UPSCALE_SPECS
        + _RUNTIME_SPECS
        + _INTEGRATION_SPECS
    )


def spec_by_key(specs: Optional[Sequence[ParamSpec]] = None) -> dict[str, ParamSpec]:
    return {p.key: p for p in (specs if specs is not None else PARAM_SPECS)}


def config_value_for_ui(cfg: Any, spec: ParamSpec) -> Any:
    """Read a Config field and normalize for the native widget."""
    parsed = parse_route_key(spec.key)
    if parsed is not None:
        idx, field_name = parsed
        return route_field_value(cfg, idx, field_name)
    parsed_vocab = parse_vocab_key(spec.key)
    if parsed_vocab is not None:
        idx, field_name = parsed_vocab
        return vocab_field_value(cfg, idx, field_name)
    val = getattr(cfg, spec.key)
    if spec.key == "t_index_list":
        if val is None:
            return ""
        return ",".join(str(x) for x in val)
    if spec.key == "audio_device" and val is None:
        return -1
    if spec.key in ("output_width", "output_height") and val is None:
        return 0
    if spec.key in ("engine_dir", "rtx_video_sdk_path", "clap_vocabs_dir",
                    "qwen_python", "audio_file", "react_routes_path") and val is None:
        return ""
    return val


def ui_value_to_config(spec: ParamSpec, raw: Any) -> Any:
    """Convert a widget value back to the Python Config type."""
    parsed = parse_route_key(spec.key)
    if parsed is not None:
        _, field_name = parsed
        if field_name == "enable":
            return bool(raw)
        if field_name in ("source", "amp", "target", "mode"):
            return str(raw)
        if field_name in ("gain", "curve"):
            return float(raw)
        return raw
    parsed_vocab = parse_vocab_key(spec.key)
    if parsed_vocab is not None:
        _, field_name = parsed_vocab
        if field_name in ("enable", "calibrate"):
            return bool(raw)
        if field_name in ("top_k", "max_tags", "calib_warmup"):
            return int(raw)
        return float(raw)
    if spec.kind == ParamKind.BOOL:
        return bool(raw)
    if spec.kind == ParamKind.INT:
        v = int(raw)
        if spec.key == "audio_device" and v < 0:
            return None
        if spec.key in ("output_width", "output_height") and v <= 0:
            return None
        return v
    if spec.kind == ParamKind.FLOAT:
        return float(raw)
    if spec.key == "t_index_list":
        s = str(raw or "").strip()
        if not s:
            return None
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    if spec.key in ("engine_dir", "rtx_video_sdk_path", "clap_vocabs_dir",
                    "qwen_python", "audio_file", "react_routes_path") and not str(raw or "").strip():
        return None
    if spec.key == "prompt_blacklist_order":
        from music_visualizer.utils.text import normalize_blacklist_order
        return normalize_blacklist_order(str(raw))
    return str(raw)
