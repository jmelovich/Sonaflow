"""Entry point for the realtime music visualizer (milestone 1).

Runs StreamDiffusion in a background thread, displays frames in an OpenCV window
on the main thread, and accepts live prompt injection from the terminal (type a
new prompt + Enter at any time). A headless ``--benchmark N`` mode is provided
for measuring throughput without a window.

Must run inside the StreamDiffusion Poetry venv (torch cu128 + streamdiffusion).
From the StreamDiffusion repo root after setup::

    cd path/to/StreamDiffusion
    $env:PYTHONUTF8 = "1"
    poetry run python path/to/Sonaflow/music_visualizer/main.py "your prompt"
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# --- bootstrap: make `import music_visualizer.*` work when run as a script ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from music_visualizer.config import Config  # noqa: E402
from music_visualizer.ui.persistence import load_defaults_dict  # noqa: E402
from music_visualizer.ui.controller import RestartCoordinator, RuntimeContext  # noqa: E402
from music_visualizer.utils.logging import get_logger, setup_logging  # noqa: E402
from music_visualizer.utils.shared_state import PromptEmbeds, SharedPrompt  # noqa: E402

logger = get_logger("music_visualizer.main")


def build_config(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(
        prog="music_visualizer",
        description="Realtime StreamDiffusion visualizer with live prompt injection.",
    )
    p.add_argument("prompt", nargs="?", default=None, help="Initial prompt.")

    gen = p.add_argument_group("generation")
    gen.add_argument("--model", dest="model_id", default=None)
    gen.add_argument("--mode", choices=["img2img", "txt2img"], default=None)
    gen.add_argument("--width", type=int, default=None)
    gen.add_argument("--height", type=int, default=None)
    gen.add_argument(
        "--acceleration", choices=["none", "xformers", "tensorrt"], default=None,
        help="tensorrt is recommended on RTX 50 (first run builds engines).",
    )
    gen.add_argument(
        "--t-index", dest="t_index_list", default=None,
        help="Comma-separated t_index_list, e.g. '32,45'. Lower = more change.",
    )
    gen.add_argument("--cfg-type", choices=["none", "full", "self", "initialize"], default=None)
    gen.add_argument("--guidance-scale", type=float, default=None)
    gen.add_argument("--delta", type=float, default=None)
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--negative-prompt", dest="negative_prompt", default=None)
    gen.add_argument(
        "--do-add-noise", dest="do_add_noise", action="store_true", default=None,
        help="Add noise between img2img feedback steps (default: off for img2img).",
    )
    gen.add_argument(
        "--no-do-add-noise", dest="do_add_noise", action="store_false", default=None,
    )
    gen.add_argument(
        "--gen-fps-cap", dest="gen_fps_cap", type=float, default=None,
        help="Cap img2img/txt2img loop rate in fps (default 30). Waits before each "
             "StreamDiffusion pass so per-frame morph stays consistent; 0 disables.",
    )

    motion = p.add_argument_group("feedback motion (img2img, Deforum-style)")
    motion.add_argument(
        "--zoom", dest="feedback_zoom", type=float, default=None,
        help="Per-frame zoom (>1 in, <1 out, 1.0 off). Default 1.0025 for img2img.",
    )
    motion.add_argument(
        "--pan-x", dest="feedback_pan_x", type=float, default=None,
        help="Per-frame horizontal drift as fraction of width (e.g. 0.01).",
    )
    motion.add_argument(
        "--pan-y", dest="feedback_pan_y", type=float, default=None,
        help="Per-frame vertical drift as fraction of height.",
    )
    motion.add_argument(
        "--rotate", dest="feedback_rotate", type=float, default=None,
        help="Per-frame rotation in degrees (positive = clockwise).",
    )
    motion.add_argument(
        "--feedback-padding", dest="feedback_padding",
        choices=["reflection", "border", "zeros"], default=None,
        help="How to fill pixels pulled in from outside the frame.",
    )
    motion.add_argument(
        "--noise-drift", dest="feedback_noise_drift", type=float, default=None,
        help="Per-frame noise walk [0..1]. Prevents the loop locking. "
             "0=locked (default), 1=fully fresh noise each frame.",
    )
    motion.add_argument(
        "--no-content-centering", dest="content_centering", action="store_false",
        default=None,
        help="Disable pan correction when zoom drifts into large flat-color regions.",
    )
    motion.add_argument(
        "--center-pan-gain", dest="center_pan_gain", type=float, default=None,
        help="Strength of flat-fill centering pan correction.",
    )

    audio = p.add_argument_group("audio (milestone 2)")
    audio.add_argument(
        "--audio", dest="audio_file", default=None, metavar="FILE",
        help="Audio file to play + analyze. Implies --audio-source file.",
    )
    audio.add_argument(
        "--audio-source", dest="audio_source",
        choices=["none", "file", "loopback", "mic"], default=None,
        help="none|file|loopback (system audio, WASAPI)|mic (default: file if --audio given).",
    )
    audio.add_argument(
        "--audio-device", dest="audio_device", type=int, default=None,
        help="sounddevice device index (default: system default).",
    )
    audio.add_argument(
        "--audio-blocksize", dest="audio_blocksize", type=int, default=None,
        help="Device buffer in samples (default 4096, ~93ms). Raise if audio "
             "glitches/turns to noise under GPU load; lower for less latency.",
    )
    audio.add_argument(
        "--no-react", dest="reactive", action="store_false", default=None,
        help="Disable audio->motion mapping (keeps manual /zoom etc.).",
    )
    audio.add_argument("--react-zoom-gain", dest="react_zoom_gain", type=float, default=None)
    audio.add_argument("--react-beat-zoom", dest="react_beat_zoom", type=float, default=None)
    audio.add_argument(
        "--react-beat-bass-scale", dest="react_beat_bass_scale", type=float, default=None,
        help="Scale beat zoom punch by bass at the hit: 0=full every beat, 1=bass-proportional.",
    )
    audio.add_argument(
        "--react-zoom-max", dest="react_zoom_max", type=float, default=None,
        help="Hard cap on reactive per-frame zoom (default 1.035).",
    )
    audio.add_argument("--react-noise-gain", dest="react_noise_gain", type=float, default=None)
    audio.add_argument("--react-rotate-gain", dest="react_rotate_gain", type=float, default=None)
    audio.add_argument("--react-beat-rotate", dest="react_beat_rotate", type=float, default=None)
    audio.add_argument("--react-pan-gain", dest="react_pan_gain", type=float, default=None)
    beat = p.add_argument_group("beat detection (aubio tempo tuning)")
    beat.add_argument(
        "--beat-threshold", dest="beat_tempo_threshold", type=float, default=None,
        help="aubio tempo peak threshold (lower=more beats, default 0.15; aubio default 0.3).",
    )
    beat.add_argument(
        "--beat-silence-db", dest="beat_silence_db", type=float, default=None,
        help="Ignore beats when input is below this level in dB (default -80).",
    )
    beat.add_argument(
        "--beat-grid", dest="beat_grid_lock", action="store_true", default=None,
        help="Fill missed beats on a BPM grid (experimental; off by default).",
    )
    beat.add_argument(
        "--no-beat-grid", dest="beat_grid_lock", action="store_false", default=None,
        help="Disable BPM grid fill-in (default).",
    )
    beat.add_argument(
        "--beat-bpm-smoothing", dest="beat_bpm_smoothing", type=float, default=None,
        help="EMA rate for BPM estimate smoothing (default 0.12).",
    )

    clap = p.add_argument_group("clap (audio -> auto prompt injection)")
    clap.add_argument(
        "--no-clap", dest="clap", action="store_false", default=None,
        help="Disable CLAP auto prompt injection (manual prompts only).",
    )
    clap.add_argument("--clap-model", dest="clap_model_id", default=None)
    clap.add_argument(
        "--clap-device", dest="clap_device", default=None,
        help="cpu (default, no generation stutter) or cuda (faster tags, may "
             "stutter video as CLAP contends for the GPU).",
    )
    clap.add_argument(
        "--clap-window", dest="clap_window_sec", type=float, default=None,
        help="Audio window (s) scored each cycle. Smaller = less lag.",
    )
    clap.add_argument(
        "--clap-interval", dest="clap_interval_sec", type=float, default=None,
        help="Seconds between CLAP scorings (default 0.75).",
    )
    clap.add_argument("--clap-top-k", dest="clap_top_k", type=int, default=None)
    clap.add_argument("--clap-max-tags", dest="clap_max_tags", type=int, default=None)
    clap.add_argument(
        "--clap-enter", dest="clap_enter_threshold", type=float, default=None,
        help="Smoothed score to activate a tag (default 0.12).",
    )
    clap.add_argument(
        "--clap-exit", dest="clap_exit_threshold", type=float, default=None,
        help="Smoothed score below which a tag drops (default 0.06).",
    )
    clap.add_argument("--clap-score-tau", dest="clap_score_tau", type=float, default=None)
    clap.add_argument("--clap-min-hold", dest="clap_min_hold_sec", type=float, default=None)
    clap.add_argument("--clap-weight-min", dest="clap_weight_min", type=float, default=None)
    clap.add_argument("--clap-weight-max", dest="clap_weight_max", type=float, default=None)
    clap.add_argument(
        "--clap-weight-step", dest="clap_weight_step", type=float, default=None,
        help="Quantize tag weights to this step (default 0.1). Larger = fewer "
             "prompt re-encodes / less stutter.",
    )
    clap.add_argument("--clap-softmax-temp", dest="clap_softmax_temp", type=float, default=None)
    clap.add_argument(
        "--no-clap-calibrate", dest="clap_calibrate", action="store_false", default=None,
        help="Disable hubness calibration (rank on raw CLAP similarity). Tags "
             "will bias toward the same few generic phrases.",
    )
    clap.add_argument(
        "--clap-calib-alpha", dest="clap_calib_alpha", type=float, default=None,
        help="EMA rate for the per-phrase baseline (default 0.0001; smaller=slower).",
    )
    clap.add_argument(
        "--clap-calib-warmup", dest="clap_calib_warmup", type=int, default=None,
        help="Cycles before trusting the calibration baseline (default 12).",
    )
    clap.add_argument(
        "--clap-calib-spike-guard", dest="clap_calib_spike_guard", type=float, default=None,
        help="Spike protection in cosine-sim units (default 0.05). Damps the "
             "baseline's upward update for phrases spiking above their baseline "
             "so a strong, unique match held for a long stretch isn't absorbed "
             "and discarded. 0 = original symmetric calibration.",
    )
    clap.add_argument(
        "--clap-vocab", dest="clap_vocab_path", default=None, metavar="JSON",
        help="JSON array of phrases to override the built-in CLAP vocabulary.",
    )

    qwen = p.add_argument_group("qwen (Qwen2-Audio sidecar captioning)")
    qwen.add_argument(
        "--qwen", dest="qwen", action="store_true", default=None,
        help="Enable Qwen2-Audio caption injection (requires sidecar venv).",
    )
    qwen.add_argument(
        "--no-qwen", dest="qwen", action="store_false", default=None,
        help="Disable Qwen2-Audio captioning (default).",
    )
    qwen.add_argument("--qwen-model", dest="qwen_model_id", default=None)
    qwen.add_argument(
        "--qwen-python", dest="qwen_python", default=None,
        help="Python interpreter for the sidecar venv (default: .venv-qwen).",
    )
    qwen.add_argument(
        "--qwen-window", dest="qwen_window_sec", type=float, default=None,
        help="Past audio window (s) fed to Qwen2-Audio (default 15).",
    )
    qwen.add_argument(
        "--qwen-interval", dest="qwen_min_interval_sec", type=float, default=None,
        help="Minimum seconds between Qwen inferences (0 = free-run).",
    )
    qwen.add_argument(
        "--qwen-instruction", dest="qwen_instruction", default=None,
        help="User instruction for visual-scene captioning.",
    )
    qwen.add_argument(
        "--qwen-system", dest="qwen_system_prompt", default=None,
        help="System prompt for Qwen2-Audio (overrides the default template).",
    )
    qwen.add_argument(
        "--qwen-max-new-tokens", dest="qwen_max_new_tokens", type=int, default=None,
    )
    qwen.add_argument(
        "--qwen-greedy", dest="qwen_do_sample", action="store_false", default=None,
        help="Use greedy decoding (deterministic, repeats the same caption).",
    )
    qwen.add_argument(
        "--qwen-temperature", dest="qwen_temperature", type=float, default=None,
        help="Sampling temperature (default 0.9; higher = more varied).",
    )
    qwen.add_argument(
        "--qwen-top-p", dest="qwen_top_p", type=float, default=None,
        help="Nucleus sampling top-p (default 0.9).",
    )
    qwen.add_argument(
        "--qwen-timeout", dest="qwen_timeout_sec", type=float, default=None,
        help="Sidecar inference timeout in seconds (default 30).",
    )
    qwen.add_argument(
        "--qwen-port", dest="qwen_port", type=int, default=None,
        help="Fixed TCP port for sidecar (0 = ephemeral).",
    )
    qwen.add_argument(
        "--no-qwen-4bit", dest="qwen_4bit", action="store_false", default=None,
        help="Load Qwen2-Audio in fp16 instead of 4-bit (needs more VRAM).",
    )
    qwen.add_argument(
        "--qwen-blend", dest="qwen_blend", action="store_true", default=None,
        help="Crossfade Qwen captions in compel embedding space (Qwen-only; "
             "incompatible with CLAP).",
    )
    qwen.add_argument(
        "--qwen-blend-sec", dest="qwen_blend_sec", type=float, default=None,
        help="Seconds to blend between Qwen captions (default: qwen-interval).",
    )
    qwen.add_argument(
        "--qwen-hold", dest="qwen_hold", action="store_true", default=None,
        help="Hold Qwen captions while CLAP audio similarity stays high.",
    )
    qwen.add_argument(
        "--no-qwen-hold", dest="qwen_hold", action="store_false", default=None,
        help="Disable Qwen prompt hold (default).",
    )
    qwen.add_argument(
        "--qwen-hold-threshold", dest="qwen_hold_threshold", type=float, default=None,
        help="Hold score threshold; score = max(0, cosine) ** sim-power (default 0.9).",
    )
    qwen.add_argument(
        "--qwen-hold-sim-power", dest="qwen_hold_sim_power", type=float, default=None,
        help="Exponent on cosine similarity for hold decisions (1=linear, 2=squared, etc).",
    )
    qwen.add_argument(
        "--qwen-hold-max-sec", dest="qwen_hold_max_sec", type=float, default=None,
        help="Force a new Qwen caption after this many seconds (default 20).",
    )
    qwen.add_argument(
        "--qwen-hold-check-interval", dest="qwen_hold_check_interval_sec",
        type=float, default=None,
        help="Seconds between similarity checks while holding (default 1).",
    )
    qwen.add_argument(
        "--qwen-hold-prefetch", dest="qwen_hold_prefetch", action="store_true",
        default=None,
        help="While holding, keep inferring each interval and publish on switch.",
    )
    qwen.add_argument(
        "--no-qwen-hold-prefetch", dest="qwen_hold_prefetch", action="store_false",
        default=None,
        help="Disable Qwen hold prefetch (default).",
    )

    ups = p.add_argument_group("upscaling (NVIDIA RTX VSR)")
    ups.add_argument(
        "--no-upscale", dest="upscale", action="store_false", default=None,
        help="Disable RTX VSR; output at native generation resolution.",
    )
    ups.add_argument(
        "--upscale-factor", dest="upscale_factor", type=int, default=None,
        help="Output = generation size x factor (default 8, so 512->4096).",
    )
    ups.add_argument("--output-width", dest="output_width", type=int, default=None)
    ups.add_argument("--output-height", dest="output_height", type=int, default=None)
    ups.add_argument(
        "--vsr-quality", dest="vsr_quality", type=int, default=None,
        help="VSR quality 0..4 (0=bicubic, 4=best/slowest; default 4).",
    )
    ups.add_argument("--rtx-video-sdk-path", dest="rtx_video_sdk_path", default=None)

    integ = p.add_argument_group("integration")
    integ.add_argument("--streamdiffusion-path", dest="streamdiffusion_path", default=None)
    integ.add_argument("--engine-dir", dest="engine_dir", default=None)

    disp = p.add_argument_group("display")
    disp.add_argument("--fullscreen", action="store_true", default=None)
    disp.add_argument("--no-overlay", dest="show_overlay", action="store_false", default=None)
    disp.add_argument("--display-scale", type=float, default=None)
    disp.add_argument(
        "--display-max-size", dest="display_max_size", type=int, default=None,
        help="Cap the displayed frame's longest side in px (VSR still runs at "
             "full res; the result is downscaled on-GPU before display). "
             "0 disables (ship full upscaled frames). Default 1600.",
    )
    disp.add_argument(
        "--no-gl-present", dest="gl_present", action="store_false", default=None,
        help="Disable the GPU presenter (CUDA->OpenGL); use the OpenCV window "
             "with a device->host copy instead.",
    )

    dbg = p.add_argument_group("debug")
    dbg.add_argument("--log-level", default=None)
    dbg.add_argument(
        "--benchmark", type=int, default=None, metavar="N",
        help="Run N frames headless, report FPS, then exit (no window).",
    )

    args = p.parse_args(argv)

    # Collect only explicitly-provided overrides so Config.__post_init__ runs once
    # with the right mode (and derives mode-dependent defaults correctly).
    overrides: dict = {}
    if args.prompt is not None:
        overrides["prompt"] = args.prompt
    for field_name in (
        "model_id", "mode", "width", "height", "acceleration", "cfg_type",
        "guidance_scale", "delta", "seed", "negative_prompt", "do_add_noise",
        "feedback_zoom", "feedback_pan_x", "feedback_pan_y", "feedback_rotate",
        "feedback_padding", "feedback_noise_drift", "content_centering",
        "center_pan_gain",
        "audio_file", "audio_source", "audio_device", "audio_blocksize", "reactive",
        "beat_tempo_threshold", "beat_silence_db", "beat_bpm_smoothing",
        "react_zoom_gain", "react_beat_zoom", "react_beat_bass_scale", "react_zoom_max", "react_noise_gain",
        "react_rotate_gain", "react_beat_rotate", "react_pan_gain",
        "clap_model_id", "clap_device", "clap_window_sec", "clap_interval_sec",
        "clap_top_k", "clap_max_tags", "clap_enter_threshold",
        "clap_exit_threshold", "clap_score_tau", "clap_min_hold_sec",
        "clap_weight_min", "clap_weight_max", "clap_weight_step",
        "clap_softmax_temp", "clap_calib_alpha", "clap_calib_warmup",
        "clap_calib_spike_guard", "clap_vocab_path",
        "qwen_model_id", "qwen_python", "qwen_window_sec", "qwen_min_interval_sec",
        "qwen_instruction", "qwen_max_new_tokens", "qwen_timeout_sec", "qwen_port",
        "qwen_temperature", "qwen_top_p", "qwen_system_prompt", "qwen_blend_sec",
        "qwen_hold_threshold", "qwen_hold_sim_power", "qwen_hold_max_sec",
        "qwen_hold_check_interval_sec",
        "upscale_factor", "output_width", "output_height", "vsr_quality",
        "rtx_video_sdk_path",
        "streamdiffusion_path", "engine_dir", "display_scale", "display_max_size",
        "gen_fps_cap",
        "log_level",
        "benchmark",
    ):
        val = getattr(args, field_name)
        if val is not None:
            overrides[field_name] = val
    if args.upscale is not None:
        overrides["upscale"] = args.upscale
    if args.gl_present is not None:
        overrides["gl_present"] = args.gl_present
    if args.clap is not None:
        overrides["clap"] = args.clap
    if args.beat_grid_lock is not None:
        overrides["beat_grid_lock"] = args.beat_grid_lock
    if args.clap_calibrate is not None:
        overrides["clap_calibrate"] = args.clap_calibrate
    if args.qwen is not None:
        overrides["qwen"] = args.qwen
    if args.qwen_4bit is not None:
        overrides["qwen_4bit"] = args.qwen_4bit
    if args.qwen_do_sample is not None:
        overrides["qwen_do_sample"] = args.qwen_do_sample
    if args.qwen_blend is not None:
        overrides["qwen_blend"] = args.qwen_blend
    if args.qwen_hold is not None:
        overrides["qwen_hold"] = args.qwen_hold
    if args.qwen_hold_prefetch is not None:
        overrides["qwen_hold_prefetch"] = args.qwen_hold_prefetch
    # --audio FILE implies file source unless an explicit source was given.
    if args.audio_file is not None and args.audio_source is None:
        overrides["audio_source"] = "file"
    if args.t_index_list is not None:
        overrides["t_index_list"] = [
            int(x) for x in args.t_index_list.split(",") if x.strip()
        ]
    if args.fullscreen is not None:
        overrides["fullscreen"] = args.fullscreen
    if args.show_overlay is not None:
        overrides["show_overlay"] = args.show_overlay

    defaults = load_defaults_dict()
    merged = {**defaults, **overrides}
    return Config(**merged)


def _make_gl_presenter(pipeline, shared_prompt, audio_state, orchestrator, cfg, base_prompt=None, ui_context=None):
    """Build the GPU presenter on the *calling* thread (GL/CUDA are thread-bound).

    Returns the presenter, or ``None`` if it couldn't initialize (caller then
    falls back to the OpenCV window).
    """
    from music_visualizer.display.gl_presenter import GLPresenter

    disp_w, disp_h = pipeline.display_size
    return GLPresenter(
        cfg, disp_w, disp_h,
        shared_prompt=shared_prompt,
        audio_state=audio_state,
        pipeline=pipeline,
        orchestrator=orchestrator,
        base_prompt=base_prompt,
        ui_context=ui_context,
    )


def _generation_loop(
    pipeline,
    shared_prompt: SharedPrompt,
    display_queue: "queue.Queue",
    shutdown: threading.Event,
    fps_state: dict,
    audio_state=None,
    mapper=None,
    gl_request: bool = False,
    gl_state: Optional[dict] = None,
    orchestrator=None,
    cfg: Optional[Config] = None,
    embeds_slot=None,
    prompt_blend=None,
    base_prompt: Optional[SharedPrompt] = None,
    ui_context: Optional[RuntimeContext] = None,
) -> None:
    logger.info("Generation loop started.")

    # The GPU presenter must be created on this thread (its window + GL/CUDA
    # contexts are thread-affine). On failure we fall back to the queue + OpenCV.
    presenter = None
    if gl_request:
        try:
            presenter = _make_gl_presenter(
                pipeline, shared_prompt, audio_state, orchestrator, cfg, base_prompt,
                ui_context=ui_context,
            )
        except Exception as exc:
            logger.warning(
                "GPU presenter unavailable (%s); using the OpenCV window.", exc
            )
            presenter = None
    if gl_state is not None:
        gl_state["active"] = presenter is not None
        gl_state["presenter"] = presenter
        gl_state["decided"].set()

    frame_count = 0
    min_interval = 0.0
    next_slot = time.perf_counter()
    prev_slot_t: Optional[float] = None
    if cfg is not None and cfg.gen_fps_cap > 0:
        min_interval = 1.0 / cfg.gen_fps_cap
        logger.info(
            "img2img capped at %.1f fps (Stable Diffusion runs at most once per slot).",
            cfg.gen_fps_cap,
        )
    try:
        while not shutdown.is_set():
            # In-process pipeline rebuild (settings UI restart).
            if ui_context is not None and ui_context.restart.gen_restart.is_set():
                ui_context.restart.gen_restart.clear()
                logger.info("Rebuilding StreamDiffusion pipeline...")
                if cfg is not None and cfg.acceleration == "tensorrt":
                    engine_path = Path(cfg.resolved_engine_dir())
                    if not engine_path.exists():
                        logger.info(
                            "First run at %dx%d with TensorRT: building a new "
                            "engine (this can take several minutes; the window "
                            "will be unresponsive until it finishes).",
                            cfg.width, cfg.height,
                        )
                try:
                    pipeline.close()
                except Exception:
                    pass
                from music_visualizer.generation.stream_pipeline import StreamPipeline

                pipeline = StreamPipeline(cfg)
                if ui_context is not None:
                    ui_context.pipeline = pipeline
                if presenter is not None:
                    presenter.pipeline = pipeline
                    presenter.orchestrator = ui_context.orchestrator
                    presenter.refresh_ui()
                if gl_state is not None:
                    gl_state["pipeline"] = pipeline
                if orchestrator is not None and hasattr(orchestrator, "set_pipeline"):
                    orchestrator.set_pipeline(pipeline)
                # Re-wire CLAP/Qwen orchestrators if present on context.
                if ui_context.clap_orchestrator is not None:
                    ui_context.clap_orchestrator.set_pipeline(pipeline)
                if ui_context.qwen_orchestrator is not None:
                    ui_context.qwen_orchestrator.set_pipeline(pipeline)
                logger.info("Pipeline rebuild complete.")

            # Live FPS cap (UI may change gen_fps_cap at runtime).
            if cfg is not None:
                if cfg.gen_fps_cap > 0:
                    min_interval = 1.0 / cfg.gen_fps_cap
                else:
                    min_interval = 0.0

            # Wait for the next frame slot *before* touching StreamDiffusion so the
            # cap gates the actual img2img pass, not just presentation afterward.
            if min_interval > 0:
                wait = next_slot - time.perf_counter()
                if wait > 0:
                    shutdown.wait(timeout=wait)
                elif wait < -min_interval:
                    # Fell behind by more than one slot; don't burst to catch up.
                    next_slot = time.perf_counter()
                next_slot += min_interval

            slot_t = time.perf_counter()
            slot_dt = (
                (slot_t - prev_slot_t) if prev_slot_t is not None
                else (min_interval if min_interval > 0 else 0.0)
            )
            if prev_slot_t is not None and slot_dt > 0:
                fps_state["gen_fps"] = 1.0 / slot_dt
            prev_slot_t = slot_t

            if prompt_blend is not None:
                embeds = prompt_blend.current_embeds()
                if embeds is not None:
                    pipeline.apply_embeds(embeds)
            elif embeds_slot is not None:
                # CLAP path: the orchestrator already encoded the prompt off the
                # render thread; just swap the ready tensor in (cheap).
                embeds = embeds_slot.take_if_changed()
                if embeds is not None:
                    pipeline.apply_embeds(embeds)
            else:
                new_prompt = shared_prompt.take_if_changed()
                if new_prompt is not None and prompt_blend is None:
                    pipeline.update_prompt(new_prompt)
                    logger.debug("Prompt -> %s", new_prompt)

            # Audio-reactive motion: dt matches the capped slot interval so morph
            # speed stays consistent when gen_fps_cap is active.
            live_audio = (
                ui_context.audio_state if ui_context is not None else audio_state
            )
            live_mapper = (
                ui_context.mapper if ui_context is not None else mapper
            )
            features = live_audio.get(consume_beat=True) if live_audio is not None else None
            pipeline.set_audio_features(features)
            if cfg is not None and cfg.reactive and live_mapper is not None and features is not None:
                params = live_mapper.update(features, slot_dt)
                pipeline.set_transform(**params)

            try:
                if presenter is not None:
                    frame = pipeline.generate_rgba_gpu()
                else:
                    frame = pipeline.generate()
            except Exception:
                logger.exception("Generation failed; stopping.")
                shutdown.set()
                break

            if presenter is not None:
                # Present straight from the GPU (no PCIe readback, no imshow).
                try:
                    presenter.present(frame, fps_state.get("gen_fps", 0.0))
                except Exception:
                    logger.exception("GPU present failed; stopping.")
                    shutdown.set()
                    break
                if presenter.should_close():
                    shutdown.set()
            else:
                # Drop-oldest so generation never blocks on a slow display.
                try:
                    display_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        display_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        display_queue.put_nowait(frame)
                    except queue.Full:
                        pass

            frame_count += 1
    finally:
        if presenter is not None:
            presenter.close()
    logger.info("Generation loop stopped (%d frames).", frame_count)


_MOTION_COMMANDS = {
    "/zoom": "feedback_zoom",
    "/panx": "feedback_pan_x",
    "/pany": "feedback_pan_y",
    "/rotate": "feedback_rotate",
    "/noise": "feedback_noise_drift",
}


def _handle_motion_command(line: str, pipeline) -> bool:
    """Handle a ``/zoom 1.05`` style motion command. Returns True if handled."""
    parts = line.split()
    cmd = parts[0].lower()
    key = _MOTION_COMMANDS.get(cmd)
    if key is None:
        return False
    if len(parts) < 2:
        logger.warning("Usage: %s <value>", cmd)
        return True
    try:
        value = float(parts[1])
    except ValueError:
        logger.warning("Invalid number for %s: %s", cmd, parts[1])
        return True
    # set_transform maps config-style names; strip the 'feedback_' prefix.
    pipeline.set_transform(**{key.replace("feedback_", ""): value})
    logger.info("Motion %s -> %.4f", cmd, value)
    return True


def _stdin_prompt_loop(
    shared_prompt: SharedPrompt, shutdown: threading.Event, ctx: Optional[RuntimeContext] = None
) -> None:
    """Read prompts/commands from stdin until EOF/shutdown. Runs as a daemon thread."""
    logger.info(
        "Type a prompt + Enter to inject it. Motion: /zoom 1.04, /panx 0.01, "
        "/pany -0.01, /rotate 0.5, /noise 0. 'q' to quit."
    )
    while not shutdown.is_set():
        try:
            line = sys.stdin.readline()
        except (EOFError, ValueError):
            break
        if line == "":  # EOF
            break
        line = line.strip()
        if not line:
            continue
        if line.lower() in {"q", "quit", "exit"}:
            shutdown.set()
            break
        pipeline = ctx.pipeline if ctx is not None else None
        if line.startswith("/") and pipeline is not None:
            if _handle_motion_command(line, pipeline):
                continue
        shared_prompt.set(line)
        if ctx is not None and ctx.base_prompt is not None:
            ctx.base_prompt.set(line, allow_empty=True)
            ctx.cfg.prompt = line.strip()
            if ctx.clap_orchestrator is not None:
                ctx.clap_orchestrator.republish_from_base()
            elif ctx.qwen_orchestrator is not None:
                ctx.qwen_orchestrator.republish_from_base()


def _stop_audio_stack(
    audio_engine,
    analyzer_thread,
    clap_orchestrator,
    qwen_orchestrator,
) -> None:
    if audio_engine is not None:
        try:
            audio_engine.stop()
        except Exception:
            pass
    for orch in (clap_orchestrator, qwen_orchestrator):
        if orch is not None:
            orch.request_stop()
    if analyzer_thread is not None:
        analyzer_thread.request_stop()
    if clap_orchestrator is not None:
        clap_orchestrator.join(timeout=3.0)
    if qwen_orchestrator is not None:
        qwen_orchestrator.join(timeout=3.0)
        # Keep the sidecar alive across in-process audio rebuilds; full shutdown
        # closes it via qwen_orchestrator.close() in main()'s finally block.
    if analyzer_thread is not None:
        analyzer_thread.join(timeout=3.0)


def _rebuild_audio_stack(
    cfg: Config,
    shutdown: threading.Event,
    base_prompt: SharedPrompt,
    active_prompt: SharedPrompt,
    embeds_slot,
    qwen_client,
    ui_context: RuntimeContext,
) -> tuple:
    """Stop and recreate the audio capture/analysis stack from current cfg."""
    logger.info("Rebuilding audio/analysis stack from settings...")
    _stop_audio_stack(
        ui_context.audio_engine,
        ui_context.analyzer_thread,
        ui_context.clap_orchestrator,
        ui_context.qwen_orchestrator,
    )

    clap_active = cfg.clap and cfg.audio_source != "none"
    qwen_active = cfg.qwen and cfg.audio_source != "none"

    if cfg.audio_source == "none":
        ui_context.audio_engine = None
        ui_context.analyzer_thread = None
        ui_context.mapper = None
        ui_context.clap_orchestrator = None
        ui_context.qwen_orchestrator = None
        ui_context.orchestrator = None
        ui_context.audio_state = None
        return (None,) * 9 + (qwen_client,)

    (
        audio_state, audio_engine, analyzer_thread, mapper,
        orchestrator, clap_orchestrator, qwen_orchestrator,
        audio_ring, qwen_active,
    ) = _start_audio(
        cfg, shutdown, base_prompt, active_prompt, clap_active, embeds_slot
    )

    pipeline = ui_context.pipeline
    if clap_orchestrator is not None and pipeline is not None:
        clap_orchestrator.set_pipeline(pipeline)

    if qwen_active and audio_ring is not None:
        from music_visualizer.analysis.qwen_client import QwenSidecarClient

        if qwen_client is None or not qwen_client.is_alive():
            if qwen_client is not None:
                try:
                    qwen_client.close()
                except Exception:
                    pass
            qwen_client = QwenSidecarClient(cfg)
            qwen_client.start()
        qwen_orchestrator, orchestrator = _start_qwen_orchestrator(
            cfg, shutdown, base_prompt, active_prompt, audio_ring,
            clap_orchestrator, qwen_client, None, clap_active,
        )
        if pipeline is not None:
            qwen_orchestrator.set_pipeline(pipeline)

    ui_context.audio_engine = audio_engine
    ui_context.analyzer_thread = analyzer_thread
    ui_context.mapper = mapper
    ui_context.clap_orchestrator = clap_orchestrator
    ui_context.qwen_orchestrator = qwen_orchestrator
    ui_context.orchestrator = orchestrator

    presenter = ui_context.presenter
    if presenter is not None:
        presenter.orchestrator = orchestrator
        presenter.audio_state = audio_state
        presenter.refresh_ui()

    ui_context.audio_state = audio_state

    return (
        audio_state, audio_engine, analyzer_thread, mapper,
        orchestrator, clap_orchestrator, qwen_orchestrator,
        audio_ring, qwen_active, qwen_client,
    )


def run_benchmark(cfg: Config, n: int) -> None:
    from music_visualizer.generation.stream_pipeline import StreamPipeline

    pipeline = StreamPipeline(cfg)
    logger.info("Warming up...")
    pipeline.warmup(3)

    logger.info("Benchmarking %d frames...", n)
    t0 = time.time()
    for _ in range(n):
        pipeline.generate()
    elapsed = time.time() - t0
    fps = n / elapsed if elapsed > 0 else 0.0
    logger.info("Benchmark: %d frames in %.2fs => %.1f fps (%.1f ms/frame)",
                n, elapsed, fps, 1000.0 * elapsed / n)
    pipeline.close()


def main(argv: list[str] | None = None) -> int:
    cfg = build_config(argv)
    setup_logging(cfg.log_level)

    if cfg.benchmark and cfg.benchmark > 0:
        run_benchmark(cfg, cfg.benchmark)
        return 0

    from music_visualizer.display.renderer import Renderer
    from music_visualizer.generation.stream_pipeline import StreamPipeline
    from music_visualizer.utils.shared_state import AudioState, PromptBlendState

    shutdown = threading.Event()
    # Two prompt channels: `base_prompt` is the manual foundation (CLI/stdin),
    # `active_prompt` is what generation renders. CLAP and/or Qwen compose onto
    # the base and publish the result to active without overwriting the base.
    clap_active = cfg.clap and cfg.audio_source != "none"
    qwen_active = cfg.qwen and cfg.audio_source != "none"
    if cfg.qwen_blend and clap_active:
        logger.warning(
            "qwen_blend is disabled while CLAP is active (both use prompt embeds). "
            "Run with --no-clap to enable Qwen caption crossfading."
        )
    qwen_blend_active = qwen_active and cfg.qwen_blend and not clap_active
    prompt_blend = PromptBlendState() if qwen_blend_active else None
    # Separate active channel when CLAP composes tags on top of the base prompt.
    base_prompt = SharedPrompt(prompt=cfg.prompt, negative_prompt=cfg.negative_prompt)
    active_prompt = (
        SharedPrompt(prompt=cfg.prompt, negative_prompt=cfg.negative_prompt)
        if (clap_active or qwen_active)
        else base_prompt
    )
    # When CLAP is active the orchestrator encodes the composed prompt off the
    # render thread and publishes the ready embedding tensor here; the generation
    # loop only swaps it in (no per-cycle CLIP encode on the hot path).
    embeds_slot = PromptEmbeds() if clap_active else None
    display_queue: "queue.Queue" = queue.Queue(maxsize=2)
    fps_state: dict = {"gen_fps": 0.0}

    restart = RestartCoordinator()
    ui_context = RuntimeContext(cfg=cfg, restart=restart)

    # Audio backend (milestone 2): start before the pipeline so capture warms up.
    audio_state: AudioState | None = None
    audio_engine = None
    analyzer_thread = None
    mapper = None
    orchestrator = None
    clap_orchestrator = None
    qwen_orchestrator = None
    audio_ring = None
    qwen_client = None
    if cfg.audio_source != "none":
        try:
            (
                audio_state, audio_engine, analyzer_thread, mapper,
                orchestrator, clap_orchestrator, qwen_orchestrator,
                audio_ring, qwen_active,
            ) = _start_audio(
                cfg, shutdown, base_prompt, active_prompt, clap_active, embeds_slot
            )
        except Exception as exc:
            logger.error("Audio backend failed to start: %s", exc)
            return 2

    # Load Qwen into VRAM before StreamDiffusion so both fit on a 16 GB card.
    if qwen_active and audio_ring is not None:
        from music_visualizer.analysis.qwen_client import QwenSidecarClient

        try:
            qwen_client = QwenSidecarClient(cfg)
            qwen_client.start()
        except Exception as exc:
            logger.error("Qwen sidecar failed to start: %s", exc)
            return 2

    pipeline = StreamPipeline(cfg)
    # img2img bootstraps and warms the latent buffer inside StreamPipeline.__init__.
    if cfg.mode == "txt2img":
        pipeline.warmup(2)

    if qwen_active and audio_ring is not None and qwen_client is not None:
        qwen_orchestrator, orchestrator = _start_qwen_orchestrator(
            cfg, shutdown, base_prompt, active_prompt, audio_ring,
            clap_orchestrator, qwen_client, prompt_blend, clap_active,
        )

    # Hand the pipeline to the CLAP orchestrator so prompt embeddings are encoded
    # on its thread (off the render hot path) and applied via the embeds slot.
    if clap_orchestrator is not None:
        clap_orchestrator.set_pipeline(pipeline)
    if qwen_orchestrator is not None:
        qwen_orchestrator.set_pipeline(pipeline)
        if prompt_blend is not None:
            seed = pipeline.compute_embeds(base_prompt.current())
            if seed is not None:
                prompt_blend.snap(seed)

    ui_context.pipeline = pipeline
    ui_context.mapper = mapper
    ui_context.base_prompt = base_prompt
    ui_context.analyzer_thread = analyzer_thread
    ui_context.clap_orchestrator = clap_orchestrator
    ui_context.qwen_orchestrator = qwen_orchestrator
    ui_context.orchestrator = orchestrator
    ui_context.audio_engine = audio_engine
    ui_context.audio_state = audio_state

    from music_visualizer.ui.controller import UIController
    from music_visualizer.ui.tk_panel import TkSettingsPanel
    from music_visualizer.display.key_poll import F1Edge

    opencv_ui = UIController(ui_context, bridge=None)
    tk_settings = TkSettingsPanel(opencv_ui)
    f1_edge = F1Edge()

    # GPU presenter (CUDA->OpenGL) when possible: windowed Windows sessions only.
    # The presenter is created inside the gen thread; gl_state reports the outcome
    # so the main thread knows whether to also open an OpenCV window.
    gl_request = bool(cfg.gl_present) and sys.platform == "win32" and not cfg.fullscreen
    if gl_request:
        from music_visualizer.display.native.build_glpresent import ensure_dll

        ensure_dll(log=logger)

    gl_state: dict = {"active": False, "decided": threading.Event(), "pipeline": pipeline}

    gen_thread = threading.Thread(
        target=_generation_loop,
        args=(pipeline, active_prompt, display_queue, shutdown, fps_state,
              audio_state, mapper, gl_request, gl_state, orchestrator, cfg,
              embeds_slot, prompt_blend, base_prompt, ui_context),
        name="generation",
        daemon=True,
    )
    # stdin edits the manual base prompt; orchestrators compose onto active_prompt.
    stdin_thread = threading.Thread(
        target=_stdin_prompt_loop,
        args=(base_prompt, shutdown, ui_context),
        name="stdin-prompt",
        daemon=True,
    )
    gen_thread.start()
    stdin_thread.start()

    # Wait for the gen thread to decide if the GPU presenter came up (it owns its
    # own window). Building the bridge DLL on first run can take a few seconds.
    if gl_request:
        gl_state["decided"].wait(timeout=120)

    try:
        if gl_state["active"]:
            presenter = gl_state.get("presenter")
            use_tk_ui = bool(
                presenter
                and presenter.ui_controller
                and (
                    presenter.ui_controller._bridge is None
                    or not presenter.ui_controller._bridge._available
                )
            )
            if use_tk_ui:
                logger.info(
                    "Native settings UI unavailable; F1 opens Tk settings panel."
                )
            logger.info(
                "Presenting from GPU. Press 'q'/Esc to quit, F1 settings, "
                "F7 overlay, F11 fullscreen."
            )
            while not shutdown.is_set():
                if use_tk_ui:
                    if f1_edge.pressed():
                        tk_settings.toggle()
                    if tk_settings.visible and tk_settings._root is not None:
                        try:
                            tk_settings._root.update()
                        except Exception:
                            pass
                if restart.audio_restart.is_set():
                    restart.audio_restart.clear()
                    try:
                        result = _rebuild_audio_stack(
                            cfg, shutdown, base_prompt, active_prompt,
                            embeds_slot, qwen_client, ui_context,
                        )
                        if result[0] is not None:
                            audio_state = result[0]
                            audio_engine = result[1]
                            analyzer_thread = result[2]
                            mapper = result[3]
                            orchestrator = result[4]
                            clap_orchestrator = result[5]
                            qwen_orchestrator = result[6]
                            audio_ring = result[7]
                            qwen_active = result[8]
                            qwen_client = result[9]
                        ui_context.restart.clear_pending()
                        presenter = gl_state.get("presenter")
                        if presenter is not None:
                            presenter.refresh_ui()
                    except Exception:
                        logger.exception("Audio stack rebuild failed.")
                shutdown.wait(0.2)
                if not gen_thread.is_alive():
                    break
        else:
            renderer = Renderer(
                cfg, active_prompt, audio_state, pipeline, orchestrator,
                base_prompt=base_prompt, tk_panel=tk_settings,
            )
            renderer.run(display_queue, shutdown, fps_state)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        shutdown.set()
        gen_thread.join(timeout=5.0)
        if audio_engine is not None:
            audio_engine.stop()
        if clap_orchestrator is not None:
            clap_orchestrator.join(timeout=2.0)
        if qwen_orchestrator is not None:
            qwen_orchestrator.join(timeout=2.0)
            qwen_orchestrator.close()
        if analyzer_thread is not None:
            analyzer_thread.join(timeout=2.0)
        try:
            pipeline.close()
        except Exception:
            pass

    return 0


def _start_audio(
    cfg: Config,
    shutdown: threading.Event,
    base_prompt: SharedPrompt,
    active_prompt: SharedPrompt,
    clap_active: bool,
    embeds_slot=None,
):
    """Build + start the audio engine, analyzer, and prompt orchestrators.

    Returns ``(audio_state, engine, analyzer_thread, mapper,
    display_orchestrator, clap_orchestrator, qwen_orchestrator)``.
    ``display_orchestrator`` is used for overlay lines (may be a composite).
    """
    from music_visualizer.audio.engine import AnalyzerThread, make_engine
    from music_visualizer.audio.reactive import ReactiveMapper
    from music_visualizer.analysis.qwen_client import CompositeOrchestrator
    from music_visualizer.utils.shared_state import AudioRing, AudioState

    qwen_active = cfg.qwen and cfg.audio_source != "none"
    audio_state = AudioState()
    block_queue: "queue.Queue" = queue.Queue(maxsize=32)
    engine = make_engine(cfg, block_queue)
    # Start the device first so a live source can report its negotiated rate.
    engine.start()

    audio_ring = None
    if clap_active or qwen_active:
        capacity = max(
            8.0,
            (cfg.clap_window_sec * 2.0) if clap_active else 0.0,
            (cfg.qwen_window_sec + 5.0) if qwen_active else 0.0,
        )
        audio_ring = AudioRing(engine.sample_rate, capacity_seconds=capacity)

    analyzer_thread = AnalyzerThread(
        cfg, block_queue, audio_state, engine.sample_rate, shutdown, audio_ring
    )
    mapper = ReactiveMapper(cfg) if cfg.reactive else None
    analyzer_thread.start()

    clap_orchestrator = None
    if clap_active and audio_ring is not None:
        clap_orchestrator = _start_clap(
            cfg, shutdown, base_prompt, active_prompt, audio_ring, embeds_slot
        )

    qwen_orchestrator = None
    if qwen_active and audio_ring is not None:
        # Qwen sidecar is started after StreamDiffusion loads (GPU contention).
        pass

    parts = [o for o in (clap_orchestrator, qwen_orchestrator) if o is not None]
    if len(parts) == 0:
        display_orchestrator = None
    elif len(parts) == 1:
        display_orchestrator = parts[0]
    else:
        display_orchestrator = CompositeOrchestrator(*parts)

    logger.info(
        "Audio backend ready: source=%s reactive=%s clap=%s qwen=%s",
        cfg.audio_source, cfg.reactive, clap_active, qwen_active,
    )
    return (
        audio_state, engine, analyzer_thread, mapper,
        display_orchestrator, clap_orchestrator, qwen_orchestrator,
        audio_ring if (clap_active or qwen_active) else None,
        qwen_active,
    )


def _start_qwen_orchestrator(
    cfg: Config,
    shutdown: threading.Event,
    base_prompt: SharedPrompt,
    active_prompt: SharedPrompt,
    audio_ring,
    clap_orchestrator,
    client,
    blend_state=None,
    clap_active: bool = False,
):
    """Start the Qwen caption thread using an already-running sidecar client."""
    from music_visualizer.analysis.qwen_client import CompositeOrchestrator, QwenOrchestrator

    similarity_embedder = None
    if cfg.qwen_hold:
        from music_visualizer.analysis.clap_analyzer import CLAPAnalyzer

        similarity_embedder = CLAPAnalyzer(
            vocabulary=[],
            model_id=cfg.clap_model_id,
            device=cfg.clap_device,
            calibrate=False,
        )

    qwen_orchestrator = QwenOrchestrator(
        cfg,
        client,
        audio_ring,
        base_prompt,
        active_prompt,
        shutdown,
        blend_state=blend_state,
        clap_composes=clap_active,
        similarity_embedder=similarity_embedder,
    )
    qwen_orchestrator.start()
    if clap_orchestrator is not None:
        clap_orchestrator.set_qwen_orchestrator(qwen_orchestrator)
    parts = [o for o in (clap_orchestrator, qwen_orchestrator) if o is not None]
    if len(parts) == 0:
        display = None
    elif len(parts) == 1:
        display = parts[0]
    else:
        display = CompositeOrchestrator(*parts)
    return qwen_orchestrator, display


def _start_clap(
    cfg: Config,
    shutdown: threading.Event,
    base_prompt: SharedPrompt,
    active_prompt: SharedPrompt,
    audio_ring,
    embeds_slot=None,
):
    """Build the CLAP analyzer + prompt builder and start the orchestrator."""
    from music_visualizer.analysis.clap_analyzer import CLAPAnalyzer
    from music_visualizer.analysis.orchestrator import PromptOrchestrator
    from music_visualizer.analysis.prompt_builder import PromptBuilder
    from music_visualizer.analysis.vocabulary import load_vocab

    vocab = load_vocab(cfg.clap_vocab_path)
    analyzer = CLAPAnalyzer(
        vocabulary=vocab,
        model_id=cfg.clap_model_id,
        device=cfg.clap_device,
        softmax_temp=cfg.clap_softmax_temp,
        calibrate=cfg.clap_calibrate,
        calib_alpha=cfg.clap_calib_alpha,
        calib_warmup=cfg.clap_calib_warmup,
        calib_spike_guard=cfg.clap_calib_spike_guard,
    )
    builder = PromptBuilder(
        base_prompt=cfg.prompt,
        enter_threshold=cfg.clap_enter_threshold,
        exit_threshold=cfg.clap_exit_threshold,
        score_tau=cfg.clap_score_tau,
        min_hold_sec=cfg.clap_min_hold_sec,
        weight_min=cfg.clap_weight_min,
        weight_max=cfg.clap_weight_max,
        weight_step=cfg.clap_weight_step,
        max_tags=cfg.clap_max_tags,
        prompt_blacklist=cfg.prompt_blacklist,
        prompt_blacklist_mode=cfg.prompt_blacklist_mode,
        prompt_blacklist_whole_words=cfg.prompt_blacklist_whole_words,
    )
    orchestrator = PromptOrchestrator(
        cfg, analyzer, builder, audio_ring, base_prompt, active_prompt, shutdown,
        embeds_slot=embeds_slot,
    )
    orchestrator.start()
    return orchestrator


if __name__ == "__main__":
    raise SystemExit(main())
