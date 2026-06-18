"""Shared overlay text construction.

Both display backends (the OpenCV :class:`~music_visualizer.display.renderer.Renderer`
and the GPU :class:`~music_visualizer.display.gl_presenter.GLPresenter`) draw the
same status overlay -- FPS, audio features, feedback motion, CLAP tags, prompt.
This module builds the line list once so the two paths can't drift apart.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from music_visualizer.audio.reactive import bpm_noise_multiplier
from music_visualizer.config import Config
from music_visualizer.utils.shared_state import AudioState, SharedPrompt

OVERLAY_WRAP = 72  # max chars per overlay line before wrapping


def wrap_overlay(text: str, width: int = OVERLAY_WRAP) -> List[str]:
    """Break long overlay strings into readable lines."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= width:
        return [text]
    words = text.split()
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def collect_overlay_lines(
    config: Config,
    shared_prompt: Optional[SharedPrompt],
    audio_state: Optional[AudioState],
    pipeline,
    orchestrator,
    gen_fps: float,
    disp_fps: float,
    base_prompt: Optional[SharedPrompt] = None,
) -> Tuple[List[str], object]:
    """Build the overlay text lines and return ``(lines, features)``.

    ``features`` is the latest audio snapshot (or ``None``) so the caller can
    drive the beat-flash indicator without fetching it again.
    """
    lines = [f"display {disp_fps:4.1f} fps  |  gen {gen_fps:4.1f} fps"]

    features = audio_state.get() if audio_state is not None else None
    if features is not None:
        lines.append(
            f"bpm {features.bpm:5.1f}  rms {features.rms:.2f}  "
            f"bass {features.bass:.2f} mid {features.mid:.2f} treb {features.treble:.2f}"
        )
        nm = bpm_noise_multiplier(
            features.bpm,
            pivot=config.react_bpm_noise_pivot,
            mult_min=config.react_bpm_noise_mult_min,
            mult_max=config.react_bpm_noise_mult_max,
            high_bpm=config.react_bpm_noise_high_bpm,
        )
        lines.append(f"noise x {nm:.2f}  (bpm scale)")

    # Current feedback-motion parameters (img2img only applies the transform).
    if pipeline is not None and config.mode == "img2img":
        m = pipeline.get_motion_params()
        lines.append(
            f"zoom {m['zoom']:.4f}  pan ({m['pan_x']:+.4f},{m['pan_y']:+.4f})"
        )
        lines.append(
            f"rot {m['rotate']:+.4f}  noise {m['noise']:.2f}"
            + ("  CENTER" if m.get("centering") else "")
        )
        if m.get("centering"):
            lines.append(
                f"cpan ({m['center_pan_x']:+.4f},{m['center_pan_y']:+.4f})"
                f"  x{m.get('center_strength', 1.0):.2f}"
            )

    if orchestrator is not None:
        for line in orchestrator.overlay_lines():
            lines.extend(wrap_overlay(line))

    if (
        base_prompt is not None
        and shared_prompt is not None
        and base_prompt is not shared_prompt
    ):
        anchor = base_prompt.current().strip()
        if anchor:
            lines.append("base:")
            lines.extend(wrap_overlay(anchor))

    if shared_prompt is not None:
        prompt = shared_prompt.current()
        lines.append("prompt:")
        lines.extend(wrap_overlay(prompt))

    return lines, features
