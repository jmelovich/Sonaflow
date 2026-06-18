"""OpenCV display renderer.

Runs on the main thread (cv2's HighGUI event loop wants the main thread). Pulls
finished BGR frames from a queue, draws an optional overlay, shows them, and
reports display FPS. Keyboard: ``q`` or ``Esc`` requests shutdown; ``F7`` toggles
the debug overlay.
"""

from __future__ import annotations

import sys
import queue
import threading
import time
import tkinter as tk
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

from music_visualizer.config import Config
from music_visualizer.display.letterbox import letterbox_frame, visible_image_scale
from music_visualizer.display.overlay import collect_overlay_lines
from music_visualizer.utils.logging import get_logger
from music_visualizer.utils.shared_state import AudioState, SharedPrompt

if sys.platform == "win32":
    from music_visualizer.display.win32_window import find_window, toggle_borderless_fullscreen
else:
    find_window = None  # type: ignore[misc, assignment]
    toggle_borderless_fullscreen = None  # type: ignore[misc, assignment]

logger = get_logger(__name__)

# Overlay layout tuned for ~512–1080p; scaled up when the window shrinks a large frame.
_OVERLAY_FONT = 0.55
_OVERLAY_LINE = 26
_OVERLAY_MARGIN = 10
_OVERLAY_Y0 = 22
_OVERLAY_THICK_OUT = 3
_OVERLAY_THICK_IN = 1
_OVERLAY_BEAT_MARGIN = 24
_OVERLAY_BEAT_RADIUS = 10
_WINDOW_FIT_CAP = 1600  # matches initial resizeWindow cap in run()
# OpenCV HighGUI on Windows: function keys via waitKeyEx (0x700000 + (n-1)*0x10000).
_KEY_F1_WIN = 0x700000
_KEY_F7_WIN = 0x760000
_KEY_F11_WIN = 0x7A0000


def _poll_key() -> int:
    """Read one key from the OpenCV window (function keys need waitKeyEx on Win)."""
    if sys.platform == "win32":
        return int(cv2.waitKeyEx(1))
    return int(cv2.waitKey(1) & 0xFF)


class Renderer:
    def __init__(
        self,
        config: Config,
        shared_prompt: Optional[SharedPrompt] = None,
        audio_state: Optional[AudioState] = None,
        pipeline=None,
        orchestrator=None,
        base_prompt: Optional[SharedPrompt] = None,
        tk_panel=None,
    ):
        self.config = config
        self.shared_prompt = shared_prompt
        self.base_prompt = base_prompt
        self.audio_state = audio_state
        self.pipeline = pipeline
        self.orchestrator = orchestrator
        self._tk_panel = tk_panel
        self._frame_times: Deque[float] = deque(maxlen=60)
        self._last_beat_wall = 0.0  # wall-clock of last observed beat, for the flash
        self._win = config.window_title
        self._fit_scale: float = 1.0  # frame->window scale set on first frame

    def _overlay_ui_scale(self, frame_w: int, frame_h: int) -> float:
        """Multiplier for overlay metrics so text stays readable when the window
        letterboxes a large frame."""
        rect = cv2.getWindowImageRect(self._win)
        dw, dh = rect[2], rect[3]
        if dw > 0 and dh > 0:
            return visible_image_scale(frame_w, frame_h, dw, dh)
        if self._fit_scale < 1.0:
            return max(1.0, 1.0 / self._fit_scale)
        cap_scale = _WINDOW_FIT_CAP / max(frame_w, frame_h)
        if cap_scale < 1.0:
            return 1.0 / cap_scale
        return 1.0

    def _measured_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span

    def _draw_overlay(self, frame: np.ndarray, gen_fps: float) -> np.ndarray:
        disp_fps = self._measured_fps()
        lines, features = collect_overlay_lines(
            self.config, self.shared_prompt, self.audio_state, self.pipeline,
            self.orchestrator, gen_fps, disp_fps, base_prompt=self.base_prompt,
        )

        h, w = frame.shape[:2]
        ui = self._overlay_ui_scale(w, h)
        font = _OVERLAY_FONT * ui
        y_step = max(12, int(round(_OVERLAY_LINE * ui)))
        y0 = max(14, int(round(_OVERLAY_Y0 * ui)))
        mx = max(6, int(round(_OVERLAY_MARGIN * ui)))
        thick_out = max(1, int(round(_OVERLAY_THICK_OUT * ui)))
        thick_in = max(1, int(round(_OVERLAY_THICK_IN * ui)))

        y = y0
        for line in lines:
            cv2.putText(
                frame, line, (mx, y), cv2.FONT_HERSHEY_SIMPLEX, font,
                (0, 0, 0), thick_out, cv2.LINE_AA,
            )
            cv2.putText(
                frame, line, (mx, y), cv2.FONT_HERSHEY_SIMPLEX, font,
                (255, 255, 255), thick_in, cv2.LINE_AA,
            )
            y += y_step

        # Beat flash: a dot top-right that lights up briefly after each beat.
        # Beats land on single hops, so latch the last beat time and hold the
        # flash ~120ms rather than testing the instantaneous flag.
        if features is not None:
            if features.is_beat:
                self._last_beat_wall = time.time()
            beat_m = max(12, int(round(_OVERLAY_BEAT_MARGIN * ui)))
            beat_r = max(5, int(round(_OVERLAY_BEAT_RADIUS * ui)))
            lit = (time.time() - self._last_beat_wall) < 0.12
            color = (0, 255, 0) if lit else (60, 60, 60)
            cv2.circle(frame, (w - beat_m, beat_m), beat_r, color, -1, cv2.LINE_AA)
        return frame

    def _handle_key(self, key: int, shutdown: threading.Event) -> None:
        if (key & 0xFF) in (ord("q"), ord("Q"), 27):
            shutdown.set()
            return
        if sys.platform == "win32" and key == _KEY_F1_WIN and self._tk_panel is not None:
            self._tk_panel.toggle()
            logger.info("Settings panel %s.", "open" if self._tk_panel.visible else "closed")
            return

        if sys.platform == "win32" and key == _KEY_F7_WIN:
            self.config.show_overlay = not self.config.show_overlay
            logger.info(
                "Debug overlay %s.", "on" if self.config.show_overlay else "off"
            )

        if sys.platform == "win32" and key == _KEY_F11_WIN and find_window is not None:
            hwnd = find_window(self._win)
            if hwnd:
                active = toggle_borderless_fullscreen(hwnd)
                logger.info("Fullscreen %s.", "on" if active else "off")

    def _present_frame(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox the frame to the current window client area."""
        rect = cv2.getWindowImageRect(self._win)
        return letterbox_frame(frame, rect[2], rect[3])

    def run(
        self,
        display_queue: "queue.Queue",
        shutdown: threading.Event,
        fps_state: Optional[dict] = None,
    ) -> None:
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        sized = False
        if self.config.fullscreen and find_window is not None:
            hwnd = find_window(self._win)
            if hwnd:
                toggle_borderless_fullscreen(hwnd)
            sized = True

        logger.info(
            "Display window open. Press 'q'/Esc to quit, F1 settings, F7 overlay, F11 fullscreen."
        )
        try:
            while not shutdown.is_set():
                if self._tk_panel is not None and self._tk_panel.visible:
                    try:
                        self._tk_panel._root.update()  # type: ignore[union-attr]
                    except tk.TclError:
                        pass
                try:
                    frame = display_queue.get(timeout=0.1)
                except queue.Empty:
                    # Keep the GUI responsive even if generation stalls.
                    self._handle_key(_poll_key(), shutdown)
                    if cv2.getWindowProperty(self._win, cv2.WND_PROP_VISIBLE) < 1:
                        shutdown.set()
                    continue

                self._frame_times.append(time.time())

                # Size the window to the first real frame (which may be upscaled),
                # capped so a 2K/4K frame still fits on screen.
                if not sized:
                    fh, fw = frame.shape[:2]
                    cap = _WINDOW_FIT_CAP
                    scale = min(1.0, cap / max(fw, fh))
                    self._fit_scale = scale
                    cv2.resizeWindow(self._win, int(fw * scale), int(fh * scale))
                    sized = True

                if self.config.display_scale != 1.0:
                    frame = cv2.resize(
                        frame,
                        None,
                        fx=self.config.display_scale,
                        fy=self.config.display_scale,
                        interpolation=cv2.INTER_CUBIC,
                    )

                if self.config.show_overlay:
                    gen_fps = float(fps_state.get("gen_fps", 0.0)) if fps_state else 0.0
                    frame = self._draw_overlay(frame, gen_fps)

                frame = self._present_frame(frame)
                cv2.imshow(self._win, frame)
                self._handle_key(_poll_key(), shutdown)
                if cv2.getWindowProperty(self._win, cv2.WND_PROP_VISIBLE) < 1:
                    shutdown.set()
        finally:
            cv2.destroyAllWindows()
            # Pump the event loop so the window actually closes on Windows.
            for _ in range(4):
                cv2.waitKey(1)
            logger.info("Display closed (avg %.1f fps).", self._measured_fps())
