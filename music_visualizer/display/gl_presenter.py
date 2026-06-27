"""GPU presenter (CUDA -> OpenGL interop).

Shows upscaled frames straight from the GPU: the VSR output is an RGBA ``uint8``
CUDA tensor, which is blitted into an OpenGL texture via CUDA interop and drawn as
a textured quad. Nothing crosses the PCIe bus on the display path -- no
``tensor.cpu()`` readback and no ``cv2.imshow``.

The optional status overlay is composited *on the GPU*: its text is rendered once
to a small RGBA strip on the CPU (cheap), uploaded, and alpha-blended over the top
of the frame tensor before presenting.

All methods must be called from a single host thread (the generation thread in the
visualizer), since the Win32 window, GL context, and CUDA context are thread-bound.
"""

from __future__ import annotations

import ctypes
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, TYPE_CHECKING

import cv2
import numpy as np
import torch

from music_visualizer.config import Config
from music_visualizer.display.letterbox import visible_image_scale
from music_visualizer.display.overlay import collect_overlay_lines
from music_visualizer.ui.ui_bridge import NativeUIBridge
from music_visualizer.utils.logging import get_logger
from music_visualizer.utils.shared_state import AudioState, SharedPrompt

if TYPE_CHECKING:
    from music_visualizer.ui.controller import RuntimeContext, UIController

logger = get_logger(__name__)

_NATIVE_DIR = Path(__file__).resolve().parent / "native"
_DLL_PATH = _NATIVE_DIR / "glpresent_bridge.dll"

# Resolve the app icon once at import time; works in both the dev tree and the
# packaged layout where resources/ sits one level above the app/ directory.
_ICON_PATH: Optional[Path] = None
for _p in [
    Path(__file__).resolve().parent.parent.parent / "resources" / "SonaflowIcon1.ico",
    Path(__file__).resolve().parent.parent.parent.parent / "resources" / "SonaflowIcon1.ico",
]:
    if _p.exists():
        _ICON_PATH = _p
        break


def _set_process_app_id() -> None:
    """Give this process its own AUMID so Windows assigns it a dedicated taskbar
    button with Sonaflow's icon instead of inheriting the python.exe button."""
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Sonaflow.Sonaflow")
    except Exception:
        pass

# Overlay layout tuned for ~512-1080p; scaled up when the window shrinks a large
# frame so text stays readable (mirrors the OpenCV renderer's metrics).
_OVERLAY_FONT = 0.55
_OVERLAY_LINE = 26
_OVERLAY_MARGIN = 10
_OVERLAY_Y0 = 22
_OVERLAY_THICK_OUT = 3
_OVERLAY_THICK_IN = 1
_OVERLAY_BEAT_MARGIN = 24
_OVERLAY_BEAT_RADIUS = 10


class GLPresenter:
    """Presents RGBA CUDA frames in a window via CUDA/OpenGL interop."""

    def __init__(
        self,
        config: Config,
        tex_w: int,
        tex_h: int,
        shared_prompt: Optional[SharedPrompt] = None,
        audio_state: Optional[AudioState] = None,
        pipeline=None,
        orchestrator=None,
        base_prompt: Optional[SharedPrompt] = None,
        device: int = 0,
        ui_context: Optional["RuntimeContext"] = None,
    ) -> None:
        self.config = config
        self.tex_w = int(tex_w)
        self.tex_h = int(tex_h)
        self.shared_prompt = shared_prompt
        self.base_prompt = base_prompt
        self.audio_state = audio_state
        self.pipeline = pipeline
        self.orchestrator = orchestrator
        self._device = torch.device(f"cuda:{device}")
        self._frame_times: Deque[float] = deque(maxlen=60)
        self._last_beat_wall = 0.0
        self._overlay_gpu: Optional[torch.Tensor] = None  # reused upload buffer
        self.ui_controller: Optional["UIController"] = None

        self._lib = self._load_library()
        self._bind_signatures()

        # Ensure torch's primary CUDA context exists + is current on this thread
        # before the native side grabs it for interop.
        torch.cuda.init()
        torch.zeros(1, device=self._device)

        _set_process_app_id()

        if _ICON_PATH is not None:
            self._lib.glpresent_set_icon_path(str(_ICON_PATH))

        title = config.window_title
        ok = self._lib.glpresent_init(title, self.tex_w, self.tex_h, int(device))
        if not ok:
            raise RuntimeError(f"GL presenter init failed: {self._last_error()}")
        logger.info("GPU presenter ready: %dx%d texture (CUDA->GL interop).",
                    self.tex_w, self.tex_h)

        if ui_context is not None:
            from music_visualizer.ui.controller import UIController

            ui_context.presenter = self
            ui_context.pipeline = pipeline
            ui_context.base_prompt = base_prompt
            bridge = NativeUIBridge(self._lib)
            self.ui_controller = UIController(ui_context, bridge=bridge)
            if bridge._available:
                logger.info("Settings UI ready (F1 toggles panel).")
            else:
                logger.warning(
                    "Settings UI not available in DLL; use OpenCV path or rebuild the bridge."
                )

    # -- library loading -----------------------------------------------------

    def _load_library(self) -> ctypes.CDLL:
        from music_visualizer.display.native.build_glpresent import ensure_dll

        dll_path = ensure_dll(log=logger)
        return ctypes.CDLL(str(dll_path))

    def _bind_signatures(self) -> None:
        lib = self._lib
        lib.glpresent_set_icon_path.argtypes = [ctypes.c_wchar_p]
        lib.glpresent_set_icon_path.restype = None
        lib.glpresent_init.argtypes = [
            ctypes.c_wchar_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.glpresent_init.restype = ctypes.c_int
        lib.glpresent_present.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
        lib.glpresent_present.restype = ctypes.c_int
        lib.glpresent_should_close.argtypes = []
        lib.glpresent_should_close.restype = ctypes.c_int
        lib.glpresent_take_overlay_toggle.argtypes = []
        lib.glpresent_take_overlay_toggle.restype = ctypes.c_int
        lib.glpresent_client_size.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        lib.glpresent_client_size.restype = ctypes.c_int
        lib.glpresent_shutdown.argtypes = []
        lib.glpresent_shutdown.restype = None
        lib.glpresent_last_error.argtypes = []
        lib.glpresent_last_error.restype = ctypes.c_char_p
        if hasattr(lib, "glpresent_set_fullscreen"):
            lib.glpresent_set_fullscreen.argtypes = [ctypes.c_int]
            lib.glpresent_set_fullscreen.restype = None

    def set_fullscreen(self, enabled: bool) -> None:
        if hasattr(self._lib, "glpresent_set_fullscreen"):
            self._lib.glpresent_set_fullscreen(1 if enabled else 0)

    def _last_error(self) -> str:
        msg = self._lib.glpresent_last_error()
        return msg.decode("utf-8", "replace") if msg else "(no error message)"

    # -- presentation --------------------------------------------------------

    def should_close(self) -> bool:
        return bool(self._lib.glpresent_should_close())

    def _poll_overlay_toggle(self) -> None:
        if self._lib.glpresent_take_overlay_toggle():
            self.config.show_overlay = not self.config.show_overlay
            logger.info(
                "Debug overlay %s.", "on" if self.config.show_overlay else "off"
            )

    def _client_size(self) -> tuple[int, int]:
        cw, ch = ctypes.c_int(0), ctypes.c_int(0)
        self._lib.glpresent_client_size(ctypes.byref(cw), ctypes.byref(ch))
        return cw.value, ch.value

    def _measured_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def present(self, rgba_gpu: torch.Tensor, gen_fps: float = 0.0) -> None:
        """Composite the overlay (if enabled) and present an RGBA GPU frame.

        ``rgba_gpu`` is a contiguous ``(H, W, 4)`` ``uint8`` CUDA tensor. It may
        be mutated in place by the overlay compositor.
        """
        self._poll_overlay_toggle()
        self._frame_times.append(time.time())

        if self.config.show_overlay:
            self._composite_overlay(rgba_gpu, gen_fps)

        h, w = rgba_gpu.shape[0], rgba_gpu.shape[1]
        # The native blit is a synchronous device->array copy, so make sure torch
        # has finished writing the frame (and overlay) before handing the pointer.
        torch.cuda.synchronize(self._device)
        ok = self._lib.glpresent_present(
            ctypes.c_uint64(rgba_gpu.data_ptr()), int(w), int(h)
        )
        if not ok:
            raise RuntimeError(f"GL present failed: {self._last_error()}")

        if self.ui_controller is not None:
            self.ui_controller.pump()

    def attach_ui_context(self, ui_context: "RuntimeContext") -> None:
        """Late-bind the settings UI (e.g. after audio stack starts)."""
        if self.ui_controller is not None:
            return
        from music_visualizer.ui.controller import UIController

        ui_context.presenter = self
        ui_context.pipeline = self.pipeline
        ui_context.base_prompt = self.base_prompt
        bridge = NativeUIBridge(self._lib)
        self.ui_controller = UIController(ui_context, bridge=bridge)

    def refresh_ui(self) -> None:
        if self.ui_controller is not None:
            self.ui_controller.push_all_values()
            self.ui_controller.clear_restart_pending()

    # -- overlay (rendered on CPU, blended on GPU) ---------------------------

    def _overlay_ui_scale(self, frame_w: int, frame_h: int) -> float:
        """How much to enlarge overlay metrics when the window letterboxes the frame."""
        cw, ch = self._client_size()
        if cw > 0 and ch > 0:
            return visible_image_scale(frame_w, frame_h, cw, ch)
        return 1.0

    def _composite_overlay(self, frame: torch.Tensor, gen_fps: float) -> None:
        h, w = frame.shape[0], frame.shape[1]
        ui = self._overlay_ui_scale(w, h)

        lines, features = collect_overlay_lines(
            self.config, self.shared_prompt, self.audio_state, self.pipeline,
            self.orchestrator, gen_fps, self._measured_fps(),
            base_prompt=self.base_prompt,
        )

        font = _OVERLAY_FONT * ui
        y_step = max(12, int(round(_OVERLAY_LINE * ui)))
        y0 = max(14, int(round(_OVERLAY_Y0 * ui)))
        mx = max(6, int(round(_OVERLAY_MARGIN * ui)))
        thick_out = max(1, int(round(_OVERLAY_THICK_OUT * ui)))
        thick_in = max(1, int(round(_OVERLAY_THICK_IN * ui)))

        # Only build a strip tall enough for the content (+ the beat dot region),
        # so the per-frame CPU draw + upload stay small.
        beat_m = max(12, int(round(_OVERLAY_BEAT_MARGIN * ui)))
        beat_r = max(5, int(round(_OVERLAY_BEAT_RADIUS * ui)))
        content_h = y0 + y_step * len(lines)
        strip_h = min(h, max(content_h, beat_m + beat_r) + y_step)
        if strip_h <= 0:
            return

        # RGBA strip, transparent where nothing is drawn. Channel order matches
        # the frame (R,G,B,A); black/white/green are channel-symmetric or correct.
        strip = np.zeros((strip_h, w, 4), dtype=np.uint8)
        y = y0
        for line in lines:
            if y - y_step > strip_h:
                break
            cv2.putText(strip, line, (mx, y), cv2.FONT_HERSHEY_SIMPLEX, font,
                        (0, 0, 0, 255), thick_out, cv2.LINE_AA)
            cv2.putText(strip, line, (mx, y), cv2.FONT_HERSHEY_SIMPLEX, font,
                        (255, 255, 255, 255), thick_in, cv2.LINE_AA)
            y += y_step

        if features is not None:
            if features.is_beat:
                self._last_beat_wall = time.time()
            lit = (time.time() - self._last_beat_wall) < 0.12
            color = (0, 255, 0, 255) if lit else (60, 60, 60, 255)
            cv2.circle(strip, (w - beat_m, beat_m), beat_r, color, -1, cv2.LINE_AA)

        # Upload the strip and alpha-blend it over the top of the frame on the GPU.
        if (self._overlay_gpu is None
                or self._overlay_gpu.shape[0] < strip_h):
            self._overlay_gpu = torch.empty(
                (strip_h, w, 4), dtype=torch.uint8, device=self._device
            )
        buf = self._overlay_gpu[:strip_h]
        buf.copy_(torch.from_numpy(strip), non_blocking=True)

        a = buf[:, :, 3:4].float() / 255.0
        region = frame[:strip_h, :, :3].float()
        blended = buf[:, :, :3].float() * a + region * (1.0 - a)
        frame[:strip_h, :, :3] = blended.clamp_(0, 255).to(torch.uint8)

    def close(self) -> None:
        lib = getattr(self, "_lib", None)
        if lib is not None:
            try:
                lib.glpresent_shutdown()
            except Exception:
                pass
            self._lib = None

    def __del__(self):
        self.close()
