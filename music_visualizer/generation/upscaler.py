"""Realtime RTX VSR upscaler.

Wraps the NVIDIA RTX Video SDK's Video Super Resolution (VSR) feature through a
small native bridge DLL (``native/rtxvsr_bridge.dll``) and exposes a single
``upscale_to_bgr`` call for the generation loop.

Data stays on the GPU until the very end: the StreamDiffusion output (an RGB
``[0,1]`` CUDA tensor) is packed into the SDK's 8-bit BGRA layout on-device, VSR
runs in-place on CUDA device pointers (shared primary context, so torch's
pointers are valid as-is), and only the final upscaled frame is copied to the CPU
as a BGR ``uint8`` array for OpenCV display.

The bridge DLL is built on first use (see :mod:`.native.build_rtxvsr`) so there's
no separate compile step to remember.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from music_visualizer.utils.logging import get_logger

logger = get_logger(__name__)

_NATIVE_DIR = Path(__file__).resolve().parent / "native"
_DLL_PATH = _NATIVE_DIR / "rtxvsr_bridge.dll"

# NVSDK_NGX_VSR_QualityLevel: 0=bicubic, 1=low, 2=medium, 3=high, 4=ultra.
QUALITY_MIN = 0
QUALITY_MAX = 4


def _feature_dll_dir(sdk_root: Path) -> Path:
    """Directory holding ``nvngx_vsr.dll`` (NGX loads it from the init path)."""
    rel = sdk_root / "bin" / "Windows" / "x64" / "rel"
    dev = sdk_root / "bin" / "Windows" / "x64" / "dev"
    if (rel / "nvngx_vsr.dll").is_file():
        return rel
    if (dev / "nvngx_vsr.dll").is_file():
        return dev
    raise RuntimeError(
        f"nvngx_vsr.dll not found under {sdk_root / 'bin' / 'Windows' / 'x64'}. "
        "Is NV_RTX_VIDEO_SDK pointing at a complete SDK install?"
    )


class RTXUpscaler:
    """RTX VSR upscaler operating on CUDA tensors.

    Parameters
    ----------
    input_width, input_height:
        Native generation resolution (the VSR input size).
    output_width, output_height:
        Target upscaled resolution.
    quality:
        VSR quality level 0..4 (4 = best/slowest).
    sdk_root:
        RTX Video SDK root. Defaults to ``$NV_RTX_VIDEO_SDK``.
    device:
        CUDA device index (must match where StreamDiffusion runs).
    """

    def __init__(
        self,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
        quality: int = 4,
        sdk_root: str | None = None,
        device: int = 0,
        display_max_size: int | None = None,
    ) -> None:
        self.in_w = int(input_width)
        self.in_h = int(input_height)
        self.out_w = int(output_width)
        self.out_h = int(output_height)
        self.quality = max(QUALITY_MIN, min(QUALITY_MAX, int(quality)))
        self.device = device
        self._torch_device = torch.device(f"cuda:{device}")

        # Optional on-GPU downscale of the VSR result before the CPU transfer.
        # VSR still runs at full out_w x out_h; we only shrink the copied-out
        # frame so the PCIe transfer + OpenCV display work on a small buffer.
        self.disp_w = self.out_w
        self.disp_h = self.out_h
        self._downscale = False
        if display_max_size and display_max_size > 0:
            longest = max(self.out_w, self.out_h)
            if longest > display_max_size:
                s = display_max_size / longest
                self.disp_w = max(1, round(self.out_w * s))
                self.disp_h = max(1, round(self.out_h * s))
                self._downscale = True

        root = sdk_root or os.environ.get("NV_RTX_VIDEO_SDK")
        if not root:
            raise RuntimeError(
                "NV_RTX_VIDEO_SDK is not set and no sdk_root was given; cannot "
                "locate the RTX Video SDK for upscaling."
            )
        self.sdk_root = Path(root)
        self._dll_dir = _feature_dll_dir(self.sdk_root)

        self._lib = self._load_library()
        self._bind_signatures()

        # Reusable output buffer (H, W, 4) BGRA uint8 on GPU.
        self._out_gpu = torch.empty(
            (self.out_h, self.out_w, 4), dtype=torch.uint8, device=self._torch_device
        )

        self._init_native()
        if self._downscale:
            logger.info(
                "RTX VSR ready: %dx%d -> %dx%d, quality=%d (display downscaled to %dx%d)",
                self.in_w, self.in_h, self.out_w, self.out_h, self.quality,
                self.disp_w, self.disp_h,
            )
        else:
            logger.info(
                "RTX VSR ready: %dx%d -> %dx%d, quality=%d",
                self.in_w, self.in_h, self.out_w, self.out_h, self.quality,
            )

    # -- library loading -----------------------------------------------------

    def _load_library(self) -> ctypes.CDLL:
        if not _DLL_PATH.is_file():
            logger.info("rtxvsr_bridge.dll not found; building it now...")
            from music_visualizer.generation.native.build_rtxvsr import build

            build()
        # Let the loader (and NGX) find nvngx_vsr.dll + CUDA runtime deps.
        os.add_dll_directory(str(self._dll_dir))
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path and (Path(cuda_path) / "bin").is_dir():
            os.add_dll_directory(str(Path(cuda_path) / "bin"))
        return ctypes.CDLL(str(_DLL_PATH))

    def _bind_signatures(self) -> None:
        lib = self._lib
        lib.rtxvsr_init.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
        lib.rtxvsr_init.restype = ctypes.c_int
        lib.rtxvsr_upscale.argtypes = [
            ctypes.c_uint64, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint64, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint,
        ]
        lib.rtxvsr_upscale.restype = ctypes.c_int
        lib.rtxvsr_shutdown.argtypes = []
        lib.rtxvsr_shutdown.restype = None
        lib.rtxvsr_last_error.argtypes = []
        lib.rtxvsr_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        msg = self._lib.rtxvsr_last_error()
        return msg.decode("utf-8", "replace") if msg else "(no error message)"

    def _init_native(self) -> None:
        # Touch CUDA so the primary context exists before NGX retains it.
        torch.cuda.init()
        torch.zeros(1, device=self._torch_device)
        ok = self._lib.rtxvsr_init(str(self._dll_dir), self.device)
        if not ok:
            raise RuntimeError(f"RTX VSR init failed: {self._last_error()}")

    # -- inference -----------------------------------------------------------

    @staticmethod
    def _to_chw01(img: torch.Tensor) -> torch.Tensor:
        """Normalize an RGB image tensor to (3, H, W) float in [0, 1]."""
        t = img
        if t.ndim == 4:
            t = t[0]
        if t.ndim != 3:
            raise ValueError(f"expected 3D/4D image tensor, got shape {tuple(img.shape)}")
        # Accept CHW or HWC.
        if t.shape[0] in (3, 4) and t.shape[-1] not in (3, 4):
            chw = t
        elif t.shape[-1] in (3, 4):
            chw = t.permute(2, 0, 1)
        else:
            chw = t  # assume CHW
        return chw[:3].float().clamp(0, 1)

    def _pack_bgra(self, img01: torch.Tensor) -> torch.Tensor:
        """RGB [0,1] -> contiguous (H, W, 4) uint8 in B,G,R,A byte order."""
        chw = self._to_chw01(img01).to(self._torch_device, non_blocking=True)
        r, g, b = chw[0], chw[1], chw[2]
        a = torch.ones_like(r)
        bgra = torch.stack((b, g, r, a), dim=-1)  # (H, W, 4), [0,1]
        return (bgra * 255.0 + 0.5).clamp_(0, 255).to(torch.uint8).contiguous()

    def _run_vsr(self, img01: torch.Tensor) -> torch.Tensor:
        """Run VSR on an RGB [0,1] tensor; return the BGRA ``_out_gpu`` buffer."""
        in_gpu = self._pack_bgra(img01)
        h, w = in_gpu.shape[0], in_gpu.shape[1]

        # The native side issues synchronous device<->array copies, so make sure
        # torch has finished writing the input before we hand over the pointer.
        torch.cuda.synchronize(self._torch_device)

        ok = self._lib.rtxvsr_upscale(
            ctypes.c_uint64(in_gpu.data_ptr()), w, h,
            ctypes.c_uint64(self._out_gpu.data_ptr()), self.out_w, self.out_h,
            self.quality,
        )
        if not ok:
            raise RuntimeError(f"RTX VSR upscale failed: {self._last_error()}")
        return self._out_gpu

    def upscale_to_bgr(self, img01: torch.Tensor):
        """Upscale an RGB [0,1] CUDA tensor and return a BGR uint8 numpy frame.

        Returns a ``(disp_h, disp_w, 3)`` ``uint8`` array ready for ``cv2.imshow``
        (``disp_*`` equals the VSR output size unless ``display_max_size`` shrank
        it for the display path).
        """
        bgra = self._run_vsr(img01)
        # Output is BGRA; drop alpha -> BGR for OpenCV.
        bgr = bgra[:, :, :3]
        if self._downscale:
            bgr = self._downscale_chlast(bgr)
        # Only the (possibly much smaller) display frame crosses the PCIe bus.
        return bgr.cpu().numpy()

    def upscale_to_rgba_gpu(self, img01: torch.Tensor) -> torch.Tensor:
        """Upscale an RGB [0,1] CUDA tensor and return an RGBA uint8 GPU tensor.

        The result stays on the GPU as a contiguous ``(disp_h, disp_w, 4)``
        ``uint8`` tensor (R,G,B,A byte order, alpha forced opaque) -- ready to be
        blitted into an OpenGL texture via CUDA interop with no device->host copy.
        """
        bgra = self._run_vsr(img01)
        # BGRA -> RGBA (advanced indexing yields a fresh contiguous tensor).
        rgba = bgra[:, :, [2, 1, 0, 3]]
        if self._downscale:
            rgba = self._downscale_chlast(rgba)
        rgba = rgba.contiguous()
        rgba[:, :, 3] = 255
        return rgba

    def _downscale_chlast(self, img: torch.Tensor) -> torch.Tensor:
        """GPU downscale a channels-last (H, W, C) uint8 frame to the display size.

        ``area`` interpolation is an averaging filter -- the right choice for
        shrinking (it anti-aliases, avoiding the shimmer bilinear gives on
        high-frequency upscaled detail).
        """
        chw = img.permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
        small = F.interpolate(
            chw, size=(self.disp_h, self.disp_w), mode="area"
        )
        return small.squeeze(0).permute(1, 2, 0).round().clamp_(0, 255).to(torch.uint8).contiguous()

    def set_quality(self, quality: int) -> None:
        self.quality = max(QUALITY_MIN, min(QUALITY_MAX, int(quality)))

    def close(self) -> None:
        if getattr(self, "_lib", None) is not None:
            try:
                self._lib.rtxvsr_shutdown()
            except Exception:
                pass
            self._lib = None

    def __del__(self):
        self.close()
