"""StreamDiffusion pipeline wrapper.

Wraps the StreamDiffusion repo's :class:`StreamDiffusionWrapper` and exposes a
small, stable surface for the visualizer:

* ``generate()`` -> one BGR ``uint8`` numpy frame ready for OpenCV display
* ``update_prompt()`` -> swap the active prompt without rebuilding anything

In ``img2img`` mode the pipeline runs a Deforum-style feedback loop: the previous
output frame is fed back as the next ``init_image`` so visuals continuously morph
rather than flicker between independent samples. ``txt2img`` mode samples a fresh
image each call.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from PIL import Image

from music_visualizer.config import Config
from music_visualizer.generation.content_centering import ContentCentering
from music_visualizer.generation.sd_wrapper import StreamDiffusionWrapper
from music_visualizer.generation.transforms import FeedbackTransform
from music_visualizer.utils.logging import get_logger

if TYPE_CHECKING:
    from music_visualizer.audio.features import AudioFeatures

logger = get_logger(__name__)

# TF32 accelerates matmul on Ampere+ (RTX 30/40/50) with negligible quality loss.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


_TRT_BUILD_DIMS = {"width": 512, "height": 512}
_TRT_COMPILE_PATCHED = False
_TRT_MAX_IMAGE_SIDE = 1024


def _trt_max_image_side(width: int, height: int) -> int:
    side = max(int(width), int(height), 256)
    return ((side + 7) // 8) * 8


def _expand_trt_model_limits(model_data, width: int, height: int) -> None:
    """Raise StreamDiffusion TRT profile caps to fit the requested resolution.

    ``BaseModel`` defaults to 1024px max; wider/taller builds assert in
    ``check_dims`` unless we widen ``max_image_shape`` / ``max_latent_shape``.
    """
    global _TRT_MAX_IMAGE_SIDE
    max_side = _trt_max_image_side(width, height)
    _TRT_MAX_IMAGE_SIDE = max(_TRT_MAX_IMAGE_SIDE, max_side)
    min_side = max(256, (min(int(width), int(height)) // 8) * 8)
    model_data.max_image_shape = max_side
    model_data.min_image_shape = min(model_data.min_image_shape, min_side)
    model_data.max_latent_shape = max_side // 8
    model_data.min_latent_shape = model_data.min_image_shape // 8


def _patch_tensorrt_compile_dims(width: int, height: int) -> None:
    """Inject generation resolution into StreamDiffusion's TensorRT compile calls.

    Upstream ``utils.wrapper`` calls ``compile_unet`` / ``compile_vae_*`` without
    ``opt_image_height`` / ``opt_image_width``, so engines default to 512x512 even
    when the pipeline runs at another size (static-shape mismatch at inference).
    """
    global _TRT_COMPILE_PATCHED
    _TRT_BUILD_DIMS["width"] = int(width)
    _TRT_BUILD_DIMS["height"] = int(height)
    if _TRT_COMPILE_PATCHED:
        return

    import streamdiffusion.acceleration.tensorrt as trt_accel

    def _inject(fn):
        def wrapped(*args, opt_batch_size=1, engine_build_options=None, **kwargs):
            opts = dict(engine_build_options or {})
            w = _TRT_BUILD_DIMS["width"]
            h = _TRT_BUILD_DIMS["height"]
            max_side = _trt_max_image_side(w, h)
            min_side = max(256, (min(w, h) // 8) * 8)
            opts.setdefault("opt_image_height", h)
            opts.setdefault("opt_image_width", w)
            opts.setdefault("max_image_resolution", max_side)
            opts.setdefault("min_image_resolution", min_side)
            if len(args) >= 2:
                _expand_trt_model_limits(args[1], w, h)
            return fn(
                *args,
                opt_batch_size=opt_batch_size,
                engine_build_options=opts,
                **kwargs,
            )

        return wrapped

    trt_accel.compile_unet = _inject(trt_accel.compile_unet)
    trt_accel.compile_vae_decoder = _inject(trt_accel.compile_vae_decoder)
    trt_accel.compile_vae_encoder = _inject(trt_accel.compile_vae_encoder)
    _TRT_COMPILE_PATCHED = True


def _resolution_stamp(config: Config) -> str:
    return f"{config.width}x{config.height}"


def _invalidate_stale_tensorrt_engines(config: Config) -> None:
    """Drop cached TRT engines when they were built for a different resolution."""
    if config.acceleration != "tensorrt":
        return
    root = Path(config.resolved_engine_dir())
    if not root.is_dir():
        return
    stamp = _resolution_stamp(config)
    engine_names = (
        "unet.engine",
        "vae_encoder.engine",
        "vae_decoder.engine",
        "unet.engine.onnx",
        "unet.engine.opt.onnx",
        "vae_encoder.engine.onnx",
        "vae_encoder.engine.opt.onnx",
        "vae_decoder.engine.onnx",
        "vae_decoder.engine.opt.onnx",
    )
    for model_dir in root.rglob("*"):
        if not model_dir.is_dir():
            continue
        has_engine = (model_dir / "unet.engine").is_file()
        has_partial = (model_dir / "unet.engine.onnx").is_file()
        if not has_engine and not has_partial:
            continue
        marker = model_dir / ".built_resolution"
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == stamp:
            continue
        if not marker.is_file() and stamp == "512x512" and has_engine:
            # Legacy engines from before resolution stamping; assume 512x512.
            marker.write_text(stamp, encoding="utf-8")
            continue
        logger.warning(
            "Removing stale TensorRT engines in %s (rebuilding for %s).",
            model_dir,
            stamp,
        )
        for name in engine_names:
            try:
                (model_dir / name).unlink()
            except OSError:
                pass
        try:
            marker.unlink()
        except OSError:
            pass


def _mark_tensorrt_engines_built(config: Config) -> None:
    if config.acceleration != "tensorrt":
        return
    root = Path(config.resolved_engine_dir())
    stamp = _resolution_stamp(config)
    if not root.is_dir():
        return
    for model_dir in root.rglob("*"):
        if model_dir.is_dir() and (model_dir / "unet.engine").is_file():
            (model_dir / ".built_resolution").write_text(stamp, encoding="utf-8")


class StreamPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.mode = config.mode
        self.width = config.width
        self.height = config.height

        cfg_type = "none" if self.mode == "txt2img" else config.cfg_type

        logger.info(
            "Loading StreamDiffusion: model=%s mode=%s %dx%d acceleration=%s",
            config.model_id,
            self.mode,
            self.width,
            self.height,
            config.acceleration,
        )

        if config.acceleration == "tensorrt":
            _patch_tensorrt_compile_dims(self.width, self.height)
            _invalidate_stale_tensorrt_engines(config)
            max_side = _trt_max_image_side(self.width, self.height)
            if max_side > 1024:
                logger.info(
                    "TensorRT engine build at %dx%d (max side %d px; StreamDiffusion "
                    "default cap is 1024, expanded for this resolution).",
                    self.width,
                    self.height,
                    max_side,
                )
            logger.info("TensorRT engine dir: %s", config.resolved_engine_dir())

        self.wrapper = StreamDiffusionWrapper(
            model_id_or_path=config.model_id,
            t_index_list=config.t_index_list,
            mode=self.mode,
            output_type="np",
            width=self.width,
            height=self.height,
            warmup=config.warmup,
            acceleration=config.acceleration,
            use_denoising_batch=True,
            cfg_type=cfg_type,
            frame_buffer_size=1,
            seed=config.seed,
            do_add_noise=config.do_add_noise,
            engine_dir=config.resolved_engine_dir(),
        )

        self.wrapper.prepare(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            num_inference_steps=50,
            guidance_scale=config.guidance_scale,
            delta=config.delta,
        )

        # Lazily-built compel processor for weighted prompts (e.g. "(neon)1.2").
        # StreamDiffusion's update_prompt uses the plain CLIP tokenizer, which
        # treats compel weight syntax as literal text, so we build embeddings
        # ourselves when a prompt carries weights.
        self._compel = None
        self._compel_unavailable = False

        # img2img keeps the feedback frame as a clamped [0,1] GPU tensor (NCHW).
        self._feedback: Optional[torch.Tensor] = None
        self._transform = FeedbackTransform.from_config(config)
        self._noise_drift = float(config.feedback_noise_drift or 0.0)
        self._centering = ContentCentering(config)
        self._centering_pan_x = 0.0
        self._centering_pan_y = 0.0
        self._audio_features: Optional["AudioFeatures"] = None
        self._last_gen_t = time.time()
        if self.mode == "img2img":
            self._bootstrap_img2img()
            if not self._transform.is_identity():
                logger.info(
                    "Feedback motion: zoom=%.3f pan=(%.3f,%.3f) rotate=%.2f deg/frame",
                    self._transform.zoom,
                    self._transform.pan_x,
                    self._transform.pan_y,
                    self._transform.rotate,
                )
            logger.info("Feedback noise drift: %.3f", self._noise_drift)
            if self._centering.enabled:
                logger.info("Content centering: on (flat-fill recovery)")

        self._upscaler = self._make_upscaler(config)

        if config.acceleration == "tensorrt":
            _mark_tensorrt_engines_built(config)

        logger.info("StreamDiffusion ready (batch_size=%d).", self.wrapper.batch_size)

    def _make_upscaler(self, config: Config):
        """Build the RTX VSR upscaler, or return None if disabled/unavailable.

        Failure here is non-fatal: the visualizer still runs at native resolution
        so a missing SDK or driver issue degrades gracefully instead of crashing.
        """
        if not config.upscale:
            return None
        out_w, out_h = config.resolved_output_size()
        if out_w <= self.width and out_h <= self.height:
            logger.info("Upscale target <= generation size; skipping RTX VSR.")
            return None
        try:
            from music_visualizer.generation.upscaler import RTXUpscaler

            upscaler = RTXUpscaler(
                input_width=self.width,
                input_height=self.height,
                output_width=out_w,
                output_height=out_h,
                quality=config.vsr_quality,
                sdk_root=config.resolved_rtx_sdk_path(),
                device=0,
                display_max_size=config.display_max_size,
            )
            return upscaler
        except Exception as exc:
            logger.warning(
                "RTX VSR upscaler unavailable (%s); running at %dx%d.",
                exc, self.width, self.height,
            )
            return None

    def _bootstrap_img2img(self) -> None:
        """Seed the feedback loop from a prompt-aligned frame, then warm the latent buffer."""
        stream = self.wrapper.stream
        decoded = stream.decode_image(
            stream.predict_x0_batch(stream.init_noise[0:1])
        ).detach()
        self._feedback = self._to_feedback01(decoded)

        passes = max(self.wrapper.batch_size, self.config.warmup)
        logger.info("img2img bootstrap: %d feedback passes", passes)
        for _ in range(passes):
            decoded = self.wrapper.stream(self._feedback).detach()
            self._feedback = self._to_feedback01(decoded)

    @staticmethod
    def _to_feedback01(decoded: torch.Tensor) -> torch.Tensor:
        """Denormalize the raw VAE output (~[-1,1], may overshoot) to a clamped [0,1].

        Clamping every cycle is what keeps the self-feedback loop stable: feeding
        the raw decoded tensor back lets values drift past [-1,1] and accumulate
        energy each frame until the image saturates to black. A clean [0,1] frame
        is also the range ``image_processor.preprocess`` expects (no double-norm).
        """
        return (decoded / 2 + 0.5).clamp(0, 1)

    def set_audio_features(self, features: Optional["AudioFeatures"]) -> None:
        """Latest audio snapshot for audio-scaled centering strength."""
        self._audio_features = features

    # Compel uses parenthesized weights like "(a phrase)1.2"; a bare prompt has
    # no such syntax and can take StreamDiffusion's faster default path.
    @staticmethod
    def _has_weight_syntax(prompt: str) -> bool:
        return ")" in prompt and any(c.isdigit() for c in prompt)

    def update_prompt(self, prompt: str) -> None:
        if self._has_weight_syntax(prompt) and self._update_prompt_weighted(prompt):
            return
        self.wrapper.stream.update_prompt(prompt)

    def update_negative_prompt(self, negative_prompt: str) -> None:
        """Swap the negative prompt and re-encode (live UI edits)."""
        self.config.negative_prompt = negative_prompt
        stream = self.wrapper.stream
        try:
            with torch.no_grad():
                encoder_output = stream.pipe.encode_prompt(
                    prompt=negative_prompt,
                    device=stream.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                # StreamDiffusion img2img with cfg_type none may not use negative;
                # store on config for next prepare/restart.
                if hasattr(stream, "negative_prompt_embeds"):
                    stream.negative_prompt_embeds = encoder_output[0]
        except Exception:
            logger.exception("Negative prompt update failed.")

    def sync_centering(self, config: Config) -> None:
        """Push live centering UI edits into the feedback centering helper."""
        self._centering.sync_from_config(config)

    def compute_embeds(self, prompt: str):
        """Encode ``prompt`` into a ready ``prompt_embeds`` tensor *without*
        assigning it. Safe to call off the generation thread so the expensive
        compel parse + CLIP text-encode never blocks the render loop.

        Returns the batched embedding tensor, or ``None`` on failure (caller
        should leave the current embeds in place).
        """
        stream = self.wrapper.stream
        try:
            with torch.no_grad():
                if self._has_weight_syntax(prompt):
                    compel = self._get_compel()
                    if compel is not None:
                        embeds = compel.build_conditioning_tensor(prompt)  # [1,77,D]
                        embeds = embeds.to(device=stream.device, dtype=stream.dtype)
                        return embeds.repeat(stream.batch_size, 1, 1)
                # Plain path (no weights, or compel unavailable).
                encoder_output = stream.pipe.encode_prompt(
                    prompt=prompt,
                    device=stream.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                return encoder_output[0].repeat(stream.batch_size, 1, 1)
        except Exception:
            logger.exception("Off-thread prompt encoding failed; keeping current.")
            return None

    def apply_embeds(self, embeds) -> None:
        """Swap in a precomputed embedding tensor (cheap; render-thread safe)."""
        self.wrapper.stream.prompt_embeds = embeds

    def _get_compel(self):
        """Build (once) a compel processor bound to the pipeline's CLIP encoder."""
        if self._compel is not None or self._compel_unavailable:
            return self._compel
        try:
            from compel import Compel

            stream = self.wrapper.stream
            self._compel = Compel(
                tokenizer=stream.pipe.tokenizer,
                text_encoder=stream.text_encoder,
                truncate_long_prompts=True,  # keep 77 tokens -> TensorRT-safe shape
            )
        except Exception as exc:
            self._compel_unavailable = True
            logger.warning(
                "compel unavailable (%s); weighted prompts fall back to plain "
                "encoding. Install with: pip install compel", exc,
            )
        return self._compel

    def _update_prompt_weighted(self, prompt: str) -> bool:
        """Encode a compel-weighted prompt and assign embeds. Returns success."""
        compel = self._get_compel()
        if compel is None:
            return False
        try:
            stream = self.wrapper.stream
            with torch.no_grad():
                embeds = compel.build_conditioning_tensor(prompt)  # [1, 77, D]
                embeds = embeds.to(device=stream.device, dtype=stream.dtype)
                stream.prompt_embeds = embeds.repeat(stream.batch_size, 1, 1)
            return True
        except Exception:
            logger.exception("Weighted prompt encoding failed; using plain path.")
            return False

    def warmup(self, n: int = 2) -> None:
        """Run a few passes so cold-start latency is excluded from FPS metrics."""
        if self.mode == "txt2img":
            for _ in range(max(0, n)):
                self.generate()

    def _img2img_step(self) -> torch.Tensor:
        """Advance the img2img feedback loop one step; return the new [0,1] frame.

        Drifts the noise, applies the (audio-driven) Deforum warp + centering,
        re-imagines the frame, and stores it as the next feedback tensor.
        """
        # Drift the (otherwise fixed) noise so the loop can't lock to a fixed point.
        self._drift_noise()
        # Corrective pan when zoom has drifted into large flat-color center regions.
        now = time.time()
        dt = now - self._last_gen_t
        self._last_gen_t = now
        cx, cy = self._centering.update(self._feedback, dt, self._audio_features)
        self._centering_pan_x, self._centering_pan_y = cx, cy
        saved_pan_x, saved_pan_y = self._transform.pan_x, self._transform.pan_y
        self._transform.pan_x += cx
        self._transform.pan_y += cy
        # img2img: warp the feedback frame (Deforum motion), then re-imagine it.
        try:
            warped = self._transform.apply(self._feedback)
        finally:
            self._transform.pan_x, self._transform.pan_y = saved_pan_x, saved_pan_y
        decoded = self.wrapper.stream(warped)
        self._feedback = self._to_feedback01(decoded.detach())
        return self._feedback

    def generate(self) -> np.ndarray:
        """Produce one frame and return it as an HxWx3 BGR uint8 array."""
        if self.mode == "txt2img":
            out = self.wrapper.txt2img()
            if self._upscaler is not None:
                return self._upscaler.upscale_to_bgr(self._np_rgb_to_cuda01(out))
            rgb = self._to_uint8_rgb(out)
            return rgb[:, :, ::-1].copy()

        # Upscale the [0,1] GPU feedback frame directly (no CPU roundtrip). The
        # low-res feedback is kept internally so the morph loop is unaffected.
        feedback = self._img2img_step()
        if self._upscaler is not None:
            return self._upscaler.upscale_to_bgr(feedback)
        rgb = self._feedback01_to_uint8_rgb(feedback)
        return rgb[:, :, ::-1].copy()

    def generate_rgba_gpu(self) -> torch.Tensor:
        """Produce one frame as a contiguous RGBA uint8 GPU tensor (H, W, 4).

        Used by the GPU presenter: the upscaled frame is handed straight to an
        OpenGL texture via CUDA interop, so it never leaves the GPU. Falls back to
        packing the native-resolution frame to RGBA when no upscaler is active.
        """
        if self.mode == "txt2img":
            img01 = self._np_rgb_to_cuda01(self.wrapper.txt2img())
        else:
            img01 = self._img2img_step()
        if self._upscaler is not None:
            return self._upscaler.upscale_to_rgba_gpu(img01)
        return self._rgb01_to_rgba_gpu(img01)

    @property
    def display_size(self) -> tuple[int, int]:
        """The (width, height) of frames returned by the GPU/CPU display paths."""
        if self._upscaler is not None:
            return self._upscaler.disp_w, self._upscaler.disp_h
        return self.width, self.height

    def _rgb01_to_rgba_gpu(self, img01: torch.Tensor) -> torch.Tensor:
        """RGB [0,1] tensor (CHW/HWC/NCHW) -> contiguous (H, W, 4) uint8 RGBA GPU."""
        t = img01
        if t.ndim == 4:
            t = t[0]
        if t.shape[0] in (3, 4) and t.shape[-1] not in (3, 4):
            t = t.permute(1, 2, 0)  # CHW -> HWC
        t = t[:, :, :3].float().clamp(0, 1)
        h, w = t.shape[0], t.shape[1]
        rgba = torch.empty((h, w, 4), dtype=torch.uint8, device=t.device)
        rgba[:, :, :3] = (t * 255.0 + 0.5).clamp_(0, 255).to(torch.uint8)
        rgba[:, :, 3] = 255
        return rgba

    def _drift_noise(self) -> None:
        """Variance-preserving random walk on the pipeline's init noise.

        ``init_noise`` is set once in ``prepare()`` and reused every frame, which
        makes img2img self-feedback deterministic (it converges to a fixed point).
        Each step we mix in fresh Gaussian noise while preserving unit variance:

            n <- sqrt(1 - a^2) * n + a * e,   e ~ N(0, 1)

        so the noise keeps moving (no fixed point) without changing magnitude.
        ``a = 0`` leaves the noise fixed; ``a = 1`` is fully fresh noise per frame.
        """
        a = self._noise_drift
        if a <= 0.0:
            return
        stream = self.wrapper.stream
        keep = math.sqrt(max(0.0, 1.0 - a * a))
        fresh = torch.randn_like(stream.init_noise)
        stream.init_noise = keep * stream.init_noise + a * fresh

    def set_transform(self, **kwargs) -> None:
        """Update feedback motion parameters at runtime (e.g. zoom/pan/rotate)."""
        for key, val in kwargs.items():
            if key == "noise_drift" and val is not None:
                self._noise_drift = max(0.0, min(1.0, float(val)))
            elif hasattr(self._transform, key) and val is not None:
                setattr(self._transform, key, val)

    def get_motion_params(self) -> dict:
        """Return the current feedback-motion parameters (for display/overlay)."""
        return {
            "zoom": self._transform.zoom,
            "pan_x": self._transform.pan_x + self._centering_pan_x,
            "pan_y": self._transform.pan_y + self._centering_pan_y,
            "rotate": self._transform.rotate,
            "noise": self._noise_drift,
            "center_pan_x": self._centering_pan_x,
            "center_pan_y": self._centering_pan_y,
            "centering": self._centering.active,
            "center_strength": self._centering.audio_strength,
        }

    def _np_rgb_to_cuda01(self, out) -> torch.Tensor:
        """Wrapper txt2img output (np/tensor/PIL) -> RGB [0,1] CUDA tensor (HWC)."""
        if isinstance(out, torch.Tensor):
            # StreamDiffusion tensors are already normalized to [0,1].
            return out.detach().to("cuda", non_blocking=True).float().clamp(0, 1)
        arr = np.asarray(self._to_uint8_rgb(out))  # HWC uint8 RGB [0,255]
        return torch.from_numpy(arr).to("cuda", non_blocking=True).float() / 255.0

    def close(self) -> None:
        upscaler = getattr(self, "_upscaler", None)
        if upscaler is not None:
            upscaler.close()
            self._upscaler = None

    @staticmethod
    def _feedback01_to_uint8_rgb(img01: torch.Tensor) -> np.ndarray:
        arr = img01.detach()
        if arr.ndim == 4:
            arr = arr[0]
        arr = arr.permute(1, 2, 0).float().cpu().numpy()  # CHW -> HWC
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_uint8_rgb(out) -> np.ndarray:
        """Normalize wrapper output (np float[0,1] / tensor / PIL) to uint8 RGB."""
        if isinstance(out, Image.Image):
            return np.asarray(out.convert("RGB"))
        if isinstance(out, torch.Tensor):
            arr = out.detach().float().cpu().numpy()
            if arr.ndim == 4:
                arr = arr[0]
            if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            out = arr
        arr = np.asarray(out)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr
