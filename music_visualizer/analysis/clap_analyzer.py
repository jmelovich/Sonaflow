"""LAION-CLAP audio->phrase retrieval.

Wraps a HuggingFace ``ClapModel`` (default ``laion/larger_clap_music``) to score
a short window of audio against a fixed vocabulary of visual/mood phrases. CLAP
embeds audio and text into a shared space; we precompute the vocabulary's text
embeddings once at startup, then each call embeds the audio window, takes cosine
similarity against every phrase, and returns the top-k as softmax probabilities.

CLAP expects mono 48 kHz audio, so windows captured at the device rate are
resampled on the fly (torchaudio if available, otherwise a linear fallback).
The model is small (~150-200M params) and runs in ~10-30 ms on a modern GPU,
cheap enough to call a few times per second alongside StreamDiffusion.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from music_visualizer.utils.logging import get_logger

logger = get_logger(__name__)

CLAP_SAMPLE_RATE = 48000


class CLAPAnalyzer:
    """Scores audio windows against a fixed phrase vocabulary."""

    def __init__(
        self,
        vocabulary: List[str],
        model_id: str = "laion/larger_clap_music",
        device: str = "cpu",
        softmax_temp: float = 0.02,
        calibrate: bool = True,
        calib_alpha: float = 0.0001,
        calib_warmup: int = 12,
        calib_spike_guard: float = 0.05,
    ):
        from transformers import AutoProcessor, ClapModel

        self.vocabulary = list(vocabulary)
        self.model_id = model_id
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.softmax_temp = float(softmax_temp)
        # Hubness calibration: CLAP has a strong per-phrase bias -- some text
        # embeddings sit near the centre of the audio-embedding space and score
        # highest for almost any audio, so the same few phrases always win. We
        # subtract each phrase's running-average similarity (an online estimate
        # of that universal prior) so ranking reflects what is *distinctively*
        # matching the current audio. Mean-centering keeps the cosine scale, so
        # softmax_temp behaves the same as before.
        self.calibrate = bool(calibrate)
        self.calib_alpha = float(calib_alpha)
        self.calib_warmup = int(calib_warmup)
        # Spike protection (in cosine-similarity units). A plain symmetric EMA
        # baseline also absorbs a phrase's *own* distinctive peaks, so a strong,
        # unique match that holds for a while slowly drags its baseline up and
        # erodes itself out of the top-k. We damp the upward baseline update in
        # proportion to how far a phrase spikes above its baseline; set to 0 to
        # restore the original symmetric behavior.
        self.calib_spike_guard = float(calib_spike_guard)
        self._sim_mean: Optional[torch.Tensor] = None
        self._calib_count = 0
        # fp16 on CUDA roughly halves CLAP's VRAM and speeds inference, leaving
        # more headroom for StreamDiffusion (less memory pressure = less drift
        # into the slow allocator path over a long session).
        self._use_fp16 = self.device == "cuda"
        # Dedicated CUDA stream for all CLAP work. StreamDiffusion runs on the
        # default stream; CLAP's per-cycle device->host sync (the .tolist() below)
        # would otherwise drain every queued generation kernel, producing a
        # discrete stutter once per CLAP interval. On a separate stream that sync
        # only waits for CLAP's own kernels, decoupling it from generation.
        self._cuda_stream = torch.cuda.Stream() if self.device == "cuda" else None
        # Cache one resampler per source rate: torchaudio.functional.resample
        # rebuilds a sinc kernel on every call, which is heavy CPU work that holds
        # the GIL and starves the realtime audio callback.
        self._resamplers: Dict[int, object] = {}

        logger.info("Loading CLAP: %s on %s (fp16=%s)", model_id, self.device, self._use_fp16)
        self._processor = AutoProcessor.from_pretrained(model_id)
        model = ClapModel.from_pretrained(model_id).to(self.device).eval()
        if self._use_fp16:
            model = model.half()
        self._model = model

        # Precompute normalized text embeddings for the whole vocabulary once.
        # Stored as fp32 so the final similarity matmul stays numerically stable.
        # Empty vocabulary => embed-only mode (audio-to-audio similarity, no retrieval).
        if self.vocabulary:
            self._text_embeds = self._encode_text(self.vocabulary)  # [V, D], unit-norm
            logger.info(
                "CLAP ready: %d vocab phrases, embed dim %d",
                len(self.vocabulary), self._text_embeds.shape[1],
            )
        else:
            self._text_embeds = None
            logger.info("CLAP embed-only mode ready.")

    def _resample(self, window: np.ndarray, src_sr: int) -> np.ndarray:
        """Resample mono float32 audio to ``CLAP_SAMPLE_RATE`` (cached kernel)."""
        if src_sr == CLAP_SAMPLE_RATE or window.size == 0:
            return window
        resampler = self._resamplers.get(src_sr)
        if resampler is None:
            try:
                import torchaudio.transforms as T

                resampler = T.Resample(src_sr, CLAP_SAMPLE_RATE)
            except Exception:
                resampler = False  # mark torchaudio unavailable; use linear path
            self._resamplers[src_sr] = resampler
        if resampler:
            t = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))
            return resampler(t).numpy()
        # Linear-interpolation fallback (lower quality but dependency-free).
        n_out = int(round(window.shape[0] * CLAP_SAMPLE_RATE / float(src_sr)))
        if n_out <= 1:
            return window.astype(np.float32)
        x_old = np.linspace(0.0, 1.0, num=window.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, window).astype(np.float32)

    @torch.inference_mode()
    def _encode_text(self, phrases: List[str]) -> torch.Tensor:
        inputs = self._processor(text=phrases, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self._model.get_text_features(**inputs)
        return torch.nn.functional.normalize(feats.float(), dim=-1)

    @torch.inference_mode()
    def embed_audio(self, window: np.ndarray, src_sr: int) -> Optional[torch.Tensor]:
        """Return a unit-norm audio embedding on CPU, or ``None`` if too short."""
        if window is None or window.shape[0] < src_sr // 4:  # < ~0.25s of audio
            return None
        audio = self._resample(np.ascontiguousarray(window, dtype=np.float32), src_sr)
        if self._cuda_stream is not None:
            with torch.cuda.stream(self._cuda_stream):
                return self._embed_audio_gpu(audio)
        return self._embed_audio_gpu(audio)

    def _embed_audio_gpu(self, audio: np.ndarray) -> torch.Tensor:
        inputs = self._processor(
            audios=audio, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
        )
        target_dtype = torch.float16 if self._use_fp16 else torch.float32
        inputs = {
            k: (v.to(self.device, target_dtype) if v.is_floating_point() else v.to(self.device))
            for k, v in inputs.items()
        }
        feats = self._model.get_audio_features(**inputs)
        return torch.nn.functional.normalize(feats.float(), dim=-1).squeeze(0).cpu()

    @torch.inference_mode()
    def score(self, window: np.ndarray, src_sr: int, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return ``[(phrase, prob), ...]`` for the top-k matches of ``window``.

        Probabilities are taken from a full-vocabulary softmax, then
        **renormalized over the top-k only** so the returned scores sum to 1.0.
        This avoids every phrase sitting near ``1/V`` when the vocab is large.

        ``window`` is mono float32 at ``src_sr``; returns an empty list if the
        window is too short to be meaningful.
        """
        if window is None or window.shape[0] < src_sr // 4:  # < ~0.25s of audio
            return []
        if self._text_embeds is None:
            return []
        audio = self._resample(np.ascontiguousarray(window, dtype=np.float32), src_sr)
        if self._cuda_stream is not None:
            with torch.cuda.stream(self._cuda_stream):
                return self._score_gpu(audio, top_k)
        return self._score_gpu(audio, top_k)

    def _score_gpu(self, audio: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        audio_embed = self._embed_audio_gpu(audio).unsqueeze(0)  # [1, D]

        sims = (audio_embed @ self._text_embeds.T).squeeze(0)  # [V]
        sims = self._apply_calibration(sims)
        probs = torch.softmax(sims / max(1e-6, self.softmax_temp), dim=-1)
        k = min(int(top_k), probs.shape[0])
        top = torch.topk(probs, k)
        idxs = top.indices.tolist()
        vals = top.values.tolist()
        # Full-vocab softmax spreads mass across ~50+ phrases, so even the best
        # match often lands near 1/V (~0.02). Renormalize within the returned
        # top-k so scores sum to 1.0 and reflect relative confidence among the
        # candidates we actually consider for injection.
        total = sum(vals)
        if total > 1e-9:
            vals = [v / total for v in vals]
        return [(self.vocabulary[i], float(v)) for i, v in zip(idxs, vals)]

    def _apply_calibration(self, sims: torch.Tensor) -> torch.Tensor:
        """Subtract each phrase's running baseline similarity (hubness removal).

        Maintains an EMA of per-phrase similarity as an online estimate of the
        universal prior, and returns the mean-centred scores so consistently
        high "hub" phrases stop dominating. During the first ``calib_warmup``
        cycles the baseline is still forming, so raw scores are returned.

        The baseline update is *spike-protected*: a phrase whose similarity sits
        far above its own baseline is spiking distinctively for the current
        audio -- the signature of a genuine, strong match rather than a hub
        (which is high for nearly all audio and so rarely spikes far above its
        own running average). Feeding such peaks into the baseline at the full
        EMA rate would slowly pull the baseline up toward the match, shrinking
        ``centered`` until even a sustained, well-fitting prompt erodes itself
        out of the top-k. So we damp the *upward* part of the update by a gate
        that shrinks as the spike grows, keeping the baseline tracking each
        phrase's background level. Downward and small moves keep the full rate,
        so true hubs still accrue a high baseline and stay suppressed.
        """
        if not self.calibrate:
            return sims
        if self._sim_mean is None:
            self._sim_mean = sims.clone()
            self._calib_count = 1
            return sims
        a = self.calib_alpha
        centered = sims - self._sim_mean
        guard = self.calib_spike_guard
        if guard > 0.0:
            rising = torch.clamp(centered, min=0.0)
            gate = guard / (guard + rising)  # 1.0 at baseline -> 0 for big spikes
            rate = torch.where(centered > 0, a * gate, torch.full_like(centered, a))
            self._sim_mean += rate * centered
        else:
            self._sim_mean += a * centered
        self._calib_count += 1
        if self._calib_count < self.calib_warmup:
            return sims
        return centered
