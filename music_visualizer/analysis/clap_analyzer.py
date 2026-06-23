"""LAION-CLAP audio->phrase retrieval.

Wraps a HuggingFace ``ClapModel`` (default ``laion/larger_clap_music``) to score
a short window of audio against one or more vocabulary lists. CLAP embeds audio
and text into a shared space; text embeddings are precomputed per list at
startup, then each cycle embeds the audio once and scores every enabled channel.

CLAP expects mono 48 kHz audio, so windows captured at the device rate are
resampled on the fly (torchaudio if available, otherwise a linear fallback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from music_visualizer.analysis.vocab_lists import VocabList, normalize_vocab_lists
from music_visualizer.utils.logging import get_logger

logger = get_logger(__name__)

CLAP_SAMPLE_RATE = 48000


@dataclass
class _ScoringChannel:
    """Runtime state for one vocabulary list."""

    vocab_id: str
    phrases: List[str]
    text_embeds: torch.Tensor
    enable: bool = True
    top_k: int = 5
    softmax_temp: float = 0.02
    calibrate: bool = True
    calib_alpha: float = 0.001
    calib_warmup: int = 8
    calib_spike_guard: float = 0.05
    sim_mean: Optional[torch.Tensor] = field(default=None, repr=False)
    calib_count: int = 0

    def sync_runtime(self, vl: VocabList) -> None:
        self.enable = vl.enable
        self.top_k = vl.top_k
        self.softmax_temp = vl.softmax_temp
        self.calibrate = vl.calibrate
        self.calib_alpha = vl.calib_alpha
        self.calib_warmup = vl.calib_warmup
        self.calib_spike_guard = vl.calib_spike_guard


class CLAPAnalyzer:
    """Scores audio windows against one or more phrase vocabularies."""

    def __init__(
        self,
        channels: Optional[List[dict]] = None,
        vocabulary: Optional[List[str]] = None,
        model_id: str = "laion/larger_clap_music",
        device: str = "cpu",
        softmax_temp: float = 0.02,
        calibrate: bool = True,
        calib_alpha: float = 0.0001,
        calib_warmup: int = 12,
        calib_spike_guard: float = 0.05,
    ):
        from transformers import AutoProcessor, ClapModel

        self.model_id = model_id
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self._default_softmax_temp = float(softmax_temp)
        self._default_calibrate = bool(calibrate)
        self._default_calib_alpha = float(calib_alpha)
        self._default_calib_warmup = int(calib_warmup)
        self._default_calib_spike_guard = float(calib_spike_guard)
        self._use_fp16 = self.device == "cuda"
        self._cuda_stream = torch.cuda.Stream() if self.device == "cuda" else None
        self._resamplers: Dict[int, object] = {}
        self._channels: Dict[str, _ScoringChannel] = {}

        logger.info("Loading CLAP: %s on %s (fp16=%s)", model_id, self.device, self._use_fp16)
        self._processor = AutoProcessor.from_pretrained(model_id)
        model = ClapModel.from_pretrained(model_id).to(self.device).eval()
        if self._use_fp16:
            model = model.half()
        self._model = model

        if channels is not None:
            self._load_channels(channels)
        elif vocabulary is not None:
            if vocabulary:
                self._load_channels([{
                    "id": "default",
                    "name": "Default",
                    "phrases": list(vocabulary),
                    "enable": True,
                    "top_k": 3,
                    "softmax_temp": self._default_softmax_temp,
                    "calibrate": self._default_calibrate,
                    "calib_alpha": self._default_calib_alpha,
                    "calib_warmup": self._default_calib_warmup,
                    "calib_spike_guard": self._default_calib_spike_guard,
                }])
            else:
                logger.info("CLAP embed-only mode ready.")
        else:
            logger.info("CLAP embed-only mode ready.")

        if self._channels:
            total = sum(len(ch.phrases) for ch in self._channels.values())
            logger.info(
                "CLAP ready: %d channel(s), %d total phrases",
                len(self._channels), total,
            )

    @property
    def vocabulary(self) -> List[str]:
        """Legacy: phrases from the first channel, or empty in embed-only mode."""
        if not self._channels:
            return []
        first = next(iter(self._channels.values()))
        return list(first.phrases)

    @property
    def softmax_temp(self) -> float:
        if self._channels:
            return next(iter(self._channels.values())).softmax_temp
        return self._default_softmax_temp

    @softmax_temp.setter
    def softmax_temp(self, value: float) -> None:
        for ch in self._channels.values():
            ch.softmax_temp = float(value)

    @property
    def calibrate(self) -> bool:
        if self._channels:
            return next(iter(self._channels.values())).calibrate
        return self._default_calibrate

    @calibrate.setter
    def calibrate(self, value: bool) -> None:
        for ch in self._channels.values():
            ch.calibrate = bool(value)

    @property
    def calib_alpha(self) -> float:
        if self._channels:
            return next(iter(self._channels.values())).calib_alpha
        return self._default_calib_alpha

    @calib_alpha.setter
    def calib_alpha(self, value: float) -> None:
        for ch in self._channels.values():
            ch.calib_alpha = float(value)

    @property
    def calib_warmup(self) -> int:
        if self._channels:
            return next(iter(self._channels.values())).calib_warmup
        return self._default_calib_warmup

    @calib_warmup.setter
    def calib_warmup(self, value: int) -> None:
        for ch in self._channels.values():
            ch.calib_warmup = int(value)

    @property
    def calib_spike_guard(self) -> float:
        if self._channels:
            return next(iter(self._channels.values())).calib_spike_guard
        return self._default_calib_spike_guard

    @calib_spike_guard.setter
    def calib_spike_guard(self, value: float) -> None:
        for ch in self._channels.values():
            ch.calib_spike_guard = float(value)

    def _load_channels(self, lists: List[dict]) -> None:
        self._channels.clear()
        for raw in normalize_vocab_lists(lists):
            vl = VocabList.from_dict(raw)
            if not vl.phrases:
                continue
            text_embeds = self._encode_text(vl.phrases)
            self._channels[vl.id] = _ScoringChannel(
                vocab_id=vl.id,
                phrases=list(vl.phrases),
                text_embeds=text_embeds,
                enable=vl.enable,
                top_k=vl.top_k,
                softmax_temp=vl.softmax_temp,
                calibrate=vl.calibrate,
                calib_alpha=vl.calib_alpha,
                calib_warmup=vl.calib_warmup,
                calib_spike_guard=vl.calib_spike_guard,
            )

    def sync_channels(self, lists: Optional[List[dict]]) -> None:
        """Apply live UI edits to per-channel runtime params."""
        if not lists:
            return
        for raw in normalize_vocab_lists(lists):
            vl = VocabList.from_dict(raw)
            ch = self._channels.get(vl.id)
            if ch is not None:
                ch.sync_runtime(vl)

    def _resample(self, window: np.ndarray, src_sr: int) -> np.ndarray:
        if src_sr == CLAP_SAMPLE_RATE or window.size == 0:
            return window
        resampler = self._resamplers.get(src_sr)
        if resampler is None:
            try:
                import torchaudio.transforms as T

                resampler = T.Resample(src_sr, CLAP_SAMPLE_RATE)
            except Exception:
                resampler = False
            self._resamplers[src_sr] = resampler
        if resampler:
            t = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))
            return resampler(t).numpy()
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
        if window is None or window.shape[0] < src_sr // 4:
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
        return torch.nn.functional.normalize(feats.float(), dim=-1).squeeze(0)

    @torch.inference_mode()
    def score(self, window: np.ndarray, src_sr: int, top_k: int = 3) -> List[Tuple[str, float]]:
        """Legacy single-channel score (first channel only)."""
        if not self._channels:
            return []
        first_id = next(iter(self._channels))
        all_results = self.score_all(window, src_sr, top_k_override={first_id: top_k})
        return all_results.get(first_id, [])

    @torch.inference_mode()
    def score_all(
        self,
        window: np.ndarray,
        src_sr: int,
        top_k_override: Optional[Dict[str, int]] = None,
        only_ids: Optional[set[str]] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Score ``window`` against selected enabled channels; one audio embed."""
        if window is None or window.shape[0] < src_sr // 4 or not self._channels:
            return {}
        audio = self._resample(np.ascontiguousarray(window, dtype=np.float32), src_sr)
        if self._cuda_stream is not None:
            with torch.cuda.stream(self._cuda_stream):
                return self._score_all_gpu(audio, top_k_override or {}, only_ids)
        return self._score_all_gpu(audio, top_k_override or {}, only_ids)

    def _score_all_gpu(
        self,
        audio: np.ndarray,
        top_k_override: Dict[str, int],
        only_ids: Optional[set[str]] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        audio_embed = self._embed_audio_gpu(audio).unsqueeze(0)
        out: Dict[str, List[Tuple[str, float]]] = {}
        for ch in self._channels.values():
            if not ch.enable:
                continue
            if only_ids is not None and ch.vocab_id not in only_ids:
                continue
            top_k = int(top_k_override.get(ch.vocab_id, ch.top_k))
            out[ch.vocab_id] = self._score_channel(audio_embed, ch, top_k)
        return out

    def _score_channel(
        self,
        audio_embed: torch.Tensor,
        ch: _ScoringChannel,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        sims = (audio_embed @ ch.text_embeds.T).squeeze(0)
        sims = self._apply_calibration(sims, ch)
        probs = torch.softmax(sims / max(1e-6, ch.softmax_temp), dim=-1)
        k = min(int(top_k), probs.shape[0])
        if k <= 0:
            return []
        top = torch.topk(probs, k)
        idxs = top.indices.tolist()
        vals = top.values.tolist()
        total = sum(vals)
        if total > 1e-9:
            vals = [v / total for v in vals]
        return [(ch.phrases[i], float(v)) for i, v in zip(idxs, vals)]

    def _apply_calibration(self, sims: torch.Tensor, ch: _ScoringChannel) -> torch.Tensor:
        if not ch.calibrate:
            return sims
        if ch.sim_mean is None:
            ch.sim_mean = sims.clone()
            ch.calib_count = 1
            return sims
        a = ch.calib_alpha
        centered = sims - ch.sim_mean
        guard = ch.calib_spike_guard
        if guard > 0.0:
            rising = torch.clamp(centered, min=0.0)
            gate = guard / (guard + rising)
            rate = torch.where(centered > 0, a * gate, torch.full_like(centered, a))
            ch.sim_mean += rate * centered
        else:
            ch.sim_mean += a * centered
        ch.calib_count += 1
        if ch.calib_count < ch.calib_warmup:
            return sims
        return centered
