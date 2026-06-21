"""Qwen2-Audio sidecar client + free-running orchestrator thread.

Spawns the sidecar subprocess (same .venv as the main app), sends rolling audio
windows over TCP, and publishes ``"<anchor>, <caption>"`` into the shared base prompt.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from music_visualizer.analysis.qwen_protocol import recv_response, send_request
from music_visualizer.config import Config
from music_visualizer.utils.logging import get_logger
from music_visualizer.utils.shared_state import AudioRing, PromptBlendState, SharedPrompt
from music_visualizer.utils.text import (
    english_only_text,
    filter_composed_addition,
    filter_prompt_addition,
)

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class QwenSidecarClient:
    """Manages the sidecar subprocess and TCP connection."""

    def __init__(self, config: Config):
        self.config = config
        self._proc: Optional[subprocess.Popen] = None
        self._sock: Optional[socket.socket] = None
        self._port: Optional[int] = None

    def start(self) -> None:
        python = self.config.resolved_qwen_python()
        if not Path(python).is_file():
            raise FileNotFoundError(
                f"Qwen sidecar Python not found: {python}\n"
                "Run setup.ps1 -Profile Full, or run with --no-qwen."
            )

        cmd = [
            python,
            "-m",
            "music_visualizer.qwen_sidecar.server",
            "--model",
            self.config.qwen_model_id,
        ]
        if self.config.qwen_4bit:
            cmd.append("--4bit")
        else:
            cmd.append("--no-4bit")
        if self.config.qwen_port:
            cmd.extend(["--port", str(self.config.qwen_port)])

        logger.info("Starting Qwen sidecar: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        deadline = time.time() + 600.0  # model load can take several minutes
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                rc = self._proc.poll()
                err = ""
                if self._proc.stderr is not None:
                    err = self._proc.stderr.read()
                raise RuntimeError(
                    f"Qwen sidecar exited before handshake (rc={rc}): {err}"
                )
            line = line.strip()
            if line.startswith("QWEN_PORT="):
                self._port = int(line.split("=", 1)[1])
                break
            logger.debug("sidecar: %s", line)
        else:
            raise TimeoutError("Qwen sidecar did not report QWEN_PORT within 600s")

        self._sock = socket.create_connection(("127.0.0.1", self._port), timeout=30.0)
        logger.info("Qwen sidecar connected on port %d.", self._port)

        # Drain stderr in the background so a full pipe doesn't stall the sidecar.
        if self._proc.stderr is not None:
            threading.Thread(
                target=self._drain_stderr,
                args=(self._proc.stderr,),
                name="qwen-sidecar-stderr",
                daemon=True,
            ).start()

    @staticmethod
    def _drain_stderr(pipe) -> None:
        for line in pipe:
            line = line.rstrip()
            if line:
                logger.debug("sidecar stderr: %s", line)

    def _ensure_connected(self) -> None:
        if self._sock is not None:
            return
        if self._port is None:
            raise RuntimeError("Qwen sidecar not started")
        self._sock = socket.create_connection(("127.0.0.1", self._port), timeout=30.0)

    def caption(
        self,
        window: np.ndarray,
        sample_rate: int,
        instruction: str,
        max_new_tokens: int,
    ) -> str:
        for attempt in range(2):
            self._ensure_connected()
            assert self._sock is not None
            try:
                send_request(
                    self._sock,
                    window,
                    sample_rate,
                    instruction,
                    max_new_tokens,
                    do_sample=self.config.qwen_do_sample,
                    temperature=self.config.qwen_temperature,
                    top_p=self.config.qwen_top_p,
                    system_prompt=self.config.qwen_system_prompt,
                )
                resp = recv_response(self._sock, timeout=self.config.qwen_timeout_sec)
                if "error" in resp:
                    raise RuntimeError(resp["error"])
                caption = english_only_text(str(resp.get("caption", "")).strip())
                if not caption:
                    raise RuntimeError("empty caption from sidecar")
                return caption
            except TimeoutError:
                # The sidecar is alive but slow (GPU contention). Reconnecting
                # would queue a *second* request behind the one still running,
                # so just drop this connection and surface the error -- the
                # orchestrator backs off and the sidecar finishes on its own.
                logger.warning(
                    "Qwen inference exceeded %.0fs timeout; skipping this cycle.",
                    self.config.qwen_timeout_sec,
                )
                self._reset_socket()
                raise
            except (ConnectionError, BrokenPipeError, OSError):
                logger.warning("Qwen sidecar connection lost; reconnecting.")
                self._reset_socket()
                if attempt == 0:
                    continue
                raise
        raise RuntimeError("unreachable")

    def _reset_socket(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None

    def is_alive(self) -> bool:
        """True when the sidecar subprocess is still running."""
        return self._proc is not None and self._proc.poll() is None

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


class QwenOrchestrator:
    """Free-running Qwen caption loop on a background thread."""

    def __init__(
        self,
        config: Config,
        client: QwenSidecarClient,
        audio_ring: AudioRing,
        base_prompt: SharedPrompt,
        active_prompt: SharedPrompt,
        shutdown: threading.Event,
        blend_state: Optional[PromptBlendState] = None,
        clap_composes: bool = False,
        stop_event: Optional[threading.Event] = None,
        similarity_embedder=None,
    ):
        self.config = config
        self._client = client
        self._ring = audio_ring
        self._base = base_prompt
        self._active = active_prompt
        self._shutdown = shutdown
        self._stop = stop_event or threading.Event()
        self._blend = blend_state
        self._clap_composes = clap_composes
        self._embedder = similarity_embedder
        self._last_composed = self._base.current().strip()
        self._pipeline = None
        self._pipeline_lock = threading.Lock()
        self._lock = threading.Lock()
        self._last_caption: str = ""
        self._status: str = "starting"
        self._infer_ms: float = 0.0
        self._infer_ms_ema: float = 0.0
        self._audio_sec: float = 0.0
        self._anchor_audio: Optional[np.ndarray] = None
        self._anchor_embed = None
        self._last_infer_time: float = 0.0
        self._last_prefetch_time: float = 0.0
        self._last_cosine: float = 1.0
        self._last_sim: float = 1.0
        self._held: bool = False
        self._pending_caption: str = ""
        self._pending_audio: Optional[np.ndarray] = None
        self._publish_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="qwen-orchestrator", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def set_pipeline(self, pipeline) -> None:
        """Attach the StreamPipeline for off-thread compel encoding."""
        with self._pipeline_lock:
            self._pipeline = pipeline

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)

    def request_stop(self) -> None:
        self._stop.set()

    def last_caption(self) -> str:
        """Latest Qwen caption (without the manual base prompt)."""
        with self._lock:
            return self._last_caption

    def close(self) -> None:
        self._client.close()

    def overlay_lines(self) -> List[str]:
        with self._lock:
            caption = self._last_caption
            status = self._status
            infer_ms = self._infer_ms_ema
            audio_sec = self._audio_sec
            held = self._held
            last_cosine = self._last_cosine
            last_sim = self._last_sim
            last_infer_time = self._last_infer_time
            pending = self._pending_caption
        lines: List[str] = []
        if audio_sec < 1.0:
            lines.append(f"Qwen: waiting for audio ({audio_sec:.1f}s / {self.config.qwen_window_sec:.0f}s)")
            return lines
        hold_suffix = ""
        if held and last_infer_time > 0.0:
            held_sec = time.time() - last_infer_time
            hold_suffix = f" hold cos={last_cosine:.2f} score={last_sim:.2f} ({held_sec:.0f}s)"
            if pending:
                hold_suffix += f" next: {pending[:60]}"
        if caption:
            blend_pct = ""
            if self._blend is not None and self._blend.is_blending():
                blend_pct = f" blend {self._blend.progress() * 100:.0f}%"
            lines.append(f"Qwen ({infer_ms:.0f}ms{blend_pct}{hold_suffix}): {caption[:120]}")
        elif held:
            held_sec = time.time() - last_infer_time if last_infer_time > 0.0 else 0.0
            lines.append(f"Qwen hold cos={last_cosine:.2f} score={last_sim:.2f} (held {held_sec:.0f}s)")
        else:
            lines.append(f"Qwen: no caption ({status})")
        return lines

    def republish_from_base(self) -> None:
        """Re-compose active prompt from the current base + last caption."""
        self._try_publish_caption(self.last_caption(), None, hold_active=False)

    def sync_blacklist_from_config(self, cfg: Config) -> None:
        """Re-filter cached captions after live blacklist edits."""
        self.config = cfg
        with self._lock:
            pending = self._pending_caption
            last = self._last_caption
        if pending:
            filtered = self._filter_caption(pending)
            with self._lock:
                if filtered is None:
                    self._pending_caption = ""
                    self._pending_audio = None
                else:
                    self._pending_caption = filtered
        if last:
            filtered = self._filter_caption(last)
            with self._lock:
                self._last_caption = filtered or ""

    def _compose(self, caption: str) -> str:
        caption = caption.strip()
        anchor = self._base.current().strip()
        if not caption:
            return anchor
        if not anchor:
            return caption
        return f"{anchor}, {caption}"

    @staticmethod
    def _hold_score(cosine: float, power: float) -> float:
        """Map raw cosine similarity to the hold decision score."""
        base = max(0.0, float(cosine))
        power = max(0.0, float(power))
        if power == 1.0:
            return base
        return base ** power

    @staticmethod
    def _freeze_window(window: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(window, dtype=np.float32).copy()

    def _embed_window(self, window: np.ndarray):
        if window is None or window.size == 0 or self._embedder is None:
            return None
        return self._embedder.embed_audio(window, self._ring.sample_rate)

    def _set_anchor_window(self, window: np.ndarray) -> None:
        """Remember the exact audio window that produced the published prompt."""
        frozen = self._freeze_window(window)
        self._anchor_audio = frozen
        self._anchor_embed = self._embed_window(frozen)

    def _hold_similarity(
        self, potential_window: np.ndarray, hold_sim_power: float
    ) -> tuple[float, float]:
        """Compare a candidate Qwen window against the published prompt window."""
        if self._anchor_audio is None or self._anchor_audio.size == 0:
            return 1.0, 1.0
        potential_embed = self._embed_window(potential_window)
        anchor_embed = self._anchor_embed
        if anchor_embed is None:
            anchor_embed = self._embed_window(self._anchor_audio)
            self._anchor_embed = anchor_embed
        if potential_embed is None or anchor_embed is None:
            return 1.0, 1.0
        cosine = float(np.dot(potential_embed.numpy(), anchor_embed.numpy()))
        return cosine, self._hold_score(cosine, hold_sim_power)

    def _filter_caption(self, caption: str) -> Optional[str]:
        caption = (caption or "").strip()
        if not caption:
            return ""
        filtered = filter_prompt_addition(
            caption,
            self.config.prompt_blacklist,
            self.config.prompt_blacklist_mode,
            whole_words=self.config.prompt_blacklist_whole_words,
        )
        if filtered is None:
            if caption:
                logger.info(
                    "Qwen caption rejected by blacklist (%s): %s",
                    self.config.prompt_blacklist_mode,
                    caption[:120],
                )
            return None
        if filtered != caption:
            logger.debug(
                "Qwen caption filtered: %s -> %s",
                caption[:120],
                filtered[:120],
            )
        return filtered

    def _prepare_composed(self, caption: str) -> Optional[str]:
        """Apply blacklist to a Qwen caption and merge with the exempt base prompt."""
        caption = (caption or "").strip()
        if not caption:
            return self._base.current().strip()
        composed = filter_composed_addition(
            self._base.current(),
            caption,
            self.config.prompt_blacklist,
            self.config.prompt_blacklist_mode,
            whole_words=self.config.prompt_blacklist_whole_words,
        )
        if composed is None and caption:
            logger.info(
                "Qwen caption rejected by blacklist (%s): %s",
                self.config.prompt_blacklist_mode,
                caption[:120],
            )
        return composed

    def _caption_from_composed(self, composed: str) -> str:
        """Extract the Qwen caption portion from a composed prompt string."""
        anchor = self._base.current().strip()
        composed = (composed or "").strip()
        if not anchor:
            return composed
        if composed == anchor:
            return ""
        prefix = f"{anchor}, "
        if composed.startswith(prefix):
            return composed[len(prefix):]
        return composed

    def _publish_prompt(self, composed: str) -> bool:
        """Push a filtered composed prompt to the generation loop (instant or blend)."""
        composed = (composed or "").strip()
        if not composed:
            return False
        with self._publish_lock:
            if self._blend is not None:
                with self._pipeline_lock:
                    pipeline = self._pipeline
                if pipeline is None:
                    self._last_composed = composed
                    if not self._clap_composes:
                        self._active.set(composed)
                    return True
                to_embeds = pipeline.compute_embeds(composed)
                if to_embeds is None:
                    logger.warning(
                        "Prompt embed encoding failed; keeping previous prompt/embeds."
                    )
                    return False
                if composed == self._last_composed:
                    self._blend.snap(to_embeds)
                    if not self._clap_composes:
                        self._active.set(composed)
                    return True
                from_embeds = pipeline.compute_embeds(self._last_composed)
                if from_embeds is None:
                    self._blend.snap(to_embeds)
                else:
                    if self._blend.is_blending():
                        mid = self._blend.current_embeds()
                        if mid is not None:
                            from_embeds = mid
                    duration = float(self.config.qwen_blend_sec)
                    self._blend.start_blend(from_embeds, to_embeds, duration)
                    logger.debug(
                        "Qwen blend started (%.1fs): %s -> %s",
                        duration, self._last_composed[:60], composed[:60],
                    )
                self._last_composed = composed
            else:
                self._last_composed = composed

            if not self._clap_composes:
                self._active.set(composed)
            return True

    def _try_publish_caption(
        self,
        caption: str,
        anchor_window: Optional[np.ndarray],
        *,
        hold_active: bool,
    ) -> bool:
        composed = self._prepare_composed(caption)
        if composed is None:
            return False
        if not self._publish_prompt(composed):
            return False
        filtered_caption = self._caption_from_composed(composed)
        if hold_active and anchor_window is not None and anchor_window.size > 0:
            self._set_anchor_window(anchor_window)
            self._last_infer_time = time.time()
            with self._lock:
                self._last_sim = 1.0
                self._held = False
                self._pending_caption = ""
                self._pending_audio = None
        with self._lock:
            self._last_caption = filtered_caption
            self._status = "ok"
        return True

    def _store_prefetch_caption(self, caption: str, window: np.ndarray) -> None:
        """Cache a hold-prefetch caption after blacklist filtering."""
        filtered = self._filter_caption(caption)
        with self._lock:
            if filtered is None:
                self._pending_caption = ""
                self._pending_audio = None
                return
            self._pending_caption = filtered
            self._pending_audio = self._freeze_window(window)
            self._status = "prefetch"

    def _record_inference(self, caption: str, infer_ms: float) -> None:
        with self._lock:
            self._infer_ms = infer_ms
            self._infer_ms_ema = (
                infer_ms if self._infer_ms_ema <= 0.0
                else 0.8 * self._infer_ms_ema + 0.2 * infer_ms
            )

    def _publish_caption(
        self,
        caption: str,
        anchor_window: Optional[np.ndarray],
        *,
        hold_active: bool,
    ) -> None:
        self._try_publish_caption(caption, anchor_window, hold_active=hold_active)

    def _run(self) -> None:
        hold_active = bool(self.config.qwen_hold and self._embedder is not None)
        logger.info(
            "Qwen orchestrator started (window=%.1fs interval=%.1fs blend=%s hold=%s prefetch=%s).",
            float(self.config.qwen_window_sec),
            max(0.0, float(self.config.qwen_min_interval_sec)),
            self._blend is not None,
            hold_active,
            bool(self.config.qwen_hold_prefetch and hold_active),
        )
        while not self._shutdown.is_set() and not self._stop.is_set():
            cfg = self.config
            window_sec = float(cfg.qwen_window_sec)
            min_interval = max(0.0, float(cfg.qwen_min_interval_sec))
            min_audio = max(1.0, window_sec * 0.5)
            hold_max_sec = max(0.0, float(cfg.qwen_hold_max_sec))
            hold_check_interval = max(0.1, float(cfg.qwen_hold_check_interval_sec))
            prefetch_active = bool(cfg.qwen_hold_prefetch and hold_active)
            loop_start = time.time()
            hold_threshold = float(cfg.qwen_hold_threshold)
            hold_sim_power = max(0.0, float(cfg.qwen_hold_sim_power))
            window = self._ring.read_latest_seconds(window_sec)
            audio_sec = window.shape[0] / float(self._ring.sample_rate)
            with self._lock:
                self._audio_sec = audio_sec

            if audio_sec < min_audio:
                with self._lock:
                    self._status = "buffering"
                self._shutdown.wait(timeout=0.5)
                continue

            holding = False
            post_wait = min_interval

            if (
                hold_active
                and self._anchor_audio is not None
                and self._anchor_audio.size > 0
            ):
                cosine, score = self._hold_similarity(window, hold_sim_power)
                now = time.time()
                held_sec = (
                    now - self._last_infer_time
                    if self._last_infer_time > 0.0
                    else 0.0
                )
                sr = float(self._ring.sample_rate)
                anchor_sec = self._anchor_audio.shape[0] / sr
                cand_sec = window.shape[0] / sr
                timed_out = held_sec >= hold_max_sec
                will_hold = score >= hold_threshold and not timed_out
                logger.info(
                    "Qwen hold check: cos=%.3f score=%.3f (pow=%.1f thr=%.2f) "
                    "anchor=%.1fs cand=%.1fs age=%.1fs%s -> %s",
                    cosine, score, hold_sim_power, hold_threshold,
                    anchor_sec, cand_sec, held_sec,
                    " TIMEOUT" if timed_out else "",
                    "hold" if will_hold else "switch",
                )
                if will_hold:
                    holding = True
                    post_wait = hold_check_interval
                    with self._lock:
                        self._last_cosine = cosine
                        self._last_sim = score
                        self._held = True
                        self._status = "hold"
                else:
                    with self._lock:
                        self._last_cosine = cosine
                        self._last_sim = score
                        self._held = False
                    if prefetch_active:
                        with self._lock:
                            pending = self._pending_caption
                            pending_audio = self._pending_audio
                        if pending and pending_audio is not None and pending_audio.size > 0:
                            try:
                                if self._try_publish_caption(
                                    pending,
                                    pending_audio,
                                    hold_active=hold_active,
                                ):
                                    logger.info(
                                        "Qwen hold switch (prefetch, sim=%.3f score=%.3f): %s",
                                        cosine, score, pending[:120],
                                    )
                                elapsed = time.time() - loop_start
                                wait = min_interval - elapsed
                                if wait > 0:
                                    self._shutdown.wait(timeout=wait)
                                continue
                            except Exception as exc:
                                logger.warning(
                                    "Qwen prefetch publish failed (%s); inferring.",
                                    exc,
                                )
                        elif not holding:
                            logger.info(
                                "Qwen hold switch (sim=%.3f score=%.3f); inferring.",
                                cosine, score,
                            )

            if holding and not prefetch_active:
                elapsed = time.time() - loop_start
                wait = post_wait - elapsed
                if wait > 0:
                    self._shutdown.wait(timeout=wait)
                continue

            if holding and prefetch_active:
                now = time.time()
                if now - self._last_prefetch_time < min_interval:
                    elapsed = time.time() - loop_start
                    wait = post_wait - elapsed
                    if wait > 0:
                        self._shutdown.wait(timeout=wait)
                    continue
                try:
                    t0 = time.perf_counter()
                    caption = self._client.caption(
                        window,
                        self._ring.sample_rate,
                        cfg.qwen_instruction,
                        cfg.qwen_max_new_tokens,
                    )
                    infer_ms = (time.perf_counter() - t0) * 1000.0
                    self._last_prefetch_time = time.time()
                    self._store_prefetch_caption(caption, window)
                    self._record_inference(caption, infer_ms)
                    with self._lock:
                        stored = self._pending_caption
                    logger.info(
                        "Qwen prefetch (%.0fms): %s",
                        infer_ms,
                        stored[:120] if stored else "(rejected by blacklist)",
                    )
                except Exception as exc:
                    logger.warning("Qwen prefetch failed (%s); retrying.", exc)
                    with self._lock:
                        self._status = "error"
                    self._shutdown.wait(timeout=2.0)
                elapsed = time.time() - loop_start
                wait = post_wait - elapsed
                if wait > 0:
                    self._shutdown.wait(timeout=wait)
                continue

            try:
                t0 = time.perf_counter()
                caption = self._client.caption(
                    window,
                    self._ring.sample_rate,
                    cfg.qwen_instruction,
                    cfg.qwen_max_new_tokens,
                )
                infer_ms = (time.perf_counter() - t0) * 1000.0
                published = self._try_publish_caption(
                    caption,
                    window if hold_active else None,
                    hold_active=hold_active,
                )
                self._record_inference(caption, infer_ms)
                with self._lock:
                    shown = self._last_caption
                if published:
                    logger.info(
                        "Qwen caption (%.0fms): %s",
                        infer_ms,
                        shown[:120] if shown else caption[:120],
                    )
                else:
                    logger.info(
                        "Qwen caption (%.0fms) rejected by blacklist: %s",
                        infer_ms,
                        caption[:120],
                    )
            except Exception as exc:
                logger.warning("Qwen caption failed (%s); retrying.", exc)
                with self._lock:
                    self._status = "error"
                self._shutdown.wait(timeout=2.0)
                continue

            if post_wait > 0:
                elapsed = time.time() - loop_start
                wait = post_wait - elapsed
                if wait > 0:
                    self._shutdown.wait(timeout=wait)
        logger.info("Qwen orchestrator stopped.")


class CompositeOrchestrator:
    """Concatenate overlay lines from multiple orchestrators."""

    def __init__(self, *orchestrators) -> None:
        self._orchestrators = orchestrators

    def overlay_lines(self) -> List[str]:
        lines: List[str] = []
        for orch in self._orchestrators:
            if orch is not None:
                lines.extend(orch.overlay_lines())
        return lines
