"""Settings UI controller: schema push, change apply, persistence, restart."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Set

from music_visualizer.config import Config
from music_visualizer.ui.persistence import (
    export_settings,
    import_settings,
    save_defaults,
    _apply_dict,
)
from music_visualizer.ui.settings_schema import (
    PARAM_SPECS,
    ApplyStrategy,
    ParamKind,
    ParamSpec,
    config_value_for_ui,
    spec_by_key,
    ui_value_to_config,
)
from music_visualizer.ui.ui_bridge import NativeUIBridge, UiAction
from music_visualizer.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Config keys that require rebuilding the audio/analysis stack.
_AUDIO_RESTART_KEYS: Set[str] = {
    "audio_source", "audio_file", "audio_device", "audio_blocksize",
    "sample_rate", "hop_size", "fft_size",
    "clap", "clap_model_id", "clap_device", "clap_vocab_path",
    "qwen", "qwen_model_id", "qwen_python", "qwen_4bit", "qwen_port", "qwen_hold",
}


@dataclass
class RestartCoordinator:
    """Signals in-process rebuild of generation and/or audio subsystems."""

    gen_restart: threading.Event = field(default_factory=threading.Event)
    audio_restart: threading.Event = field(default_factory=threading.Event)
    pending_gen: bool = False
    pending_audio: bool = False

    def mark_key(self, key: str) -> None:
        spec = spec_by_key().get(key)
        if spec is None or not spec.requires_restart:
            return
        if key in _AUDIO_RESTART_KEYS:
            self.pending_audio = True
        else:
            self.pending_gen = True

    @property
    def pending(self) -> bool:
        return self.pending_gen or self.pending_audio

    def trigger(self) -> None:
        if self.pending_gen:
            self.gen_restart.set()
        if self.pending_audio:
            self.audio_restart.set()
        self.pending_gen = False
        self.pending_audio = False

    def clear_pending(self) -> None:
        self.pending_gen = False
        self.pending_audio = False


@dataclass
class RuntimeContext:
    """Mutable handles the UI controller reads/writes each frame."""

    cfg: Config
    pipeline: Any = None
    mapper: Any = None
    base_prompt: Any = None
    analyzer_thread: Any = None
    audio_engine: Any = None
    clap_orchestrator: Any = None
    qwen_orchestrator: Any = None
    orchestrator: Any = None
    presenter: Any = None
    audio_state: Any = None
    restart: RestartCoordinator = field(default_factory=RestartCoordinator)
    on_audio_restart: Optional[Callable[[], None]] = None


class UIController:
    """Drives the native ImGui settings panel from Python."""

    _PROMPT_REPUBLISH_SEC = 0.35

    def __init__(self, ctx: RuntimeContext, bridge: Optional[NativeUIBridge] = None) -> None:
        self._bridge = bridge
        self._ctx = ctx
        self._specs = list(PARAM_SPECS)
        self._key_to_idx = {s.key: i for i, s in enumerate(self._specs)}
        self._pending_prompt_republish = False
        self._prompt_republish_at = 0.0
        if bridge is not None and bridge._available:
            bridge.set_schema(self._specs)
            self.push_all_defaults()
            self.push_all_values()
            bridge.set_restart_pending(ctx.restart.pending)
        elif bridge is not None and not bridge._available:
            logger.warning(
                "Native settings UI unavailable in glpresent_bridge.dll; "
                "rebuild with: python -m music_visualizer.display.native.build_glpresent --force"
            )

    @property
    def specs(self) -> list[ParamSpec]:
        return self._specs

    def push_all_values(self) -> None:
        if self._bridge is None or not self._bridge._available:
            return
        cfg = self._ctx.cfg
        for i, spec in enumerate(self._specs):
            self._bridge.push_value(i, config_value_for_ui(cfg, spec))

    def push_all_defaults(self) -> None:
        """Push Config dataclass defaults to native UI for double-click reset."""
        if self._bridge is None or not self._bridge._available:
            return
        defaults = Config()
        for i, spec in enumerate(self._specs):
            self._bridge.push_default(i, config_value_for_ui(defaults, spec))

    def reset_key(self, key: str) -> None:
        """Reset one setting to its Config default."""
        spec = spec_by_key().get(key)
        if spec is None:
            return
        defaults = Config()
        ui_val = config_value_for_ui(defaults, spec)
        self.apply_key(key, ui_val)
        if self._bridge is not None and self._bridge._available:
            idx = self._key_to_idx.get(key)
            if idx is not None:
                self._bridge.push_value(idx, ui_val)

    def apply_key(self, key: str, val: Any) -> None:
        """Apply one setting change (used by Tk fallback and tests)."""
        spec = spec_by_key().get(key)
        if spec is None:
            return
        self._set_config_value(spec, ui_value_to_config(spec, val))

    def clear_restart_pending(self) -> None:
        self._ctx.restart.clear_pending()
        if self._bridge is not None and self._bridge._available:
            self._bridge.set_restart_pending(False)

    def pump(self) -> None:
        """Poll native UI once per frame; apply changes and actions."""
        if self._bridge is None or not self._bridge._available:
            return
        self._apply_numeric_changes()
        self._apply_text_changes()
        self._mirror_live_prompt_text()
        self._flush_prompt_republish()
        self._handle_actions()
        if self._bridge is not None and self._bridge._available:
            self._bridge.set_restart_pending(self._ctx.restart.pending)

    def _apply_numeric_changes(self) -> None:
        for idx, raw in self._bridge.take_numeric_changes():
            spec = self._specs[idx]
            val = ui_value_to_config(spec, raw)
            self._set_config_value(spec, val)

    def _apply_text_changes(self) -> None:
        for idx in self._bridge.take_text_changes():
            spec = self._specs[idx]
            text = self._bridge.get_text(idx)
            val = ui_value_to_config(spec, text)
            if spec.key == "prompt":
                self._apply_prompt_change(str(val), immediate=True)
                continue
            self._set_config_value(spec, val)

    def _set_config_value(self, spec: ParamSpec, val: Any) -> None:
        old = getattr(self._ctx.cfg, spec.key)
        if old == val:
            return
        setattr(self._ctx.cfg, spec.key, val)
        logger.info("Setting %s -> %s", spec.key, val)

        if spec.apply == ApplyStrategy.RESTART:
            self._ctx.restart.mark_key(spec.key)
            return
        self._apply_live(spec.key, val)

    def _apply_live(self, key: str, val: Any) -> None:
        cfg = self._ctx.cfg
        ctx = self._ctx

        if key == "prompt":
            self._apply_prompt_change(str(val))
            return

        if key.startswith("prompt_blacklist"):
            if ctx.clap_orchestrator is not None:
                ctx.clap_orchestrator.sync_blacklist_from_config(cfg)
            self._republish_prompts()
            return

        if key == "negative_prompt" and ctx.pipeline is not None:
            if hasattr(ctx.pipeline, "update_negative_prompt"):
                ctx.pipeline.update_negative_prompt(str(val))
            return

        motion_keys = {
            "feedback_zoom": "zoom",
            "feedback_pan_x": "pan_x",
            "feedback_pan_y": "pan_y",
            "feedback_rotate": "rotate",
            "feedback_noise_drift": "noise_drift",
        }
        if key in motion_keys:
            kw = {motion_keys[key]: val}
            if ctx.mapper is not None:
                base_kw = {}
                if key == "feedback_zoom":
                    base_kw["zoom"] = val
                elif key == "feedback_pan_x":
                    base_kw["pan_x"] = val
                elif key == "feedback_pan_y":
                    base_kw["pan_y"] = val
                elif key == "feedback_rotate":
                    base_kw["rotate"] = val
                elif key == "feedback_noise_drift":
                    base_kw["noise_drift"] = val
                ctx.mapper.set_base(**base_kw)
            if ctx.pipeline is not None:
                ctx.pipeline.set_transform(**kw)
            return

        if key.startswith("react_") and ctx.mapper is not None:
            ctx.mapper.sync_from_config()
            return

        if key.startswith("beat_") and ctx.analyzer_thread is not None:
            ctx.analyzer_thread.sync_from_config(cfg)
            return

        if key.startswith("clap_") and ctx.clap_orchestrator is not None:
            ctx.clap_orchestrator.sync_from_config(cfg)
            return

        if key.startswith("center_") or key == "content_centering":
            if ctx.pipeline is not None and hasattr(ctx.pipeline, "sync_centering"):
                ctx.pipeline.sync_centering(cfg)
            return

        if key == "show_overlay":
            cfg.show_overlay = bool(val)
            return

        if key == "log_level":
            setup_logging(str(val))
            return

        if key == "fullscreen" and ctx.presenter is not None:
            if hasattr(ctx.presenter, "set_fullscreen"):
                ctx.presenter.set_fullscreen(bool(val))
            return

    def _apply_prompt_change(self, text: str, *, immediate: bool = False) -> None:
        """Push a manual base-prompt edit into shared state and orchestrators."""
        ctx = self._ctx
        if ctx.base_prompt is not None:
            ctx.base_prompt.set(text, allow_empty=True)
        ctx.cfg.prompt = text
        logger.info("Setting prompt -> %s", text[:120] if text else "(empty)")
        if immediate:
            self._republish_prompts()
            return
        self._pending_prompt_republish = True
        self._prompt_republish_at = time.monotonic() + self._PROMPT_REPUBLISH_SEC

    def _republish_prompts(self) -> None:
        """Re-compose the active generation prompt from the current base text."""
        ctx = self._ctx
        if ctx.clap_orchestrator is not None:
            ctx.clap_orchestrator.republish_from_base()
        elif ctx.qwen_orchestrator is not None:
            ctx.qwen_orchestrator.republish_from_base()

    def _flush_prompt_republish(self) -> None:
        if not self._pending_prompt_republish:
            return
        if time.monotonic() < self._prompt_republish_at:
            return
        self._pending_prompt_republish = False
        self._republish_prompts()

    def _mirror_live_prompt_text(self) -> None:
        """Keep cfg/base in sync while the prompt field is being edited."""
        if self._bridge is None or not self._bridge._available:
            return
        idx = self._key_to_idx.get("prompt")
        if idx is None:
            return
        text = self._bridge.get_text(idx)
        val = ui_value_to_config(self._specs[idx], text)
        if self._ctx.cfg.prompt == val:
            return
        self._apply_prompt_change(str(val))

    def _sync_text_from_ui(self) -> None:
        """Copy current text/enum widget contents into cfg (even mid-edit)."""
        if self._bridge is None or not self._bridge._available:
            return
        for i, spec in enumerate(self._specs):
            if spec.kind not in (ParamKind.TEXT, ParamKind.ENUM):
                continue
            text = self._bridge.get_text(i)
            val = ui_value_to_config(spec, text)
            setattr(self._ctx.cfg, spec.key, val)

    def _handle_actions(self) -> None:
        if self._bridge is None or not self._bridge._available:
            return
        actions = self._bridge.take_actions()
        if not actions:
            return
        if actions & UiAction.SAVE_DEFAULTS:
            self._sync_text_from_ui()
            path = save_defaults(self._ctx.cfg)
            logger.info("Saved defaults to %s", path)
        if actions & UiAction.EXPORT:
            self._sync_text_from_ui()
            path = self._bridge.open_file_dialog(save=True)
            if path:
                export_settings(self._ctx.cfg, __import__("pathlib").Path(path))
                logger.info("Exported settings to %s", path)
        if actions & UiAction.IMPORT:
            path = self._bridge.open_file_dialog(save=False)
            if path:
                data = import_settings(__import__("pathlib").Path(path))
                _apply_dict(self._ctx.cfg, data)
                self.push_all_values()
                self._flag_restart_for_all_restart_fields()
                logger.info("Imported settings from %s", path)
        if actions & UiAction.RESTART:
            self._ctx.restart.trigger()

    def _flag_restart_for_all_restart_fields(self) -> None:
        for spec in self._specs:
            if spec.requires_restart:
                self._ctx.restart.mark_key(spec.key)

    def import_dict(self, data: dict) -> None:
        _apply_dict(self._ctx.cfg, data)
        self.push_all_values()
        self._flag_restart_for_all_restart_fields()
