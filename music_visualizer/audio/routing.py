"""User-configurable audio routing matrix for feedback motion.

Each :class:`Route` binds an audio *source* (optionally scaled by a second
*amp* source) to a motion *target* with gain, curve, and accumulation mode.
The default route set reproduces the legacy hard-wired :class:`ReactiveMapper`
behavior.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from music_visualizer.audio.features import AudioFeatures
from music_visualizer.config import Config
from music_visualizer.utils.logging import get_logger

logger = get_logger(__name__)

ROUTES_GROUP = "Audio Routes"


def _ema_alpha(dt: float, tau: float) -> float:
    """Time-constant EMA coefficient: fraction to move toward target over ``dt``."""
    if tau <= 1e-6:
        return 1.0
    return 1.0 - math.exp(-max(0.0, dt) / tau)


def bpm_noise_multiplier(
    bpm: float,
    pivot: float = 80.0,
    mult_min: float = 0.8,
    mult_max: float = 1.2,
    high_bpm: float = 200.0,
) -> float:
    """Map tempo to a direct noise scale in ``[mult_min, mult_max]``."""
    if bpm <= 0:
        return 1.0
    if bpm < pivot:
        t = bpm / pivot
        return mult_min + t * (1.0 - mult_min)
    if bpm >= high_bpm:
        return mult_max
    t = (bpm - pivot) / (high_bpm - pivot)
    return 1.0 + t * (mult_max - 1.0)


class Source(str, Enum):
    NONE = "none"
    ONE = "one"
    RMS = "rms"
    BASS = "bass"
    MID = "mid"
    TREBLE = "treble"
    ONSET = "onset"
    CENTROID = "centroid"
    ROLLOFF = "rolloff"
    FLUX = "flux"
    SPREAD = "spread"
    ZCR = "zcr"
    PITCH = "pitch"
    PITCH_CONF = "pitch_conf"
    BPM_MAPPED = "bpm_mapped"
    BPM_NOISE_MULT = "bpm_noise_mult"
    BEAT = "beat"
    BEAT_BASS = "beat_bass"
    BEAT_LOUD = "beat_loud"
    ORBIT_SIN = "orbit_sin"
    ORBIT_COS = "orbit_cos"


class Target(str, Enum):
    NONE = "none"
    ZOOM = "zoom"
    PAN_X = "pan_x"
    PAN_Y = "pan_y"
    ROTATE = "rotate"
    NOISE_DRIFT = "noise_drift"


class RouteMode(str, Enum):
    ADD = "add"
    MUL = "mul"
    MAX = "max"


SOURCE_OPTIONS: tuple[str, ...] = tuple(s.value for s in Source)
TARGET_OPTIONS: tuple[str, ...] = tuple(t.value for t in Target if t != Target.NONE)
ROUTE_MODE_OPTIONS: tuple[str, ...] = tuple(m.value for m in RouteMode)

# Continuous sources smoothed with react_smoothing_tau.
_SMOOTH_SOURCES = frozenset({
    Source.RMS, Source.BASS, Source.MID, Source.TREBLE, Source.ONSET,
    Source.CENTROID, Source.ROLLOFF, Source.FLUX, Source.SPREAD,
    Source.ZCR, Source.PITCH, Source.PITCH_CONF,
})


@dataclass
class Route:
    """A single source -> target binding.

    The *structural* fields (``source``, ``amp``, ``target``, ``curve``,
    ``mode``) define what the route does and are edited in the external routes
    file. The *runtime* fields (``gain``, ``enable``) are exposed as labeled UI
    controls. The remaining fields are UI presentation hints.
    """

    # Structural (defined in the routes file).
    source: str = Source.NONE.value
    amp: str = Source.ONE.value
    target: str = Target.ZOOM.value
    curve: float = 1.0
    mode: str = RouteMode.ADD.value
    # Runtime-tunable (exposed in the UI).
    gain: float = 0.0
    enable: bool = False
    # UI presentation hints.
    label: str = ""
    tooltip: str = ""
    group: str = ROUTES_GROUP
    gain_min: float = -1.0
    gain_max: float = 1.0
    gain_step: float = 0.001

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Route":
        return cls(
            source=str(data.get("source", Source.NONE.value)),
            amp=str(data.get("amp", Source.ONE.value)),
            target=str(data.get("target", Target.ZOOM.value)),
            curve=float(data.get("curve", 1.0)),
            mode=str(data.get("mode", RouteMode.ADD.value)),
            gain=float(data.get("gain", 0.0)),
            enable=bool(data.get("enable", False)),
            label=str(data.get("label", "")),
            tooltip=str(data.get("tooltip", "")),
            group=str(data.get("group", ROUTES_GROUP)),
            gain_min=float(data.get("gain_min", -1.0)),
            gain_max=float(data.get("gain_max", 1.0)),
            gain_step=float(data.get("gain_step", 0.001)),
        )

    def display_label(self, index: int) -> str:
        return self.label or f"Route {index + 1}"


def normalize_routes(routes: Optional[List[dict]]) -> List[dict]:
    """Round-trip each route dict through :class:`Route` (fills defaults)."""
    if not routes:
        return []
    return [Route.from_dict(item).to_dict() for item in routes]


def build_default_routes(cfg: Config) -> List[dict]:
    """Build the legacy-equivalent labeled route set from ``react_*`` scalars."""
    s = float(cfg.react_beat_bass_scale)
    beat_zoom = float(cfg.react_beat_zoom)
    routes: List[Route] = [
        Route(
            label="Bass -> Zoom", enable=True,
            source=Source.BASS.value, target=Target.ZOOM.value,
            gain=float(cfg.react_zoom_gain),
            gain_min=0.0, gain_max=0.05, gain_step=0.001,
            tooltip="Bass energy adds a steady forward zoom.",
        ),
        Route(
            label="Beat Zoom Punch", enable=True,
            source=Source.BEAT.value, amp=Source.ONE.value, target=Target.ZOOM.value,
            gain=beat_zoom * (1.0 - s),
            gain_min=0.0, gain_max=0.3, gain_step=0.005,
            tooltip="Flat zoom impulse on every detected beat.",
        ),
        Route(
            label="Beat Zoom Punch (bass-scaled)", enable=s > 1e-9,
            source=Source.BEAT.value, amp=Source.BEAT_BASS.value, target=Target.ZOOM.value,
            gain=beat_zoom * s,
            gain_min=0.0, gain_max=0.3, gain_step=0.005,
            tooltip="Zoom impulse scaled by bass level at the beat hit.",
        ),
        Route(
            label="Onset -> Noise", enable=True,
            source=Source.ONSET.value, target=Target.NOISE_DRIFT.value,
            gain=float(cfg.react_noise_gain),
            gain_min=0.0, gain_max=2.0, gain_step=0.05,
            tooltip="Transients add noise drift (more re-imagining on attacks).",
        ),
        Route(
            label="Tempo Noise Scale", enable=True,
            source=Source.BPM_NOISE_MULT.value, amp=Source.ONE.value,
            target=Target.NOISE_DRIFT.value, mode=RouteMode.MUL.value,
            gain=1.0,
            gain_min=0.0, gain_max=2.0, gain_step=0.05,
            tooltip="Multiplies noise drift by the tempo curve (slow dips, fast lifts).",
        ),
        Route(
            label="Treble -> Rotate", enable=True,
            source=Source.TREBLE.value, target=Target.ROTATE.value,
            gain=float(cfg.react_rotate_gain),
            gain_min=-0.2, gain_max=0.2, gain_step=0.005,
            tooltip="Brightness drives rotation speed.",
        ),
        Route(
            label="Loud Beat -> Rotate", enable=True,
            source=Source.BEAT_LOUD.value, target=Target.ROTATE.value,
            gain=float(cfg.react_beat_rotate),
            gain_min=0.0, gain_max=0.2, gain_step=0.005,
            tooltip="Extra rotation on loud beats in the current spin direction.",
        ),
        Route(
            label="Mid Pan Orbit X", enable=True,
            source=Source.ORBIT_COS.value, amp=Source.MID.value, target=Target.PAN_X.value,
            gain=float(cfg.react_pan_gain),
            gain_min=0.0, gain_max=0.1, gain_step=0.001,
            tooltip="Mid energy drives a circular pan (horizontal component).",
        ),
        Route(
            label="Mid Pan Orbit Y", enable=True,
            source=Source.ORBIT_SIN.value, amp=Source.MID.value, target=Target.PAN_Y.value,
            gain=float(cfg.react_pan_gain),
            gain_min=0.0, gain_max=0.1, gain_step=0.001,
            tooltip="Mid energy drives a circular pan (vertical component).",
        ),
        Route(
            label="Tempo -> Zoom (constant)", enable=False,
            source=Source.BPM_MAPPED.value, target=Target.ZOOM.value,
            gain=0.005,
            gain_min=0.0, gain_max=0.05, gain_step=0.001,
            tooltip="Constant zoom that scales with detected tempo. Disabled by default.",
        ),
    ]
    return [r.to_dict() for r in routes]


# --- Routes file IO --------------------------------------------------------

def routes_file_path(cfg: Config) -> Path:
    """Resolve the external routes file path (cfg override or data dir)."""
    if getattr(cfg, "react_routes_path", None):
        return Path(cfg.react_routes_path)
    from music_visualizer.utils.paths import data_dir

    return data_dir() / "audio_routes.json"


def load_routes_file(path: Path) -> Optional[List[dict]]:
    """Load and normalize routes from ``path``; return None if missing/invalid."""
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read routes file %s (%s); using defaults.", path, exc)
        return None
    routes = data.get("routes") if isinstance(data, dict) else data
    if not isinstance(routes, list):
        logger.warning("Routes file %s has no 'routes' list; using defaults.", path)
        return None
    return normalize_routes(routes)


def save_routes_file(path: Path, routes: List[dict]) -> Path:
    """Write the routes list to ``path`` as ``{"routes": [...]}``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": (
            "Audio routing matrix. Each route binds a 'source' (optionally "
            "scaled by 'amp') to a motion 'target'. Edit structure here; tune "
            "'gain'/'enable' live in the UI. See docs/audio_routing.md."
        ),
        "routes": normalize_routes(routes),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


def parse_route_key(key: str) -> Optional[tuple[int, str]]:
    """Parse ``route3_gain`` -> ``(3, 'gain')``. Only gain/enable are UI-exposed."""
    if not key.startswith("route") or len(key) < 6:
        return None
    rest = key[5:]
    idx_str = ""
    for ch in rest:
        if ch.isdigit():
            idx_str += ch
        else:
            break
    if not idx_str or not rest[len(idx_str):].startswith("_"):
        return None
    field_name = rest[len(idx_str) + 1:]
    if field_name not in ("enable", "gain"):
        return None
    return int(idx_str), field_name


def route_field_value(cfg: Config, index: int, field_name: str) -> Any:
    routes = normalize_routes(cfg.react_routes)
    if index >= len(routes):
        return None
    route = Route.from_dict(routes[index])
    return getattr(route, field_name)


def set_route_field(cfg: Config, index: int, field_name: str, value: Any) -> None:
    routes = normalize_routes(cfg.react_routes)
    if index >= len(routes):
        return
    route = Route.from_dict(routes[index])
    if field_name == "enable":
        value = bool(value)
    elif field_name in ("source", "amp", "target", "mode"):
        value = str(value)
    elif field_name in ("gain", "curve"):
        value = float(value)
    setattr(route, field_name, value)
    routes[index] = route.to_dict()
    cfg.react_routes = routes


def _shape(value: float, curve: float) -> float:
    c = max(1e-6, float(curve))
    v = max(0.0, float(value))
    if abs(c - 1.0) < 1e-6:
        return v
    return v ** c


def _bpm_mapped(bpm: float, lo: float, hi: float) -> float:
    if bpm <= 0 or hi <= lo:
        return 0.0
    t = (bpm - lo) / (hi - lo)
    return max(0.0, min(1.0, t))


class RoutingMapper:
    """Stateful routing engine. Call :meth:`update` once per generated frame."""

    def __init__(self, config: Config):
        self.config = config
        self._base_zoom = float(config.feedback_zoom or 1.0)
        self._base_noise = float(config.feedback_noise_drift or 0.0)
        self._base_pan_x = float(config.feedback_pan_x)
        self._base_pan_y = float(config.feedback_pan_y)
        self._base_rotate = float(config.feedback_rotate)

        self._smooth_tau = float(config.react_smoothing_tau)
        self._beat_tau = float(config.react_beat_decay_tau)
        self._orbit_rate = float(config.react_pan_orbit_rate)

        self._smooth: Dict[str, float] = {s.value: 0.0 for s in _SMOOTH_SOURCES}
        self._beat_env = 0.0
        self._beat_bass = 1.0
        self._beat_rotate_env = 0.0
        self._beat_rotate_dir = 1.0
        self._phase = 0.0
        self._last_beat_ts = -1.0
        self._bpm_noise_mult = 1.0
        self._bpm = 0.0

        self._routes: List[Route] = []
        self.sync_from_config()

    def set_base(self, **kwargs) -> None:
        if "zoom" in kwargs and kwargs["zoom"] is not None:
            self._base_zoom = float(kwargs["zoom"])
        if "pan_x" in kwargs and kwargs["pan_x"] is not None:
            self._base_pan_x = float(kwargs["pan_x"])
        if "pan_y" in kwargs and kwargs["pan_y"] is not None:
            self._base_pan_y = float(kwargs["pan_y"])
        if "rotate" in kwargs and kwargs["rotate"] is not None:
            self._base_rotate = float(kwargs["rotate"])
        if "noise_drift" in kwargs and kwargs["noise_drift"] is not None:
            self._base_noise = float(kwargs["noise_drift"])

    def sync_from_config(self) -> None:
        cfg = self.config
        self._smooth_tau = float(cfg.react_smoothing_tau)
        self._beat_tau = float(cfg.react_beat_decay_tau)
        self._orbit_rate = float(cfg.react_pan_orbit_rate)
        routes_data = normalize_routes(cfg.react_routes)
        cfg.react_routes = routes_data
        self._routes = [Route.from_dict(r) for r in routes_data]

    def _update_smoothed(self, features: AudioFeatures, dt: float) -> None:
        cfg = self.config
        a = _ema_alpha(dt, self._smooth_tau)
        mapping = {
            Source.RMS: features.rms,
            Source.BASS: features.bass,
            Source.MID: features.mid,
            Source.TREBLE: features.treble,
            Source.ONSET: features.onset_strength,
            Source.CENTROID: features.spectral_centroid,
            Source.ROLLOFF: features.spectral_rolloff,
            Source.FLUX: features.spectral_flux,
            Source.SPREAD: features.spectral_spread,
            Source.ZCR: features.zcr,
            Source.PITCH: features.pitch,
            Source.PITCH_CONF: features.pitch_confidence,
        }
        for src, raw in mapping.items():
            key = src.value
            self._smooth[key] += a * (float(raw) - self._smooth[key])

        target_bpm_mult = bpm_noise_multiplier(
            features.bpm,
            pivot=cfg.react_bpm_noise_pivot,
            mult_min=cfg.react_bpm_noise_mult_min,
            mult_max=cfg.react_bpm_noise_mult_max,
            high_bpm=cfg.react_bpm_noise_high_bpm,
        )
        self._bpm_noise_mult += a * (target_bpm_mult - self._bpm_noise_mult)
        self._bpm = float(features.bpm)

        if features.is_beat and features.timestamp != self._last_beat_ts:
            self._beat_env = 1.0
            self._beat_bass = max(0.0, min(1.0, float(features.bass)))
            self._last_beat_ts = features.timestamp
            loud = (
                features.rms >= cfg.react_beat_loud_rms
                and features.mid >= cfg.react_beat_loud_mid
                and features.treble >= cfg.react_beat_loud_treb
            )
            if loud:
                trend = self._base_rotate + self._smooth[Source.TREBLE.value] * float(
                    cfg.react_rotate_gain
                )
                self._beat_rotate_dir = (
                    math.copysign(1.0, trend) if abs(trend) > 1e-6 else 1.0
                )
                self._beat_rotate_env = 1.0

    def _source_values(self) -> Dict[str, float]:
        cfg = self.config
        return {
            Source.NONE.value: 0.0,
            Source.ONE.value: 1.0,
            Source.RMS.value: self._smooth[Source.RMS.value],
            Source.BASS.value: self._smooth[Source.BASS.value],
            Source.MID.value: self._smooth[Source.MID.value],
            Source.TREBLE.value: self._smooth[Source.TREBLE.value],
            Source.ONSET.value: self._smooth[Source.ONSET.value],
            Source.CENTROID.value: self._smooth[Source.CENTROID.value],
            Source.ROLLOFF.value: self._smooth[Source.ROLLOFF.value],
            Source.FLUX.value: self._smooth[Source.FLUX.value],
            Source.SPREAD.value: self._smooth[Source.SPREAD.value],
            Source.ZCR.value: self._smooth[Source.ZCR.value],
            Source.PITCH.value: self._smooth[Source.PITCH.value],
            Source.PITCH_CONF.value: self._smooth[Source.PITCH_CONF.value],
            Source.BPM_MAPPED.value: _bpm_mapped(
                self._bpm, cfg.react_bpm_map_min, cfg.react_bpm_map_max,
            ),
            Source.BPM_NOISE_MULT.value: self._bpm_noise_mult,
            Source.BEAT.value: self._beat_env,
            Source.BEAT_BASS.value: self._beat_bass,
            Source.BEAT_LOUD.value: self._beat_rotate_dir * self._beat_rotate_env,
            Source.ORBIT_SIN.value: math.sin(self._phase),
            Source.ORBIT_COS.value: math.cos(self._phase),
        }

    def update(self, features: Optional[AudioFeatures], dt: float) -> dict:
        cfg = self.config
        dt = max(0.0, float(dt))

        if features is not None:
            self._update_smoothed(features, dt)

        decay = (
            math.exp(-dt / self._beat_tau) if self._beat_tau > 1e-6 else 0.0
        )
        self._beat_env *= decay
        self._beat_rotate_env *= decay
        self._phase += dt * self._orbit_rate

        sources = self._source_values()
        add = {t.value: 0.0 for t in Target if t != Target.NONE}
        mul = {t.value: 1.0 for t in Target if t != Target.NONE}
        peak = {t.value: 0.0 for t in Target if t != Target.NONE}
        has_peak = {t.value: False for t in Target if t != Target.NONE}

        for route in self._routes:
            if not route.enable:
                continue
            if route.source == Source.NONE.value or route.target == Target.NONE.value:
                continue
            src_val = sources.get(route.source, 0.0)
            amp_val = sources.get(route.amp, 1.0)
            contrib = float(route.gain) * _shape(src_val, route.curve) * amp_val
            tgt = route.target
            mode = route.mode
            if mode == RouteMode.MUL.value:
                mul[tgt] *= contrib
            elif mode == RouteMode.MAX.value:
                peak[tgt] = contrib if not has_peak[tgt] else max(peak[tgt], contrib)
                has_peak[tgt] = True
            else:
                add[tgt] += contrib

        zoom = self._base_zoom + add[Target.ZOOM.value]
        if has_peak[Target.ZOOM.value]:
            zoom += peak[Target.ZOOM.value]
        zoom *= mul[Target.ZOOM.value]

        pan_x = self._base_pan_x + add[Target.PAN_X.value]
        if has_peak[Target.PAN_X.value]:
            pan_x += peak[Target.PAN_X.value]
        pan_x *= mul[Target.PAN_X.value]

        pan_y = self._base_pan_y + add[Target.PAN_Y.value]
        if has_peak[Target.PAN_Y.value]:
            pan_y += peak[Target.PAN_Y.value]
        pan_y *= mul[Target.PAN_Y.value]

        rotate = self._base_rotate + add[Target.ROTATE.value]
        if has_peak[Target.ROTATE.value]:
            rotate += peak[Target.ROTATE.value]
        rotate *= mul[Target.ROTATE.value]

        noise = self._base_noise + add[Target.NOISE_DRIFT.value]
        if has_peak[Target.NOISE_DRIFT.value]:
            noise += peak[Target.NOISE_DRIFT.value]
        noise *= mul[Target.NOISE_DRIFT.value]

        zoom_min = self._base_zoom
        zoom_max = float(cfg.react_zoom_max)
        rot_max = float(cfg.react_rotate_max)
        pan_max = float(cfg.react_pan_max)

        return {
            "zoom": max(zoom_min, min(zoom_max, zoom)),
            "pan_x": max(-pan_max, min(pan_max, pan_x)),
            "pan_y": max(-pan_max, min(pan_max, pan_y)),
            "rotate": max(-rot_max, min(rot_max, rotate)),
            "noise_drift": max(0.0, min(1.0, noise)),
        }
