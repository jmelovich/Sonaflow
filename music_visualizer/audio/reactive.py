"""Maps :class:`AudioFeatures` onto Deforum feedback-motion parameters.

This module re-exports :class:`RoutingMapper` as :class:`ReactiveMapper` for
backward compatibility. See :mod:`music_visualizer.audio.routing` for the
user-configurable routing matrix.
"""

from __future__ import annotations

from music_visualizer.audio.routing import RoutingMapper, _ema_alpha, bpm_noise_multiplier


ReactiveMapper = RoutingMapper
